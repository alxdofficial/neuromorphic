"""Benchmark sequential v11 with various optimization approaches."""

import torch
import torch.nn.functional as F
import time
import sys

from src.v11.config import V11Config
from src.v11.model import V11Model
from src.v11.memory_graph import CellMemoryGraph, _HAS_TRITON_GATHER, _combined_cell_border_gather

torch.set_float32_matmul_precision('high')
device = torch.device('cuda')

results = []

def bench(label, model, BS_list=[8, 16, 32, 48, 64]):
    config = model.config if hasattr(model, 'config') else model.memory.config
    T = config.T
    print(f'\n=== {label} ===')
    for BS in BS_list:
        try:
            model.initialize_states(BS)
            ids = torch.randint(0, 100, (BS, T), device=device)
            # Warmup
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model.forward_chunk(ids)
            out['logits'].sum().backward()
            model.detach_states()
            model.zero_grad()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            n = 3
            t0 = time.time()
            for _ in range(n):
                ids = torch.randint(0, 100, (BS, T), device=device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model.forward_chunk(ids)
                out['logits'].sum().backward()
                model.detach_states()
                model.zero_grad()
            torch.cuda.synchronize()
            elapsed = (time.time() - t0) / n
            tok = BS * T / elapsed
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f'  BS={BS}: {tok:.0f} tok/s, {elapsed*1000:.0f}ms, {vram:.1f} GB')
            results.append((label, BS, tok, vram))
        except torch.cuda.OutOfMemoryError:
            print(f'  BS={BS}: OOM')
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'  BS={BS}: ERROR {e}')
            torch.cuda.empty_cache()


# =====================================================
# Approach 0: Baseline (current sequential, no tricks)
# =====================================================
config = V11Config.tier_a()
config.validate()
model = V11Model(config).to(device)
model.train()
bench('0-Baseline sequential R=1', model)
del model; torch.cuda.empty_cache()

# =====================================================
# Approach 1: Gradient checkpointing (save VRAM for bigger BS)
# =====================================================
config = V11Config.tier_a()
config.validate()
model = V11Model(config).to(device)
model.train()

# Monkey-patch to add checkpointing
original_fs = model.memory.forward_segment.__func__

def forward_with_ckpt(self, cc_signals):
    """Wrap each token step in checkpoint."""
    import types
    # Use the original but with checkpoint
    BS = cc_signals.shape[0]
    T_seg = cc_signals.shape[1]
    NC = self.config.N_cells
    C = self.config.C_neurons
    D = self.config.D_neuron
    alpha = self.config.alpha

    if not self._initialized:
        raise RuntimeError("Not initialized")

    h = self.h.detach()
    msg = self.prev_messages.detach()

    w_conn, w_conn_border, decay_logit, primitives = self._run_modulator(
        h, self.hebbian_traces.detach(),
        self.decay_logit.detach(),
        self.primitives_state.detach())

    w_conn_sig = torch.sigmoid(w_conn)
    w_conn_border_sig = torch.sigmoid(w_conn_border)
    dt = h.dtype

    nid = self.neuron_id.to(dt)
    state_w_in = self.state_w1[:, :D].to(dt)
    state_w_prim = self.state_w1[:, D:2*D].to(dt)
    state_w_id = self.state_w1[:, 2*D:3*D].to(dt)
    state_w_decay = self.state_w1[:, 3*D:].to(dt)
    state_w2 = self.state_w2.to(dt)
    state_b1 = self.state_b1.to(dt)
    state_b2 = self.state_b2.to(dt)
    msg_w_h = self.msg_w1[:, :D].to(dt)
    msg_w_prim = self.msg_w1[:, D:2*D].to(dt)
    msg_w_id = self.msg_w1[:, 2*D:].to(dt)
    msg_w2 = self.msg_w2.to(dt)
    msg_b1 = self.msg_b1.to(dt)
    msg_b2 = self.msg_b2.to(dt)

    state_const = (
        F.linear(primitives, state_w_prim) +
        F.linear(nid, state_w_id).unsqueeze(0) +
        F.linear(decay_logit.unsqueeze(-1), state_w_decay) +
        state_b1
    )
    msg_const = (
        F.linear(primitives, msg_w_prim) +
        F.linear(nid, msg_w_id).unsqueeze(0) +
        msg_b1
    )

    decay = torch.sigmoid(decay_logit)
    d = decay.unsqueeze(-1)
    omd = 1 - d

    border_idx_exp = self.border_indices.unsqueeze(0).unsqueeze(-1).expand(BS, -1, -1, D)
    inject_idx_exp = self.inject_indices.unsqueeze(0).unsqueeze(-1).expand(BS, -1, -1, D)
    readout_idx_exp = self.readout_indices.unsqueeze(0).unsqueeze(-1).expand(BS, -1, -1, D)

    def _step(h_in, msg_in, inject_sig):
        if _HAS_TRITON_GATHER and msg_in.is_cuda:
            received = _combined_cell_border_gather(
                msg_in, w_conn_sig, self.conn_indices,
                w_conn_border_sig, self.border_conn_indices, alpha)
        else:
            received = self._cell_gather(msg_in, w_conn_sig)
            border_received = self._border_gather(msg_in, w_conn_border_sig)
            received = received.scatter_add(2, border_idx_exp, border_received.to(received.dtype))

        inject_addend = torch.zeros_like(received)
        inject_addend.scatter_(2, inject_idx_exp, inject_sig.to(received.dtype))
        input_vec = received + inject_addend

        hidden = torch.tanh(F.linear(input_vec, state_w_in) + state_const)
        update = torch.tanh(F.linear(hidden, state_w2, state_b2))
        h_out = d * h_in + omd * update

        msg_hidden = torch.tanh(F.linear(h_out, msg_w_h) + msg_const)
        msg_out = torch.tanh(F.linear(msg_hidden, msg_w2, msg_b2)) + nid
        return h_out, msg_out

    readouts = []
    for t in range(T_seg):
        inject_raw = cc_signals[:, t].reshape(BS, NC, D)
        inject_signal = inject_raw.unsqueeze(2).expand(-1, -1, alpha, -1)
        h, msg = torch.utils.checkpoint.checkpoint(
            _step, h, msg, inject_signal, use_reentrant=False)
        readout_msgs = torch.gather(msg, 2, readout_idx_exp)
        readouts.append(readout_msgs.mean(dim=2).reshape(BS, NC * D))

    mem_out = torch.stack(readouts, dim=1)

    with torch.no_grad():
        msg_norms = msg.detach().norm(dim=-1)
        self.h = h.detach().to(self.dtype)
        self.prev_messages = msg.detach().to(self.dtype)
        self.w_conn = w_conn.detach().to(self.dtype)
        self.w_conn_border = w_conn_border.detach().to(self.dtype)
        self.primitives_state = primitives.detach().to(self.dtype)
        self.decay_logit = decay_logit.detach().to(self.dtype)
        self.hebbian_traces = (msg_norms.unsqueeze(-1) * w_conn_sig.detach()).to(self.dtype)
        self.msg_magnitude = ((1 - 0.05) * self.msg_magnitude + 0.05 * msg_norms).to(self.dtype)

    return mem_out

import types
model.memory.forward_segment = types.MethodType(forward_with_ckpt, model.memory)
bench('1-Checkpointed sequential R=1', model, BS_list=[8, 16, 32, 48, 64, 96])
del model; torch.cuda.empty_cache()


# =====================================================
# Approach 2: Fewer neurons (C=64 instead of 124)
# =====================================================
# Fewer neurons = smaller tensors = bigger BS
config2 = V11Config.tier_a(C_neurons=64, cell_mod_hidden=24)
config2.validate()
model2 = V11Model(config2).to(device)
model2.train()
print(f'\n(C=64, N_total={config2.N_total}, params={model2.param_count()/1e6:.1f}M)')
bench('2-Fewer neurons C=64', model2, BS_list=[8, 16, 32, 48, 64])
del model2; torch.cuda.empty_cache()


# =====================================================
# Approach 3: Larger D_neuron (D=16, NC=128, fewer cells)
# =====================================================
config3 = V11Config.tier_a(D_neuron=16, N_cells=128, C_neurons=124, cell_mod_hidden=24)
config3.validate()
model3 = V11Model(config3).to(device)
model3.train()
print(f'\n(D=16, NC=128, N_total={config3.N_total}, params={model3.param_count()/1e6:.1f}M)')
bench('3-Wider neurons D=16 NC=128', model3, BS_list=[8, 16, 32, 48])
del model3; torch.cuda.empty_cache()


# =====================================================
# Approach 4: Shorter segments (T=64 instead of 128)
# =====================================================
config4 = V11Config.tier_a(T=64)
config4.validate()
model4 = V11Model(config4).to(device)
model4.train()
bench('4-Shorter segments T=64', model4, BS_list=[8, 16, 32, 48, 64])
del model4; torch.cuda.empty_cache()


# =====================================================
# Summary
# =====================================================
print('\n\n=== SUMMARY ===')
print(f'{"Approach":<40s} {"BS":>4s} {"tok/s":>8s} {"VRAM":>6s}')
print('-' * 62)
for label, bs, tok, vram in results:
    print(f'{label:<40s} {bs:>4d} {tok:>8.0f} {vram:>5.1f}G')
