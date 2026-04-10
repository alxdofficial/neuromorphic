"""Smoke test: run a few training steps and check for instability."""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.model import Model


def smoke_test(steps=20, bs=4, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    c = Config.tier_a()
    c.vocab_size = 32000
    c.validate()

    model = Model(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    print(f"Config: D={c.D}, D_n={c.D_n}, Cn={c.neurons_per_cell}, NC={c.N_cells}")
    print(f"  d_inner={c.d_inner}, Hmod={c.cell_mod_hidden}, BS={bs}, T={c.T}")
    print(f"  Params: {model.param_count()/1e6:.1f}M "
          f"(LM={model.lm_param_count()/1e6:.1f}M, Mem={model.memory_param_count()/1e6:.1f}M)")
    print()

    losses = []
    for step in range(steps):
        input_ids = torch.randint(1, c.vocab_size, (bs, c.T), device=device)
        target_ids = torch.randint(0, c.vocab_size, (bs, c.T), device=device)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = model.forward_chunk(input_ids, target_ids=target_ids)

        loss = result["loss"]
        aux = result["aux_loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # === Gradient diagnostics ===
        lm_grad_norm = torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0).item()
        mem_grad_norm = torch.nn.utils.clip_grad_norm_(model.memory.parameters(), 1.0).item()

        # Check for NaN/Inf
        has_nan = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"  WARNING: NaN/Inf gradient in {name}")
                    has_nan = True

        optimizer.step()
        model.detach_states()

        loss_val = loss.item()
        aux_val = aux.item()
        losses.append(loss_val)

        # === Per-step diagnostics ===
        mg = model.memory
        if mg.is_initialized:
            h_norm = mg.h.float().norm().item()
            msg_norm = mg.msg.float().norm().item()
            W_norm = mg.W.float().norm().item()
            W_sparsity = (mg.W.float().abs() < 1e-4).float().mean().item()
            W_max = mg.W.float().abs().max().item()
            decay_mean = mg.decay.float().mean().item()
            s_mem = mg.s_mem_live.float().mean().item()
            s_fast = mg.s_mem_ema_fast.float().mean().item()
        else:
            h_norm = msg_norm = W_norm = W_sparsity = W_max = 0
            decay_mean = s_mem = s_fast = 0

        # Check for explosions
        flags = []
        if loss_val != loss_val:  # NaN
            flags.append("NaN_LOSS")
        if loss_val > 100:
            flags.append("HIGH_LOSS")
        if lm_grad_norm > 10:
            flags.append("HIGH_LM_GRAD")
        if mem_grad_norm > 10:
            flags.append("HIGH_MEM_GRAD")
        if h_norm > 1000:
            flags.append("H_EXPLOSION")
        if W_max > 100:
            flags.append("W_EXPLOSION")
        if has_nan:
            flags.append("NaN_GRAD")

        flag_str = " ".join(flags) if flags else "OK"

        if step % 5 == 0 or flags:
            print(f"[{step:3d}] loss={loss_val:.3f} mem={aux_val:.4f} "
                  f"lm_gn={lm_grad_norm:.3f} mem_gn={mem_grad_norm:.3f} | "
                  f"h={h_norm:.1f} msg={msg_norm:.1f} W={W_norm:.1f} "
                  f"W_sp={W_sparsity:.2f} W_max={W_max:.3f} "
                  f"dec={decay_mean:.3f} s_mem={s_mem:.2f} fast={s_fast:.2f} | "
                  f"{flag_str}")

    print()
    print("=== Summary ===")
    print(f"Steps: {steps}")
    print(f"Loss: {losses[0]:.3f} → {losses[-1]:.3f} (delta={losses[-1]-losses[0]:.3f})")
    print(f"Loss trend: {'decreasing' if losses[-1] < losses[0] else 'increasing or flat'}")
    print(f"Min loss: {min(losses):.3f}, Max loss: {max(losses):.3f}")
    any_nan = any(l != l for l in losses)
    print(f"NaN detected: {any_nan}")
    print(f"Final h_norm: {h_norm:.1f}")
    print(f"Final W_norm: {W_norm:.1f}, W_max: {W_max:.3f}, W_sparsity: {W_sparsity:.2f}")
    print(f"Final s_mem: live={s_mem:.2f} fast={s_fast:.2f}")

    if any_nan:
        print("\nFAILED: NaN in loss")
        return False
    if max(losses) > 100:
        print("\nWARNING: Very high loss values")
    if h_norm > 1000 or W_max > 100:
        print("\nWARNING: Potential explosion in neuron state or W")
    print("\nPASSED: No critical issues detected")
    return True


if __name__ == "__main__":
    smoke_test()
