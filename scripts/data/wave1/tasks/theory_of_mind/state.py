"""Theory of Mind — scene + event-log state model."""

from __future__ import annotations

from dataclasses import dataclass, field


CHARACTER_NAMES = (
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Isla", "Jack", "Kira", "Leo", "Mia", "Noah", "Olive", "Paul",
)

OBJECTS = (
    "ball", "book", "key", "letter", "phone", "umbrella", "watch", "ring",
    "notebook", "envelope", "vase", "candle",
)

CONTAINERS = (
    "red box", "blue box", "wooden drawer", "metal cabinet",
    "wicker basket", "leather bag", "glass jar", "cardboard box",
    "tin can", "paper envelope", "plastic bin", "fabric pouch",
)


@dataclass
class TomEvent:
    """One observable event in the scene."""
    event_type: str            # "place" | "move" | "leave" | "return"
    actor: str                 # character performing the action
    obj: str | None = None     # object moved/placed (None for leave/return)
    src: str | None = None     # source container (for "move")
    dst: str | None = None     # destination container (for "place"/"move")
    witnesses: tuple[str, ...] = ()  # set at build-time


@dataclass
class TomScenario:
    scenario_idx: int
    chars: tuple[str, ...]
    objects: tuple[str, ...]
    containers: tuple[str, ...]
    events: tuple[TomEvent, ...] = ()
    # Derived ground truth (filled by build):
    final_location: dict[str, str] = field(default_factory=dict)  # obj -> container
    belief: dict[tuple[str, str], str] = field(default_factory=dict)
    # belief[(char, obj)] = container where char *thinks* obj is.
    # Set only after they witnessed at least one event for that obj.


def _present(in_room: set[str]) -> tuple[str, ...]:
    return tuple(sorted(in_room))


def build_scenario(
    rng, scenario_idx: int,
    *, n_chars: int = 2, n_objects: int = 1,
) -> TomScenario:
    """Build a 1st-order false-belief scene.

    Sequence (when n_objects=1, n_chars=2):
      1. char_A and char_B in the room
      2. char_A places object in container_X (both witness)
      3. char_A leaves (witnessed by char_B)
      4. char_B moves object from X to Y (only char_B witnesses)
      5. char_A returns

    After: char_A still thinks object is in X (false belief).
            char_B knows object is in Y (true belief).
    """
    n_chars = max(2, n_chars)
    n_containers_needed = 2 * n_objects
    chars = tuple(rng.sample(CHARACTER_NAMES, n_chars))
    objects = tuple(rng.sample(OBJECTS, n_objects))
    containers = tuple(rng.sample(CONTAINERS, n_containers_needed))

    in_room: set[str] = set(chars)
    events: list[TomEvent] = []
    final_location: dict[str, str] = {}
    belief: dict[tuple[str, str], str] = {}

    # Pick: for each object, char_A places it; char_A leaves; char_B moves it.
    # If multiple objects: each "moved object" gets a different (A, B) pair if possible.
    char_pairs: list[tuple[str, str]] = []
    for i in range(n_objects):
        # Rotate which char is the absent one.
        a = chars[i % n_chars]
        b = chars[(i + 1) % n_chars]
        char_pairs.append((a, b))

    for i, obj in enumerate(objects):
        a, b = char_pairs[i]
        c_init = containers[2 * i]
        c_final = containers[2 * i + 1]
        # 1. A places obj in c_init (witnesses = everyone currently in room)
        place_evt = TomEvent(
            event_type="place", actor=a, obj=obj, dst=c_init,
            witnesses=_present(in_room),
        )
        events.append(place_evt)
        for w in place_evt.witnesses:
            belief[(w, obj)] = c_init
        final_location[obj] = c_init

        # 2. A leaves (witnesses = current room)
        leave_evt = TomEvent(
            event_type="leave", actor=a, witnesses=_present(in_room),
        )
        events.append(leave_evt)
        in_room.discard(a)

        # 3. B moves obj from c_init → c_final
        move_evt = TomEvent(
            event_type="move", actor=b, obj=obj, src=c_init, dst=c_final,
            witnesses=_present(in_room),
        )
        events.append(move_evt)
        for w in move_evt.witnesses:
            belief[(w, obj)] = c_final
        final_location[obj] = c_final

        # 4. A returns
        return_evt = TomEvent(
            event_type="return", actor=a, witnesses=_present(in_room | {a}),
        )
        events.append(return_evt)
        in_room.add(a)

    return TomScenario(
        scenario_idx=scenario_idx,
        chars=chars,
        objects=objects,
        containers=containers,
        events=tuple(events),
        final_location=final_location,
        belief=belief,
    )


def config_space_size() -> int:
    """16 names × 12 objects × 10 containers, picking 2 chars × 1 object × 2 containers.

    P(16,2) × 12 × P(10,2) ≈ 240 × 12 × 90 ≈ 259,200 base scenes.
    With 2 objects: × ~4M more combinations. Use base = 260K to be conservative.
    """
    return 260_000
