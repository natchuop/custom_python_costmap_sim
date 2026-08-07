# Fixed scenario implementation report

The presets use the simulator-loaded `uint8` grid representation. All listed
points are free, have a free 3x3 neighbourhood, and are mutually reachable by
the shared four-neighbour A* planner.

| Preset | Shape | SHA-256 | Robot starts (0, 1, 2) | Delivery points |
| --- | --- | --- | --- | --- |
| `warehouse_002` | 188x192 | `b8f295ab98f5ebef3970dafa537a63468ed8c0f417ec5a47223328a90e447b3b` | (101,105), (37,15), (119,28) | (42,139), (66,63), (40,89), (123,133) |
| `warehouse_005` | 48x80 | `784921874e5fbaad904527cc019bf0d960b600820de1fb1800f075905804e523` | (36,35), (24,6), (16,52) | (14,25), (44,15), (45,51), (29,21) |
| `warehouse_005_rotated` | 35x52 | `6a10880abafdd65e8b0a38d3d4463164b87cd4fbbb873742cbd110df4ee5fe62` | (26,5), (4,48), (32,37) | (9,21), (14,39), (25,25), (25,49) |

Relevant A* path lengths (each start to the four delivery points, in delivery
point order) are:

- `warehouse_002`: robot 0 `[93,77,77,50]`; robot 1 `[177,77,129,204]`; robot 2 `[188,88,140,131]`.
- `warehouse_005`: robot 0 `[32,28,25,21]`; robot 1 `[29,29,66,20]`; robot 2 `[29,65,36,44]`.
- `warehouse_005_rotated`: robot 0 `[33,46,21,45]`; robot 1 `[32,19,44,26]`; robot 2 `[39,20,19,19]`.

Validation and task generation are seed-independent. The attacker receives one
deterministic navigation task because the existing legacy rollout requires a
task for every robot; it does not receive a benign delivery queue.

Verification:

- `python -m pytest -q`: **23 passed**.
- All three requested 100-step smoke commands completed successfully.
- Manifest-only authoring completed for `warehouse_005`.
- Four-method fairness replay completed with `--max-steps 100`; shared manifest
  fields and method configurations were asserted equal.
- The exact full-length comparison command was attempted but exceeded the
  120-second command limit; the bounded replay provides the same manifest
  fairness check without changing replay semantics.
