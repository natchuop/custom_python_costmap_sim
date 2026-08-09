# Implementation Plan: Attack/Obstacle Hardening Follow-up

## Target branch

Implement this work on:

`feature/attack-obstacle-hardening`

Do not merge or rebase `main` as part of this task unless explicitly required. Preserve any unrelated local/user changes.

Before editing:

```bash
git status
git branch --show-current
git diff main...HEAD
```

Confirm the current branch is `feature/attack-obstacle-hardening` and record the pre-change test status.

---

# 1. Goals and locked requirements

This task is a follow-up to the existing attack-obstacle-hardening work. The implementation must satisfy all requirements below together, not individually in a way that causes the visualization, planner, manifest, or tests to disagree.

## 1.1 Attack scheduler

- During the attack phase, schedule one attack attempt every **30 simulation steps by default**.
- At each scheduled attack time, choose **one** attack type using the seeded RNG from the feasible enabled attack types:
  - `fake_obstacle`
  - `false_clearance`
  - `stale_reassertion`
- The choice must remain deterministic for a fixed seed/configuration.
- Do not run multiple attack types simultaneously just because the clock reached a scheduled attack step.
- If a type is impossible at that step, select from the currently feasible types rather than creating a malformed event.
- Preserve CLI/config override support for attack intervals. The default should become 30/30, not a hard-coded value that cannot be changed.

## 1.2 Latest-attack indicator

In the live Matplotlib simulation window, add a small status indicator near the bottom that shows the latest attack event at or before the currently displayed simulation step.

Example:

`Latest attack: False Clearance - Step 630`

Requirements:

- Before the first attack: `Latest attack: None`
- Use friendly names:
  - Fake Obstacle
  - False Clearance
  - Stale Reassertion
- The displayed step must be the actual simulation step from the manifest/log.
- The indicator must work correctly when animation playback skips multiple steps at a time.
- It must reflect the fixed manifest events during normal modular runs, not infer the type from report colors.

## 1.3 Two map display modes

The existing per-robot map is not a complete combined belief because peer evidence is kept in the defense/fusion runner rather than written into the local direct-observation grid.

Rename the existing visualization concept to:

**Local Observation Map**

Add a second visualization mode:

**Combined Belief Map**

The startup Tkinter GUI must allow the user to choose one of these modes. Do not display both versions simultaneously.

Default selection: **Combined Belief Map**.

Also add a CLI equivalent for reproducibility/headless configuration, for example:

```text
--map-view combined
--map-view local
```

The exact option name may vary, but use a small validated enum/string rather than a free-form value.

### Local Observation Map semantics

The Local Observation Map should represent the robot's static prior plus its own direct sensor observations only.

It must not visually incorporate peer reports.

The underlying planner behavior must not be changed simply to make this display easier.

### Combined Belief Map semantics

The Combined Belief Map must represent the information the robot currently uses for navigation:

- static prior
- robot's own direct observations
- currently active/effective peer information from the selected defense/fusion method
- current trust/source-linked weighting where applicable
- no rejected/expired/non-operational peer reports

The Combined Belief Map must be derived from the same effective data used by path planning. Do not maintain a second independent approximation that can drift away from planner behavior.

Core invariant:

> If peer information is currently influencing that robot's navigation, the Combined Belief Map must visually reflect it. If a report was rejected or is no longer operational, it must not be shown as active peer belief.

Do **not** solve this by blindly writing accepted peer claims into the local `RobotBeliefMap.belief` array. That would mix local sensing with peer fusion and could change existing source-linked/decay behavior. Prefer a read-only effective/planning-belief snapshot assembled from the local map plus the defense runner's current effective state.

## 1.4 Belief-map colors/provenance

For benign robots:

- Direct/self-observed dynamic obstacle information: **green**.
- Effective peer-derived information: **yellow**.
- Fake obstacles accepted from the malicious robot: **yellow**, because the benign robot believes them as peer information.
- Accepted stale-reassertion information: **yellow** on benign robots.
- Accepted false-clearance information: use the peer/yellow provenance treatment on the combined belief view where it is useful to communicate that the effective/free belief is peer-derived.
- Rejected peer information: do not show it as active belief.
- Do not color malicious information red on benign robot belief maps merely because the simulator knows it is malicious.

For the malicious robot:

- Attack artifacts may continue to use **red**.

For ground truth:

- Keep actual physical truth intact.
- Use red/red-family overlays only for attack visualization on the ground-truth/debug panel.
- Do not alter `world.grid` just to color an attack.
- Fake obstacle attack overlays may be red as they are now.
- False-clearance attack regions should have a distinguishable light-red/red-family overlay while the attack indicator is active/relevant, without pretending the physical obstacle disappeared from ground truth.
- Stale-reassertion attack regions should use the red attack-overlay convention on ground truth when active/relevant.
- No extra red outline/debug marker is needed on benign belief maps.

Keep the map legend/title semantics clear enough that a screenshot is interpretable without reading code.

## 1.5 Fake and temporary obstacle sizes

Both fake obstacles and real temporary obstacles must use varying rectangular dimensions.

Allowed dimensions:

- height: 1 through 5 cells
- width: 1 through 5 cells

Minimum effective/reportable cells per object: **4**.

Valid examples include:

- 1x4
- 4x1
- 1x5
- 5x1
- 2x2
- 2x4
- 3x4
- 2x5
- 5x5

Invalid examples include footprints with fewer than four cells such as 1x1, 1x2, 1x3, 2x1, or 3x1.

Implementation guidance:

- Use a single deterministic helper for sampling dimensions from the seeded RNG.
- Sample height and width independently from 1..5.
- Reject/regenerate dimension pairs whose area is below 4.
- For temporary obstacles, the full footprint must fit on valid physical free cells and therefore contain at least four cells.
- For fake obstacles, the visual rectangle may overlap non-reportable cells if existing visualization semantics allow it, but the malicious event must contain at least four valid/reportable target cells.
- Do not silently clip a generated 4-cell shape down to 1-3 reported cells and still accept it.
- Record actual chosen dimensions in candidate/event metadata if practical so tests and experiment debugging can verify variation.

Remove the fixed 6x9 fake footprint behavior.

## 1.6 Attack target selection / bottlenecks

Do not explicitly prioritize narrow hallways/bottlenecks as the first attack-ranking criterion.

Restore the broader behavior where traffic/route relevance determines useful candidates without strongly forcing nearly every attack into the same narrow areas.

At minimum:

- Remove the branch change that made bottleneck score the first candidate sort key.
- Restore the earlier ordering where affected victims and route overlap take priority over bottleneck score, or remove bottleneck weighting entirely if it is no longer needed.
- Do not add a hard `bottleneck_score > 0` filter.
- Candidate selection must still permit ordinary non-bottleneck regions.
- Preserve attack diversity/spacing rules so repeated events do not keep selecting the same location.

After reverting the prioritization, inspect topology-related constants/helpers. Remove only those that become truly unused. Do not leave dead constants or stale comments claiming that attacks deliberately target bottlenecks when they no longer do.

## 1.7 Temporary obstacle movement

Temporary obstacles must support both movement styles:

1. **Shift**
2. **Teleport**

At every temporary-obstacle movement update, make a deterministic seeded **50/50 choice** between shift and teleport for each object (or equivalent per-object movement unit).

### Shift behavior

- Preserve the same object footprint dimensions.
- Shift the object by **1 to 3 cells**.
- The direction and distance must come from the seeded RNG.
- Only accept moves whose entire footprint remains valid and does not overlap forbidden robot cells.
- A diagonal shift is acceptable only if the existing simulation intentionally permits it for environmental objects; otherwise use cardinal movement. Be consistent and test the chosen rule.

### Teleport behavior

- Preserve the object's footprint dimensions.
- Move it to a genuinely different valid region/location.
- Do not call a one-cell or two-cell move a teleport.
- Require the new footprint to be non-identical and preferably non-overlapping with the previous footprint; ensure the center is meaningfully displaced.
- It must still obey static-map validity, robot occupancy, action-point restrictions, spacing rules, and map boundaries.

### Failure/fallback behavior

If the chosen movement type cannot find a valid placement after a bounded number of attempts:

- Try the alternate movement type.
- If that also fails, keep the object in its prior valid position for that update.
- Never create an invalid or partially clipped physical obstacle just to satisfy the 50/50 request.

Expose or log the movement decision (`shift`, `teleport`, `unchanged`) in a testable way. The final simulation behavior must remain seed-deterministic.

`map_poisoning.temp_obstacles.export_temp_episodes()` relies on `TemporaryBlockageManager`, so make sure exported fixed-manifest obstacle episodes correctly capture both movement modes.

## 1.8 Navigation must actually react to attacks

Double-check and test that accepted attack information can change benign navigation when it is relevant to the route.

Expected semantics:

- `fake_obstacle`: accepted BLOCKED peer evidence should increase routing cost or hard-block cells according to the active defense method and trigger replanning when relevant.
- `false_clearance`: accepted malicious FREE evidence must participate in fusion correctly. It should be capable of reducing/canceling peer-derived blocked belief when the defense method allows it. A robot's own direct observation remains authoritative where the current planner explicitly treats it that way.
- `stale_reassertion`: accepted stale BLOCKED evidence after a real obstacle clears must remain capable of influencing the victim planner until contradicted, rejected, decayed, or reweighted away according to the selected method.

Do not require every attack event to force a replan; attacks outside the robot's current route may legitimately have no immediate route effect.

Tests must verify the positive case by constructing/finding events that do intersect a benign robot's effective route.

## 1.9 Robot information sharing

Verify the full chain, not only LiDAR sensing:

`temporary obstacle -> direct observation -> honest report -> broadcast -> receiving robot -> defense/fusion -> Combined Belief Map -> replanning if route-relevant`

A real temporary obstacle detected by Robot A should appear as yellow peer-derived blocked information on Robot B's Combined Belief Map if Robot B accepts/uses that report.

If the report is rejected or operationally expired, it must not remain displayed as active peer belief.

## 1.10 Delivery tasks

All robots, including the malicious robot, must have delivery tasks.

The current authoring paths already appear intended to create task queues for every robot; preserve this and add explicit tests so a future change cannot accidentally exclude robot 0.

Do not make the malicious robot stationary unless another existing experiment rule requires it.

## 1.11 Runtime animation speed selector

Do **not** put the speed selector in the startup Tkinter GUI.

Add it to the actual Matplotlib simulation window, below the maps/status area.

Label it clearly, for example:

`Steps per frame:`

Selectable values:

- 1
- 3
- 5
- 10

Default: **1**.

This is a visualization/playback control. The simulation rollout is currently computed before `animate()` displays the log, so changing the selector must:

- change how many recorded simulation steps the animation advances per visual update
- not change simulation physics
- not change attack timing
- not change random seeds
- not change output metrics
- not drop/alter log records

Use Matplotlib widgets (for example `RadioButtons`) or an equivalent in-window control.

Because the user can change the value while the animation is running, do not simply create a fixed `range(0, max_frames, stride)` once. Maintain a mutable current-frame index and read the selected stride each timer tick.

When frames are skipped, all displayed state should come from the destination simulation step. The latest-attack indicator must show the newest attack whose step is `<= current displayed step`.

---

# 2. Current code structure to work with

The current branch uses a modular front end while replaying through validated `sim2.py` behavior.

Important paths/functions to inspect before implementation:

## `map_poisoning/config.py`

Relevant objects:

- `AttackConfig`
- `VisualizationConfig`
- `SimulationConfig`

Changes expected:

- default attack interval 30/30
- add validated map-view setting, default combined

## `map_poisoning/cli.py`

Changes expected:

- default attack interval 30/30
- add map-view CLI option
- propagate it into `VisualizationConfig`

## `map_poisoning/ui.py`

Changes expected:

- add startup selector for Local Observation Map vs Combined Belief Map
- default Combined Belief Map
- do not add playback speed here

## `map_poisoning/scenario.py`

Relevant functions:

- `author_manifest()`
- `author_warehouse_manifest()`

Changes expected:

- preserve seeded one-of-three attack choice
- default timing comes from config at 30-step cadence
- variable fake obstacle dimensions
- no over-prioritization of bottlenecks
- event/candidate metadata should describe actual footprint sizes where useful
- all robots retain delivery queues
- ensure false-clearance/stale episode references remain valid

Be careful with `author_manifest()` temporary episodes: verify that every generated episode has `appearance_step < clearance_step`. Do not leave episodes that can never become active.

## `map_poisoning/temp_obstacles.py`

Relevant function:

- `export_temp_episodes()`

It delegates to `sim2.TemporaryBlockageManager`. Verify it still exports correct fixed episodes after mixed shift/teleport behavior is introduced.

## `defense_method_runner.py`

Relevant state/API:

- `claims_by_cell`
- `active_claims`
- `claims_for()`
- `evidence()`
- `occupancy_probability()`
- `routing_cost()`
- `is_hard_blocked()`

Recommended change:

Add a small read-only helper/API that summarizes the **current effective peer state for visualization** instead of making animation code inspect private internals.

For example, conceptually:

```python
@dataclass(frozen=True)
class EffectivePeerCell:
    claim: int | None
    has_active_evidence: bool
    hard_blocked: bool
    routing_cost: float
    evidence: float
```

The exact shape is flexible. The point is to centralize method-aware interpretation so the Combined Belief Map uses the same source of truth as planning.

Do not expose historical/rejected claims as current effective state.

## `sim2.py`

Relevant areas:

- configuration constants
- `choose_temporary_object_footprints()`
- `TemporaryBlockageManager`
- fake footprint/candidate helpers
- `RobotBeliefMap`
- `GridRobot.process_inbox()`
- route-affect/replanning logic
- per-step logging in `run_simulation()`
- display helper functions
- `animate()`

Expected work:

- variable 1..5 rectangular footprints with min 4 cells
- mixed shift/teleport temp movement
- restore broader fake-candidate ranking
- add planner/effective-belief snapshot support
- log both local and effective visualization state (or sufficient provenance data)
- log attack events/overlays needed for deterministic playback
- implement local vs combined display selection
- implement colors/provenance
- add latest-attack status text
- add runtime steps-per-frame selector

Avoid unnecessary planner rewrites. The goal is to expose and verify the existing defense/planning model, not replace it.

## `map_poisoning/rollout.py`

Relevant function:

- `replay_manifest()`

Changes expected:

- pass the selected visualization/map-view option to animation if necessary
- keep headless behavior unchanged

## Tests

Current tests are concentrated in `tests/test_core.py`. It is acceptable, and preferable if clarity improves, to split new tests into focused files such as:

- `tests/test_attack_scheduler.py`
- `tests/test_obstacles.py`
- `tests/test_sharing_and_beliefs.py`
- `tests/test_visualization.py`

Do not overfit tests to one magic seed unless the test specifically documents why that seed is used.

---

# 3. Recommended implementation sequence

## Phase A - Baseline and safety

1. Confirm branch/status.
2. Run the full existing test suite before changes.
3. Save the failure/pass result in Codex's final report.
4. Run at least one short headless simulation to establish the current command works.
5. Do not delete current code until replacement behavior is covered by tests.

Suggested commands:

```bash
python -m pytest -q
python main.py --headless --no-animation --max-steps 100 --deliveries-per-robot 1
```

If dependencies are missing, use the repository's existing installer rather than introducing ad-hoc packages:

```bash
python install_dependencies.py --install --include-dev
```

## Phase B - Configuration and scheduler defaults

1. Change `AttackConfig.interval_min` and `interval_max` defaults from 50 to 30.
2. Change CLI defaults to 30.
3. Align legacy/no-manifest attack interval constants where they are still active so behavior does not unexpectedly differ between modular and internal paths.
4. Add map-view config (`combined` / `local`) with validation.
5. Add GUI selector, default combined.
6. Add CLI option for map view.
7. Add unit tests for defaults and invalid map-view values.

## Phase C - Variable rectangle helper

Create one reusable seeded rectangle-dimension sampler with these invariants:

```text
1 <= height <= 5
1 <= width <= 5
height * width >= 4
```

Use it for both temporary and fake obstacle generation.

Do not duplicate slightly different dimension logic in two modules unless there is a strong reason.

For fake obstacle candidate generation, make footprint dimensions explicit inputs rather than relying on global fixed `MALICIOUS_FAKE_OBJECT_ROWS/COLS` constants.

Remove the old fixed 6x9 constants once no longer used.

Tests:

- generate many shapes over a deterministic set of seeds
- every dimension is 1..5
- every footprint area is >=4
- confirm thin rectangles such as 1x4/4x1 or 1x5/5x1 are reachable across a seed sweep
- confirm at least several distinct dimension pairs appear

## Phase D - Temporary obstacle movement

Refactor `TemporaryBlockageManager.refresh_active_blockages()` so movement decisions are explicit and testable.

Recommended helper decomposition:

- `_try_shift_footprint(cells, forbidden_cells) -> new_cells | None`
- `_try_teleport_footprint(cells, forbidden_cells, other_active_footprints) -> new_cells | None`
- `_move_footprint(...) -> (new_cells, movement_type)`

Use `self.rng` only; do not create a new unseeded RNG.

Shift:

- choose distance 1..3
- choose valid direction
- preserve shape

Teleport:

- preserve shape
- choose a materially different location
- validate full footprint
- avoid robot cells
- avoid illegal overlap/spacing

Record recent movement decisions in a small manager field or return structure so tests can assert behavior without parsing console output.

Update `export_temp_episodes()` tests to verify the exported schedule reflects moved footprints.

Replace the current `test_temporary_objects_shift_without_teleporting` with tests that prove both modes occur across deterministic seeds.

## Phase E - Fake target ranking and footprint generation

1. Restore candidate priority so bottleneck score is not first.
2. Prefer the previous/main ordering where affected victims and route overlap are ahead of bottleneck score, or remove topology weighting if it no longer serves a needed purpose.
3. Make fake dimensions event-specific and seeded.
4. Require at least four reportable cells after filtering.
5. If a sampled size has no valid candidate, retry a bounded number of size/candidate samples rather than emitting an undersized attack.
6. Keep diversity/spacing logic.
7. Update candidate metadata with at least:
   - center
   - footprint height
   - footprint width
   - reportable cell count
   - traffic score
   - route overlap
   - bottleneck score only if it remains relevant diagnostics

Do not state in comments/logs that attacks intentionally target bottlenecks if that is no longer policy.

## Phase F - Effective Combined Belief Map

This is the most important architectural part.

### F1. Keep local sensing separate

Preserve the current local/direct belief grid as the robot's local observation state.

If practical, rename display/log concepts without forcing a risky rename of every internal `belief_map` planner reference.

Recommended logging compatibility:

- keep existing `rlog["belief"]` temporarily if downstream metrics/scripts may depend on it
- add `rlog["local_observation"]`
- add `rlog["combined_belief"]` or a richer display snapshot
- document `belief` as a compatibility alias if retained

### F2. Add method-aware effective peer state

Do not decide peer display status based only on `report.is_malicious` or `accepted_reports` counters.

Use current defense-runner state after:

- report admission/storage
- pruning/expiration
- source-linked current trust weighting
- majority/hard-threshold/soft method semantics

Provide a read-only helper to answer, for each cell at a step, whether current peer evidence is effectively FREE, BLOCKED, neutral, or absent, and what routing effect it has.

### F3. Build a display snapshot

Add a pure helper such as:

```python
build_combined_belief_display(robot, step)
```

or equivalent that returns display state/provenance without modifying planner state.

Precedence should be deliberate:

1. static map/semantic cells
2. robot's direct/self observation
3. effective peer fusion where direct observation does not already override it under planner rules
4. path/goal/robot overlays handled by visualization

The exact precedence must mirror `RobotBeliefMap.is_blocked_for_planning()` and `traversal_cost()`.

For each peer-derived cell, store enough provenance to render yellow.

For a direct dynamic obstacle, render green.

### F4. Validate planner/display consistency

Create tests that compare display classification against planner behavior. Examples:

- peer BLOCKED creates non-default routing cost/hard block -> Combined Belief Map marks peer/yellow
- rejected report -> no yellow peer state
- expired report -> no active yellow peer state
- direct observation overrides peer state where planner does so -> display follows direct/self state
- source-linked trust reduction removes/weakens operational peer influence -> Combined Belief Map updates accordingly

## Phase G - Attack overlays and latest-attack logging

When `run_simulation()` receives fixed `attack_events`, store a serialization-friendly event summary in `log`, for example:

```python
log["attack_events"] = [
    {
        "event_id": ...,
        "step": ...,
        "attack_type": ...,
        "cells": [...],
        "claim": ...,
    },
]
```

For legacy/no-manifest attacks, add the same log representation when an attack is generated so animation code has one uniform source.

Do not infer latest attack from `malicious_fake_objects`, because that loses false-clearance/stale-reassertion type information.

Add display-only ground-truth attack overlay data by attack type. Keep it separate from `truth_grid`.

Suggested display rules:

- fake obstacle: red attack overlay
- false clearance: lighter red attack overlay on the attacked real-obstacle cells
- stale reassertion: red-family overlay on the stale claimed cells

The exact shades may use existing red plus one lighter red; do not add red to benign combined beliefs.

## Phase H - Runtime speed control

Refactor `animate()` away from assuming the animation callback's `frame` argument is always the simulation step.

Recommended state:

```python
playback = {
    "frame": 0,
    "steps_per_frame": 1,
}
```

Create a timer/tick callback that:

1. renders `playback["frame"]`
2. reads the currently selected steps-per-frame value
3. increments/clamps the frame for the next tick
4. stops animation at the final recorded frame

Use Matplotlib `RadioButtons` or a similarly simple control positioned under the map panels.

Values: `1`, `3`, `5`, `10`.

Keep the existing status line and allocate enough bottom margin so controls/text do not overlap.

Add a second concise line for latest attack if that is visually cleaner than appending it to the already long status string.

Do not put the speed selector in Tkinter.

Test the frame-advance logic as a pure helper/state function instead of trying to require an interactive GUI in CI.

## Phase I - All robots have tasks

Add explicit assertions in both authoring paths:

- every robot start has a task queue
- malicious robot ID appears in the task queues
- each queue length equals `deliveries_per_robot` when enough action points exist and repair succeeds

If the current code already satisfies this, no unnecessary behavior change is required; add regression tests.

---


## Phase J - Original branch-audit regression pass

Before final cleanup, explicitly revisit the issues identified in the original `main` vs `feature/attack-obstacle-hardening` audit. Do not assume they are covered merely because the new feature tests pass.

1. Compare the final branch against `main` again:

```bash
git diff --stat main...HEAD
git diff main...HEAD -- map_poisoning/scenario.py map_poisoning/temp_obstacles.py sim2.py tests
```

2. Verify the normal warehouse path and the generic/custom-map path separately. The earlier branch tests leaned on `author_manifest()` even though ordinary warehouse runs use `author_warehouse_manifest()`. Required coverage must exercise both where the behavior exists.
3. Preserve valid earlier fixes unless the new requirements intentionally supersede them, including correct false-clearance ground-truth/audit semantics and seeded one-of-three attack authoring.
4. Replace or strengthen any old test whose name overstates what it proves. In particular, do not retain a test named like "each attack type causes replanning" if it only checks that some malicious report existed and some unrelated replan happened afterward.
5. Attribute route changes to the specific attack under test. Capture the pre-event route/cost state, process the event, then assert the relevant victim's effective planning state/replan/path changed because of that event when the scenario was intentionally constructed to be route-relevant.
6. Check that generic/custom-map temporary episodes can actually become active; every authored episode must satisfy valid temporal ordering and must not have `appearance_step >= clearance_step`.
7. Re-run a fixed-manifest replay after all changes to ensure authoring and replay agree on attack type, attack cells, temporary-obstacle positions, and timing.
8. Inspect the final diff for accidental unrelated changes. If a changed constant/helper is no longer needed after reverting bottleneck-first targeting or fixed 6x9 footprints, remove it only after confirming no other path uses it.

# 4. Required automated tests

Do not consider the task complete if only manual visualization looks correct.

## 4.1 Scheduler/determinism tests

- Same seed/config => identical manifest.
- Default attack interval is 30.
- With fixed 30/30 interval, scheduled attack steps are 30 simulation steps apart after the initial attack-phase offset logic.
- Each scheduled event contains exactly one attack type.
- Across a reasonable seed sweep, all three attack types appear.
- Stale reassertion is never authored before its referenced obstacle clears.
- False clearance is never authored against an episode that is not active at the event step.
- Every temporary episode satisfies `appearance_step < clearance_step`.

## 4.2 Fake obstacle geometry tests

For generated fake events:

- dimensions are each 1..5
- at least four reportable cells
- no report cell is an invalid static blocked cell
- multiple different dimension pairs appear across a seed sweep
- no fixed 6x9 assumption remains

## 4.3 Temporary obstacle geometry/movement tests

- dimensions are each 1..5
- area >=4
- entire footprint remains valid physical free space at placement time
- shift preserves dimensions and moves 1..3 cells
- teleport preserves dimensions and moves to a genuinely different region
- 50/50 selection is driven by the seeded RNG
- across deterministic seeds, at least one shift and at least one teleport can be observed
- same seed produces the same movement sequence
- no movement places an obstacle on a robot footprint or outside map bounds
- exported obstacle episodes match manager state

Do not assert an exact statistical 50.000% ratio in a small test sample; test the decision mechanism and both outcomes deterministically.

## 4.4 Sharing/effective-belief integration test

Construct a small deterministic scenario:

1. Robot A sees a real temporary obstacle with LiDAR.
2. A creates an honest BLOCKED report.
3. Report is broadcast to Robot B.
4. B processes it.
5. Assert B's local observation map does not falsely claim it was self-sensed.
6. Assert B's Combined Belief Map marks it as peer-derived/yellow.
7. Assert B's planner cost/block state matches the combined display.
8. If the report intersects B's route, assert a relevant replan occurs.

This test is required; the existing LiDAR-only test is not sufficient.

## 4.5 Rejected-report display test

Use a defense/admission setup that rejects or makes a report non-operational.

Assert:

- it is not shown as active yellow peer belief
- planner does not use it
- local map remains unchanged

## 4.6 Fake obstacle navigation test

Create/author a fake obstacle whose report cells intersect a benign remaining route.

Assert:

- victim receives effective malicious BLOCKED peer evidence
- Combined Belief Map shows those relevant peer cells as yellow
- at least one attacked route cell has increased traversal cost or is hard blocked according to the method
- the robot replans because of that event
- where an alternative path exists, verify the new route differs from the previous route, not merely that a replan counter increments

## 4.7 False-clearance test

Create a scenario where:

- a real temporary obstacle exists
- peer-derived BLOCKED evidence for it exists
- a malicious false-clearance FREE report is then accepted

Assert the false-clearance report changes the fusion/effective belief in the expected direction and can alter the route when route-relevant.

Also test that a direct self-sensed physical BLOCKED observation remains authoritative if the current planner rules specify that behavior.

## 4.8 Stale-reassertion test

Create a scenario where:

- a temporary obstacle existed and was learned
- it has physically cleared in ground truth
- the malicious robot sends stale BLOCKED evidence afterward

Assert:

- the event occurs after clearance
- accepted stale evidence appears yellow on benign Combined Belief Map
- ground-truth attack overlay is red-family and does not modify physical truth
- planner is affected when the cell lies on the route

## 4.9 Visualization helper tests

Keep these non-interactive:

- local mode returns local-only state
- combined mode applies peer provenance
- benign fake/stale peer evidence is yellow, not red
- malicious/debug attack overlay is red only on malicious/ground-truth views
- latest-attack lookup returns the correct event for frames before, at, between, and after attack steps
- playback advancement produces 1/3/5/10 step increments and clamps at the last frame

## 4.10 Delivery task tests

For both `author_manifest()` and `author_warehouse_manifest()` where practical:

- every robot ID has a task queue
- malicious robot has tasks
- no task pickup/dropoff is invalid/static-blocked

---


## 4.11 Original-audit regression tests

These tests explicitly close the weaknesses found in the earlier branch review.

### Normal warehouse authoring path

Test `author_warehouse_manifest()` directly rather than relying only on `author_manifest()`:

- across a deterministic seed sweep, verify fake-obstacle, false-clearance, and stale-reassertion events can each be authored when feasible
- verify exactly one attack type is selected at each scheduled attack step
- verify every false-clearance event references a physically active temporary obstacle at that step
- verify every stale-reassertion event occurs after the referenced physical obstacle has cleared
- verify the event cells and referenced episode IDs remain valid after variable-size/moving temporary obstacles are introduced

### Specific-event replan attribution

For each attack type, build or author a deterministic route-relevant positive case and assert the effect of the **specific event**, not merely "a replan sometime after an attack":

1. Capture victim route, traversal cost/hard-block state, and replan counter immediately before the event.
2. Process the event/report through the same inbox/defense path used by the simulation.
3. Assert the event becomes operational/accepted for the victim under the selected test defense method.
4. Assert at least one attacked route cell changes effective planning state in the expected direction.
5. Assert the victim replans at that event step or the next planner update caused by that event.
6. Where an alternate path exists, assert `new_path != old_path` or a measurable path-cost difference rather than relying only on a counter increment.

### Fake obstacle before direct self-observation

Add a targeted integration test for the original practical problem where robots seemed to drive through fake obstacles:

- place the malicious fake footprint on the victim's remaining route but initially outside that victim's direct LiDAR/self-observation range
- ensure the malicious peer report is accepted/effective
- assert the cells appear yellow on the victim's Combined Belief Map before the victim personally observes them
- assert the planner reroutes or changes effective route cost before the victim reaches direct sensing range
- then advance until the victim can directly sense that the fake cells are physically free
- assert the Local Observation Map records the direct free observation correctly
- assert the Combined Belief Map and planner reconcile the conflicting self-vs-peer evidence according to the selected defense method instead of permanently displaying a stale yellow ghost unless that method's documented semantics intentionally keep it operational

This distinguishes two behaviors that must not be confused:

- "the fake report never affected navigation" = bug for an accepted route-relevant attack
- "the robot later personally observes the area is free and corrects its belief" = potentially correct behavior, depending on the defense method

### SourceLinked accepted/rejected pair

Because SourceLinked motivated the visualization concern, include two deterministic SourceLinked cases:

- a peer report whose trust/evidence is high enough to become operational: it appears yellow and can affect planning
- a peer report that is rejected/non-operational: it does not appear yellow and does not affect planning

Do not hard-code colors or display state from `is_malicious`; derive visibility from current effective defense state.

### Sharing path distinct from attacks

Keep at least one test where an **honest benign robot** is the source of a temporary-obstacle report. This prevents a passing malicious-report test from being mistaken for proof that ordinary robot-to-robot sharing works.

### Visualization-only invariance

Using the same fixed manifest/log:

- Local Observation vs Combined Belief display selection must not change planner results or metrics
- playback speed 1/3/5/10 must not change planner results, attack schedule, logs, or metrics
- rendering helpers must be read-only with respect to robot belief, defense state, and world truth

# 5. Manual verification by running the program

After automated tests pass, perform at least one normal interactive run using `source_linked`, because that was the configuration that motivated the visualization concern.

Recommended:

```bash
python main.py
```

In the startup GUI:

- Defense method: `source_linked`
- Map view: `Combined Belief Map` (should already be default)
- Keep all three attacks enabled
- Use a fixed seed and record it
- Enable animation

Verify visually:

## Combined Belief Map

- Panel titles say `Combined Belief Map`, not the old generic label.
- Real dynamic obstacle personally sensed by a robot is green on that robot.
- The same real obstacle can appear yellow on another robot after sharing, before that second robot personally sees it.
- Accepted fake obstacles appear yellow on benign victim maps.
- Fake/stale attack information does not appear red on benign maps.
- Rejected/non-operational malicious reports do not remain displayed as active yellow belief.
- The route drawn on the panel is consistent with the displayed effective obstacle information.

## Local Observation Map

Run again selecting Local Observation Map.

- Panel titles say `Local Observation Map`.
- Peer-only fake obstacles do not appear as local sensor observations.
- Local LiDAR-discovered temporary obstacles do appear.
- Selecting local view changes visualization only; it must not change metrics or planner behavior for the same fixed manifest/config.

## Ground truth

- Real temporary obstacles remain physical truth.
- Fake obstacle attack overlays are red.
- False-clearance locations have the intended lighter-red/red-family attack overlay while ground truth still shows the actual physical obstacle.
- Stale-reassertion attack locations have the intended red-family overlay after the physical obstacle has cleared.

## Latest attack indicator

Watch multiple attacks and confirm it updates, e.g.:

```text
Latest attack: Fake Obstacle - Step 480
Latest attack: False Clearance - Step 510
Latest attack: Stale Reassertion - Step 540
```

Exact types depend on seed/feasibility.

## Playback speed

While animation is running:

- switch between 1, 3, 5, and 10 steps per frame
- confirm the control is below the Matplotlib maps, not in Tkinter
- confirm switching works without restarting
- confirm status step increments accordingly
- confirm the latest-attack text catches up correctly even when an attack step was skipped visually
- confirm final frame is not skipped past/crashed

## Temporary obstacle motion

Use a seed/run long enough to observe multiple change periods.

- some obstacles should shift locally by 1-3 cells
- some should teleport to clearly different valid areas
- object rectangles should visibly vary in dimensions
- no obstacle should appear inside static walls, action points, or robot footprints

---

# 6. Headless/regression verification

Run the full suite:

```bash
python -m pytest -q
```

Then run a short modular headless simulation:

```bash
python main.py --headless --no-animation --seed 15 --max-steps 150 --deliveries-per-robot 1
```

Author a fixed manifest and replay it:

```bash
python main.py --headless --manifest-only --seed 15 --no-animation --output-directory outputs/plan_verify_manifest
python main.py --headless --manifest outputs/plan_verify_manifest/scenario_manifest.json --seed 15 --no-animation --output-directory outputs/plan_verify_replay
```

If the repository's documented PowerShell launcher is being tested on Windows, also run the documented quick path:

```powershell
.\run_sim.ps1 -NoAnimation -MaxSteps 100 -DeliveriesPerRobot 1
```

Verify:

- no exceptions
- deterministic manifest replay
- all three robot task queues exist
- metrics files still write correctly
- visualization-only map-view option does not affect headless experiment outcomes
- changing animation playback speed does not alter saved metrics or logs

For one fixed manifest, replay with Local vs Combined visualization mode if convenient and compare summary/metrics files. They should be identical except visualization configuration metadata.

---

# 7. Bug-hunting checklist

Before declaring completion, explicitly inspect for the following failure modes.

## Scheduler bugs

- attack interval accidentally remains 50 in CLI while config says 30
- legacy path still injects fake attacks at 20 and creates confusing divergent behavior
- attack type selection uses an unseeded random source
- stale attack references an obstacle that has not cleared
- false clearance references a non-active obstacle
- one event accidentally emits multiple attack types
- tests cover only `author_manifest()` while the normal warehouse run uses `author_warehouse_manifest()`
- a test attributes an unrelated later replan to an earlier attack without proving event causality

## Geometry bugs

- 1x1/1x2/1x3 objects leak through
- fake visual footprint is >=4 but actual report cells are <4
- dimension sampling clips at map edges instead of retrying
- temporary obstacle shape changes dimensions during movement

## Temporary movement bugs

- 'teleport' only moves a few cells
- shifts can exceed 3 cells
- forbidden robot cells are ignored
- movement changes nondeterministically between same-seed runs
- `export_temp_episodes()` reads stale pool footprints instead of `current_footprints`

## Belief/display bugs

- peer reports are written into local sensor state and erase provenance
- rejected reports are still yellow
- expired source-linked claims remain yellow forever
- benign fake obstacles are red because display logic keys off `is_malicious`
- planner avoids a peer-derived cell that Combined Belief Map does not show
- Combined Belief Map shows a block the planner treats as normal cost 1 without a documented reason
- direct self observation and peer evidence use inconsistent precedence between display and planner
- an accepted route-relevant fake report is visible but has no planner cost/block effect before direct self-observation
- a victim later self-senses fake cells as free but Combined Belief keeps an unexplained permanent yellow ghost inconsistent with defense semantics
- rendering Local/Combined view mutates planner or defense state

## False-clearance bugs

- false FREE reports never have any operational effect because only BLOCKED peer evidence was considered in combined visualization/fusion tests
- false clearance visually changes ground truth instead of adding a display-only attack overlay
- self-sensed BLOCKED truth is incorrectly overridden if planner semantics say self sensing is authoritative

## Animation bugs

- speed control changes only a label but not frame advancement
- changing speed during playback resets animation
- `FuncAnimation` frame argument and simulation frame become confused
- skipped frames make latest attack indicator stale
- last frame causes an out-of-range access
- controls overlap status text or are clipped by `tight_layout`

## Task bugs

- malicious robot has no tasks in one authoring path
- task validation uses a step-0 dynamic blockage and incorrectly rejects valid manifest tasks

---

# 8. Cleanup and code quality pass

After functionality and tests pass, perform a deliberate cleanup pass.

## Remove/rename stale code and comments

Search for references to the old behavior:

```bash
grep -R "6x9\|6 x 9\|without teleport\|bottleneck lies\|Belief Map" -n .
```

Adjust wording to match the new behavior.

Examples:

- comments claiming temp objects never teleport
- comments claiming attack policy deliberately reinforces bottlenecks
- comments claiming benign belief maps store peer fake obstacles directly in the local grid
- fixed 6x9 constants no longer used
- tests named `shift_without_teleporting`

## Remove unused helpers/constants only after search

Potential candidates include topology-specific weights or fixed-size constants that become unused after the changes.

Use a project-wide search before deleting anything.

Do not remove topology scoring if it still serves diagnostic metadata or another experiment path.

## Avoid duplicated sources of truth

There should be one authoritative place for:

- default attack interval
- valid map-view values
- rectangle size constraints
- effective peer-state interpretation
- playback stride values

Do not scatter magic values `30`, `5`, or `4` across many unrelated functions if a named config/helper makes sense.

## Keep logs/backward compatibility reasonable

If external scripts may rely on `log["robots"][rid]["belief"]`, retain it as a compatibility alias or migrate carefully.

Do not silently change metrics calculations merely because visualization semantics were renamed.

## Formatting/lint sanity

At minimum run:

```bash
python -m compileall map_poisoning sim2.py defense_method_runner.py tests
python -m pytest -q
```

If the repository already has a formatter/linter configured, run it. Do not introduce a new formatting tool solely for this task.

---

# 9. Definition of done / acceptance checklist

Codex should not report this task complete until all applicable boxes are satisfied.

- [ ] Working branch is `feature/attack-obstacle-hardening`.
- [ ] Existing tests were run before changes and final tests pass.
- [ ] Default attack cadence is one seeded attack attempt every 30 steps.
- [ ] Each event chooses one feasible type from fake / false-clearance / stale-reassertion.
- [ ] Same seed creates the same attack schedule and obstacle movement.
- [ ] Latest attack + step is visible at the bottom of animation.
- [ ] Startup GUI has Local Observation vs Combined Belief selector.
- [ ] Combined Belief is the default.
- [ ] Playback speed is not in startup GUI.
- [ ] Matplotlib window has runtime 1/3/5/10 steps-per-frame selector.
- [ ] Playback speed does not change simulation results.
- [ ] Existing local map is clearly named Local Observation Map.
- [ ] Combined Belief Map uses local + currently effective peer information.
- [ ] Rejected/non-operational peer reports are not shown.
- [ ] Benign direct dynamic observations are green.
- [ ] Benign effective peer information is yellow.
- [ ] Accepted fake obstacles are yellow on benign maps.
- [ ] Attack overlays are red/red-family on ground truth and red on malicious robot where appropriate.
- [ ] No red attack-debug outline is added to benign maps.
- [ ] Fake obstacle dimensions vary 1..5 in each dimension with >=4 reportable cells.
- [ ] Temporary obstacle dimensions vary 1..5 in each dimension with >=4 cells.
- [ ] Fixed 6x9 fake objects are removed.
- [ ] Bottleneck score is no longer the dominant first-priority attack ranking criterion.
- [ ] Non-bottleneck attack candidates remain possible.
- [ ] Temp obstacles choose shift vs teleport using seeded 50/50 logic.
- [ ] Shift distance is 1..3 cells.
- [ ] Teleport moves to a genuinely different valid area.
- [ ] Both shift and teleport are covered by tests.
- [ ] Robot-to-robot sharing is covered by an end-to-end test.
- [ ] `author_warehouse_manifest()` has direct attack-type/feasibility tests, not only generic `author_manifest()` coverage.
- [ ] Each attack's route effect is attributed to that specific event in a constructed route-relevant positive case.
- [ ] Accepted fake-obstacle evidence is proven to affect planning before the victim directly senses the fake area.
- [ ] Later direct sensing of fake cells as free reconciles Combined Belief/planner state according to the active defense method.
- [ ] SourceLinked has both accepted/operational and rejected/non-operational visualization/planning tests.
- [ ] Visualization helpers are read-only and cannot alter simulation/planner outcomes.
- [ ] Fake-obstacle route replanning is tested with a route-relevant event.
- [ ] False-clearance fusion/navigation effect is tested.
- [ ] Stale-reassertion fusion/navigation effect is tested.
- [ ] Combined belief visualization agrees with planner-effective peer state.
- [ ] All robots, including malicious robot, have delivery tasks.
- [ ] All generated temporary episodes have valid appearance/clearance ordering.
- [ ] Headless run, manifest authoring, and fixed-manifest replay still work.
- [ ] `compileall` passes.
- [ ] Dead constants/comments/tests from superseded behavior are cleaned up.

---

# 10. Final report Codex should return

After implementation, provide a concise report containing:

1. Files changed and the purpose of each.
2. Exact behavioral changes implemented.
3. Tests added/updated.
4. Exact test commands run and pass/fail counts.
5. Manual simulation command/seed used and what was visually verified.
6. Any requirement that could not be completed, with a specific reason.
7. Any leftover technical debt or compatibility concern.
8. A short final `main...HEAD` audit confirming no unrelated behavioral changes or stale superseded code remain.

Do not claim an attack affects navigation merely because a malicious report exists. Cite the specific test/assertion proving planner cost/path/replan behavior.

Do not claim sharing works merely because LiDAR detects a temporary obstacle. Cite the end-to-end sender-to-recipient test.

