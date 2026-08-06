# Map-Poisoning Simulator Methodology Audit and Implementation Plan

## 1. Purpose

This plan updates the `modularize` branch so that the end-to-end behavior of all defense methods is correct, reproducible, and scientifically interpretable.

The immediate goal is not to make `source_linked` win. The goal is to make each method implement its stated semantics, replay the same scenario fairly, expose enough evidence to explain results, and prevent outcome-driven parameter tuning.

The four primary methods are:

1. `full_trust`
2. `majority_vote`
3. `trust_fused`
4. `source_linked`

The experiment uses exactly three robots:

- Robot `0`: delayed attacker
- Robot `1`: benign robot
- Robot `2`: benign robot

The attacker is globally aware for attack authoring, broadcasts to both benign robots, behaves honestly during reconnaissance, injects malicious claims during poisoning, and returns to fully honest reporting during recovery.

This plan is based on:

- The `modularize` branch of `custom_map_poisoning_costmap`
- The project description in `Physical AI Notes(1).docx`
- The decisions made during the design discussion
- The exploratory fixed-manifest result that produced different rankings across the four methods

## 2. Scientific guardrails

These rules are requirements.

### 2.1 Do not tune toward a preferred ranking

Do not change fusion parameters, attack placement, replan thresholds, trust priors, claim lifetimes, or phase lengths because one method performed badly or another method did not win.

A code change is allowed before the protocol is frozen only when it is one of the following:

- It fixes a mismatch between code and the written method definition.
- It fixes a reproducibility or configuration bug.
- It removes hidden ground-truth leakage from operational logic.
- It adds logging needed to diagnose behavior.
- It fixes an invariant that can be tested independently of mission outcomes.

Every behavior-changing fix must have a focused test that would fail before the fix and pass after it.

### 2.2 Separate exploratory and reportable results

All current results are exploratory until the audit requirements in this plan pass.

Do not assert any expected winner in tests. Tests may assert mechanism-level differences, such as source-linked evidence decreasing after a trust drop while trust-fused evidence remains unchanged. Tests must not assert that source-linked completes more deliveries.

### 2.3 Freeze a named protocol before evaluation

Create named protocols instead of silently choosing between parameter sets:

- `legacy_cli_protocol`: parameters corresponding to the public legacy CLI behavior
- `legacy_internal_protocol`: parameters corresponding to the legacy runner's internal defaults
- `paper_v1`: the first frozen modular protocol after correctness validation
- `custom`: any user-edited configuration

A protocol must resolve every effective parameter. Once `paper_v1` is frozen, do not change it in place. Create `paper_v2` for later revisions.

### 2.4 Keep audit labels out of operational code

The following may appear in manifests, event logs, and evaluation code only:

- `is_malicious`
- `attack_type`
- true obstacle episode identity
- true obstacle appearance and clearance steps
- ground-truth state used to label an attack

Trust, admission, fusion, planning, and robot behavior must never read these audit fields.

## 3. Current branch issues that must be addressed

The implementation agent must verify these against the checked-out branch before editing. If the branch has already changed, preserve the intent of the fix and update the test accordingly.

### 3.1 Incorrect modular team size

`map_poisoning/scenario.py` currently authors benign robot IDs `(1, 2, 3)`. Change the modular experiment to exactly:

```text
attacker: 0
benign:   1, 2
```

All starts, task queues, recipients, metrics, GUI labels, and tests must use this team definition.

### 3.2 Two competing fusion implementations

The branch currently has method logic in both:

- `defense_method_runner.py`
- `map_poisoning/fusion.py`

Do not maintain two independently evolving definitions of the same methods.

Make `map_poisoning/fusion.py` the canonical implementation. Convert `defense_method_runner.py` into a thin legacy compatibility wrapper, or remove its duplicated algorithmic logic after the legacy simulator is no longer needed.

During migration, both entry points must pass the same method-contract tests.

### 3.3 Configured and effective parameters can differ

The modular config currently contains fusion defaults that differ from the legacy defense runner defaults. The legacy adapter can run without passing the modular fusion settings, while `resolved_config.json` still records the modular values.

This is a reproducibility bug.

Every run must write both:

- `requested_config.json`: what the CLI or GUI requested
- `effective_config.json`: the exact values instantiated by the engine

The effective file must include the trust model class, priors, threshold, admission policy, method name, all fusion parameters, phase boundaries, team IDs, sensor settings, communication cadence, and protocol name.

Add a runtime assertion that the effective configuration read back from each robot agrees with the logged configuration.

### 3.4 Legacy trust selection is not reliably wired

The legacy simulator defines its own global trust-model selection. The legacy adapter must explicitly apply and restore:

- trust model name
- Bayesian prior alpha and beta, or scalar initial value
- trust threshold
- any trust reward or penalty parameters used by the selected model

Do not rely on the legacy module's imported defaults.

Add a test that selects Bayesian trust through the modular configuration and verifies that every benign legacy robot actually owns a Bayesian model with the requested prior.

### 3.5 Majority voting is not one vote per robot

Current implementations can count every stored report as another vote. Repeated reports from one robot can therefore outvote other robots.

The primary majority method must use the latest active claim from each sender for each cell. A robot gets at most one vote per cell.

Operational majority state for a recipient is based on:

- the recipient's latest valid direct observation as its self vote, when available
- the latest non-expired report from each other robot

Tie behavior must be explicit:

- blocked majority: cell is blocked
- free majority: cell is free
- tie or no votes: unknown; no peer-added route penalty and no hard block

A repeated same-state report from the same sender refreshes that sender's vote timestamp; it does not add another vote.

### 3.6 Majority voting currently behaves like a soft evidence method

The written majority baseline marks a cell occupied or free according to the vote. It should not pass the vote count through the same continuous logistic cost as source-linked.

Implement majority voting as a discrete state method:

```text
positive majority -> hard blocked
negative majority -> free
zero/tie          -> unknown
```

Keep the downstream A* interface common by returning either infinity or base traversal cost.

### 3.7 Admission can double-count trust

The current `auto_soft` admission policy can multiply a report by sender trust before a trust-based fusion method multiplies it by trust again. This can produce trust-squared behavior.

For the primary method comparison, use `admission_policy=method_default` and resolve it as follows:

| Method | Primary admission behavior |
|---|---|
| `full_trust` | accept all valid reports with influence 1 |
| `majority_vote` | accept all valid reports as one sender vote |
| `trust_fused` | accept all valid reports; trust is applied once at report time |
| `source_linked` | accept all valid reports; current trust is applied once at planning time |

Keep `accept_all`, `hard_reject`, and a soft-gating policy as optional ablations. Do not mix a soft gate into the primary four-method comparison unless the paper explicitly defines the comparison as a compound admission-plus-fusion defense.

If soft gating remains available, rename it clearly, document its formula, and ensure it does not unintentionally square trust.

### 3.8 Direct observations do not fully override peer evidence

The direct belief map currently hard-blocks directly observed obstacles but can still apply peer risk to a cell directly observed as free.

Implement fresh direct-observation precedence:

- Direct `BLOCKED`: hard block for that recipient.
- Direct `FREE`: base traversal cost for that recipient, suppressing peer blockage influence while the direct observation is considered current.
- Static obstacles: always hard block.

Keep the latest direct observation and step. Define a simple direct-observation validity rule in the protocol. Recommended `paper_v1` rule:

- A direct observation is authoritative until replaced by a newer direct observation.
- A newer peer report does not overwrite direct state; it is stored separately and can matter after later direct sensing changes the state.

This matches the current known-map, local-verification experiment and avoids adding an unvalidated direct-sensor decay model. If direct-observation decay is later studied, make it a separate protocol.

### 3.9 Report refreshes can stack without limit

Repeated reports from the same sender about the same cell should not create unlimited linear evidence merely because the sender transmits frequently.

Use two stores:

- immutable report history for audit and trust analysis
- active claim index keyed by `(sender_id, target_cell)` for operational fusion

A new report from the same sender and cell supersedes the previous active report. The old report remains in history but no longer contributes to current map evidence.

This rule applies to full trust, trust-fused, and source-linked. It naturally gives majority voting one vote per sender. A stale reassertion remains meaningful because it becomes the sender's latest active claim.

### 3.10 Honest-stale classification is too simplistic

Current modular verification can classify any contradiction older than a fixed number of steps as honest stale and can slightly reward trust. That can misclassify a malicious false obstacle that simply remained unverified for a long time.

Do not infer honesty solely from age.

Use the following simple operational outcomes:

- `confirmed_fresh`: direct observation agrees with a still-current claim; reward trust
- `contradicted_fresh`: direct observation contradicts a still-current claim; penalize trust
- `temporally_ambiguous_or_expired`: contradiction occurs after the claim validity window; remove or weaken map influence but do not reward or penalize trust
- `unresolved`: no direct verification

The claim-validity window is a protocol parameter and may differ by claim type if later justified. For `paper_v1`, use one explicit value and do not tune it by method.

Audit code may later label the true report as current-valid, honest-stale, or malicious, but that label must not decide the operational trust update.

### 3.11 Recovery is not fully honest in the legacy behavior

During recovery, robot `0` must return to the same honest reporting behavior used in reconnaissance, including both `BLOCKED` and `FREE` observations.

Use an explicit phase check:

```text
RECONNAISSANCE -> honest reports only
ATTACK         -> honest reports plus scheduled malicious reports
RECOVERY       -> honest reports only
```

Do not use a boolean meaning "attack has ever started" to decide reporting behavior.

### 3.12 The fixed manifest does not yet fix the full attacker stream

The same malicious events are replayed, but the attacker trajectory and honest report stream must also be fixed across methods.

The scenario manifest must include:

- attacker start
- attacker per-step position or deterministic scripted route
- attacker task queue, if task behavior is used
- each honest attacker report, with observation, sent, and delivery step
- all malicious attack events

During replay, robot `0` is a scripted actor. Its motion and honest reports do not adapt to defense-dependent benign routes.

Benign robots remain live and defense-dependent. Their routes, sensing, verification opportunities, and honest reports may differ naturally across methods.

### 3.13 Attack candidate concentration

Legacy authoring can repeatedly sample from only a few top candidates. A concentrated repeated attack can be a valid stress test, but it should not be the only evaluation.

Add manifest diversity constraints:

- `max_uses_per_footprint`
- `min_center_spacing`
- `min_unique_footprints`
- per-corridor reuse count
- seeded weighted sampling without always choosing rank 1

Save candidate metadata in the manifest:

- candidate ID
- center and footprint cells
- route overlap
- traffic score
- bottleneck score
- estimated detour score
- rank
- selection probability or weight
- prior use count

If a map cannot meet the requested diversity, fail authoring with a clear error or mark the manifest as a concentrated stress scenario. Do not silently relax all constraints.

### 3.14 No-path results lack cause information

The exploratory result contains no-path totals that align with entire temporary-obstacle periods. Add no-path provenance so the result can be explained.

When a robot has no path, classify the step using counterfactual planning:

1. `truth_disconnected`: no path exists in the current ground-truth dynamic world.
2. `direct_belief_disconnected`: a ground-truth path exists, but no path exists using static plus the robot's direct observations only.
3. `peer_fusion_disconnected`: a direct-belief path exists, but peer fusion makes the operational map disconnected.
4. `planner_or_state_error`: a path should exist under the operational map but planning failed.

Blocked moves and robot-to-robot collision waits are separate event types, not no-path causes.

### 3.15 Source-linked-specific replan heuristics can confound the method

Any source-linked-only cooldown, trust-drop threshold, or route-risk threshold is part of the proposed method and must be declared, logged, and tested. It must not be introduced only because source-linked churn looked bad.

Preferred primary rule: use one method-independent replanning policy for all four methods.

Replan when one of these occurs:

- no path exists
- the next path cell becomes directly blocked
- a task transition changes the goal
- a newly admitted or revised claim changes the current route's feasibility
- the total current-route cost changes by at least one fixed, protocol-level threshold

The route-cost threshold and cooldown, if any, must be shared across methods. A source-linked trust update can trigger this common rule because it changes route cost retroactively.

## 4. Target architecture

Keep the package flat and small. Do not create a framework of factories and interfaces beyond what this experiment needs.

Recommended structure:

```text
map_poisoning/
    __init__.py
    application.py       # orchestration shared by CLI and GUI
    cli.py               # argument parsing only
    ui.py                # Tkinter form only
    config.py            # immutable requested configuration
    protocols.py         # named protocol defaults and resolution
    models.py            # dataclasses and enums
    rng.py               # named deterministic random streams
    map_io.py            # map loading
    world.py             # static map and temporary obstacle truth
    scenario.py          # manifest schema, authoring, loading, validation
    trust.py             # Bayesian and scalar trust models
    verification.py      # claim verification outcome policy
    admission.py         # optional pre-fusion report gate
    fusion.py            # canonical four primary methods and optional legacy methods
    belief.py            # direct observations plus peer-fusion view
    planning.py          # A* and path-cost helpers
    robot.py             # one benign robot's state and behavior
    simulation.py        # fixed-manifest replay loop
    metrics.py           # event, time-series, and summary writers
    audit.py             # invariant and cross-run checks

defense_method_runner.py # temporary compatibility wrapper only
sim2.py                  # legacy reference during migration only
main.py                  # application entry point
verify_end_to_end.py     # one command to run verification
install_dependencies.py

tests/
    test_config_protocols.py
    test_manifest.py
    test_trust.py
    test_verification.py
    test_admission.py
    test_fusion_contract.py
    test_majority_vote.py
    test_direct_precedence.py
    test_robot_independence.py
    test_replay_determinism.py
    test_recovery_behavior.py
    test_logging_audit.py
    test_micro_scenarios.py
```

Only add `protocols.py`, `verification.py`, and `audit.py` if equivalent responsibilities do not already exist. Do not split files merely to make them shorter.

## 5. Canonical data model

### 5.1 Operational report

Use one report type everywhere:

```python
@dataclass(frozen=True)
class ClaimReport:
    report_id: str
    sender_id: int
    target_cell: Cell
    claim: ClaimType
    observation_step: int
    sent_step: int
    received_step: int
    confidence: float = 1.0
    scenario_event_id: str | None = None
```

Do not include `is_malicious` or `attack_type` in the operational object.

Validate:

- unique report ID within a scenario
- valid sender and recipient IDs
- in-bounds target cell
- `observation_step <= sent_step <= received_step`
- confidence in `[0, 1]`

### 5.2 Audit label

Keep evaluation-only information in a separate lookup keyed by report ID:

```python
@dataclass(frozen=True)
class ReportAuditLabel:
    report_id: str
    is_malicious: bool
    attack_type: AttackType | None
    obstacle_episode_id: str | None
    actual_state_at_observation: ClaimType
    actual_state_at_reception: ClaimType
    original_obstacle_appearance_step: int | None
    original_obstacle_clearance_step: int | None
```

### 5.3 Active claim and history

The fusion engine should expose:

```text
report_history: report_id -> stored report
active_claims: (sender_id, cell) -> active stored report
```

A superseded report remains available for audit but contributes zero operational weight.

### 5.4 Per-recipient state

Each benign robot independently owns:

- direct belief map
- inbox
- report history and active claims
- pending verifications
- trust model
- admission policy
- fusion engine
- planner and current path
- task queue and task state
- behavior counters

No mutable trust, fusion, inbox, or belief object may be shared between robots `1` and `2`.

## 6. Exact method contracts

Use the same claim impact values and downstream cost mapping for the three evidence-based primary methods. This isolates the weighting difference.

Let:

```text
impact(BLOCKED) = +1
impact(FREE)    = -1
```

Ignore `CONGESTED` in the primary paper protocol unless it is explicitly part of an experiment.

### 6.1 Full trust

For each active claim `i` about cell `c`:

```text
weight_i = confidence_i
```

Evidence:

```text
E(c, t) = sum(weight_i * impact_i)
```

Later trust changes do not alter evidence. Claim age does not alter weight within the common active-claim validity rule.

### 6.2 Majority vote

For each sender, use only its latest active claim for the cell. Include the recipient's current direct state as a self vote when available.

```text
vote(BLOCKED) = +1
vote(FREE)    = -1
```

The majority result is discrete and does not depend on confidence or trust.

### 6.3 Trust fused

Snapshot sender trust at admission:

```text
weight_i = confidence_i * trust_at_report_i
```

Later trust changes must not revise the stored weight.

### 6.4 Source linked

Recompute sender trust at planning time:

```text
weight_i(t) = confidence_i * current_trust(sender_i, t) * age_weight_i(t)
```

Recommended age function:

```text
age_weight_i(t) = exp(-decay_rate * max(0, t - observation_step_i))
```

Later trust changes must revise old active claims. Age decay must be applied independently of trust.

### 6.5 Evidence-to-cost mapping

Use one mapping for `full_trust`, `trust_fused`, and `source_linked`:

```text
P_occ(c, t) = sigmoid(E(c, t))
risk(c, t) = max(0, 2 * (P_occ(c, t) - 0.5))
cost(c, t) = base_cost + cost_scale * risk(c, t) ^ cost_exponent
```

An empty or balanced evidence set must produce base cost.

Do not make peer evidence a hard wall in these three methods unless a separately named hard-threshold ablation is selected.

### 6.6 Cell-level invariants

For the same active blocked claims, confidence values, and time:

```text
full_trust evidence >= trust_fused evidence
```

when all trust values are in `[0, 1]`.

After a sender trust drop:

```text
full_trust:   unchanged
majority:     unchanged
trust_fused:  unchanged
source_linked: decreases
```

These are mechanism invariants. Mission outcomes are not required to follow the same ordering because planning and dynamics are nonlinear.

## 7. Trust and verification behavior

### 7.1 Bayesian trust remains the default

Use a Beta reputation model:

```text
initial alpha = 7
initial beta  = 3
trust = alpha / (alpha + beta) = 0.70
```

Recommended `paper_v1` updates:

```text
confirmed_fresh:       alpha += 1
contradicted_fresh:    beta += 1
temporally_ambiguous:  no change
unresolved:            no change
```

Do not reward a stale contradiction.

Keep scalar trust as an optional baseline, with its parameters logged.

### 7.2 Recipient-specific verification

A report is verified only when that recipient directly observes the target cell.

Robot `1` and robot `2` may update trust at different steps and may hold different trust values for robot `0`. This is intended.

A direct contradiction must:

1. update the recipient's direct map
2. classify the verification outcome without audit labels
3. update only that recipient's trust model
4. update that recipient's active claim state
5. cause source-linked evidence to be reevaluated
6. trigger replanning only through the common replan policy

### 7.3 Trust recovery

During recovery, robot `0` sends honest local observations from its fixed trajectory.

Do not guarantee trust recovery. Measure whether it occurs from genuinely verified honest reports.

Log:

- trust at attack start
- minimum trust during attack
- step first below distrust threshold
- trust at recovery start
- first step trust returns above threshold
- final trust
- number of confirmed and contradicted reports per recipient

## 8. Scenario manifest version 2

Increase the manifest schema version and reject incompatible versions with a clear message.

### 8.1 Required manifest contents

```text
schema version
scenario ID
protocol ID
master seed
named derived seeds
map content and hash
phase boundaries
robot IDs and roles
robot starts
task queues per robot
temporary obstacle episodes
attacker scripted positions or route
attacker honest report stream
malicious attack events
report audit labels
candidate-selection metadata
authoring warnings, including concentration warnings
```

### 8.2 Fixed and live behavior

Fixed across defense methods:

- static map
- temporary obstacle schedule
- starts and tasks
- attacker trajectory
- attacker honest reports
- malicious attack reports
- observation, sent, and received steps for attacker reports

Live and defense-dependent:

- benign trajectories
- benign direct sensing
- benign honest reports
- recipient-specific verification
- trust trajectories
- replanning and task performance

### 8.3 Attack types

Support these independently and in combination:

- `fake_obstacle`: blocked claim on a truly free cell
- `false_clearance`: free claim on an active real temporary obstacle
- `stale_reassertion`: fresh blocked claim after a real temporary obstacle has cleared

For stale reassertion, the transmitted observation step equals the malicious action step. The original obstacle episode is audit-only.

### 8.4 Scheduling

Use one global malicious-action budget.

Default interval:

```text
seeded integer in [40, 80]
```

Use a seeded shuffled bag for enabled attack types. If a type is infeasible, try another feasible type and record the deferral.

### 8.5 Evaluation rollout sequence

Do not enable all complexity at once.

1. Audit using clean runs and one fake-obstacle micro-scenario.
2. Validate a diverse fixed fake-obstacle manifest.
3. Add and validate false clearance in isolation.
4. Add and validate stale reassertion in isolation.
5. Run pairwise combinations.
6. Run all three together.

This sequence is for debugging, not for selecting favorable results.

## 9. Replanning and movement

### 9.1 Shared replan policy

Implement one `ReplanPolicy` used by every method. It receives facts and decides whether to replan; it must not branch on method name.

Required replan reasons:

- `initial_plan`
- `task_transition`
- `direct_blocked_next_cell`
- `direct_state_changed_on_route`
- `peer_claim_changed_route_feasibility`
- `peer_claim_changed_route_cost`
- `source_trust_changed_route_cost`
- `no_path_retry`

The source-linked method can produce the final reason because its evidence changes when trust changes, but the threshold is shared.

### 9.2 Productive replan definition

Do not define productive as merely `old_path != new_path`.

Log separate booleans:

- `path_changed`
- `restored_path_from_no_path`
- `avoided_new_direct_block`
- `reduced_estimated_path_cost`
- `reduced_ground_truth_remaining_cost` for audit only
- `productive_operational` using only operational information

Define the primary productive metric before evaluation. Recommended:

```text
productive_operational =
    restored_path_from_no_path
    OR avoided_new_direct_block
    OR new operational path cost < old operational path cost - epsilon
```

### 9.3 Distance metrics

Do not interpret greater total distance as automatically worse when delivery counts differ.

Record:

- total distance
- distance per completed delivery
- detour ratio per delivery
- distance during reconnaissance, attack, and recovery
- movement steps
- stopped/no-path steps

## 10. Logging and output contract

Keep the three requested CSV files and add small JSON metadata files.

### 10.1 Required files

```text
scenario_manifest.json
requested_config.json
effective_config.json
run_metadata.json
events.csv
robot_timeseries.csv
run_summary.csv
audit_report.json
```

### 10.2 Run metadata

Include:

- timestamp
- git commit hash and dirty flag, when available
- Python version
- dependency versions
- operating system
- engine name
- protocol ID
- scenario ID and manifest hash
- method
- trust model
- admission policy

### 10.3 Event logging

At minimum, log:

- phase transition
- temporary obstacle appeared or cleared
- attacker position
- honest report sent
- malicious report sent
- report received
- admission accepted or rejected
- active claim superseded
- direct observation changed
- report verified
- trust updated
- fusion evidence changed
- route risk changed
- replan requested
- replan completed
- no-path started and ended
- no-path cause
- blocked move
- pickup
- drop-off
- delivery completed

Every report-related event must carry stable join keys:

```text
report_id
scenario_event_id
sender_id
recipient_id
target_cell
observation_step
sent_step
received_step
```

### 10.4 Time-series logging

For each benign robot, include:

- step and phase
- position and goal
- task ID and carrying state
- path length and operational path cost
- deliveries completed
- total distance
- movement steps
- no-path steps by cause
- blocked moves
- replans and productive replans
- attacker trust
- active attacker claims
- route evidence and route risk
- map error rate

### 10.5 Run summary

Primary metrics:

- deliveries completed
- average delivery time
- detour ratio
- no-path time and cause breakdown
- false blockage persistence
- trust detection delay
- operational recovery time

Secondary metrics:

- replans per delivery
- productive replans
- blocked moves
- map error rate
- false acceptance rate
- false rejection rate
- planning time
- route stability
- distance per delivery
- final and minimum attacker trust

For metrics that cannot be computed yet, write an explicit empty value plus a `metric_unavailable_reason`; do not silently omit them.

## 11. Experiment-audit layer

Create `map_poisoning/audit.py` and `verify_end_to_end.py`.

### 11.1 Manifest audit

Check:

- team is exactly `0, 1, 2`
- robot `0` is attacker
- recipients are exactly `1, 2`
- all report IDs are unique
- timestamps are ordered
- every target is in bounds
- attack labels agree with ground truth
- fake obstacles target free cells
- false clearances target active obstacles
- stale reassertions target cleared obstacle episodes
- attacker honest stream is identical across method runs
- candidate diversity requirements are met or warnings are explicit

### 11.2 Cross-method replay audit

For one scenario, verify all methods received identical attacker-originated streams:

- report IDs
- cells
- claims
- observation steps
- sent steps
- received steps
- recipients

Benign reports are not expected to match after routes diverge.

### 11.3 Effective-configuration audit

Verify:

- selected trust model class matches configuration
- trust priors match
- admission policy matches the resolved method profile
- fusion parameters match instantiated objects
- phase boundaries match the manifest
- logs identify the same protocol and manifest hash

### 11.4 Result sanity audit

Do not reject unusual rankings. Flag diagnostic conditions:

- full-trust cell evidence lower than trust-fused for identical active blocked claims
- source-linked evidence unchanged after a trust drop
- trust-fused evidence changed after a trust drop
- repeated majority reports from one sender increase vote count
- peer-fusion no-path events without any blocking peer state
- results from different methods have different attacker streams
- requested and effective configs differ without an explicit adapter mapping

## 12. Test plan

### 12.1 Unit tests

#### Configuration and protocols

- Named protocol resolves to a complete immutable config.
- `paper_v1` cannot be partially overridden without becoming `custom`.
- Requested and effective configs serialize deterministically.

#### Team and manifest

- Exactly three robots: `0`, `1`, `2`.
- Same seed and protocol produce byte-equivalent manifests.
- Different named RNG streams do not interfere with one another.

#### Trust

- Bayesian prior starts at `0.70` for alpha 7, beta 3.
- Fresh confirmation increases trust.
- Fresh contradiction decreases trust.
- Ambiguous/expired contradiction leaves trust unchanged.
- Recipients update independently.

#### Admission

- Primary method defaults apply trust exactly once.
- Hard reject rejects below threshold.
- Optional soft gate has a documented, non-duplicated effect.

#### Fusion

- Full trust ignores later trust changes.
- Trust fused snapshots trust.
- Source linked uses current trust.
- Source linked applies age decay.
- Active claim replacement prevents same-sender stacking.
- Free and blocked claims have opposite effects.
- Empty or balanced evidence yields base cost.

#### Majority

- Repeated sender reports count once.
- Latest sender report replaces prior vote.
- One blocked and one free vote tie.
- Direct self vote is included when present.
- Positive majority hard-blocks; negative majority is free.

#### Direct precedence

- Direct blocked overrides peer free reports.
- Direct free overrides peer blocked reports.
- A later direct state replaces an earlier direct state.
- Direct state is recipient-local.

#### Recovery

- Robot `0` sends honest blocked and free observations in reconnaissance.
- Robot `0` sends honest plus malicious reports during attack.
- Robot `0` sends honest blocked and free observations in recovery.

### 12.2 Mechanism micro-scenarios

Build tiny deterministic maps that finish in seconds.

#### Micro-scenario A: one route-targeted fake obstacle

- A single corridor and a longer alternate route.
- Attacker submits one blocked claim on the short route.
- Later, a benign robot directly verifies the cell as free.

Check exact evidence, trust, path, and replan values at each step for all methods.

#### Micro-scenario B: false clearance

- A real temporary obstacle blocks the nominal route.
- Attacker reports free.
- Check whether a method routes toward the obstruction, records a blocked move, and corrects after direct sensing.

#### Micro-scenario C: stale reassertion

- A real obstacle appears and later clears.
- Attacker sends a falsely fresh blocked claim after clearance.
- Verify that the report is not treated as honest stale merely because the old obstacle episode existed.

#### Micro-scenario D: majority disagreement

- Attacker reports blocked.
- Other benign robot reports free.
- Recipient has no direct observation, producing a tie.
- Then recipient directly observes free, producing a free majority/direct override.

### 12.3 Deterministic integration tests

- Author one manifest.
- Replay it twice with the same method.
- Compare summaries and event streams after removing wall-clock timing fields.
- Replay all four methods and verify identical attacker-originated streams.
- Verify that at least one mechanism-level quantity differs where expected; do not require a mission-performance ordering.

### 12.4 Legacy parity tests

Before retiring `sim2.py`, run a small deterministic clean case through legacy and modular engines with equivalent configuration.

Compare only behavior that should match:

- map and starts
- tasks
- obstacle schedule
- basic sensing radius/cadence
- delivery state transitions
- no-attack path feasibility

Do not force exact path or metric parity when movement models differ. Document intentional differences.

## 13. Evaluation sequence after correctness passes

### 13.1 Calibration suite

Use at least:

```text
2 maps x 5 seeds
```

For each scenario, run:

- clean no-attack baseline
- fixed fake-obstacle manifest
- all four primary methods

Report every seed. Do not select representative favorable seeds.

### 13.2 Attack ablations

After the calibration suite passes:

- fake obstacle only
- false clearance only
- stale reassertion only
- all pairwise combinations
- all three

Use the same protocol and predeclared scenario-generation rules.

### 13.3 Trust and admission ablations

Keep these separate from the main four-method comparison:

- Bayesian versus scalar trust
- method-default admission versus hard reject
- optional soft gate

Do not combine all ablations into one initial sweep.

### 13.4 Reporting

Use paired comparisons by scenario ID and seed.

For each primary metric, report:

- every seed-level value
- mean and median
- spread or confidence interval
- paired difference from clean baseline
- paired difference between methods

Explain unusual cases using event traces and no-path provenance, not by changing parameters.

## 14. GUI and CLI updates

### 14.1 Main GUI tab

Show only the common settings:

- map selector
- seed
- protocol selector
- defense method
- trust model
- admission policy
- attack checkboxes
- generate or load manifest
- animation checkbox
- output directory
- Start Simulation
- Run Selected Defenses

Default values should come from the selected named protocol.

### 14.2 Advanced tab

Place these settings under Advanced or More Options:

- phase lengths
- attack interval
- candidate top K
- diversity limits
- Bayesian priors and threshold
- claim validity window
- decay rate
- cost scale and exponent
- maximum claim age
- replan cost threshold and cooldown
- communication cadence
- time-series logging cadence
- engine selection during migration

When a frozen protocol is selected, editing any advanced field must change the protocol display to `custom`.

### 14.3 CLI

Support:

```text
--protocol
--seed
--defense-method
--trust-model
--admission-policy
--attacks
--manifest
--manifest-only
--compare
--headless
--output-directory
```

Advanced protocol overrides may remain available, but any override must cause the effective protocol ID to become `custom`.

## 15. Implementation order

Follow this order. Do not skip directly to large comparison runs.

### Phase 0: Snapshot and baseline

1. Record the current commit hash.
2. Preserve the current exploratory outputs unchanged.
3. Add a note labeling them exploratory.
4. Run existing tests and record results.
5. Add a short deterministic no-attack legacy baseline.

Exit condition: current behavior is reproducible and preserved for reference.

### Phase 1: Protocol and effective configuration

1. Add named protocols.
2. Add requested/effective config serialization.
3. Wire trust and fusion settings into the legacy adapter.
4. Log instantiated parameters.
5. Add configuration audit tests.

Exit condition: logs accurately describe the objects that ran.

### Phase 2: Team and manifest v2

1. Fix team IDs to `0`, `1`, `2`.
2. Add starts and task queues to the manifest.
3. Add fixed attacker trajectory.
4. Add fixed honest attacker reports.
5. Separate audit labels from reports.
6. Add manifest validation.

Exit condition: all methods receive the same complete attacker stream.

### Phase 3: Canonical fusion implementation

1. Make one canonical fusion engine.
2. Add active claim replacement.
3. Implement exact method contracts.
4. Fix majority voting.
5. Remove trust double-counting.
6. Add fusion contract tests.

Exit condition: all cell-level invariants pass.

### Phase 4: Direct belief and verification

1. Implement direct free and blocked precedence.
2. Move verification policy into its own module.
3. Remove age-only honesty inference.
4. Make trust updates recipient-specific.
5. Add direct-precedence and verification tests.

Exit condition: a micro-scenario trace is explainable report by report.

### Phase 5: Recovery and replan policy

1. Make recovery fully honest.
2. Use one shared replan policy.
3. Remove or explicitly protocolize method-specific thresholds.
4. Improve productive-replan logging.

Exit condition: recovery reports and replan reasons are verified by tests.

### Phase 6: Metrics and no-path provenance

1. Add no-path cause classification.
2. Expand events and time series.
3. Add effective config and metadata files.
4. Implement missing primary metrics where possible.

Exit condition: every surprising mission result can be traced to events and local map state.

### Phase 7: Manifest diversity

1. Add footprint identity and reuse counts.
2. Add spacing and minimum-diversity constraints.
3. Save candidate scores and rank.
4. Add concentration warnings.

Exit condition: authored scenarios satisfy declared diversity rules or fail clearly.

### Phase 8: End-to-end verifier

1. Implement `verify_end_to_end.py`.
2. Run unit tests.
3. Run micro-scenarios.
4. Author and replay a fixed manifest.
5. Audit cross-method attacker streams.
6. Validate CSV and JSON output schemas.
7. Produce a human-readable audit summary.

Exit condition: one command gives a pass/fail report without checking for a preferred method ranking.

### Phase 9: Freeze `paper_v1`

1. Review all resolved parameters.
2. Document the rationale for each.
3. Freeze the protocol.
4. Tag the commit.
5. Run the calibration suite on previously unused seeds.

Exit condition: evaluation begins only after the protocol is frozen.

## 16. `verify_end_to_end.py` behavior

The script should run from the repository root:

```bash
python verify_end_to_end.py
```

It should:

1. Run `pytest`.
2. Run the three or four tiny mechanism scenarios.
3. Author one deterministic manifest.
4. Replay all four primary methods.
5. Check attacker stream equality.
6. Check requested/effective config equality.
7. Check method invariants from logged cell evidence.
8. Check team IDs and recovery behavior.
9. Check output files and required columns.
10. Write `verification_outputs/audit_summary.md` and `audit_report.json`.

The script exits nonzero on correctness failures.

It may print metric tables for inspection, but it must not fail because source-linked is not best or trust-fused is worst.

## 17. Acceptance criteria

The audit is complete only when all of the following are true:

- Exactly three robots exist, with attacker `0` and benign robots `1` and `2`.
- Each benign robot owns independent trust, admission, fusion, belief, inbox, path, and task state.
- The same complete attacker stream is replayed across methods.
- Operational code cannot read malicious audit labels.
- Requested and effective configurations match or contain an explicit documented adapter mapping.
- Bayesian selection actually creates Bayesian trust models.
- Full trust, trust fused, and source linked satisfy their weighting contracts.
- Majority voting gives one vote per sender and uses discrete majority state.
- Direct free and blocked observations override peer fusion locally.
- Repeated same-sender reports supersede rather than stack.
- Recovery includes honest free and blocked reports.
- No-path time is broken down by cause.
- Source-linked-specific heuristics are removed or explicitly frozen as method parameters.
- All tests pass on Windows and macOS.
- Same seed, protocol, and manifest produce deterministic results.
- The calibration suite uses unused seeds after protocol freeze.
- No acceptance test requires a particular method to win.

## 18. Interpretation of the current exploratory table

Keep the current table as an exploratory artifact only.

It is plausible that source-linked outperforms trust-fused because source-linked can reduce the influence of earlier high-trust lies after trust drops, while trust-fused cannot. It is also possible for trust-fused to have worse mission outcomes than full trust because small cell-cost differences can cause different routes, sensing opportunities, obstacle encounters, and long no-path periods.

However, the current table cannot establish that ranking until the implementation and audit issues above are resolved. In particular, majority semantics, effective parameter logging, trust wiring, recovery behavior, direct precedence, repeated claim handling, attacker-stream completeness, and no-path provenance can all affect interpretation.

The correct response to a surprising ranking is to trace and test the mechanism, not to tune the method until the ranking looks expected.

