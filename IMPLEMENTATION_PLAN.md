# Modular Map-Poisoning Simulator: Implementation and Verification Plan

## 1. Purpose

Refactor the existing `custom_map_poisoning_costmap` project into a modular, repeatable experimental framework for comparing multi-robot map-sharing defenses against a delayed attacker.

The implementation must preserve the current simulator's useful behavior while adding:

- Three independently selectable attack types.
- Seeded, fixed attack manifests replayed identically across defense methods.
- Swappable trust models, admission policies, and map-fusion methods.
- A three-phase experiment: reconnaissance, poisoning, and recovery.
- CSV event, time-series, and summary logging.
- A Tkinter launcher with a simple main tab and an advanced-options tab.
- Headless command-line operation for automated experiments.
- Tests and verification scripts that prove deterministic and method-specific behavior.

The code must remain simple to understand and extend. Avoid building a general robotics framework or adding abstractions that are not needed by this experiment.

---

## 2. Source Material and Existing Project

Implementation must be grounded in:

1. Current repository:
   `https://github.com/natchuop/custom_map_poisoning_costmap`
2. Project description document:
   `Physical AI Notes(1).docx`
3. Decisions recorded in the conversation that produced this plan.

Current repository facts that must be considered during migration:

- `sim2.py` contains most simulator behavior and is approximately 4,870 lines.
- `defense_method_runner.py` already supports:
  - `hard_threshold`
  - `soft_probability`
  - `time_decay`
  - `trust_fused`
  - `source_linked`
- Existing runtime requirements are:
  - NumPy
  - Matplotlib
  - Pillow
  - PyYAML
- Tkinter will be added for the launcher. Tkinter is part of many normal Python installations but must be checked separately because it is not installed through `pip` on every system.
- The current attacker chooses fake-obstacle cells using global map and route information. It is not limited to cells that the attacker directly senses. Preserve that attacker-awareness assumption for all three attack types.
- The current attack code broadcasts reports to the other robots. Preserve broadcast behavior.
- The current Bayesian trust model is a Beta-reputation model with a default prior of `alpha=7`, `beta=3`, producing an initial trust of `0.70`.
- The current `should_accept()` trust method is not consistently used by inbox processing. The refactor must intentionally separate trust estimation, admission, and map fusion instead of accidentally relying on an unused threshold.

---

## 3. Locked Experimental Decisions

These are requirements, not suggestions.

### 3.1 Default experiment phases

Use simulation steps only.

| Phase | Default range | Length | Attacker behavior |
|---|---:|---:|---|
| Reconnaissance | `0-499` | 500 | Fully honest; observes benign movement and builds traffic knowledge |
| Poisoning | `500-1699` | 1200 | Continues normal reporting and injects malicious claims |
| Recovery | `1700-2499` | 800 | Returns to fully honest reporting, including both `BLOCKED` and `FREE` observations |
| Total | `0-2499` | 2500 | Fixed-length run unless all delivery tasks finish first |

The phase lengths must be configurable in the advanced GUI tab and through CLI arguments. Total steps are the sum of the three phase lengths.

### 3.2 Attack types

Support these independently selectable attacks:

1. `fake_obstacle`
   - Claim that currently free cells are blocked.
2. `false_clearance`
   - Claim that cells belonging to a currently active real temporary obstacle are free.
   - The attacker may target an obstacle it learned directly, through another report, or through its configured global awareness.
3. `stale_reassertion`
   - After a real temporary obstacle has cleared, claim that it is still blocked.
   - Send a falsely fresh observation timestamp equal to the attack step.
   - Store the true original obstacle episode and clearance step only in audit/ground-truth metadata, never in the operational report presented to a benign robot.

All enabled attack types must work alone or in any combination. With three attack types, the system must support all eight ablations, including no attack.

### 3.3 Attack scheduling

- Use one shared attack-action budget, not a separate timer per attack type.
- Default interval between attack actions: a seeded random integer from 40 through 80 steps.
- Draw intervals when authoring the scenario manifest. Replay the recorded steps exactly.
- Use a seeded shuffled bag for attack-type selection so all enabled and feasible types occur regularly without a fixed order.
- If the next bag item is temporarily infeasible:
  1. Try another enabled feasible item.
  2. Keep the infeasible item available for a later opportunity.
  3. Record a skipped/deferred scheduling event.
- A repeated refresh of a false-clearance claim is allowed and consumes one full attack action.
- One attack action may create multiple cell-level reports when the target is a multi-cell footprint.

### 3.4 Attack target selection

Keep the current global-awareness threat model for all attacks.

The attack-authoring process may use:

- The full static map.
- The full temporary-obstacle episode history.
- Benign nominal routes from the clean authoring rollout.
- The benign traffic heatmap.
- Current and future temporary-obstacle schedule information when generating a fixed scenario.
- Bottleneck and chokepoint scores.
- Ground-truth free/blocked state for candidate validity.

Benign robots must not receive any of this hidden information.

Target selection must be disruptive but not deterministic in the sense of always selecting the single maximum-scoring cell. Use seeded, impact-weighted sampling from the top candidates.

Recommended default target-scoring components:

- Nominal benign route overlap.
- Traffic frequency, preferring useful medium-to-high traffic instead of only the hottest cells.
- Bottleneck/chokepoint score.
- Estimated route detour or path loss if the claim is believed.
- Distance outside immediate benign lidar verification range.
- Likelihood that the false claim persists before direct verification.
- For false clearance: likelihood that the lie causes planning through a real obstruction.
- For stale reassertion: previous operational importance of the cleared obstacle episode.

Normalize component scores before combining them. Put weights in advanced configuration. Select from the top `K` candidates using seeded rank-weighted or softmax sampling. Default `K=12`.

### 3.5 Fixed attack manifests

The exact same attack chain must be replayed across defense methods.

The scenario-authoring rollout is a canonical clean rollout:

1. Generate map, robot starts, task queues, and temporary-obstacle schedule from the master seed.
2. Run with the attacker behaving honestly and with no malicious reports.
3. Record nominal benign traffic and routes.
4. Generate attack events against that nominal behavior.
5. Save a versioned JSON manifest.
6. Replay the same manifest unchanged against every defense method.

The manifest must fix at least:

- Map identity and content hash.
- Master seed and named derived seeds.
- Robot IDs, roles, starts, and task queues.
- Temporary-obstacle episodes, cells, appearance steps, and clearance steps.
- Phase boundaries.
- Attack event steps.
- Attack types.
- Target cells and grouped footprint IDs.
- Claimed state.
- Falsified operational observation step.
- Original obstacle episode ID where relevant.
- Broadcast recipients.
- Report IDs.
- Scenario schema version.

A defense replay must not call attack target-selection code. It may only replay recorded attack events.

### 3.6 Default trust and admission

- Default trust model: `bayesian`.
- Keep `scalar` as a supported baseline.
- Default Bayesian prior:
  - `alpha=7.0`
  - `beta=3.0`
  - initial trust `0.70`
- Default distrust threshold: `0.55`.
- Default admission behavior: soft/low influence, meaning reports are not discarded solely because trust is low; trust-aware fusion methods reduce their influence.
- Admission must be independently selectable and method-aware.
- Hard rejection must remain an optional policy.

### 3.7 Primary defense methods

Primary comparison set:

1. `full_trust`
2. `majority_vote`
3. `trust_fused` (MATE-style)
4. `source_linked`

Supplementary/legacy methods that must remain functional because they already exist:

- `hard_threshold`
- `soft_probability`
- `time_decay`

Do not prioritize new research work on the supplementary methods. Preserve them, test them, and keep CLI/GUI support.

### 3.8 Communication and recipient behavior

- Malicious reports are broadcast to all benign robots.
- Honest reports continue to use the same report format.
- The operational report format must not reveal whether a report is malicious.
- Simulation steps remain the timestamp unit.

### 3.9 Logging outputs

Every experiment output directory must contain exactly these three primary CSV files:

1. `run_summary.csv`
2. `robot_timeseries.csv`
3. `events.csv`

Additional non-CSV files are allowed, including:

- `scenario_manifest.json`
- `resolved_config.json`
- figures
- optional console log

---

## 4. Design Principles

1. **Behavior-preserving refactor first.** Move code before changing algorithms.
2. **One responsibility per module.** Do not recreate `sim2.py` as another large file.
3. **No hidden global configuration.** Runtime behavior comes from immutable configuration objects passed explicitly.
4. **No circular imports.** Shared dataclasses and enums live in a low-level models module.
5. **No truth leakage.** Defense and trust code cannot inspect `attack_type`, `is_malicious`, true obstacle history, or scenario audit labels.
6. **Deterministic experiments.** Same seed and config must reproduce the same scenario manifest and attack chain.
7. **Named RNG streams.** Adding a new random subsystem must not silently alter unrelated existing random streams.
8. **Headless core.** Simulation logic must not import Tkinter or require a display.
9. **Thin UI.** The GUI only edits a `SimulationConfig` and invokes application services.
10. **Simple extension points.** A new trust model, attack, fusion method, or admission policy should require one class plus one registry entry.
11. **No defense-specific attacker adaptation.** This study compares defenses against fixed scenarios.
12. **Preserve direct sensing authority.** A robot's fresh direct observation must remain separate from peer-fused evidence.
13. **Do not overfit to one map or seed.** No map-coordinate constants inside attack or defense logic.

---

## 5. Target Repository Structure

Use this structure unless an equivalent structure is clearly simpler. Avoid splitting tiny helpers into separate files without reason.

```text
custom_map_poisoning_costmap/
├── main.py
├── check_environment.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── convert_maps.py
├── run_sim.ps1                 # update for new entry point
├── run_sim.command             # optional macOS convenience launcher
├── sim2.py                     # temporary compatibility wrapper after migration
├── map_poisoning/
│   ├── __init__.py
│   ├── config.py               # immutable config dataclasses, defaults, validation
│   ├── models.py               # enums, reports, events, tasks, obstacle episodes
│   ├── rng.py                  # stable named RNG derivation
│   ├── map_io.py               # NPY, MovingAI, demo-map loading
│   ├── world.py                # truth grid and temporary-obstacle manager
│   ├── sensing.py              # lidar/raycast and direct observations
│   ├── communication.py        # broadcast, inboxes, optional delay hooks
│   ├── planning.py             # A*, route costs, path utilities
│   ├── robot.py                # robot state and orchestration of components
│   ├── simulation.py           # simulation loop and phase transitions
│   ├── scenario.py             # manifest schema, authoring, loading, replay
│   ├── visualization.py        # Matplotlib animation and static diagnostics
│   ├── cli.py                  # argument parsing and config merging
│   ├── application.py          # run one simulation or comparison batch
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── base.py             # AttackStrategy and read-only AttackContext
│   │   ├── scheduler.py        # interval and shuffled-bag scheduling
│   │   ├── fake_obstacle.py
│   │   ├── false_clearance.py
│   │   └── stale_reassertion.py
│   ├── trust/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── scalar.py
│   │   ├── bayesian.py
│   │   └── factory.py
│   ├── admission/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── policies.py
│   │   └── factory.py
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── full_trust.py
│   │   ├── majority_vote.py
│   │   ├── trust_fused.py
│   │   ├── source_linked.py
│   │   ├── legacy.py
│   │   └── factory.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── collector.py
│   │   ├── calculations.py
│   │   └── csv_writer.py
│   └── ui/
│       ├── __init__.py
│       ├── app.py
│       ├── basic_tab.py
│       ├── advanced_tab.py
│       └── validation.py
└── tests/
    ├── unit/
    ├── integration/
    ├── fixtures/
    └── golden/
```

Keep `defense_method_runner.py` during migration. Once all methods are moved and regression-tested, either:

- replace it with a compatibility import wrapper, or
- remove it in a final cleanup commit after all callers are updated.

Do not delete `sim2.py` at the start. After successful migration, reduce it to a small compatibility wrapper that invokes `main.py` and prints a deprecation notice.

---

## 6. Core Data Models

Place shared models in `map_poisoning/models.py`. They must not import robot, world, trust, fusion, GUI, or visualization modules.

### 6.1 Required enums

```python
class ClaimType(IntEnum):
    FREE = 0
    BLOCKED = 1
    CONGESTED = 2  # retained for legacy support

class SimulationPhase(str, Enum):
    RECONNAISSANCE = "reconnaissance"
    ATTACK = "attack"
    RECOVERY = "recovery"

class AttackType(str, Enum):
    FAKE_OBSTACLE = "fake_obstacle"
    FALSE_CLEARANCE = "false_clearance"
    STALE_REASSERTION = "stale_reassertion"

class VerificationOutcome(str, Enum):
    CONFIRMED = "confirmed"
    CONTRADICTED_FRESH = "contradicted_fresh"
    HONEST_STALE_OR_EXPIRED = "honest_stale_or_expired"
    UNRESOLVED = "unresolved"
```

Retain existing cell-state values where compatibility requires them.

### 6.2 Operational claim report

Do not include `is_malicious` or `attack_type` in the report object given to robots.

```python
@dataclass(frozen=True)
class ClaimReport:
    report_id: str
    sender_id: int
    target_cell: Cell
    claim: ClaimType
    observation_step: int
    sent_step: int
    received_step: int | None = None
    confidence: float = 1.0
    scenario_event_id: str | None = None
```

`received_step` may be filled by the communication layer by returning a copied dataclass.

### 6.3 Audit label kept outside operational code

```python
@dataclass(frozen=True)
class ReportAuditLabel:
    report_id: str
    is_malicious: bool
    attack_type: AttackType | None
    obstacle_episode_id: str | None
    actual_state_at_observation: ClaimType
    original_obstacle_appearance_step: int | None
    original_obstacle_clearance_step: int | None
```

The metrics collector may access audit labels. Trust, admission, fusion, planning, and benign robot code may not.

### 6.4 Other required models

Define typed dataclasses for:

- `DeliveryTask`
- `DirectObservation`
- `PendingVerification`
- `TemporaryObstacleEpisode`
- `AttackEvent`
- `ScenarioManifest`
- `SimulationEvent`
- `RunIdentity`
- `RunResult`
- `AdmissionDecision`
- `TrustUpdate`
- `ReplanRecord`

Every event and report needs a stable ID so events can be joined across CSV files.

---

## 7. Configuration System

### 7.1 Single source of truth

Create immutable nested dataclasses in `config.py`, for example:

```python
@dataclass(frozen=True)
class PhaseConfig: ...

@dataclass(frozen=True)
class AttackConfig: ...

@dataclass(frozen=True)
class TrustConfig: ...

@dataclass(frozen=True)
class FusionConfig: ...

@dataclass(frozen=True)
class LoggingConfig: ...

@dataclass(frozen=True)
class VisualizationConfig: ...

@dataclass(frozen=True)
class SimulationConfig: ...
```

No simulator module should read module-level configuration constants after migration.

### 7.2 Required defaults

```text
seed = 15
trust_model = bayesian
bayesian_prior_alpha = 7.0
bayesian_prior_beta = 3.0
trust_threshold = 0.55
admission_policy = auto_soft
single_defense_method = source_linked
comparison_methods = full_trust, majority_vote, trust_fused, source_linked
attacks_enabled = fake_obstacle, false_clearance, stale_reassertion
recon_steps = 500
attack_steps = 1200
recovery_steps = 800
attack_interval_min = 40
attack_interval_max = 80
attack_candidate_top_k = 12
broadcast_attacks = true
attacker_global_awareness = true
recovery_attacker_is_fully_honest = true
communication_period_steps = 4
temporary_blockage_change_period_steps = 400
animation = true
timeseries_period_steps = 5
output_directory = outputs
```

Preserve current source-linked defaults unless validation shows a reason to change them:

```text
decay_rate = 0.006
cost_scale = 14.0
cost_exponent = 1.5
blocked_probability_threshold = 0.70
max_claim_age = 900
congested_impact = 0.50
duplicate_window_steps = 0
```

### 7.3 Validation

Reject invalid configurations before the simulation starts:

- Negative seed is allowed only if explicitly supported; otherwise require nonnegative.
- Phase lengths must be positive, except attack length may be zero for a clean run.
- Minimum attack interval must be at least one.
- Maximum interval must be greater than or equal to minimum.
- At least one defense must be selected for comparison mode.
- A replay run requires a manifest path.
- A loaded manifest's map hash and critical scenario configuration must match the run.
- Probabilities and trust thresholds must be in `[0, 1]`.
- Decay and cost scales must be nonnegative.
- GUI errors must be displayed clearly without a stack trace.
- Headless errors must return a nonzero exit code.

---

## 8. Stable Randomness and Reproducibility

Do not rely on a single shared global RNG.

Derive named streams from the master seed with a stable hash, not by positional `spawn()` ordering. This ensures adding a new random subsystem does not change older streams.

Example named streams:

```text
map_generation
robot_start_positions
delivery_tasks
temporary_obstacle_pool
temporary_obstacle_schedule
attack_schedule
attack_type_order
fake_obstacle_targets
false_clearance_targets
stale_reassertion_targets
visual_only
```

Recommended derivation:

1. Create bytes from `f"{master_seed}:{stream_name}"`.
2. Hash with SHA-256.
3. Convert the first 64 or 128 bits to an integer.
4. Pass it to `numpy.random.default_rng()`.

Requirements:

- Same master seed and config produce byte-identical manifests.
- Adding a new named RNG stream does not alter existing streams.
- Visualization must never consume an experiment RNG stream.
- Replay mode must not generate new attack randomness.

---

## 9. Scenario Authoring and Manifest Replay

### 9.1 Authoring service

Implement a `ScenarioAuthor` in `scenario.py`.

Responsibilities:

1. Resolve map and hash its static-grid contents.
2. Generate robot starts and fixed delivery-task queues.
3. Generate the full temporary-obstacle episode schedule for all 2,500 default steps.
4. Run the clean canonical rollout.
5. Build nominal benign route and traffic traces.
6. Invoke the seeded attack scheduler and strategies to generate `AttackEvent` objects.
7. Validate every event against world truth at its authoring step.
8. Write a versioned JSON manifest using sorted keys and stable formatting.

### 9.2 Manifest replay service

Implement `ScenarioReplayer`.

Responsibilities:

- Load and validate the manifest.
- Recreate identical starts, tasks, obstacle schedule, phases, and attack events.
- Inject attack reports at recorded sent steps.
- Broadcast to recorded recipients.
- Never rescore or retarget attacks.
- Produce report IDs identical across defense runs.

### 9.3 Comparison batch

`application.py` must support:

```python
run_comparison(config, manifest, defense_methods) -> list[RunResult]
```

For each defense:

- Reset all mutable simulation state.
- Use the same manifest.
- Use a unique `run_id` but the same `scenario_id`.
- Write rows to the same three experiment CSVs.
- Do not show four animations sequentially by default. Comparison runs should default to headless or summary visualization.

### 9.4 Clean baseline

For every scenario used in a paper-quality comparison, also support a clean baseline replay with no malicious events. Use the same world and tasks. Baseline data is required for recovery and performance comparisons.

---

## 10. Attack Architecture

### 10.1 Interface

```python
class AttackStrategy(ABC):
    attack_type: AttackType

    @abstractmethod
    def feasible_candidates(self, context: AttackContext) -> list[AttackCandidate]: ...

    @abstractmethod
    def build_event(
        self,
        candidate: AttackCandidate,
        context: AttackContext,
        rng: np.random.Generator,
    ) -> AttackEvent: ...
```

`AttackContext` must be read-only and available only to scenario authoring. It may expose global truth because global awareness is part of the threat model.

### 10.2 Fake obstacle

Candidate rules should preserve useful current behavior:

- Target currently free/action cells.
- Candidate footprint may visually overlap walls, but only traversable target cells receive false reports.
- Prefer route overlap and useful traffic.
- Avoid goals.
- Avoid cells currently within benign lidar range.
- Avoid repeatedly selecting nearly identical centers unless reinforcement is intentionally allowed.
- Use the existing rectangular footprint defaults initially.

Output one grouped `AttackEvent` containing one `BLOCKED` report per valid footprint cell.

### 10.3 False clearance

- Candidate must belong to an active real temporary-obstacle episode.
- Claim `FREE` for selected cells while truth is blocked.
- Prefer obstacles that currently influence nominal benign routes.
- Prefer targets outside immediate direct verification where possible, but do not require impossible geometry.
- Allow refreshing the same false-clearance episode later; each refresh is a separate action and separate report IDs.
- Keep the operational observation step fresh at the action step.

### 10.4 Stale reassertion

- Candidate must belong to a temporary-obstacle episode that previously existed and has cleared.
- Claim `BLOCKED` with `observation_step=attack_step`.
- Audit metadata stores the old episode's appearance and clearance steps.
- Prefer former obstacles that were route-relevant and whose cells are currently free.
- Reject a candidate if a new real obstacle now occupies the same cells.

### 10.5 Scheduler

The scheduler owns:

- attack intervals,
- shuffled bag,
- feasibility fallback,
- global action count,
- event ordering,
- deterministic IDs.

Strategies do not own timing.

---

## 11. Trust Models

### 11.1 Interface

```python
class TrustModel(ABC):
    def score(self, sender_id: int) -> float: ...
    def observe_verification(self, result: VerificationResult) -> TrustUpdate | None: ...
    def snapshot(self) -> dict[str, object]: ...
```

Do not put admission decisions inside the trust model.

### 11.2 Scalar trust

Preserve current behavior as a baseline:

- Unknown sender starts at `0.70`.
- Confirmed report adds configurable reward, initially `0.02`.
- Fresh contradiction subtracts configurable penalty, initially `0.06`.
- Clamp to `[0,1]`.

### 11.3 Bayesian trust

Default method:

- Each sender has Beta parameters `(alpha, beta)`.
- Score is `alpha / (alpha + beta)`.
- Start at `(7,3)`.
- Confirmed report increments `alpha`.
- Fresh malicious-looking contradiction increments `beta`.
- Store alpha, beta, score, and change step in logs.

### 11.4 Honest-stale versus malicious contradiction

Implement in two stages.

#### Stage A: modularization compatibility

During initial refactor, preserve the existing verification outcome so behavior can be regression-tested.

#### Stage B: temporal-aware verification

Add a separate `VerificationPolicy` or `ClaimValidityEvaluator` that classifies a contradiction using only information available to the receiving robot:

- claim age,
- observation and reception steps,
- expected obstacle persistence,
- corroboration or contradiction from other robots,
- later direct sensing,
- repeated reassertion behavior,
- reporter trust history.

Operational behavior:

- `CONFIRMED`: map influence remains or strengthens; source trust increases.
- `HONEST_STALE_OR_EXPIRED`: map influence decreases/expires; source trust changes little or not at all.
- `CONTRADICTED_FRESH`: map influence decreases; source trust decreases.
- `UNRESOLVED`: no trust update yet.

Do not let this classifier read the audit label. The audit label is used only to evaluate classifier accuracy.

The falsely fresh stale-reassertion attack should tend to look like `CONTRADICTED_FRESH` after direct verification, which is intentional.

---

## 12. Admission Policies

Separate admission from trust and fusion.

### 12.1 Interface

```python
class AdmissionPolicy(ABC):
    def decide(
        self,
        report: ClaimReport,
        sender_trust: float,
        method_name: str,
    ) -> AdmissionDecision: ...
```

### 12.2 Required policies

1. `method_default` / GUI label `Soft/low influence (recommended)`
   - `full_trust`: accept.
   - `majority_vote`: accept.
   - `trust_fused`: accept; fusion applies trust-at-entry.
   - `source_linked`: accept; fusion applies current trust.
   - supplementary methods: accept unless their own documented behavior says otherwise.
2. `accept_all`
   - Always accept syntactically valid reports.
3. `hard_reject`
   - Reject reports below the configured threshold.

A rejected report must still be recorded in `events.csv` with a reason, but must not enter the fusion method.

Do not multiply trust twice. Admission either accepts or rejects. Trust-aware weighting belongs to fusion.

---

## 13. Fusion and Defense Methods

### 13.1 Common interface

```python
class FusionMethod(ABC):
    name: str

    def add_report(self, report: ClaimReport, trust_at_receive: float) -> bool: ...
    def observe_direct(self, observation: DirectObservation) -> None: ...
    def set_step(self, step: int) -> None: ...
    def cell_risk(self, cell: Cell) -> float: ...
    def traversal_cost(self, cell: Cell) -> float: ...
    def is_hard_blocked(self, cell: Cell) -> bool: ...
    def snapshot(self) -> dict[str, object]: ...
```

Direct observation storage remains separate from peer evidence. A fresh direct observation should override or strongly dominate peer claims for the locally observed cell.

### 13.2 Full trust

Primary baseline:

- Treat all peer claims equally.
- Ignore sender trust and claim age except for a configurable absolute expiry that prevents permanent storage.
- Occupied claims contribute positive evidence.
- Free claims contribute negative evidence.
- Convert evidence to occupancy probability and continuous cost using the common probability/cost transformation.

For backward compatibility, `soft_probability` may share the same implementation or be a documented alias. Keep the CLI name working.

### 13.3 Majority vote

- One active vote per sender per cell.
- Use the sender's newest active report for that cell.
- Equal vote weight.
- Expire votes after a configurable validity window.
- Majority occupied -> occupied risk.
- Majority free -> free/no peer risk.
- Tie or no votes -> neutral/no peer risk.
- A robot cannot gain multiple votes by spamming duplicate reports.
- Direct local observation is not just another peer vote; it remains authoritative locally.

### 13.4 MATE-style trust-fused

- Weight a report by sender trust at the moment it enters the map.
- Later trust changes must not alter its existing contribution.
- Operational per-cell fused evidence should not need the source identity after fusion.
- Source identity may remain in audit logs but cannot be consulted to recalculate the operational map.
- Use consistent occupied/free impacts and cost conversion so comparisons isolate the evidence-revision difference.

### 13.5 Source-linked proposed method

Store claims separately and retain source linkage.

Default claim weight:

```text
weight_i(t) = current_trust(sender_i, t)
              * exp(-decay_rate * claim_age)
              * confidence_i
```

Impacts:

```text
BLOCKED = +1
FREE = -1
CONGESTED = configurable legacy impact
```

Per-cell evidence:

```text
E(cell, t) = sum(weight_i(t) * impact_i)
```

Convert to occupancy probability with a numerically stable logistic function. Preserve the current useful behavior that no claims means zero peer risk rather than a default `0.5` cost penalty.

Routing cost:

```text
cost = 1 + cost_scale * occupied_risk ** cost_exponent
```

Requirements:

- Current trust changes retroactively alter old claim influence.
- Claim age changes map influence without automatically reducing source trust.
- Fresh direct verification can invalidate or supersede a claim.
- The method can expose source-attributed route risk for replan decisions and logging.

### 13.6 Supplementary methods

Preserve:

- `hard_threshold`
- `soft_probability`
- `time_decay`

Place them in `fusion/legacy.py` or share common implementation helpers. Do not duplicate the entire fusion pipeline.

---

## 14. Robot, Sensing, Communication, and Planning

### 14.1 Robot composition

`GridRobot` should coordinate injected components:

- trust model,
- admission policy,
- fusion method,
- planner,
- sensor,
- inbox/outbox,
- task state.

It should not choose attack targets or write CSV files directly.

### 14.2 Sensor behavior

Preserve current lidar/raycast behavior during refactor. Emit typed `DirectObservation` objects containing:

- robot ID,
- cell,
- observed state,
- step,
- confidence.

### 14.3 Honest reporting by phase

- Reconnaissance: attacker and benign robots report honest `BLOCKED` and `FREE` observations according to the same communication rules.
- Attack: attacker continues honest normal reports and additionally broadcasts manifest attacks.
- Recovery: attacker reports fully honestly again, including both `BLOCKED` and `FREE`.

### 14.4 Planning

Move A* and route utilities to `planning.py` without changing behavior first.

Planner input should be an interface that can answer:

- static/dynamic direct block state,
- peer-derived traversal cost,
- hard-block condition if applicable.

Instrument planning with `time.perf_counter()` and expanded-node counts.

---

## 15. Metrics and CSV Logging

### 15.1 Logging architecture

Use a synchronous `MetricsCollector` called explicitly by the simulator. Do not add a complex asynchronous event bus.

- `MetricsCollector.record(event)` stores or streams events.
- `CsvWriter` writes incrementally.
- Flush periodically and on shutdown.
- Use standard-library `csv` and `json`; do not add pandas as a runtime dependency.

### 15.2 Output directory layout

```text
outputs/
└── <timestamp>_<scenario_id>/
    ├── resolved_config.json
    ├── scenario_manifest.json
    ├── run_summary.csv
    ├── robot_timeseries.csv
    ├── events.csv
    └── figures/
```

### 15.3 Common identifiers

Every CSV row should include the relevant subset of:

```text
experiment_id
scenario_id
manifest_hash
run_id
seed
defense_method
trust_model
admission_policy
attack_configuration
step
phase
robot_id
```

### 15.4 `events.csv`

Use fixed common columns plus a `details_json` column for event-specific data.

Recommended columns:

```text
event_id
experiment_id
scenario_id
run_id
step
phase
event_type
robot_id
sender_id
recipient_id
report_id
scenario_event_id
cell_row
cell_col
claim
attack_type_audit
is_malicious_audit
accepted
reason
trust_before
trust_after
path_length_before
path_length_after
planning_time_seconds
details_json
```

The `_audit` fields are written by the metrics layer after joining on `report_id`. They are never put into the operational `ClaimReport`.

Required event types include:

```text
phase_started
phase_ended
temporary_obstacle_appeared
temporary_obstacle_cleared
report_sent
report_received
report_admitted
report_rejected
report_stored
report_pruned
claim_verified
claim_expired
attack_action_injected
attack_opportunity_deferred
trust_changed
distrust_threshold_crossed
path_planned
path_replanned
no_path_started
no_path_ended
delivery_started
delivery_completed
route_reversal
false_claim_stopped_affecting_map
false_claim_stopped_affecting_route
simulation_finished
```

### 15.5 `robot_timeseries.csv`

Default sample period: every 5 steps. Events preserve exact change timestamps.

Recommended columns:

```text
experiment_id
scenario_id
run_id
step
phase
robot_id
is_attacker
row
col
goal_row
goal_col
carrying_item
completed_deliveries
path_length_remaining
has_valid_path
no_path_active
replan_count
distance_traveled
map_error_rate
active_peer_claims
attacker_trust
attacker_alpha
attacker_beta
malicious_claim_cells_on_route
cumulative_hesitation_steps
cumulative_planning_time_seconds
```

For the attacker robot, `attacker_trust` as viewed by itself may be blank. For each benign robot, it is that robot's current trust in the attacker.

### 15.6 `run_summary.csv`

One row per run/defense.

Required metrics from the project document:

- Deliveries completed.
- Average delivery time.
- Detour ratio.
- No-path time.
- Replans per delivery.
- Hesitation time.
- Route reversals.
- False blockage persistence.
- Map error rate.
- False acceptance rate.
- False rejection rate.
- Trust detection delay.
- Recovery time.
- Planning time.
- Route stability.
- Safety events, left blank or zero in the discrete Python simulator and reserved for Gazebo.

Also record:

- total malicious actions,
- total malicious cell reports,
- counts by attack type,
- attacker trust at phase boundaries,
- whether trust recovered above a configurable rehabilitation threshold,
- recovery step if it occurred,
- reports accepted/rejected/stored/pruned,
- route exposure to malicious claims,
- simulation completion reason,
- wall-clock runtime.

### 15.7 Precise metric definitions

Use explicit definitions and record them in README/methodology documentation.

- **Average delivery time:** mean steps from `delivery_started` to `delivery_completed` for benign robots.
- **Detour ratio:** actual traveled cells for a completed delivery divided by the shortest ground-truth path length computed at delivery start. Null if no oracle path exists at delivery start. Record whether truth changed during the delivery.
- **No-path time:** count of steps during which a delivery goal is active and no valid planned path exists.
- **Replans per delivery:** benign replan count divided by completed benign deliveries.
- **Hesitation time:** steps with active goal and unchanged position while a valid path exists; report no-path waiting separately.
- **Route reversal:** replan where initial route heading changes by more than a configurable threshold, default 120 degrees.
- **False blockage map persistence:** steps from injection of a false `BLOCKED` report until its target cells no longer exceed the method's configured peer-risk relevance threshold.
- **False blockage route persistence:** steps from injection until none of its target cells influence the selected route or route cost.
- **Map error rate:** fraction of dynamic-map cells where robot estimate differs from current ground truth; report per-robot mean and aggregate mean.
- **False acceptance rate:** malicious reports admitted/stored divided by malicious reports received. Also provide a separate route-influence rate.
- **False rejection rate:** legitimate reports rejected by admission divided by legitimate reports received.
- **Trust detection delay:** first malicious report step to first step trust falls below threshold, separately per benign robot and aggregate.
- **Recovery time:** steps after attack phase ends until a rolling benign performance measure returns within a default 10% of the clean baseline for a configurable stable window. If not recovered by run end, leave null and set `recovered=false`.
- **Planning time:** mean and total wall-clock planning duration.
- **Route stability:** overlap between consecutive planned routes, using Jaccard cell overlap or another single documented metric.

Do not change metric definitions between defense methods.

---

## 16. Tkinter GUI

### 16.1 Startup behavior

- `python main.py` opens the GUI.
- CLI arguments are parsed first and prefill GUI values.
- The user may edit the values and then click a start button.
- `python main.py --headless ...` bypasses Tkinter entirely.
- GUI and headless modes must both build the same `SimulationConfig`.

### 16.2 Main tab

Keep the main tab uncluttered. Include:

- Map selector with browse button.
- Seed field, default `15`.
- Run mode:
  - Single simulation.
  - Compare selected defenses.
  - Generate manifest only.
  - Replay manifest.
- Single defense dropdown, default `source_linked`.
- Comparison defense checkboxes, default all four primary methods selected.
- Trust model dropdown, default `bayesian`.
- Admission policy dropdown, default `Soft/low influence (recommended)`.
- Attack checkboxes, all checked by default:
  - Fake obstacle.
  - False clearance.
  - Stale reassertion.
- Scenario manifest path and browse button when replay is selected.
- Animation checkbox, default checked for single simulation.
- Output-directory selector.
- `Start Simulation` button.
- `Run Selected Defenses` button.
- A small status/progress label.

### 16.3 Advanced tab

Place detailed controls here:

- Reconnaissance, attack, and recovery lengths.
- Attack interval minimum and maximum.
- Candidate top-K.
- Attack scoring weights.
- Traffic percentiles.
- Fake-obstacle footprint size and spacing.
- Lidar range, ray count, and communication period.
- Temporary-obstacle count and change period.
- Bayesian alpha/beta and scalar reward/penalty.
- Trust threshold and rehabilitation threshold.
- Claim decay, max age, cost scale/exponent, block threshold.
- Majority vote validity window.
- Direct-observation freshness window.
- Expected obstacle persistence.
- Timeseries sample period.
- Deliveries/tasks per robot.
- Animation interval and diagnostics.
- Optional supplementary defense methods.
- Debug logging.
- `Reset advanced defaults` button.

Every field must have a visible default value and validation.

### 16.4 Threading and responsiveness

Do not run a long comparison directly on the Tkinter UI thread.

Use a simple worker thread or subprocess with a thread-safe progress queue. All widget updates must occur on the Tkinter main thread via `after()` polling.

Single Matplotlib animation may run in the main process. Batch comparison should default to headless execution and update progress in the GUI.

### 16.5 GUI error handling

- Show validation errors in a message box.
- Show output directory when finished.
- Restore buttons after failure.
- Preserve entered settings after an error.
- Print a traceback to console/log for developers, but show a readable message to the user.

---

## 17. Command-Line Interface

### 17.1 Core arguments

Support at least:

```text
--headless
--map <path>
--map-npy <path>                 # compatibility
--map-movingai <path>            # compatibility
--seed 15
--run-mode single|compare|generate-manifest|replay
--defense-method source_linked
--compare-defense full_trust      # repeatable
--compare-defense majority_vote
--trust-model bayesian|scalar
--admission-policy method_default|accept_all|hard_reject
--attack fake_obstacle            # repeatable
--attack false_clearance
--attack stale_reassertion
--no-attacks
--scenario-in <manifest.json>
--scenario-out <manifest.json>
--output-dir <directory>
--no-animation
```

### 17.2 Advanced arguments

Support corresponding CLI flags for all advanced GUI settings, including:

```text
--recon-steps 500
--attack-steps 1200
--recovery-steps 800
--attack-interval-min 40
--attack-interval-max 80
--attack-candidate-top-k 12
--bayes-alpha 7
--bayes-beta 3
--trust-threshold 0.55
--decay-rate 0.006
--cost-scale 14
--cost-exponent 1.5
--max-claim-age 900
--timeseries-period 5
--deliveries-per-robot 100
```

### 17.3 Precedence

Configuration precedence:

1. Built-in defaults.
2. CLI values.
3. Manifest-locked scenario fields in replay mode.
4. User edits in GUI, except fields locked by a loaded manifest.

Print or save the fully resolved config before every run.

### 17.4 Example commands

Open GUI prefilled with selected values:

```bash
python main.py \
  --seed 15 \
  --trust-model bayesian \
  --defense-method source_linked \
  --attack fake_obstacle \
  --attack false_clearance \
  --attack stale_reassertion
```

Generate a manifest headlessly:

```bash
python main.py --headless \
  --run-mode generate-manifest \
  --seed 15 \
  --scenario-out scenarios/seed_15.json
```

Replay all primary defenses:

```bash
python main.py --headless \
  --run-mode compare \
  --scenario-in scenarios/seed_15.json \
  --compare-defense full_trust \
  --compare-defense majority_vote \
  --compare-defense trust_fused \
  --compare-defense source_linked
```

---

## 18. Cross-Platform Installation

### 18.1 Environment checker

Place the provided `check_environment.py` in the repository root.

It must remain standard-library-only and support:

```bash
python check_environment.py
python check_environment.py --install
python check_environment.py --skip-gui
python check_environment.py --include-dev
```

The checker verifies:

- Python 3.10 or newer.
- NumPy.
- Matplotlib.
- Pillow.
- PyYAML.
- Tkinter import and test-window creation.
- Basic NumPy deterministic RNG behavior.
- Matplotlib image output.
- Pillow image output.
- YAML parsing.
- Filesystem write access.
- Expected project files and map assets.

### 18.2 Requirements files

Keep runtime dependencies in `requirements.txt`:

```text
numpy
matplotlib
Pillow
PyYAML
```

Add `requirements-dev.txt`:

```text
-r requirements.txt
pytest
```

Do not add pandas or a GUI framework dependency.

### 18.3 Windows and macOS documentation

README must show:

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py
.\.venv\Scripts\python.exe main.py
```

macOS Terminal:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_environment.py
python main.py
```

Mention that a Python installation lacking Tk support must be repaired or replaced; Tkinter is not generally fixed with `pip install tkinter`.

---

## 19. Incremental Implementation Sequence

Use small commits. Do not combine refactor, new attacks, GUI, and metrics in one change.

### Step 0: Freeze and document the baseline

- Check out the current main branch.
- Run `check_environment.py`.
- Record current command lines and outputs.
- Create deterministic short baseline runs with seed 15:
  - demo map, clean, 50 steps,
  - demo map, attack, 200 steps,
  - loaded NPY map if available.
- Save console summaries and selected internal values as golden fixtures.
- Add tests around existing `defense_method_runner.py` before moving it.

Acceptance condition: the current code can be reproduced from a documented command.

### Step 1: Add package skeleton and configuration

- Create `map_poisoning/` package.
- Add typed configuration dataclasses and validation.
- Add CLI parser that can reproduce current defaults.
- Keep `sim2.py` running unchanged.

Acceptance condition: config unit tests pass and resolved config serializes to JSON.

### Step 2: Extract low-risk utilities

Move without algorithm changes:

- enums/dataclasses,
- map I/O,
- grid helpers,
- A*,
- visualization helpers,
- deterministic utility functions.

Update imports gradually.

Acceptance condition: baseline outputs remain equivalent within documented nondeterministic timing fields.

### Step 3: Extract trust and fusion

- Move scalar and Bayesian trust to `trust/`.
- Move existing defense runner behavior to `fusion/`.
- Add factories/registries.
- Preserve existing method names.
- Add explicit admission policies, but use compatibility behavior first.

Acceptance condition: unit tests reproduce existing trust and defense numeric results.

### Step 4: Expand report timestamps and remove truth leakage

- Replace the one `timestamp` field with observation, sent, and received steps.
- Add stable report IDs.
- Move malicious labels to the audit store.
- Update communication, pending verification, fusion, and logs.

Acceptance condition: no trust/fusion/planning module imports or reads audit labels.

### Step 5: Extract world, sensing, robot, and simulation loop

- Move temporary blockage manager to `world.py`.
- Move lidar to `sensing.py`.
- Move robot to `robot.py` with injected components.
- Move the loop to `simulation.py`.
- Implement explicit phase state machine.

Acceptance condition: a headless compatibility run matches the baseline behavior.

### Step 6: Implement named RNG streams and scenario manifests

- Add stable named RNG derivation.
- Make starts, tasks, temporary obstacles, and phase boundaries manifestable.
- Implement manifest JSON round trip and hash.
- Implement clean canonical authoring rollout.

Acceptance condition: same seed/config produces byte-identical manifests on repeated runs.

### Step 7: Implement attack strategies and scheduler

- Move current fake-obstacle behavior first.
- Add false clearance.
- Add stale reassertion with false fresh timestamps.
- Add seeded shuffled-bag scheduler and 40-80 step global interval.
- Add impact-weighted top-K sampling.
- Broadcast events to all benign robots.

Acceptance condition: all eight attack ablations run and generate valid manifests.

### Step 8: Implement primary fusion methods

- Add explicit `full_trust`.
- Add `majority_vote`.
- Confirm `trust_fused` is non-retroactive.
- Confirm `source_linked` is retroactive.
- Preserve legacy methods.

Acceptance condition: focused tests demonstrate the expected difference after a sender's trust drops.

### Step 9: Add temporal-aware verification

- Add `VerificationOutcome` evaluator.
- Prevent old honest contradictions from receiving the same penalty as fresh contradictions.
- Keep the feature configurable so compatibility behavior can still be tested.
- Log predicted report state and audit ground-truth state separately.

Acceptance condition: an honest old obstacle report can expire without a large trust penalty, while a falsely fresh stale reassertion receives a penalty after contradiction.

### Step 10: Add metrics and three CSV writers

- Implement event capture first.
- Add time-series sampling.
- Add summary calculations.
- Add clean-baseline pairing for recovery.

Acceptance condition: three CSV files are produced, parse correctly, and contain all required identifiers and metrics.

### Step 11: Add GUI

- Build main and advanced tabs.
- Wire CLI prefill.
- Add single, comparison, manifest-generation, and replay modes.
- Use a worker for batch runs.

Acceptance condition: GUI and headless runs with the same resolved config produce the same manifest and numerical outputs.

### Step 12: Compatibility cleanup

- Update README and launch scripts.
- Convert `sim2.py` into a compatibility wrapper.
- Convert `defense_method_runner.py` into a wrapper or remove it after tests.
- Remove dead globals, duplicate code, and debug prints.

Acceptance condition: new entry point is documented and old common command still gives a clear migration path.

---

## 20. Verification Plan

The implementing AI must run the tests and inspect output files, not only write code.

### 20.1 Environment verification

Windows and macOS users run:

```bash
python check_environment.py
```

Expected:

- Exit code `0` when required runtime and Tkinter checks pass.
- Clear failure and installation guidance when a dependency is missing.

CI/headless:

```bash
python check_environment.py --skip-gui --include-dev
```

### 20.2 Static checks

- `python -m compileall main.py map_poisoning tests`
- Confirm no circular import errors.
- Confirm no operational module references:
  - `is_malicious`
  - `attack_type_audit`
  - true obstacle episode labels
  except metrics/scenario-authoring modules.
- Search for old runtime globals and ensure they are no longer read by core logic.

### 20.3 Unit tests

Create tests for at least the following.

#### Configuration

- Defaults match this plan.
- CLI overrides defaults.
- GUI model receives CLI-prefilled values.
- Invalid phase and interval values fail validation.
- Resolved config JSON round trips.

#### Named RNG

- Same seed/name gives same sequence.
- Different names give different sequences.
- Creating an additional stream does not alter existing streams.

#### Trust

- Bayesian initial score is `0.7`.
- Confirmed report increments alpha and score.
- Contradicted fresh report increments beta and lowers score.
- Scalar trust clamps to `[0,1]`.
- Honest stale outcome has zero or small configured trust effect.

#### Admission

- `accept_all` never rejects valid reports.
- `hard_reject` rejects below threshold.
- method-default policy does not double-apply trust weighting.

#### Full trust

- Equal occupied reports give equal influence regardless of sender trust.
- Free claims provide negative evidence.

#### Majority vote

- One vote per sender.
- Newest active vote replaces older vote from that sender/cell.
- Duplicate spam does not create extra votes.
- Expired votes disappear.
- Tie produces neutral peer risk.

#### Trust fused

- A report uses trust at insertion.
- Later trust changes do not change old evidence.

#### Source linked

- A report uses current trust.
- Later trust drop reduces old evidence.
- Age decay reduces old evidence.
- Free and blocked claims oppose one another.
- No claims produce zero peer risk.

#### Attacks

- Fake obstacle only targets currently free report cells.
- False clearance only targets active real obstacle cells.
- Stale reassertion only targets cleared episodes.
- Stale reassertion operational timestamp equals the fresh attack step.
- Audit metadata retains original episode information.
- Attack strategies never attach audit labels to operational reports.

#### Scheduler

- Intervals are between 40 and 80 by default.
- Same seed gives same schedule.
- One global attack budget is enforced.
- Shuffled bag distributes feasible enabled types.
- Infeasible types are deferred and logged.

#### Manifest

- JSON round trip preserves all fields.
- Hash is stable.
- Map mismatch is rejected.
- Replay never invokes target selection.

#### CSV

- All three files have stable headers.
- Rows include scenario and run IDs.
- `details_json` is valid JSON.
- Audit join works by report ID.

### 20.4 Integration tests

#### Minimal smoke test

Run a short headless demo:

```bash
python main.py --headless --run-mode single --seed 15 \
  --recon-steps 20 --attack-steps 40 --recovery-steps 20 \
  --deliveries-per-robot 1 --no-animation
```

Assert:

- Exit code `0`.
- All three CSVs exist.
- Manifest and resolved config exist.
- At least one row appears in time series and events.

#### Deterministic authoring

Generate the same manifest twice with identical seed/config.

Assert byte-for-byte equality or equality after excluding an explicitly nonsemantic creation timestamp. Prefer no creation timestamp in the hashed content.

#### Different-seed test

Generate with seed 15 and seed 16.

Assert at least one scenario component differs.

#### Same attack chain across defenses

Replay one manifest against all four primary methods.

From `events.csv`, filter `attack_action_injected` and `report_sent` malicious-audit events. Assert equality across defenses for:

- step,
- scenario event ID,
- report ID,
- attack type,
- target cell,
- claim,
- sender,
- recipients.

Defense outputs such as admission, trust, routes, and metrics are allowed to differ.

#### Attack ablations

Run:

```text
none
fake only
false clearance only
stale only
fake + clearance
fake + stale
clearance + stale
all three
```

Assert only enabled attack types appear.

#### Recovery behavior

Assert after the attack phase ends:

- no new malicious audit events are injected,
- attacker honest `BLOCKED` and `FREE` reports still occur,
- trust may increase only through verified honest behavior,
- recovery metrics are populated or explicitly marked not recovered.

#### Method distinction test

Construct a tiny deterministic scenario:

1. Attacker begins with high trust.
2. Sends one false blocked claim.
3. Claim initially affects route.
4. Later direct verification lowers attacker trust.

Assert:

- `trust_fused` old evidence remains numerically unchanged except normal expiry rules.
- `source_linked` old evidence weakens immediately when current trust changes.

#### Honest-stale test

1. Real obstacle appears.
2. Honest robot reports blocked.
3. Obstacle later clears.
4. Receiver verifies free after sufficient age.

Assert map influence decreases, but trust is not penalized like a fresh malicious contradiction.

#### False-fresh stale attack test

1. Real obstacle appears and clears.
2. Attacker reasserts it with fresh timestamp.
3. Benign robot verifies free shortly afterward.

Assert map influence decreases and attacker trust is penalized.

### 20.5 GUI verification

Manual checks on both Windows and macOS:

1. Run `python main.py`.
2. Confirm all main-tab defaults match this plan.
3. Confirm advanced settings are hidden on the second tab.
4. Launch with CLI arguments and verify fields are prefilled.
5. Change a GUI value and confirm resolved config reflects the edit.
6. Run one short animated simulation.
7. Run a four-defense comparison and confirm the UI remains responsive.
8. Load an existing manifest and confirm locked fields cannot be changed inconsistently.
9. Confirm readable error for missing map or invalid number.
10. Confirm output directory opens or is clearly displayed after completion.

### 20.6 Output inspection

For one default-seed comparison, inspect:

- phase transition steps are 500 and 1700,
- total run ends at or before 2500,
- malicious events only occur during the attack phase,
- recovery contains honest attacker reports,
- attack events are identical across defenses,
- attacker trust timeline differs only because of defense/admission/verification behavior,
- summary rows contain no unexplained missing values,
- map error is between 0 and 1,
- false acceptance/rejection rates are between 0 and 1,
- planning times are nonnegative,
- report counts reconcile between events and summary.

### 20.7 Performance sanity check

Run a four-defense comparison for the full 2,500 steps without animation. Record wall-clock runtime and peak memory.

Do not optimize prematurely. Optimize only after profiling identifies a bottleneck. Avoid changing algorithm semantics for speed without a regression test.

---

## 21. Acceptance Criteria

The implementation is complete only when all statements are true.

1. `python main.py` opens a functional Tkinter launcher.
2. `python main.py --headless ...` runs without importing or initializing Tkinter.
3. CLI arguments prefill the GUI.
4. All user-visible settings have defaults.
5. Advanced settings are separated from the main tab.
6. Same seed/config produces the same manifest.
7. Same manifest produces the same malicious report chain across all defenses.
8. All three attack types work independently and together.
9. The attacker returns to fully honest reporting during recovery.
10. Malicious reports broadcast to all benign robots.
11. Bayesian trust is the default and scalar trust remains supported.
12. Admission policy is modular and low-influence behavior is the default.
13. Full trust, majority vote, trust-fused, and source-linked methods are selectable.
14. Existing hard-threshold, soft-probability, and time-decay methods remain functional.
15. Source-linked evidence changes when current trust changes.
16. Trust-fused old evidence does not change when current trust changes.
17. Operational reports contain no malicious labels.
18. `run_summary.csv`, `robot_timeseries.csv`, and `events.csv` are created for every run.
19. Trust changes include exact steps in events.
20. Required document metrics are calculated or explicitly marked unavailable with a reason.
21. Environment checker works on Windows and macOS Python installations.
22. Unit and integration tests pass.
23. README documents install, GUI, headless, manifest, comparison, and verification workflows.
24. Core modules are small enough to understand independently and do not depend on GUI code.
25. No map-specific coordinates or seed-specific branches are added.

---

## 22. Guardrails for the Implementing AI

- Do not redesign the research question.
- Do not replace A* with another planner.
- Do not add ROS or Gazebo dependencies to the Python simulator.
- Do not use hidden ground truth in benign trust, admission, fusion, or planning logic.
- Do not let the attacker adapt separately to each defense during comparison runs.
- Do not silently change default phase lengths or trust priors.
- Do not make the attack frequency multiply by the number of enabled attacks.
- Do not punish every old contradictory report as malicious; preserve the honest-stale distinction.
- Do not make a long inheritance hierarchy when a small interface and composition are enough.
- Do not add a database, message broker, pandas, Qt, or web UI.
- Do not delete the legacy working entry point before the replacement passes regression tests.
- Do not claim verification without running commands and inspecting generated outputs.

---

## 23. Final Implementation Report Required from the AI

After implementation, the AI must provide:

1. A concise summary of architectural changes.
2. A file tree of new and changed files.
3. Exact installation commands used.
4. Exact test commands run.
5. Test results and any skipped tests.
6. Example commands for GUI, single headless, manifest generation, and comparison replay.
7. Paths to example output CSVs and manifest.
8. A deterministic comparison proving attack events match across defenses.
9. A focused numeric example proving trust-fused is non-retroactive and source-linked is retroactive.
10. Known limitations and remaining work, especially any methodology from the document not yet fully implemented.
