# Implementation Verification

This build implements the agreed five-method experiment in this order:

1. Latest Report
2. Majority Vote
3. Full Trust
4. Trust Fused
5. Source Memory

## Baseline defaults

- Simulation phases: 300 reconnaissance / 1700 attack / 500 recovery (2500 total)
- Attack interval default: uniformly sampled from 35 through 40 steps
- One 2500-step deterministic attack-free reference heatmap per seed
- LiDAR: 360-degree Euclidean line-of-sight, radius 5 cells
- Sensor confidence by distance 1..5: 1.00, 0.90, 0.80, 0.70, 0.60
- Direct-observation and peer-report lifetime: 300 steps
- Confidence resend threshold: 0.10
- Bayesian trust: prior alpha=9, beta=1, evidence cap=12
- Confirmation evidence multiplier: 0.25
- Contradiction evidence multiplier: 6.0
- Distrust threshold: 0.50
- Source Memory recovery rate: 0.05 per positive trust-update batch
- Majority tie/no-consensus cost: 3
- Common periodic route optimization check: 25 steps, in addition to immediate event-driven replanning
- Admission policy: accept_all for receipt/audit; Trust Fused and Source Memory give reports zero operational map influence once their applicable trust is below 0.50

## Verification completed

- `python -m pytest -q`: 124 tests passed, including peer FREE/BLOCKED display fusion, latest-report categorical behavior, strict future visibility, seeded deliveries, trust recovery, collision-safe temporary-obstacle activation, popup labels, and the earlier contracts.
- Seed 15 completed all five 2500-step methods on one identical audited manifest. It had zero robot overlaps, deadlocks, or physically blocked moves in every method.
- Held-out seeds 16 and 17 completed all ten requested 2500-step cells with zero failures or missing runs. Both manifests passed audit; every fake target's first visibility was within 15--40 future steps.
- Fake-obstacle stress regression completed: default warehouse seeds 24 and 25 x four methods x 2500 steps = 8 valid full runs.
- Each stress manifest contains 45 attacks at 35--40-step spacing, one 5000-count clean heatmap, positive finite reference detours, and targets outside the intended victim's current LiDAR view.
- All eight stress runs completed deliveries with zero physical blocked moves; batch validation reports zero failed or missing cells.
- Source Memory ignored reports had zero operational weight and had zero sampled below-threshold states where the attacker still affected a route.
- Independently recomputed run/aggregate metrics matched `multiseed_runs.csv` and `method_comparison_table.csv` exactly.
- A full 2500-step Source Memory replay completed successfully.
- A full 2500-step four-method comparison completed successfully on one shared manifest in the required order.
- The four-method run replayed the same 353 attack-report actions for every method.
- Benign-to-benign trust remained above the 0.50 distrust threshold in the verified full comparison.
- Both benign recipients distrusted the attacker in the verified full comparison.
- Source Memory showed immediate trust-memory drops and slower recovery than current Bayesian trust.
- Full multiseed regression completed: 4 maps x 2 seeds x 4 methods x 2500 steps = 32 full runs. All four batch validations report `valid: true`, with no failed/missing cells and no robot-overlap violations.
- Every full run completed deliveries; totals ranged from 26 deliveries on the largest Map 002 cases to 91 on rotated Map 005.
- Under the final contradiction multiplier of 6.0, Source Memory drove both benign recipients below 0.50 on all eight map/seed cases, while continuing to complete deliveries.
- While Source Memory effective trust was below 0.50, full-run timeseries contained zero samples where attacker claims still affected the victim route or contributed influential fake claims.
- Source Memory operationally ignored malicious report deliveries during distrust (for example 220/172 on the two default-map seeds, 62/31 on Map 002, 111/57 on Map 005, and 38/28 on rotated Map 005).
- Comparison plots were generated successfully.
- The configuration GUI and live simulation windows were smoke-tested under a virtual display. A final attack-phase render verified gray unknown/expired cells, white valid free cells, temporary obstacles, LOS rays, route overlays, attack panel, and Source Memory Trust/Memory/ACTIVE-or-IGNORED state. The combined map hides unsupported attacker BLOCKED claims while the source is operationally ignored.
- The attack-free reference heatmap popup uses the static warehouse geometry as its background; it no longer presents the temporary-obstacle state from a single reconnaissance step as representative of all 2500 clean steps.
- Replan events and run summaries separately attribute temporary physical obstacles, robots detected on-route, malicious reports, and each exact attack type. Malicious attribution is post-decision audit metadata and is never supplied to admission, fusion, trust, or planning.
- Generated report schema uses `sensor_confidence` and does not contain `sent_step` or `received_step`.
- `robot_timeseries.csv` separates `planning_checks` from `path_changes` and records attacker Source Memory.

## Final fresh-seed stress regression

A final regression was run on previously unused default-warehouse seeds 21, 22, and 23 after the batch/performance fixes. All four methods completed the full 2500 steps for every seed: 3 seeds x 4 methods = 12 complete full runs. The simulator's own multiseed `--resume` validation then accepted all 12 cells and regenerated the combined aggregate tables with `valid: true`, zero failed cells, and zero missing cells.

Across these 12 runs:

- every method completed deliveries on every seed (46-55 benign deliveries in these three default-map seeds);
- every run reached exactly 2500 steps with zero robot-overlap violations and zero blocked physical moves for benign robots;
- both benign robots distrusted the attacker in every Source Memory seed;
- Source Memory operationally ignored 67, 291, and 204 malicious report deliveries on seeds 21, 22, and 23 respectively while the source was below the operational threshold;
- ignored malicious reports had zero operational weight;
- Source Memory had zero sampled states where attacker memory was below 0.50 but the attacker still affected the victim route or contributed influential fake claims;
- attack/trust/verification-related replans produced real path changes in every Source Memory seed; and
- independently recomputed aggregate means matched `multiseed_summary.csv` with zero discrepancies.

Two seed-dependent performance issues were found and fixed during this final pass. First, `active_fake_claim_count` no longer rebuilds/scans the full fusion claim map every step; FusionEngine now maintains the count incrementally. Second, pending trust-validation reports are indexed by cell so each scan checks only reports that can actually be validated instead of rescanning thousands of unrelated pending reports every step. The multiseed worker now also writes completion metadata atomically before immediate process termination, making full cells resumable even when a parent batch process is interrupted after outputs are written.

## Held-out method-separation audit

After the display and attribution corrections, previously unused default-warehouse
seeds 26--29 were run for all four methods: 16 full 2500-step replays. Batch
validation reported 16 complete, zero failed, and zero missing cells. All methods
used the identical per-seed manifest, and each attack-free heatmap summed to 5000.

- All 16 runs had zero robot-overlap violations and zero blocked physical moves.
- Traffic coordination prevented 37.0--47.5 conflicts per run on average.
- Temporary physical obstacles caused real path changes in every run.
- Trust repeatedly rose and fell, crossed below 0.50 for both victims, and recovered
  during clean evidence; Source Memory retained a measurable recovery lag.
- Source Memory averaged 1.00 attacker route penalty and 114.25 attacker-affected
  steps, versus 2.70 and 241.50 for Full Trust.
- Source Memory operationally ignored 302.5 malicious report deliveries per run on
  average while Full Trust and Majority Vote intentionally ignored none.
- Mean delivery totals remained close (53.75--55.00), confirming that delivery count
  is too coarse by itself to measure method separation over four seeds.

All three attack types are implemented and enabled by default. Exact per-type replan
and path-change counters are written to `run_summary.csv`. `--attacks none` also
authors and replays a valid attack-free manifest without requiring attack candidates.

## Parameters to sensitivity-test

Change one family at a time using the same seeds:

- Observation/report lifetime: 150, 300, 450
- Bayesian contradiction multiplier: baseline 6; sensitivity test 4, 5, 6 (and optionally 7 if future maps are harder)
- Bayesian evidence cap: 10, 12, 15
- Source Memory recovery rate: 0.025, 0.05, 0.10
- Periodic route-check interval: 10, 25, 50
- Confidence resend threshold: 0.05, 0.10, 0.20

The baseline values above are starting values, not claims that they are globally optimal.

## Full-map tuning findings

The original contradiction multiplier of 4.0 was too conservative on several full 2500-step converted-map manifests: one benign recipient could verify many attacks yet never cross the 0.50 distrust threshold. After correcting temporal validation so moved temporary obstacles no longer create false benign contradictions, multipliers 4, 5, and 6 were replayed on the same borderline manifests. A multiplier of 6.0 was the first tested value that pushed both benign recipients below 0.50 on all of those cases without introducing benign contradiction penalties or materially reducing deliveries. It is therefore the new baseline.

Reports are still received under `accept_all` so experiments can audit all traffic. Operationally, `source_memory` gives all reports from a source zero weight once its effective trust/memory is below 0.50; `trust_fused` ignores new reports received while the source is below 0.50 but intentionally preserves older reports that were accepted while trusted. `majority_vote` and `full_trust` remain trust-agnostic baselines.
