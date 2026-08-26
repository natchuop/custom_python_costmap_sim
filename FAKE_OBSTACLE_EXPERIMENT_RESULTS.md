# Method-Separation and Navigation Audit

## Conclusion

The small differences in total deliveries are mostly a measurement and sample-size
effect, not evidence that the four methods are behaving identically. Delivery count
is coarse: a 2500-step run produces only about 50--56 benign deliveries, physical
obstacles and traffic affect every method, and clean recovery lets trust return toward
one. Route-level metrics show the expected ordering much more clearly.

Two display/observability defects were corrected during this audit:

- the 2500-step attack-free heatmap now uses static warehouse geometry instead of
  displaying the temporary-obstacle snapshot from step 299; and
- real-obstacle, robot-on-route, honest-peer, malicious-report, and exact attack-type
  replans are now labeled and counted separately. The old generic peer-replan label
  could not prove which source changed a route.

Neither correction changes defense decisions. Malicious attribution is used only
after decisions for audit metrics.

## Held-out experiment

- Map: default warehouse
- Previously unused seeds: 26, 27, 28, 29
- Methods: Majority Vote, Full Trust, Trust Fused, Source Memory
- Steps: 300 reconnaissance / 1700 attack / 500 recovery
- Attacks: all three types, every 35--40 steps
- Runs: 4 seeds x 4 methods x 2500 steps = 16 complete runs
- Manifest authoring: one deterministic 2500-step attack-free reference per seed
- Heatmap: one shared 5000-count heatmap per seed (2 benign robots x 2500 clean steps)
- Fairness: all four methods replay the identical manifest within each seed

Each manifest contained 45 attacks. Every run reached 2500 steps; batch validation
reported 16 complete, zero failed, and zero missing cells.

## Held-out means

| Method | Deliveries | Cycle p95 | No-path steps | Attack penalty | Route-affected steps | Ignored malicious reports |
|---|---:|---:|---:|---:|---:|---:|
| Majority Vote | 54.25 | 122.31 | 90.25 | 1.87 | 173.50 | 0.00 |
| Full Trust | 54.50 | 120.58 | 38.25 | 2.70 | 241.50 | 0.00 |
| Trust Fused | 53.75 | 125.05 | 113.50 | 1.50 | 145.00 | 253.75 |
| Source Memory | 55.00 | 119.93 | 38.75 | 1.00 | 114.25 | 302.50 |

Source Memory reduced mean attacker route penalty by about 63% and attacker-affected
route steps by about 53% relative to Full Trust. Delivery means differ by at most
1.25 deliveries because delivery count is a lagging, low-resolution outcome. With
only four held-out seeds these are validation findings, not a claim of statistical
significance.

## Mechanism checks

| Method | Malicious-report replans | Malicious path changes | Fake-obstacle replans | Fake path changes | Physical-obstacle path changes |
|---|---:|---:|---:|---:|---:|
| Majority Vote | 8.25 | 6.50 | 6.75 | 5.00 | 53.75 |
| Full Trust | 11.00 | 8.75 | 9.75 | 7.50 | 39.00 |
| Trust Fused | 8.25 | 5.75 | 7.00 | 5.25 | 40.50 |
| Source Memory | 5.50 | 4.50 | 4.50 | 4.00 | 38.50 |

False-clearance reports normally do not force a route replan because a peer FREE
claim cannot override current physical LiDAR or materially increase route cost. They
are still fused, later verified, and included in trust evidence. Stale reassertions
caused occasional route changes. Temporary physical obstacles changed paths in every
run, proving that real obstacles affect navigation independently of fake claims.

## Trust and traffic verification

- Trust was not flat. Across the held-out sampled victim trajectories, each method
  recorded about 204--213 downward moves and 1707--1830 upward moves.
- Victim trust spent hundreds of sampled states below 0.50, then recovered during
  clean evidence. Source Memory lagged recovered current trust by as much as 0.317.
- Source Memory ignored an average of 302.5 malicious report deliveries with zero
  operational weight.
- Traffic coordination prevented an average of 37.0--47.5 conflicts per run.
- All 16 runs had zero robot overlaps and zero blocked physical moves.
- No robot-on-route replans occurred in these runs because frozen-intent traffic
  coordination prevented that conflict before motion; this is expected, not a stuck
  condition.
- The final automated suite passed 117 tests, including an attack-free CLI smoke contract.

P95 means the 95th percentile: 95% of completed delivery cycles finished in that
many steps or fewer. It is useful here because it exposes slow-tail deliveries that
the mean or total delivery count can hide.
