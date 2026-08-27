# Latest Report and Peer-Fusion Verification

## Implemented behavior

- Added `latest_report` as the first primary comparison method. It auto-accepts
  the newest active categorical FREE/BLOCKED report without trust or occupancy
  probability. Exact-time conflicting newest reports resolve to unknown.
- Combined belief maps now apply operationally accepted peer FREE and BLOCKED
  information. FREE paints white; BLOCKED keeps the reporter's robot color.
  Trust-aware methods suppress reports using their normal operational gate.
- Current local LiDAR remains authoritative. Majority Vote ties remain unknown,
  so a false clearance is not painted white when honest BLOCKED votes tie or
  outvote it.
- Popup trust columns are `Reporter`, `Observed by`, score fields, and state.
- Red dotted attack perimeters now appear on victim combined maps for fake
  obstacles, false clearances, and stale reassertions.
- Bayesian confirmation gain changed from 1.0 to 0.25; scalar positive reward
  changed from 0.02 to 0.005. Contradiction strength is unchanged.
- Default warehouse action points and task order are reproducible by seed.
  Each task block favors long cross-warehouse routes while retaining corridor,
  medium, and short trips.
- Fake-obstacle targets must first enter the clean-reference victim's LiDAR in
  15--40 future steps, stay relevant to the same delivery leg, remain reachable,
  and produce a positive finite detour. Invalid stale targets that are physically
  occupied again by a newer episode are rejected.

## Verification results

- Automated suite: 123 passed.
- Seed 15: five methods x 2500 steps, identical audited manifest; audit passed.
- Held-out seeds 16--17: ten additional 2500-step runs; batch validation passed
  with zero failed and zero missing cells.
- All 15 full method runs had zero robot-overlap violations, zero deadlocks, and
  zero benign moves into physically blocked cells.
- Seed 15 showed method separation in attacker-affected route steps:
  `latest_report` 157, `majority_vote` 94, `full_trust` 227,
  `trust_fused` 114, and `source_memory` 49.
- Seed 15 operationally ignored malicious reports: `latest_report` 4,
  `majority_vote` 0, `full_trust` 0, `trust_fused` 69, and
  `source_memory` 112. The few Latest Report non-influential deliveries are
  categorical replacement/timing effects, not trust gating.
- Visual playback QA at false-clearance step 340 confirmed the reordered table,
  correct latest-attack text, and red dotted perimeters on both victim maps.

The strict visibility filter transparently skips scheduled fake attacks when no
candidate satisfies every preregistered constraint. This avoids silently relaxing
the 15--40 window or selecting a noncausal target.

## Additional held-out audit: seeds 18--19

- Ten more 2500-step cells completed across all five methods; both manifests and
  the aggregate batch validation passed with zero failed or missing cells.
- An independent event reconstruction found that the original temporary-obstacle
  onset could place a multi-cell rectangle around a robot already in its footprint,
  producing an artificial 150-step stall. The runtime now defers that obstacle's
  activation until its complete footprint is empty.
- Corrected replays had zero physical robot/obstacle overlaps, zero robot/robot
  overlaps, zero blocked moves, and every detected traffic deadlock recovered.
- Every method reacted to real obstacles with productive route changes. Fake
  obstacles changed routes in the trust-agnostic methods; Trust Fused and Source
  Memory suppressed many of them after distrust, as intended.
- Both recipients distrusted the attacker on both seeds. Mean minimum attacker
  trust ranged from 0.0097 to 0.0904. Pure contradiction batches always reduced
  trust, and pure confirmation batches never reduced it.
- Seed 19 Latest Report's remaining 488 benign no-path steps were classified as
  409 peer-fusion disconnections and 79 genuine truth-map disconnections, with
  zero direct-belief or planner/state errors. This is expected vulnerability of
  the categorical auto-accept baseline, not a traffic deadlock.
