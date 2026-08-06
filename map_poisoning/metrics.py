"""CSV logging with the three stable primary output files."""
from __future__ import annotations
import csv, json
from pathlib import Path

class CsvMetrics:
    def __init__(self): self.events=[]; self.timeseries=[]
    def event(self, step, kind, **data): self.events.append({"step":step,"kind":kind,**data})
    def trust_update(self, step, *, method, report_id, sender_id, recipient_id, outcome, old_trust, new_trust):
        """Record every trust change with a stable report and recipient join key."""
        self.event(step, "trust_update", method=method, report_id=report_id,
                   sender_id=sender_id, recipient_id=recipient_id,
                   outcome=outcome, old_trust=old_trust, new_trust=new_trust)
    def fusion_effect(self, step, *, method, report_id, sender_id, recipient_id, cell,
                      evidence_before, evidence_after, probability_before, probability_after,
                      outcome, phase, observation_age, scenario_event_id):
        """Record whether a trust change revised already stored peer evidence."""
        self.event(step, "fusion_effect", method=method, report_id=report_id,
                   sender_id=sender_id, recipient_id=recipient_id, target_cell=cell,
                   evidence_before=evidence_before, evidence_after=evidence_after,
                   evidence_delta=evidence_after-evidence_before,
                   absolute_evidence_delta=abs(evidence_after-evidence_before),
                   probability_before=probability_before,
                   probability_after=probability_after,
                   probability_delta=probability_after-probability_before,
                   outcome=outcome, phase=phase, observation_age=observation_age,
                   scenario_event_id=scenario_event_id,
                   malicious_audit=scenario_event_id is not None)
    def sample(self, **data): self.timeseries.append(data)
    def write(self, directory, summary):
        root=Path(directory); root.mkdir(parents=True, exist_ok=True)
        self._write(root/"events.csv",self.events); self._write(root/"robot_timeseries.csv",self.timeseries); self._write(root/"run_summary.csv",[summary])
    @staticmethod
    def _write(path, rows):
        keys=sorted({key for row in rows for key in row}) or ["empty"]
        with path.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    @staticmethod
    def config(path, data): Path(path).write_text(json.dumps(data,indent=2,sort_keys=True),encoding="utf-8")
