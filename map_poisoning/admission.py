"""Separate policy for report admission from trust estimation and fusion."""
from __future__ import annotations
from .models import AdmissionDecision

def decide(policy: str, trust: float, threshold: float) -> AdmissionDecision:
    if policy == "hard_reject" and trust < threshold: return AdmissionDecision(False, 0., "below_trust_threshold")
    if policy == "auto_soft": return AdmissionDecision(True, trust if trust < threshold else 1., "soft_low_influence" if trust < threshold else "accepted")
    return AdmissionDecision(True, 1., "accepted")
