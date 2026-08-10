"""Small, dependency-light statistics helpers for paired seed experiments."""
from __future__ import annotations
import math
from statistics import mean, median

# Two-sided .975 Student-t critical values, df 1..30; the normal limit is
# sufficiently accurate for larger studies and avoids making SciPy required.
_T975 = {
    1: 12.7062047364, 2: 4.3026527299, 3: 3.1824463053,
    4: 2.7764451052, 5: 2.5705818366, 6: 2.4469118511,
    7: 2.3646242511, 8: 2.3060041352, 9: 2.2621571629,
    10: 2.2281388519, 11: 2.2009851601,
}

def t_critical_975(df: int) -> float:
    if df < 1:
        raise ValueError("degrees of freedom must be positive")
    if df in _T975:
        return _T975[df]
    # Normal-limit expansion; error is negligible for df > 30.
    z = 1.959963984540054
    return z + (z**3 + z) / (4 * df) + (5*z**5 + 16*z**3 + 3*z) / (96 * df**2)

def summarize(values) -> dict:
    clean = [float(value) for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]
    n = len(clean)
    result = {"n": n, "mean": None, "sample_std": None, "sem": None,
              "ci95_low": None, "ci95_high": None, "median": None,
              "min": None, "max": None}
    if not clean:
        return result
    result.update(mean=mean(clean), median=median(clean), min=min(clean), max=max(clean))
    if n >= 2:
        variance = sum((value - result["mean"]) ** 2 for value in clean) / (n - 1)
        std = math.sqrt(variance); sem = std / math.sqrt(n)
        margin = t_critical_975(n - 1) * sem
        result.update(sample_std=std, sem=sem, ci95_low=result["mean"] - margin, ci95_high=result["mean"] + margin)
    return result

def paired_summary(source, baseline) -> dict:
    differences = [float(a) - float(b) for a, b in zip(source, baseline)]
    result = summarize(differences)
    result["n_pairs"] = result.pop("n")
    result["mean_difference"] = result.pop("mean")
    result["sample_std_difference"] = result.pop("sample_std")
    result["sem_difference"] = result.pop("sem")
    result["ci95_difference_low"] = result.pop("ci95_low")
    result["ci95_difference_high"] = result.pop("ci95_high")
    return result
