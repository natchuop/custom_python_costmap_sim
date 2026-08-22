"""Small, dependency-light statistics helpers for paired seed experiments."""
from __future__ import annotations
import math
from statistics import mean, median

# Fixed two-sided 95% Student-t critical values, df 1..30.  For n >= 2:
#   sample SD = sqrt(sum((x - mean)^2) / (n - 1))
#   SEM = sample SD / sqrt(n)
#   CI95 = mean +/- t(0.975, n - 1) * SEM
# The normal-limit expansion is used only beyond df 30.
_T975 = {
    1: 12.7062047364, 2: 4.3026527299, 3: 3.1824463053,
    4: 2.7764451052, 5: 2.5705818366, 6: 2.4469118511,
    7: 2.3646242511, 8: 2.3060041352, 9: 2.2621571629,
    10: 2.2281388519, 11: 2.2009851601, 12: 2.1788128297,
    13: 2.1603686565, 14: 2.1447866879, 15: 2.1314495456,
    16: 2.1199052992, 17: 2.1098155778, 18: 2.1009220402,
    19: 2.0930240544, 20: 2.0859634473, 21: 2.0796138447,
    22: 2.0738730679, 23: 2.0686576104, 24: 2.0638985616,
    25: 2.0595385528, 26: 2.0555294386, 27: 2.0518305165,
    28: 2.0484071418, 29: 2.0452296421, 30: 2.0422724563,
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
