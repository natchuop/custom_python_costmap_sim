import math

from map_poisoning.statistics import paired_summary, summarize


def test_summary_uses_sample_standard_deviation_and_student_t_ci():
    result = summarize([1, 2, 3])
    assert result["n"] == 3
    assert math.isclose(result["sample_std"], 1.0)
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]


def test_single_observation_leaves_dispersion_undefined():
    result = summarize([4])
    assert result["mean"] == result["median"] == result["min"] == result["max"] == 4.0
    assert result["sample_std"] is None
    assert result["sem"] is None
    assert result["ci95_low"] is None
    assert result["ci95_high"] is None


def test_missing_values_are_not_converted_to_zero():
    result = summarize([None, float("nan"), 5])
    assert result["n"] == 1
    assert result["mean"] == 5.0


def test_paired_differences_keep_positive_direction():
    result = paired_summary([2, 3, 4], [1, 2, 3])
    assert result["n_pairs"] == 3
    assert result["mean_difference"] == 1.0
