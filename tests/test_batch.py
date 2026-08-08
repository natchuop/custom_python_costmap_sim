import math

from map_poisoning.batch import parse_seed_spec
from map_poisoning.statistics import paired_summary, summarize, t_critical_975


def test_parse_seed_spec_ranges_lists_and_deduplicates():
    assert parse_seed_spec("1-3") == (1, 2, 3)
    assert parse_seed_spec("1,5,10") == (1, 5, 10)
    assert parse_seed_spec("1-3,7,2") == (1, 2, 3, 7)


def test_parse_seed_spec_rejects_invalid_ranges():
    for value in ("-1", "3-1", "abc", "1-"):
        try:
            parse_seed_spec(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)


def test_student_t_summary_known_values():
    result = summarize([1, 2, 3])
    assert result["mean"] == 2
    assert result["sample_std"] == 1
    assert math.isclose(result["sem"], 1 / math.sqrt(3))
    margin = t_critical_975(2) / math.sqrt(3)
    assert math.isclose(result["ci95_low"], 2 - margin)
    assert math.isclose(result["ci95_high"], 2 + margin)


def test_statistics_missing_and_single_value_are_not_zero_filled():
    assert summarize([None, 4])["n"] == 1
    result = summarize([4])
    assert result["mean"] == 4
    assert result["sample_std"] is None
    assert result["ci95_low"] is None


def test_paired_summary_uses_seed_differences():
    result = paired_summary([10, 12, 14], [8, 13, 10])
    assert result["n_pairs"] == 3
    assert result["mean_difference"] == 5 / 3
    assert result["sample_std_difference"] > 0
