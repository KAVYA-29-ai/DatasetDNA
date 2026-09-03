import pandas as pd

from datasetdna.profiler.cardinality import check_cardinality
from datasetdna.profiler.correlations import check_correlations
from datasetdna.profiler.duplicates import check_duplicates
from datasetdna.profiler.missing import check_missing
from datasetdna.profiler.numerical import check_numerical
from datasetdna.profiler.outliers import check_outliers
from datasetdna.profiler.target import check_target

from datasetdna.scoring.health_score import (
    calculate_data_quality_score,
    calculate_health_score,
    get_statistical_signals,
)


def build_results(
    df: pd.DataFrame,
    target: str | None = "target",
) -> dict:

    return {
        "missing": check_missing(df),
        "duplicates": check_duplicates(df),
        "cardinality": check_cardinality(df),
        "numerical": check_numerical(df),
        "outliers": check_outliers(df),
        "correlations": check_correlations(df),
        "target": check_target(df, target),
    }


# =============================================================
# BASIC HEALTH SCORE TESTS
# =============================================================

def test_healthy_dataset_scores_100():

    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [10, 20, 30, 40, 50],
            "target": ["A", "B", "A", "B", "A"],
        }
    )

    results = build_results(df)

    score = calculate_data_quality_score(results)

    assert score == 100


def test_missing_values_reduce_score():

    df = pd.DataFrame(
        {
            "feature_a": [
                1,
                2,
                3,
                4,
                None,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ],
            "target": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
        }
    )

    results = build_results(df)

    score = calculate_data_quality_score(results)

    # 1 / 20 = 5% missing → -5
    assert score == 95


def test_severe_missing_values_reduce_score_more():

    df = pd.DataFrame(
        {
            "feature_a": [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                16,
                17,
                18,
                19,
            ],
            "target": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
            ],
        }
    )

    results = build_results(df)

    score = calculate_data_quality_score(results)

    # 16 / 20 = 80% missing → -15
    assert score == 85


def test_duplicates_reduce_score():

    df = pd.DataFrame(
        {
            "feature": [
                1,
                1,
                1,
                2,
                3,
            ],
            "target": [
                "A",
                "A",
                "A",
                "B",
                "B",
            ],
        }
    )

    results = build_results(df)

    score = calculate_data_quality_score(results)

    assert score < 100


def test_target_imbalance_reduces_score():

    df = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "target": [
                "A",
                "A",
                "A",
                "A",
                "A",
                "B",
            ],
        }
    )

    results = build_results(df)

    score = calculate_data_quality_score(results)

    # 5:1 imbalance → -5
    assert score == 95


def test_constant_column_reduces_score():

    df = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
            ],
            "constant": [
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )

    results = build_results(
        df,
        target=None,
    )

    score = calculate_data_quality_score(results)

    assert score == 95


# =============================================================
# STATISTICAL SIGNAL TESTS
# =============================================================

def test_high_correlation_does_not_reduce_score():

    df = pd.DataFrame(
        {
            "feature_a": [
                1,
                2,
                3,
                4,
                5,
            ],
            "feature_b": [
                2,
                4,
                6,
                8,
                10,
            ],
        }
    )

    results = build_results(
        df,
        target=None,
    )

    score = calculate_data_quality_score(results)

    assert score == 100


def test_correlation_is_reported_as_signal():

    df = pd.DataFrame(
        {
            "feature_a": [
                1,
                2,
                3,
                4,
                5,
            ],
            "feature_b": [
                2,
                4,
                6,
                8,
                10,
            ],
        }
    )

    results = build_results(
        df,
        target=None,
    )

    health = calculate_health_score(
        results
    )

    signals = health[
        "statistical_signals"
    ]["signals"]

    correlation_signals = [
        signal
        for signal in signals
        if signal["type"] == "correlation"
    ]

    assert correlation_signals


def test_outliers_are_informational_only():

    df = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                100,
            ],
        }
    )

    results = build_results(
        df,
        target=None,
    )

    score = calculate_data_quality_score(
        results
    )

    signals = get_statistical_signals(
        results
    )

    assert score == 100

    assert any(
        signal["type"] == "outliers"
        for signal in signals
    )


def test_skewness_is_informational_only():

    df = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                100,
            ],
        }
    )

    results = build_results(
        df,
        target=None,
    )

    score = calculate_data_quality_score(
        results
    )

    signals = get_statistical_signals(
        results
    )

    assert score == 100

    assert any(
        signal["type"] == "skewness"
        for signal in signals
    )


# =============================================================
# CORRELATION EDGE CASES
# =============================================================

def test_constant_columns_are_not_correlated():

    df = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
            ],
            "constant": [
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )

    results = build_results(
        df,
        target=None,
    )

    correlations = results[
        "correlations"
    ]["numeric"]

    assert correlations == []


# =============================================================
# SIGNAL STRUCTURE
# =============================================================

def test_statistical_signals_have_required_fields():

    df = pd.DataFrame(
        {
            "feature_a": [
                1,
                2,
                3,
                4,
                5,
            ],
            "feature_b": [
                2,
                4,
                6,
                8,
                10,
            ],
        }
    )

    results = build_results(
        df,
        target=None,
    )

    signals = get_statistical_signals(
        results
    )

    assert signals

    required_fields = {
        "type",
        "severity",
        "message",
    }

    for signal in signals:

        assert required_fields.issubset(
            signal.keys()
        )


# =============================================================
# COMPARISON TEST
# =============================================================

def test_bad_dataset_scores_lower_than_healthy_dataset():

    healthy_df = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
            ],
            "target": [
                "A",
                "B",
                "A",
                "B",
                "A",
            ],
        }
    )

    bad_df = pd.DataFrame(
        {
            "feature": [
                1,
                1,
                None,
                1,
                1,
            ],
            "constant": [
                1,
                1,
                1,
                1,
                1,
            ],
            "target": [
                "A",
                "A",
                "A",
                "A",
                "B",
            ],
        }
    )

    healthy_results = build_results(
        healthy_df
    )

    bad_results = build_results(
        bad_df
    )

    healthy_score = calculate_data_quality_score(
        healthy_results
    )

    bad_score = calculate_data_quality_score(
        bad_results
    )

    assert healthy_score > bad_score


# =============================================================
# FINAL HEALTH REPORT
# =============================================================

def test_health_report_contains_quality_and_signals():

    df = pd.DataFrame(
        {
            "feature_a": [
                1,
                2,
                3,
                4,
                5,
            ],
            "feature_b": [
                2,
                4,
                6,
                8,
                10,
            ],
            "target": [
                "A",
                "A",
                "B",
                "B",
                "B",
            ],
        }
    )

    results = build_results(df)

    health = calculate_health_score(
        results
    )

    assert "score" in health
    assert "grade" in health
    assert "data_quality" in health
    assert "statistical_signals" in health

    assert "issues" in health[
        "data_quality"
    ]

    assert "signals" in health[
        "statistical_signals"
    ]

    assert isinstance(
        health["score"],
        int,
    )

    assert 0 <= health["score"] <= 100