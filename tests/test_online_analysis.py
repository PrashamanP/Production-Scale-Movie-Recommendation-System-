from datetime import datetime, timedelta

import pytest

from src.evaluation.online_als import (
    RECO_LOG_PATTERN,
    RecsByUser,
    WatchesByUser,
    calculate_watch_through_rate,
    two_proportion_z_test,
)


def _dt(hours: int) -> datetime:
    return datetime(2024, 1, 1, hour=hours)


def test_reco_log_pattern_handles_bucket():
    line = (
        "2025-11-11T01:11:50Z,100,recommendation request /recommend/100, "
        "status 200, variant=als_model, bucket=0.611009, result: m1,m2, 5 ms"
    )
    match = RECO_LOG_PATTERN.search(line)
    assert match is not None
    assert match.group("variant") == "als_model"
    assert match.group("bucket") == "0.611009"


def test_calculate_watch_through_rate_with_variants():
    recs_by_user: RecsByUser = {
        "42": [
            (_dt(10), {"m1", "m2"}, "als"),
            (_dt(11), {"m3"}, "popgen"),
        ]
    }
    watches_by_user: WatchesByUser = {
        "42": [
            (_dt(10) + timedelta(minutes=30), "m2"),
            (_dt(12), "m4"),
        ]
    }

    total, converted, wtr, variant_details = calculate_watch_through_rate(
        recs_by_user,
        watches_by_user,
        attribution_window_hours=1,
    )

    assert total == 2
    assert converted == 1
    assert pytest.approx(wtr) == 0.5
    assert pytest.approx(variant_details["als"]["watch_through_rate"]) == 1.0
    assert pytest.approx(variant_details["popgen"]["watch_through_rate"]) == 0.0


def test_calculate_watch_through_rate_assigns_unknown_variant():
    recs_by_user: RecsByUser = {
        "7": [
            (_dt(9), {"mx"}, "unknown"),
        ]
    }
    watches_by_user: WatchesByUser = {
        "7": [
            (_dt(9) + timedelta(minutes=5), "mx"),
        ]
    }

    _, _, _, variant_details = calculate_watch_through_rate(
        recs_by_user,
        watches_by_user,
        attribution_window_hours=1,
    )

    assert "unknown" in variant_details
    assert variant_details["unknown"]["converted_recommendation_events"] == 1


def test_two_proportion_z_test_basic():
    result = two_proportion_z_test(success_a=40, total_a=100, success_b=20, total_b=100)
    assert result is not None
    assert "z_score" in result
    assert pytest.approx(result["effect_size"], rel=1e-3) == 0.2
    assert result["p_value"] < 0.05
