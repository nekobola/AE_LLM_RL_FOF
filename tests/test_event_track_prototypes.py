import numpy as np

from src.compute.event_track import EventTrack


def test_event_track_crisis_prefers_defensive_assets():
    track = EventTrack()
    returns_5d = np.array(
        [
            [0.03, -0.04, 0.05, -0.02, 0.04],
            [0.06, -0.08, 0.09, -0.03, 0.07],
            [0.003, 0.002, -0.001, 0.002, 0.001],
            [0.004, -0.002, 0.003, 0.002, -0.001],
            [0.0001, 0.0002, 0.0000, -0.0001, 0.0001],
        ],
        dtype=float,
    )

    weights = track.compute(
        returns_5d,
        llm_macro=30.0,
        llm_sentiment=25.0,
        llm_risk=90.0,
    )

    defensive_weight = weights[2] + weights[3] + weights[4]
    equity_weight = weights[0] + weights[1]
    assert defensive_weight > equity_weight
    assert weights[2] >= 0.25


def test_event_track_growth_prefers_satellite_equity():
    track = EventTrack()
    returns_5d = np.array(
        [
            [0.01, 0.012, 0.008, 0.011, 0.009],
            [0.025, 0.020, 0.027, 0.023, 0.021],
            [0.001, 0.000, 0.002, 0.001, 0.000],
            [0.002, 0.001, 0.001, 0.002, 0.001],
            [0.000, 0.000, 0.000, 0.000, 0.000],
        ],
        dtype=float,
    )

    weights = track.compute(
        returns_5d,
        llm_macro=78.0,
        llm_sentiment=82.0,
        llm_risk=35.0,
    )

    assert weights[1] > weights[2]
    assert weights[1] > weights[3]
