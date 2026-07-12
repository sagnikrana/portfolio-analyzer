"""Monthly performance reconstruction.

Locks the statement identity  Ending = Beginning + Deposits + Market Gain + Income
and the column set (the redundant "Personal Investment Returns" column was
dropped; Cumulative Returns must remain and still be driven by price + income).
"""

import pandas as pd

from portfolio_analyzer.app import build_monthly_performance_frame


def _sample_inputs():
    """Three month-ends with one active middle month.

    Jan: opening month, value 10,000 (no gain by construction).
    Feb: +1,000 deposit, +50 dividend, one $500 buy, ends at 11,500.
         market gain = 11,500 - 10,000 - 1,000 - 50 = 450.
    Mar: no activity, drifts 11,500 -> 12,000, market gain = 500.
    """
    timeseries = [
        {"date": "2024-01-31", "account_value": 10_000.0},
        {"date": "2024-02-29", "account_value": 11_500.0},
        {"date": "2024-03-31", "account_value": 12_000.0},
    ]
    transactions = pd.DataFrame(
        [
            {"Activity Date": "2024-02-05", "Trans Code": "ACH", "Amount_num": 1_000.0},
            {"Activity Date": "2024-02-10", "Trans Code": "CDIV", "Amount_num": 50.0},
            {"Activity Date": "2024-02-15", "Trans Code": "Buy", "Amount_num": -500.0},
        ]
    )
    return transactions, timeseries


def _row(frame: pd.DataFrame, month: str) -> pd.Series:
    return frame.loc[frame["Month"] == month].iloc[0]


class TestColumns:
    def test_personal_column_dropped_but_breakdown_kept(self):
        tx, ts = _sample_inputs()
        frame = build_monthly_performance_frame(tx, ts)
        assert "Personal Investment Returns" not in frame.columns
        for col in ("Market Gain / Loss", "Income Returns", "Cumulative Returns", "Investment Amount"):
            assert col in frame.columns

    def test_empty_timeseries_returns_empty_frame(self):
        assert build_monthly_performance_frame(pd.DataFrame(), []).empty


class TestStatementMath:
    def test_active_month_values(self):
        tx, ts = _sample_inputs()
        frame = build_monthly_performance_frame(tx, ts)
        feb = _row(frame, "02/2024")
        assert feb["Deposits / Withdrawals"] == 1_000.0
        assert feb["Income Returns"] == 50.0
        assert feb["Investment Amount"] == 500.0  # a $500 buy shown as a positive deploy
        assert feb["Market Gain / Loss"] == 450.0

    def test_quiet_month_is_pure_price_move(self):
        tx, ts = _sample_inputs()
        frame = build_monthly_performance_frame(tx, ts)
        mar = _row(frame, "03/2024")
        assert mar["Market Gain / Loss"] == 500.0
        assert mar["Income Returns"] == 0.0

    def test_ending_identity_holds_every_row(self):
        tx, ts = _sample_inputs()
        frame = build_monthly_performance_frame(tx, ts)
        for _, r in frame.iterrows():
            expected_ending = (
                r["Beginning Balance"]
                + r["Deposits / Withdrawals"]
                + r["Market Gain / Loss"]
                + r["Income Returns"]
            )
            assert r["Ending Balance"] == round(expected_ending, 2)

    def test_cumulative_is_running_sum_of_price_plus_income(self):
        tx, ts = _sample_inputs()
        frame = build_monthly_performance_frame(tx, ts)
        # Jan personal 0, Feb 450+50=500, Mar 500  ->  cumulative 0, 500, 1000
        assert _row(frame, "02/2024")["Cumulative Returns"] == 500.0
        assert _row(frame, "03/2024")["Cumulative Returns"] == 1_000.0
