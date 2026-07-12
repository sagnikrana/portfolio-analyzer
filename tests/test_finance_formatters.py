"""Number parsing / formatting and ratio scaling.

Regression home for the VOO "3.00%" expense-ratio bug: yfinance reports
`netExpenseRatio` as a percent number (0.03 == 0.03%), the app stores ratios as
decimal fractions, and the display layer multiplies by 100. Getting any link in
that chain wrong re-inflates expense ratios 100x.
"""

import math

import pytest

from portfolio_analyzer.app import (
    _expense_ratio_text,
    _pe_text,
    money_text,
    number_text,
    parse_display_number,
    parse_float,
    pct_text,
    percent_display,
)
from portfolio_analyzer.buy_candidates import _expense_ratio_to_fraction, _normalize_ratio


class TestExpenseRatioScaling:
    """The exact bug the user caught: VOO must read 0.03%, never 3.00%."""

    @pytest.mark.parametrize(
        "yf_percent, expected_fraction",
        [(0.03, 0.0003), (0.18, 0.0018), (0.06, 0.0006), (1.0, 0.01)],
    )
    def test_yfinance_percent_becomes_fraction(self, yf_percent, expected_fraction):
        assert _expense_ratio_to_fraction(yf_percent) == pytest.approx(expected_fraction)

    @pytest.mark.parametrize("bad", [None, 0, 0.0, -0.5])
    def test_missing_or_nonpositive_is_none(self, bad):
        assert _expense_ratio_to_fraction(bad) is None

    def test_nan_is_none(self):
        assert _expense_ratio_to_fraction(float("nan")) is None

    def test_voo_end_to_end_reads_three_basis_points(self):
        # netExpenseRatio -> stored fraction -> display string
        fraction = _expense_ratio_to_fraction(0.03)
        text = _expense_ratio_text(fraction)
        assert text == "0.03%"
        assert text != "3.00%"
        # the guard a one-line test would have caught the bug with:
        assert fraction < 0.01  # i.e. under 1% for a broad-market index ETF

    def test_display_of_a_plain_fraction(self):
        assert _expense_ratio_text(0.0003) == "0.03%"
        assert _expense_ratio_text(None) == "N/A"


class TestNormalizeRatio:
    """Legacy ratio normalizer: values > 1 are percents, <= 1 already fractions."""

    @pytest.mark.parametrize(
        "raw, expected",
        [(25, 0.25), (1.5, 0.015), (0.5, 0.5), (0.0103, 0.0103), (100, 1.0)],
    )
    def test_scaling(self, raw, expected):
        assert _normalize_ratio(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("bad", [None, float("nan"), "not-a-number"])
    def test_bad_inputs_are_none(self, bad):
        assert _normalize_ratio(bad) is None


class TestParseFloat:
    @pytest.mark.parametrize("raw, expected", [(3, 3.0), (2.5, 2.5), ("4.2", 4.2)])
    def test_valid(self, raw, expected):
        assert parse_float(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("bad", [None, "abc", float("nan"), object()])
    def test_invalid_is_none(self, bad):
        assert parse_float(bad) is None


class TestParseDisplayNumber:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("$1,234.50", 1234.50),
            ("12.5%", 12.5),
            ("1,000", 1000.0),
            (1000, 1000.0),
            (12.5, 12.5),
            ("-3.25", -3.25),
        ],
    )
    def test_strips_and_parses(self, raw, expected):
        assert parse_display_number(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("empty", [None, "", "N/A", "  ", "abc"])
    def test_empty_or_bad_is_none(self, empty):
        assert parse_display_number(empty) is None


class TestDisplayHelpers:
    def test_pct_text(self):
        assert pct_text(0.1234) == "12.34%"
        assert pct_text(-0.05) == "-5.00%"
        assert pct_text(None) == "N/A"
        assert pct_text(float("nan")) == "N/A"

    def test_money_text(self):
        assert money_text(1234.5) == "$1,234.50"
        assert money_text(0) == "$0.00"
        assert money_text(None) == "N/A"

    def test_pe_text(self):
        assert _pe_text(21.34) == "21.3x"
        assert _pe_text(None) == "N/A"
        assert _pe_text("bad") == "N/A"

    def test_number_and_percent_display(self):
        assert number_text(1234.567, 2) == "1,234.57"
        assert number_text(None) == "N/A"
        assert percent_display(0.0625, 2) == "6.25%"
        assert percent_display(None) == "N/A"
