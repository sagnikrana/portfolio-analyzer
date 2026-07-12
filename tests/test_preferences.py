"""Buy-preference model + the single vehicle-selector mapping.

Guards the "ETFs only" feature: the dashboard exposes one Radio whose value is
derived from the preference flags, and the apply handler maps the string back.
The two directions must stay consistent.
"""

import pytest

from portfolio_analyzer.app import _vehicle_preference_value
from portfolio_analyzer.diagnosis import PortfolioPreferences


def _prefs(**flags) -> PortfolioPreferences:
    return PortfolioPreferences(**flags)


class TestVehiclePreferenceMapping:
    def test_none_is_blend(self):
        assert _vehicle_preference_value(None) == "Blend"

    def test_etfs_only(self):
        assert _vehicle_preference_value(_prefs(allow_etfs=True, allow_single_stocks=False)) == "ETFs only"

    def test_single_stocks_only(self):
        assert (
            _vehicle_preference_value(_prefs(allow_etfs=False, allow_single_stocks=True))
            == "Single stocks only"
        )

    def test_prefer_single_stocks(self):
        prefs = _prefs(allow_etfs=True, allow_single_stocks=True, single_stocks_preferred=True)
        assert _vehicle_preference_value(prefs) == "Prefer single stocks"

    def test_prefer_etfs(self):
        prefs = _prefs(allow_etfs=True, allow_single_stocks=True, single_stocks_preferred=False)
        assert _vehicle_preference_value(prefs) == "Prefer ETFs"

    def test_blend_when_both_allowed_no_lean(self):
        prefs = _prefs(allow_etfs=True, allow_single_stocks=True, single_stocks_preferred=None)
        assert _vehicle_preference_value(prefs) == "Blend"


class TestPreferenceSchema:
    """Fields the buy engine / vehicle mapping read by name — renaming any of
    these silently breaks buy-idea generation, so pin them down."""

    @pytest.mark.parametrize(
        "field",
        [
            "allow_etfs",
            "allow_single_stocks",
            "single_stocks_preferred",
            "buy_idea_limit",
            "include_existing_holdings",
            "prefer_high_dividend_etfs",
            "prefer_low_expense_for_dividend_etfs",
            "vehicle_preference_label",
        ],
    )
    def test_field_exists(self, field):
        assert field in PortfolioPreferences.model_fields
