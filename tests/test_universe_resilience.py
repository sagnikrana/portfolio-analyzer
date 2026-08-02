"""Universe-refresh resilience + weekly-job failure alerts.

Regression home for two production incidents:
  * the Nasdaq-100 Wikipedia table moving → bare StopIteration froze the whole
    weekly universe refresh for 5 weeks (buy candidates went stale silently);
  * silent failures generally (RH session expiry sat unnoticed ~3 weeks).
"""

import pandas as pd
import pytest

from portfolio_analyzer.buy_candidates import (
    _column_matching,
    _find_constituents_table,
    _safe_constituents,
)


class TestFlexibleTableMatching:
    def test_column_matching_is_case_insensitive(self):
        cols = ["Company", "Ticker", "GICS Sector"]
        assert _column_matching(cols, {"ticker", "symbol"}) == "Ticker"
        assert _column_matching(cols, {"Symbol"}) is None  # not present
        assert _column_matching(cols, {"company"}) == "Company"

    def test_finds_constituents_table_by_shape(self):
        good = pd.DataFrame({"Ticker": ["A"] * 60, "Company": ["x"] * 60})
        noise = pd.DataFrame({"Year": [2020, 2021], "Close": [1, 2]})
        table = _find_constituents_table(
            [noise, good], ticker_aliases={"Ticker", "Symbol"}, name_aliases={"Company"}
        )
        assert table is good

    def test_accepts_symbol_alias(self):
        good = pd.DataFrame({"Symbol": ["A"] * 55, "Security": ["x"] * 55})
        table = _find_constituents_table(
            [good], ticker_aliases={"Ticker", "Symbol"}, name_aliases={"Company", "Security"}
        )
        assert table is good

    def test_returns_none_when_no_table_matches(self):
        # too few rows / missing name column -> the StopIteration scenario
        small = pd.DataFrame({"Ticker": ["A", "B"], "Company": ["x", "y"]})
        wrong = pd.DataFrame({"Foo": range(100), "Bar": range(100)})
        assert (
            _find_constituents_table(
                [small, wrong], ticker_aliases={"Ticker"}, name_aliases={"Company"}
            )
            is None
        )


class TestSafeConstituents:
    def test_returns_live_data_when_fetch_succeeds(self, tmp_path):
        live = pd.DataFrame({"ticker": ["AAA", "BBB"]})
        out = _safe_constituents(lambda: live, saved_path=tmp_path / "x.csv", label="X")
        assert list(out["ticker"]) == ["AAA", "BBB"]

    def test_falls_back_to_saved_csv_when_fetch_raises(self, tmp_path):
        saved = tmp_path / "saved.csv"
        pd.DataFrame({"ticker": ["OLD1", "OLD2", "OLD3"]}).to_csv(saved, index=False)

        def boom():
            raise LookupError("page structure changed")

        out = _safe_constituents(boom, saved_path=saved, label="Nasdaq-100")
        assert list(out["ticker"]) == ["OLD1", "OLD2", "OLD3"]  # last-good preserved

    def test_falls_back_when_fetch_returns_empty(self, tmp_path):
        saved = tmp_path / "saved.csv"
        pd.DataFrame({"ticker": ["KEEP"]}).to_csv(saved, index=False)
        out = _safe_constituents(lambda: pd.DataFrame(), saved_path=saved, label="X")
        assert list(out["ticker"]) == ["KEEP"]

    def test_returns_empty_when_fetch_fails_and_no_saved(self, tmp_path):
        def boom():
            raise RuntimeError("down")

        out = _safe_constituents(boom, saved_path=tmp_path / "missing.csv", label="X")
        assert out.empty  # never raises; degrades to empty


class TestFailureAlerts:
    @pytest.fixture
    def wj(self, monkeypatch):
        from automation import weekly_job as wj

        store: dict = {}
        monkeypatch.setattr(wj, "_load_state", lambda: dict(store))
        monkeypatch.setattr(wj, "_save_state", lambda s: store.update(s))
        sent: list[tuple[str, str]] = []
        monkeypatch.setattr(wj, "send_alert", lambda subj, body: sent.append((subj, body)))
        return wj, sent

    def test_alert_sends_once_then_throttles_same_key(self, wj):
        mod, sent = wj
        mod._alert("boom", "body", key="k", throttle_hours=12)
        mod._alert("boom again", "body2", key="k", throttle_hours=12)
        assert len(sent) == 1  # duplicate within window suppressed
        assert "boom" in sent[0][0]

    def test_different_keys_both_send(self, wj):
        mod, sent = wj
        mod._alert("a", "b", key="k1")
        mod._alert("a", "b", key="k2")
        assert len(sent) == 2

    def test_alert_never_raises_if_send_fails(self, wj, monkeypatch):
        mod, _sent = wj

        def boom(*_a, **_k):
            raise RuntimeError("smtp down")

        monkeypatch.setattr(mod, "send_alert", boom)
        # must not propagate — alerting can never crash the job
        mod._alert("x", "y", key="kx")
