"""Multi-account (combined household) ledger merge + account slugging.

The scraper pulls each Robinhood account's report and merges the ledgers into one
combined portfolio before analysis.
"""

import pandas as pd

from automation.ingest import robinhood_scraper as rh


class TestAccountSlug:
    def test_slugs_are_filesystem_safe(self):
        assert rh._account_slug("Sagnik Invest") == "sagnik_invest"
        assert rh._account_slug("Joint investing") == "joint_investing"
        assert rh._account_slug("Roth IRA (2024)") == "roth_ira_2024"
        assert rh._account_slug("") == ""
        assert rh._account_slug(None) == ""


class TestMergeActivityCsvs:
    def _write(self, path, rows):
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_merges_both_ledgers_and_sorts_newest_first(self, tmp_path):
        a = self._write(tmp_path / "a.csv", [
            {"Activity Date": "2026-01-10", "Trans Code": "Buy", "Instrument": "AAPL", "Amount": "-100"},
            {"Activity Date": "2026-03-01", "Trans Code": "Buy", "Instrument": "MSFT", "Amount": "-200"},
        ])
        b = self._write(tmp_path / "b.csv", [
            {"Activity Date": "2026-02-15", "Trans Code": "Buy", "Instrument": "NVDA", "Amount": "-300"},
        ])
        out = rh.merge_activity_csvs([a, b], out_path=tmp_path / "combined.csv")
        assert out is not None
        combined = pd.read_csv(out)
        assert len(combined) == 3  # all rows from both accounts
        assert set(combined["Instrument"]) == {"AAPL", "MSFT", "NVDA"}
        # newest activity first
        assert list(combined["Activity Date"])[0] == "2026-03-01"

    def test_skips_missing_and_empty(self, tmp_path):
        good = self._write(tmp_path / "good.csv", [
            {"Activity Date": "2026-01-01", "Trans Code": "Buy", "Instrument": "VOO", "Amount": "-50"},
        ])
        out = rh.merge_activity_csvs(
            [tmp_path / "does_not_exist.csv", None, good], out_path=tmp_path / "c.csv"
        )
        assert out is not None and len(pd.read_csv(out)) == 1

    def test_nothing_to_merge_returns_none(self, tmp_path):
        assert rh.merge_activity_csvs([None, tmp_path / "missing.csv"]) is None


class TestAccountsConfig:
    def test_default_accounts_present(self):
        # The two real accounts to combine (overridable via RH_ACCOUNTS).
        assert "Sagnik Invest" in rh.ACCOUNTS
        assert any("Joint" in a for a in rh.ACCOUNTS)
