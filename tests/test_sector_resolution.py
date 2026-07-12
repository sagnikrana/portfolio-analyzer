"""Sector-label resolution.

Regression home for the "sectors collapse into one bucket / Unclassified" bug:
`fetch_yahoo_sector_label` used to depend entirely on a flaky live yfinance call
and cached even failed responses for 14 days. The fix added a deterministic
override map (checked first, no network) and stopped caching failures.
"""

import portfolio_analyzer.app as app
from portfolio_analyzer.app import SECTOR_OVERRIDES, _cache_key, fetch_yahoo_sector_label

# yfinance's canonical sector names. Override labels MUST be drawn from this set,
# otherwise an override-resolved ticker and a live-resolved ticker land in
# different buckets and the breakdown fragments.
CANONICAL_SECTORS = {
    "Technology",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Energy",
    "Utilities",
    "Real Estate",
    "Basic Materials",
    "ETF / Fund",
}


class TestOverrideMap:
    def test_known_megacaps_map_correctly(self):
        expected = {
            "AAPL": "Technology",
            "AMD": "Technology",
            "AVGO": "Technology",
            "NVDA": "Technology",
            "GOOGL": "Communication Services",
            "META": "Communication Services",
            "AMZN": "Consumer Cyclical",
            "TSLA": "Consumer Cyclical",
            "JPM": "Financial Services",
            "XOM": "Energy",
            "NEE": "Utilities",
            "VOO": "ETF / Fund",
            "QQQ": "ETF / Fund",
        }
        for ticker, sector in expected.items():
            assert SECTOR_OVERRIDES[ticker] == sector

    def test_all_override_labels_are_canonical(self):
        # Guards against a typo like "Tech" that would splinter the aggregation.
        assert set(SECTOR_OVERRIDES.values()) <= CANONICAL_SECTORS

    def test_map_is_non_trivial(self):
        assert len(SECTOR_OVERRIDES) >= 150


class TestDeterministicResolution:
    def test_known_ticker_resolves_without_network(self, monkeypatch):
        """Override short-circuits before any yfinance call — so even if the
        network is dead, well-known holdings still get the right sector."""

        def _boom(*_a, **_k):
            raise RuntimeError("network should not be touched for a known ticker")

        monkeypatch.setattr(app.yf, "Ticker", _boom)
        fetch_yahoo_sector_label.cache_clear()
        assert fetch_yahoo_sector_label("AAPL") == "Technology"
        assert fetch_yahoo_sector_label("googl") == "Communication Services"  # case-insensitive
        assert fetch_yahoo_sector_label("VOO") == "ETF / Fund"

    def test_failed_unknown_lookup_is_not_cached(self, monkeypatch):
        """A transient/rate-limited failure must NOT be persisted, so a later run
        can still resolve the real sector (this was the 14-day poisoning bug)."""

        def _boom(*_a, **_k):
            raise RuntimeError("simulated yfinance outage")

        monkeypatch.setattr(app.yf, "Ticker", _boom)
        fetch_yahoo_sector_label.cache_clear()

        bogus = "ZZZ_NOTREAL_TEST"
        cache_file = app.SECTOR_CACHE_DIR / f"{_cache_key('sector', bogus)}.json"
        if cache_file.exists():
            cache_file.unlink()

        assert fetch_yahoo_sector_label(bogus) == "Unclassified"
        assert not cache_file.exists(), "failed lookups must not be cached"


class TestAggregationMapping:
    def test_eight_stock_portfolio_groups_as_expected(self):
        """The exact holdings from the sector-bug investigation group into the
        correct sectors (not everything into Communication Services)."""
        holdings = ["AAPL", "AMD", "AMZN", "AVGO", "GOOGL", "META", "NVDA", "TSLA"]
        by_sector: dict[str, list[str]] = {}
        for t in holdings:
            by_sector.setdefault(SECTOR_OVERRIDES[t], []).append(t)

        assert sorted(by_sector["Technology"]) == ["AAPL", "AMD", "AVGO", "NVDA"]
        assert sorted(by_sector["Communication Services"]) == ["GOOGL", "META"]
        assert sorted(by_sector["Consumer Cyclical"]) == ["AMZN", "TSLA"]
