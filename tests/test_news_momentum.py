"""Recent-performance (momentum) tilt + local-LLM market-news sentiment.

These inputs were added so the buy ranking reflects current performance and news,
not only multi-year relative returns (the "very static" feedback). The LLM only
classifies headlines into a bounded score; the ranking rule stays deterministic.
"""

import types

import pytest

from portfolio_analyzer import buy_candidates as bc
from portfolio_analyzer import diagnosis as dg


class TestKeywordSentimentFallback:
    def test_positive_headlines(self):
        r = bc._score_news_sentiment_keyword(["Company beats estimates, raises guidance"])
        assert r["score"] > 0 and r["source"] == "keyword"

    def test_negative_headlines(self):
        r = bc._score_news_sentiment_keyword(["Firm faces lawsuit and analyst downgrade"])
        assert r["score"] < 0

    def test_neutral_headlines(self):
        r = bc._score_news_sentiment_keyword(["Company to present at a conference next week"])
        assert r["score"] == 0.0

    def test_score_is_bounded(self):
        r = bc._score_news_sentiment_keyword(["surge record upgrade beats wins rally"])
        assert -1.0 <= r["score"] <= 1.0


class TestFetchNewsSentiment:
    def test_no_headlines_is_neutral(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bc, "NEWS_SENTIMENT_CACHE_DIR", tmp_path)
        monkeypatch.setattr(bc, "fetch_candidate_news_signals", lambda *a, **k: [])
        r = bc.fetch_candidate_news_sentiment("XYZ")
        assert r["score"] == 0.0 and r["count"] == 0

    def test_falls_back_to_keyword_when_llm_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bc, "NEWS_SENTIMENT_CACHE_DIR", tmp_path)
        monkeypatch.setattr(bc, "fetch_candidate_news_signals", lambda *a, **k: ["big lawsuit and downgrade"])
        monkeypatch.setattr(bc, "_score_news_sentiment_llm", lambda *a, **k: None)
        r = bc.fetch_candidate_news_sentiment("XYZ")
        assert r["source"] == "keyword" and r["score"] < 0 and r["count"] == 1

    def test_uses_llm_score_when_available(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bc, "NEWS_SENTIMENT_CACHE_DIR", tmp_path)
        monkeypatch.setattr(bc, "fetch_candidate_news_signals", lambda *a, **k: ["some headline"])
        monkeypatch.setattr(
            bc, "_score_news_sentiment_llm",
            lambda *a, **k: {"score": 0.7, "summary": "positive earnings", "source": "llm"},
        )
        r = bc.fetch_candidate_news_sentiment("XYZ")
        assert r["source"] == "llm" and r["score"] == 0.7


class TestMomentumTilt:
    def test_recent_winner_gets_positive_tilt(self):
        bonus, note = dg._momentum_tilt({"relative_3m_return_pct": 0.30, "relative_1m_return_pct": 0.10, "relative_1y_return_pct": 0.20})
        assert bonus > 0 and "3-month" in (note or "")

    def test_recent_laggard_gets_negative_tilt(self):
        bonus, _ = dg._momentum_tilt({"relative_3m_return_pct": -0.25, "relative_1m_return_pct": -0.10})
        assert bonus < 0

    def test_is_capped(self):
        bonus, _ = dg._momentum_tilt({"relative_3m_return_pct": 5.0, "relative_1m_return_pct": 5.0, "relative_1y_return_pct": 5.0})
        assert bonus <= 20.0
        low, _ = dg._momentum_tilt({"relative_3m_return_pct": -5.0})
        assert low >= -15.0

    def test_falls_back_to_1y_when_short_windows_missing(self):
        # older enriched universe without 1M/3M still works off 1Y
        bonus, note = dg._momentum_tilt({"relative_1y_return_pct": 0.30})
        assert bonus > 0 and "12-month" in (note or "")

    def test_no_data_is_zero(self):
        assert dg._momentum_tilt({})[0] == 0.0

    def test_recency_dominates(self):
        # strong recent gain outweighs a weak year -> net positive
        bonus, _ = dg._momentum_tilt({"relative_3m_return_pct": 0.40, "relative_1y_return_pct": -0.10})
        assert bonus > 0


class TestNewsSentimentTilt:
    def test_two_sided_and_bounded(self):
        assert dg._news_sentiment_tilt(1.0) == pytest.approx(15.0)
        assert dg._news_sentiment_tilt(-1.0) == pytest.approx(-15.0)
        assert dg._news_sentiment_tilt(0.0) == 0.0
        assert dg._news_sentiment_tilt(None) == 0.0
        assert -15.0 <= dg._news_sentiment_tilt(2.0) <= 15.0  # clamped


class TestApplyNewsToSlate:
    def _slate(self):
        return [
            {"entry": types.SimpleNamespace(market_data_symbol="AAA"), "fit_score": 70.0,
             "why_it_fits": "solid fit", "score_breakdown": {"final_fit_score": 70.0}},
            {"entry": types.SimpleNamespace(market_data_symbol="BBB"), "fit_score": 68.0,
             "why_it_fits": "also fits", "score_breakdown": {"final_fit_score": 68.0}},
        ]

    def test_positive_news_lifts_score_and_can_reorder(self, monkeypatch):
        sent = {
            "AAA": {"score": -0.6, "summary": "lawsuit filed", "count": 3},
            "BBB": {"score": 0.8, "summary": "strong earnings beat", "count": 3},
        }
        monkeypatch.setattr(dg, "fetch_candidate_news_sentiment", lambda sym: sent[sym])
        out = dg._apply_news_sentiment_to_slate(self._slate())
        by = {it["entry"].market_data_symbol: it for it in out}
        assert by["BBB"]["fit_score"] > by["AAA"]["fit_score"]      # news flipped the order
        assert by["AAA"]["news_sentiment_score"] == -0.6
        assert by["BBB"]["score_breakdown"]["news_sentiment"] > 0
        assert "supportive" in by["BBB"]["why_it_fits"]

    def test_no_news_leaves_score_untouched(self, monkeypatch):
        monkeypatch.setattr(dg, "fetch_candidate_news_sentiment",
                            lambda sym: {"score": 0.0, "summary": "", "count": 0})
        out = dg._apply_news_sentiment_to_slate(self._slate())
        assert out[0]["fit_score"] == 70.0 and out[1]["fit_score"] == 68.0
