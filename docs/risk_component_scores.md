# Risk Metric Guide

This document explains the current observed portfolio risk model in plain English.

The goal of this guide is simple:

- help a user understand what each risk metric is trying to say
- explain why each metric matters
- show how the component metrics roll up into the final score

This is not a full financial-planning or suitability model. It is an observed risk model based on:

- current holdings
- portfolio construction
- recent market behavior
- actual trading behavior

It does not yet include:

- outside assets
- debt
- income stability
- taxes
- options or leverage exposure
- sector-specific macro modeling
- valuation or fundamentals

So the right interpretation is:

> This is an observed portfolio risk score, not a complete personal finance risk profile.

## How The Score Works

The model has 3 dimensions:

- `concentration_risk`
- `market_risk`
- `behavioral_risk`

Current weights:

- concentration: `40%`
- market: `40%`
- behavior: `20%`

That means the final score is:

```text
overall_risk_score =
  0.4 * concentration_risk
+ 0.4 * market_risk
+ 0.2 * behavioral_risk
```

Why those weights:

- concentration matters because a portfolio can look strong but still be fragile if too much money sits in a few names
- market behavior matters because a portfolio can be risky even when it is diversified if it behaves much more aggressively than the market
- behavior matters because investor actions can add risk even when the holdings themselves look reasonable

## 1. Concentration Metrics

These metrics ask a simple question:

> Is too much of the portfolio riding on too few holdings?

### `concentration::single_position_weight`

**Label in the app**

- Largest position size

**Plain-English question**

- How dependent is the portfolio on one stock?

**What it means**

- This looks at the biggest holding in the portfolio and asks whether that one name is large enough to dominate results.

**Why it matters**

- Even a great company can become a portfolio risk if it gets too large.
- If one position is oversized, one bad earnings report, one bad year, or one re-rating can hit the whole account hard.

**How to read the score**

- Low score: no single stock dominates the account
- High score: one stock can materially move the whole portfolio

**Bigger picture**

- A high score here means the portfolio is fragile to stock-specific risk.

### `concentration::top_5_weight`

**Label in the app**

- Top 5 holdings dominance

**Plain-English question**

- Are a few holdings driving most of the portfolio?

**What it means**

- This looks at how much of the portfolio sits in the top 5 positions combined.

**Why it matters**

- A portfolio can look diversified because it owns many tickers, but still be effectively concentrated if most of the money sits in the top few names.

**How to read the score**

- Low score: money is spread more broadly
- High score: a small cluster of stocks is doing most of the work

**Bigger picture**

- A high score here means diversification is thinner than the ticker count suggests.

### `concentration::effective_holdings`

**Label in the app**

- True diversification

**Plain-English question**

- How many holdings really matter once concentration is taken into account?

**What it means**

- This tries to answer:
  - “How many equally sized positions would this portfolio behave like?”

- It does not just count tickers.
- It adjusts for the fact that some positions are much bigger than others.

**Why it matters**

- A portfolio with 25 tickers can still behave like a 10-holding portfolio if most of the money is unevenly concentrated.

**How to read the score**

- Low score: the portfolio behaves like it has many meaningful positions
- High score: the portfolio behaves like fewer holdings really matter

**Bigger picture**

- A high score here means diversification is weaker than it first appears.

## 2. Behavior Metrics

These metrics ask:

> Does the investor’s actual behavior look patient and long-term, or more reactive and churn-heavy?

### `behavior::turnover`

**Label in the app**

- Trading churn

**Plain-English question**

- How much of the portfolio are you trading instead of holding?

**What it means**

- Turnover measures how much of the portfolio is being rotated through selling over time.

**Why it matters**

- More turnover usually means more decisions, more timing risk, and more opportunities for behavior to hurt outcomes.
- High turnover does not automatically mean “bad.”
- It does mean behavior is playing a bigger role.

**How to read the score**

- Low score: calmer, more buy-and-hold behavior
- High score: more active, more tactical, more churn

**Bigger picture**

- A high score here means the investor’s style is more active and decision-heavy than a patient long-term style.

### `behavior::short_holding_period`

**Label in the app**

- How long your dollars stay invested

**Plain-English question**

- How long do your invested dollars usually stay in positions?

**What it means**

- This measures how long larger investments are held, not just how long tiny trades are held.
- The current model uses a capital-weighted holding period so bigger dollar decisions matter more than small flips.

**Why it matters**

- A plain median holding period can be distorted by tiny speculative trades.
- Weighting by invested dollars makes the metric better reflect real investor behavior.

**How to read the score**

- Low score: bigger dollars are usually being held for longer
- High score: bigger dollars are being rotated out faster

**Bigger picture**

- A high score here means meaningful capital is being cycled faster than a long-term investing profile would suggest.

## 3. Market Metrics

These metrics ask:

> How has the portfolio behaved relative to the S&P 500, and how much of the account is exposed to that behavior?

### `market::relative_volatility_to_benchmark`

**Label in the app**

- Volatility vs S&P 500

**Plain-English question**

- Has the portfolio been swinging around more than the market?

**What it means**

- This compares the portfolio’s rolling volatility to the S&P 500 over comparable periods.
- It is benchmark-relative on purpose.

**Why it matters**

- If the whole market was volatile, that should not automatically count against the portfolio.
- This metric only penalizes the portfolio for being more volatile than the S&P 500, not for living through a rough market.

**How to read the score**

- Low score: volatility is market-like or calmer
- High score: the portfolio has been materially more volatile than the market

**Bigger picture**

- A high score here means the ride has been rougher than a simple S&P 500 portfolio.

### `market::relative_drawdown_to_benchmark`

**Label in the app**

- Downside depth vs S&P 500

**Plain-English question**

- When the portfolio goes through a bad stretch, is the drop deeper than the S&P 500’s?

**What it means**

- This compares rolling 6-month drawdown depth in the portfolio to rolling 6-month drawdown depth in the S&P 500.
- Recent periods matter more than old periods in the score.

**Why it matters**

- Investors usually feel drawdowns more than volatility.
- A portfolio should not be called “risky” just because the market had a bad stretch.
- It should be called riskier if its downside was worse than the market’s downside.

**How to read the score**

- Low score: recent drawdowns have been in line with or better than the benchmark
- High score: recent downside has been deeper than the market’s

**Bigger picture**

- A high score here means the painful periods have been worse than what a simple S&P 500 investor experienced.

### `market::relative_downside_capture_to_benchmark`

**Label in the app**

- Bad-day behavior vs S&P 500

**Plain-English question**

- On bad market days, does the portfolio hold up better than the S&P 500, about the same, or worse?

**What it means**

- This looks specifically at market down days and asks whether the portfolio tends to lose less than the benchmark, about the same, or more.

**Why it matters**

- This is a very intuitive stress metric.
- It tells you what the portfolio tends to do on the days people care about most: the red days.

**How to read the score**

- Low score: the portfolio usually holds up at least as well as the S&P 500 on bad days
- High score: the portfolio tends to lose more than the market during stress

**Bigger picture**

- A high score here means the portfolio amplifies pain when the market is already weak.

### `market::relative_market_sensitivity_to_benchmark`

**Label in the app**

- Market sensitivity vs S&P 500

**Plain-English question**

- When the S&P 500 moves, does the portfolio usually move about the same, less, or more?

**What it means**

- This is the portfolio’s recent market sensitivity relative to the S&P 500.
- It is based on rolling 6-month windows, with recent periods emphasized.

**Why it matters**

- Some portfolios do not just move with the market. They amplify it.
- This metric captures whether the portfolio has recently been behaving like a higher-octane version of the S&P 500.

**How to read the score**

- Low score: the portfolio is moving roughly in line with the market or a bit more defensively
- High score: the portfolio is amplifying broad market moves

**Bigger picture**

- A high score here means the account has been more market-sensitive than a simple benchmark portfolio.

### `market::equity_exposure`

**Label in the app**

- How fully invested you are

**Plain-English question**

- How much of the account is actually exposed to market risk right now?

**What it means**

- This compares invested holdings to total account value.
- In simple terms, it asks how much of the account is in the market versus sitting in cash.

**Why it matters**

- Two investors can own similar holdings, but the one who is fully invested has more immediate market exposure than the one with a meaningful cash buffer.

**How to read the score**

- Low score: more cash cushion and less immediate market exposure
- High score: most of the account is participating directly in market moves

**Bigger picture**

- A high score here means more of the account is riding market gains and losses right now.

## How The Metrics Roll Up

### Concentration Risk

Average of:

- `concentration::single_position_weight`
- `concentration::top_5_weight`
- `concentration::effective_holdings`

### Market Risk

Average of:

- `market::relative_volatility_to_benchmark`
- `market::relative_drawdown_to_benchmark`
- `market::relative_downside_capture_to_benchmark`
- `market::relative_market_sensitivity_to_benchmark`
- `market::equity_exposure`

### Behavioral Risk

Average of:

- `behavior::turnover`
- `behavior::short_holding_period`

## Final Reminder

This score is best used to answer:

- How concentrated is the portfolio?
- How has it behaved relative to the S&P 500?
- Does the investor’s behavior look patient or churn-heavy?

It is not yet the final word on:

- personal suitability
- future returns
- whether the portfolio is “good” or “bad”

It is a structured way of saying:

> Based on the holdings, behavior, and market history we can observe, how much risk does this portfolio appear to be taking?
