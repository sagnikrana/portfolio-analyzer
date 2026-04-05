# Risk Component Scores

This document explains the current observed portfolio risk model used by the portfolio analyzer.

It covers:

- what each risk component metric means
- what high and low values indicate
- how the component metrics roll up into dimension scores
- how the dimension scores roll up into the final observed risk score

## What This Risk Model Is

The current model is an **observed portfolio risk score**.

It is designed to answer:

- How risky does the portfolio look based on actual holdings and behavior?
- What specific factors are driving that risk?
- Is the observed portfolio risk aligned with the user’s stated risk score?

It is **not** a full financial suitability model.

It does not yet include:

- outside assets
- debt
- income stability
- tax constraints
- options or leverage exposure
- sector-specific macro risk
- fundamentals or valuation risk
- liquidity needs

So this should be interpreted as a **portfolio- and behavior-based observed risk model**, not a complete financial planning framework.

## High-Level Structure

The model has three dimensions:

- `concentration_risk`
- `market_risk`
- `behavioral_risk`

Each dimension is made up of several component metrics.

Each component is normalized into a value between `0` and `1`, then displayed in the UI as a `0-100` score.

The current weighting is:

- `concentration_risk`: `40%`
- `market_risk`: `40%`
- `behavioral_risk`: `20%`

The final score is:

```text
overall_risk_score =
  0.4 * concentration_risk
+ 0.4 * market_risk
+ 0.2 * behavioral_risk
```

## Concentration Metrics

These metrics measure how clustered the portfolio is.

### `concentration::single_position_weight`

**What it measures**

- The weight of the single largest holding in the portfolio.

**Plain-English question**

- How dependent is the portfolio on one stock?

**Why it matters**

- If one stock becomes too large, a negative move in that one name can heavily damage the total portfolio.

**How the current model uses it**

- It is normalized against roughly `22%`.
- Around `22%` or above in one position pushes this metric toward high risk.

**Low value means**

- No single stock dominates the portfolio.

**High value means**

- One stock is carrying a large share of the risk.

**How it influences overall risk**

- It raises `concentration_risk`.
- Since concentration is `40%` of the total score, this metric can materially increase the final risk score.

### `concentration::top_5_weight`

**What it measures**

- The total portfolio weight held in the top 5 positions.

**Plain-English question**

- Are just a few holdings driving most of the portfolio?

**Why it matters**

- A portfolio can have many tickers but still be effectively concentrated if the top 5 names dominate the value.

**How the current model uses it**

- It is normalized against roughly `65%`.
- If the top 5 holdings are around `65%` or more of the portfolio, this metric becomes high risk.

**Low value means**

- Capital is spread across more holdings.

**High value means**

- A small set of positions controls most of the portfolio outcome.

**How it influences overall risk**

- It raises `concentration_risk`.

### `concentration::effective_holdings`

**What it measures**

- The effective number of holdings, derived from concentration using `1 / HHI`.

**Plain-English question**

- How many equally weighted positions does this portfolio behave like?

**Why it matters**

- A portfolio with many tickers can still behave like a very concentrated portfolio if most capital is clustered in a few names.

**How the current model uses it**

- The score is now relative to the number of **meaningful holdings** in the portfolio.
- A meaningful holding is currently defined as a position with at least `1%` portfolio weight.
- The concentration risk contribution is:

```text
effective_holdings_risk = 1 - (effective_holdings / meaningful_holdings_count)
```

with clipping between `0` and `1`.

**Low value means**

- The portfolio’s effective holdings count is close to the number of meaningful positions.
- In other words, diversification is broad relative to the portfolio’s true size.

**High value means**

- The portfolio behaves like it has far fewer meaningful holdings than it appears to have.
- In other words, capital is clustered in a small subset of names.

**How it influences overall risk**

- Lower effective holdings relative to meaningful holdings raises `concentration_risk`.

## Market Metrics

These metrics measure how risky the portfolio’s market behavior is.

### `market::portfolio_volatility`

**What it measures**

- Annualized volatility of the invested portfolio return series.

**Plain-English question**

- How much does the portfolio swing around over time?

**Why it matters**

- Higher volatility usually means a rougher ride and more short-term uncertainty.

**How the current model uses it**

- It is normalized against roughly `32%` annualized volatility.

**Low value means**

- Smoother portfolio behavior.

**High value means**

- Larger swings and higher uncertainty.

**How it influences overall risk**

- It raises `market_risk`.

### `market::drawdown`

**What it measures**

- The maximum peak-to-trough decline in the invested portfolio series.

**Plain-English question**

- How bad was the worst loss period?

**Why it matters**

- Drawdown reflects the actual depth of losses the investor would have experienced.

**How the current model uses it**

- It is normalized against roughly `40%`.

**Low value means**

- The portfolio has historically held up better.

**High value means**

- The portfolio has suffered deeper losses.

**How it influences overall risk**

- It raises `market_risk`.

### `market::downside_capture`

**What it measures**

- How the portfolio behaves on negative benchmark days relative to the benchmark.

**Plain-English question**

- When the S&P 500 falls, does this portfolio fall less, about the same, or more?

**Why it matters**

- It measures how painful bad market days are for this portfolio relative to the market.

**How the current model uses it**

- It is normalized against roughly `1.15`.

**Interpretation**

- Below `1`: the portfolio tends to lose less than the benchmark on bad days
- Around `1`: similar downside behavior to the benchmark
- Above `1`: tends to lose more than the benchmark on bad days

**Low value means**

- More defensive downside behavior.

**High value means**

- More aggressive downside behavior.

**How it influences overall risk**

- It raises `market_risk`.

### `market::beta`

**What it measures**

- Sensitivity of portfolio returns to benchmark returns.

**Plain-English question**

- How strongly does the portfolio move with the market?

**Why it matters**

- Beta captures whether the portfolio amplifies or dampens broad market moves.

**How the current model uses it**

- It is normalized against roughly `1.25`.

**Interpretation**

- `beta < 1`: less sensitive than the market
- `beta ~ 1`: behaves roughly like the market
- `beta > 1`: more sensitive than the market

**Low value means**

- Lower market sensitivity.

**High value means**

- Higher market sensitivity and more aggressive market exposure.

**How it influences overall risk**

- It raises `market_risk`.

### `market::equity_exposure`

**What it measures**

- The invested portfolio value divided by total account value estimate.

**Plain-English question**

- How much of the account is actually exposed to market risk instead of sitting in cash?

**Why it matters**

- A fully invested account is more exposed to market moves than an account holding significant idle cash.

**How the current model uses it**

- It is normalized against `1.0`.

**Low value means**

- More cash buffer and less immediate market exposure.

**High value means**

- Most or all capital is currently exposed to the market.

**How it influences overall risk**

- It raises `market_risk`.

## Behavioral Metrics

These metrics measure how aggressive the investor’s behavior looks based on actual trading patterns.

### `behavior::turnover`

**What it measures**

- Annualized turnover proxy based on sell activity relative to buy activity.

**Plain-English question**

- How actively is the investor trading?

**Why it matters**

- Higher turnover often suggests a more active, tactical, or speculative style.

**How the current model uses it**

- It is normalized against roughly `0.50`.

**Low value means**

- Low churn and more accumulation-and-hold behavior.

**High value means**

- More active trading and more speculative behavior.

**How it influences overall risk**

- It raises `behavioral_risk`.

### `behavior::short_holding_period`

**What it measures**

- How short the capital-weighted median holding period is across both open and closed lots.

**Plain-English question**

- How long are the investor’s dollars typically held before being sold or before today if still open?

**Why it matters**

- A plain trade-count median can be distorted by tiny speculative positions.
- Weighting by cost basis makes larger capital commitments matter more than tiny flips.

**How the current model uses it**

- It uses an `18-month` target, which is about `548 days`.
- It computes a cost-basis-weighted median holding duration across:
  - closed lots: `sell_date - buy_date`
  - open lots: `today - buy_date`
- Shorter capital-weighted holding periods increase risk.

The current normalization is:

```text
short_holding_period_risk =
1 - min(capital_weighted_median_holding_days / 548, 1)
```

**Low value means**

- The investor’s dollars are generally held for long enough to resemble patient, longer-term investing.

**High value means**

- The investor’s dollars are typically rotated out more quickly.
- This suggests more active or tactical behavior.

**How it influences overall risk**

- It raises `behavioral_risk`.

## How Component Metrics Roll Up

The component metrics are averaged inside each dimension.

### `concentration_risk`

Average of:

- `concentration::single_position_weight`
- `concentration::top_5_weight`
- `concentration::effective_holdings`

### `market_risk`

Average of:

- `market::portfolio_volatility`
- `market::drawdown`
- `market::downside_capture`
- `market::beta`
- `market::equity_exposure`

### `behavioral_risk`

Average of:

- `behavior::turnover`
- `behavior::short_holding_period`

## How Dimension Scores Roll Up Into Final Risk

Once each dimension score is computed, the final score is:

```text
overall_risk_score =
  0.4 * concentration_risk
+ 0.4 * market_risk
+ 0.2 * behavioral_risk
```

So:

- concentration contributes `40%`
- market contributes `40%`
- behavior contributes `20%`

## Example

If the dimension scores are:

- `concentration_risk = 53.4`
- `market_risk = 71.5`
- `behavioral_risk = 2.3`

Then the approximate final score is:

```text
0.4 * 53.4 = 21.36
0.4 * 71.5 = 28.60
0.2 * 2.3  = 0.46

Total ≈ 50.42
```

That gives an overall observed risk score around `50.4/100`.

If the UI shows something like `50.7`, that is usually because:

- the dimension scores shown in the interface are rounded
- the final risk score is calculated from unrounded internal values

## How To Interpret High and Low Values

### Low values across most metrics

- better diversification
- smoother portfolio behavior
- less aggressive trading style
- lower overall observed risk

### High concentration metrics

- portfolio is more dependent on a few names
- stock-specific risk matters more

### High market metrics

- portfolio behaves more aggressively in real market conditions
- higher volatility, bigger drawdowns, higher benchmark sensitivity

### High behavioral metrics

- investor behavior itself appears more aggressive or reactive
- more trading-driven risk

## Important Caveat

These scores describe:

- observed portfolio risk
- observed market behavior
- observed trading behavior

They do **not** fully describe:

- future return potential
- investor suitability
- risk capacity
- whether the portfolio is fundamentally strong or weak

This is a practical MVP risk framework, not a complete wealth-management risk engine.
