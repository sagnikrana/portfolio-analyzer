---
name: feedback-style
description: How the user wants to collaborate — what to avoid, what to keep doing
metadata:
  type: feedback
---

**Don't claim something is fixed without verifying it in the live running app.**  
**Why:** Codex repeatedly said "fixed" but the user still saw broken UI in the browser. This eroded trust significantly.  
**How to apply:** For UI bugs especially, verify the actual rendered behavior (start the server, check it) before reporting done. If you can't verify, say so explicitly.

**No fluff, no ego, no promises — execution only.**  
**Why:** User is direct and collaborative. Wants progress, not commentary.  
**How to apply:** Short responses. State what changed and what's next. Skip summaries of what you just did if the diff is visible.

**Prefer simpler controls over styled-but-fragile ones.**  
**Why:** Gradio checkbox/radio styling kept washing out in the light theme, causing repeated UI regressions.  
**How to apply:** When adding Backtesting or similar controls, use `gr.Dropdown` with explicit choices (e.g., "No"/"Yes") rather than checkboxes or radio buttons.

**Metrics must match their evidence charts.**  
**Why:** The user caught that the volatility score and volatility chart were derived from different data paths, making the score untrustworthy. Same issue was found for drawdown and market sensitivity.  
**How to apply:** Whenever a risk metric has a supporting chart, verify both come from the same computation path. Score ≠ chart = broken trust.

**Don't add features or tabs unless they clearly add value.**  
**Why:** User explicitly pushed back on tab proliferation and unnecessary UI clutter.  
**How to apply:** Ask before adding a new tab. Prefer enriching existing tabs.

**Recommendation language: use "Review / Trim / Hold / Add" not "Sell" or "Buy" imperatively.**  
**Why:** Financial advice framing creates legal/trust risk and overstates model certainty.  
**How to apply:** All sell/buy recommendations should be framed as suggestions with evidence, not commands.
