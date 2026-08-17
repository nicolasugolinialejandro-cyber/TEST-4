# AI-Use Disclosure

## Tools used

| Tool | What it was used for |
|---|---|
| Lovable / Claude (LLM coding assistant) | Drafting the app structure, the vectorised optimiser, docstrings and this documentation; explaining trade-offs between drawdown-minimisation approaches |
| GitHub Copilot-style autocomplete (if used by the team) | Small code completions |
| ChatGPT (if used) | Explaining tracking error, Ulcer index and managed-futures splicing conventions |

No AI service is called by the deployed app at runtime; there are no API keys in
this repository.

## How the team verified the AI-assisted work

1. `verify.py` recomputes headline results **outside** the app's pipeline: total
   return rebuilt from raw prices with plain pandas, max drawdown by brute-force
   double loop, tracking error from first principles with NumPy, and the fast
   matrix engine cross-checked against the readable per-portfolio engine (agreement
   to 1e-10).
2. `pytest` unit tests pin down known-answer cases (a -50% return must give a -50%
   drawdown; a 100% SPY portfolio must reproduce SPY exactly; the weight grid must
   sum to 1 and respect bounds).
3. Two edge cases are tested: a crisis window with no data, and a tracking-error
   budget that cannot be met - both degrade gracefully with a visible warning.
4. Crisis dates and drawdown magnitudes were sanity-checked against published
   figures (SPY lost roughly 55% in 2007-2009 and roughly 34% in the COVID crash).
5. Every team member read the code and can explain the module they present.

## Known limitations of the AI-assisted work

* The LLM proposed the grid-search design; the team chose it over a solver because
  max drawdown is non-convex, and is responsible for that choice.
* LLM-suggested numbers were never trusted directly - all figures in the app and
  presentation come from executed code in this repository.
* Documentation text was drafted with AI and edited by the team for accuracy.
