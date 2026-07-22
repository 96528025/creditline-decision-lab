# Executive Summary — CreditLine Decision Lab

*One page, for a business audience. Important context: the experiment and its dollar
figures are a clearly-labeled **simulation** built to demonstrate and stress-test the
decision process; they describe no real portfolio.*

## The question

Should we offer credit-line increases — and to whom — given that more available credit
means more spending **and** more default risk?

## What we did

1. **Scored risk.** Built and validated a default-risk model on 150,000 anonymized
   credit records. Two versions agree closely: a bank-style scorecard (easy to audit)
   and a slightly stronger machine-learning model constrained to always treat more
   delinquency and higher utilization as riskier.
2. **Defined who is even eligible.** Customers with recent serious delinquency, bad
   data, or the riskiest 20% of scores are excluded up front: 94,291 of 120,000
   (79%) remain. This rule was locked before any experiment ran and cannot be changed
   after the fact — the pipeline physically refuses.
3. **Ran a (simulated) randomized test** on eligibles, half receiving an increase.
   Before looking at results we committed to the success criteria: spend must rise by
   at least **$12 per customer**, and the default rate may rise by at most **0.50
   percentage points** — a risk tolerance set by (hypothetical) risk management.

## What we found

- **Spending rose ~$25 per eligible customer** (95% confidence roughly $23–27) —
  comfortably above the $12 bar.
- **Default risk stayed inside tolerance.** Our worst-plausible-case estimate of the
  default increase is **0.32 percentage points, below the 0.50 tolerance** — so the
  safety bar is met with statistical confidence, not just on average.
- Had risk management demanded a tighter 0.30-point tolerance, this experiment would
  have been **too small to prove safety either way** — we say so plainly rather than
  claim comfort we cannot support. Proving safety at 0.30 would need roughly four
  times the sample.
- Because the test was simulated, we could grade the procedure against the known
  answer: **on this test, both safety calls it made — the portfolio-wide one and the
  one for the chosen policy — matched the truth.** That is a check on one design, not
  a guarantee about how the method behaves in general.

## Who should get the increase

Targeting models identify who spends more when given credit; they are **not reliable
enough to predict each individual's default reaction, and we do not use them that
way** — safety is enforced at the portfolio level by the experiment itself. The best
policy found is **broad coverage**: offer the increase to ~98% of eligible customers,
trimming only the riskiest edge, where the *estimated* spend benefit is smallest
anyway (roughly $35 per customer in the safest fifth versus $12 in the riskiest —
model estimates, not separately measured). Fancier targeting beat "offer to all
eligibles" by only about $0.41 per customer, which is well inside the margin of
error: we did **not** show that selective targeting is better than simply offering to
everyone eligible, so we recommend the simple, defensible version. The final test
result confirms the chosen policy beats *doing nothing*; it was not designed to rank
it against "offer to all."

## Recommendation

Within this simulated exercise, the process yields: **proceed with credit-line
increases for the eligible population under the 0.50pp risk guardrail**, monitored
the same way it was tested — keep a randomized holdout and track the portfolio-level
default gap against tolerance. Do not adopt a tighter 0.30pp tolerance without a
substantially larger test. Any real decision would need this run on real experimental
data, plus the governance work listed in the README.

## Caveats in one breath

Public dataset of unknown origin; experiment simulated; dollar figures illustrative
of the *method*, not any market; individual-level risk effects deliberately not
relied upon; full limitations in the README.
