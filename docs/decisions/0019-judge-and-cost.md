# 0019. The judge: rubrics, effort and cost

Date: 2026-09-24. Status: accepted for the rubrics, scales and cost guard; the model, effort and
price are superseded by 0022 (GPT-6 Luna at high effort).

## Context

Groundedness and correctness are judged by gemini-3.8-flash. The free tier allows 20 requests a
day for this model, and about 8,500 judgements are needed.

## Options

- **Groundedness scale:** three levels per answer, claim by claim, or yes or no.
- **Correctness scale:** two or three levels.
- **Thinking level:** low, medium or high ("minimal" is not supported by this model).
- **Paying:** billing on the API key, a local judge, or an IDE's chat agents.

## Decision

- Groundedness: supported, partly supported, not supported, judged from the passages without the
  reference. Correctness: correct or incorrect, against the reference answers. Separate calls,
  labels constrained by a response schema, default temperature as Google recommends for Gemini 3.
- Medium thinking, on a billed key: about $20 estimated at the list price of 24 September 2026
  ($0.75 per million input tokens, $3.75 per million output tokens, thinking billed as output).
  The real cost is projected after 50 calls, and the run stops if it heads more than half again
  over the approved budget.
- Rejected: an interactive coding agent or IDE chat agent as the judge. Its requests cannot be
  pinned, recorded or replayed, it has seen the reference answers, and it cannot be called at
  scale.

## Consequences

- Every request and response is cached, so verdicts are reproducible after the model is retired.
- The verdicts artefact records the rubrics, the dated price, the budget and the real spend.
