# 0022. The judge moves to GPT-6 Luna

Date: 2026-09-25. Status: accepted. Supersedes the judge model in 0008 and the model, effort and
price in 0019; the rubrics, scales and cost guard of 0019 are unchanged.

## Context

The judge was to be gemini-3.8-flash on a billed key. The key turned out to be on the free tier,
which allows 20 requests a day for that model, and setting up billing was blocked for one to two
days. Judging is on the critical path right after answer writing. No answer had been judged, so
the model could change without touching any result.

## Options

- **Wait for Gemini billing**, then gemini-3.8-flash at medium thinking: about $20 estimated, at
  $0.75 and $3.75 per million input and output tokens.
- **GPT-6 Luna** (`gpt-6-luna`, released 23 September 2026): $0.10 and $0.50 per million input
  and output tokens, structured outputs, reasoning effort from none to max, and Tier 1 limits of
  500 requests and 500,000 tokens a minute. OpenAI's lightest GPT-6 model.
- **GPT-6 Sol**, the stronger model: about $50 estimated at medium effort.

## Decision

GPT-6 Luna at high reasoning effort, through the Responses API with a strict JSON schema and
storage off. High effort offsets Luna being the lightest model, and still costs about a quarter
of the Gemini plan (about $5 estimated, list price of 25 September 2026). The judge still comes
from a different company than the Qwen writer.

## Consequences

- Luna has a single version with no dated snapshot, so the provider can update it in place; the
  response cache keeps every published verdict reproducible regardless.
- Reasoning models take no temperature, so a fresh call can give a different label; the cache is
  the record.
- How well Luna judges is not assumed: agreement with the 200 blind hand labels measures it.
