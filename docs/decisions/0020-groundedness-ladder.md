# 0020. The groundedness ladder and judge agreement

Date: 2026-09-24. Status: accepted.

## Context

A language-model judge is only worth its cost if it agrees with people better than cheaper
methods do. The project measures that rather than assuming it.

## Options

- **Middle rung:** a general NLI classifier or a groundedness-specific checker.
- **Scores to labels:** cut-offs fixed in advance, tuned on labels, or no labels at all.
- **Hand labels:** how many, how chosen, and how shown.

## Decision

- Rungs, cheapest first: word overlap on the words an answer adds beyond the question;
  MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (MIT, pinned) taking the strongest
  entailment across the passages; the Gemini judge; 200 hand labels.
- Cut-offs fixed before any label: NLI supported at entailment 0.5 or more, word overlap only at
  1.0. Also ROC AUC, which needs no cut-off.
- 200 hand labels, 40 per stage-two setup, seeded and shuffled, labelled blind to setup, reference
  and judge in a terminal tool that saves each label with a digest of exactly what was shown.
- Agreement is Cohen's kappa with a 95% bootstrap interval.

## Consequences

- The judge is compared with people on its own three-level scale; the cheaper rungs on "fully
  supported or not".
- With 200 labels, a kappa interval is roughly a tenth wide either way, and it is reported with
  every kappa.
