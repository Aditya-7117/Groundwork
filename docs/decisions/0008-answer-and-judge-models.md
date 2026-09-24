# 0008. Answer-writing and judging models

Date: 2026-09-19. Status: accepted for the writer; the judge model is superseded by 0022
(GPT-6 Luna), because billing for the Gemini key was blocked when judging was due.

## Context

Generation needs a model that writes answers from retrieved passages, and groundedness scoring
needs a model that judges them. Three constraints shape the choice:

- **Self-preference.** A language model grades text written by its own model family more kindly,
  so the writer and the judge must come from different families.
- **Reproducibility.** Anyone rerunning the evaluation must get the same answers and verdicts.
- **Hardware.** The development machine has 24 GB of unified memory shared by the CPU and GPU.

## Options

- **Writer:** a local open-weights model (Qwen3.8-27B), or a hosted API model.
- **Judge:** Gemini 3.8 Flash (stable), Gemini 3.1 Pro (preview), or a local model.
- **Build of the local model:** the official Ollama `qwen3.8:27b` tag (about 18 GB, including an
  image encoder), or a smaller text-only GGUF quantisation.

## Decision

- **Writer: Qwen3.8-27B, running locally through Ollama** as `qwen3.8-27b-iq4xs`. This is the
  unsloth `Qwen3.8-27B-UD-IQ4_XS.gguf` build, 14,252,845,984 bytes, SHA-256
  `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`, with the official tag's chat
  renderer and sampling defaults. It is called with thinking disabled and temperature 0. Licence
  Apache-2.0.
- **Judge: `gemini-3.8-flash`**, the newest stable Gemini model. A preview model was rejected,
  because Google shuts previews down, which would leave published numbers impossible to rerun.
  Every judge request and response is cached to disk, so verdicts can be re-scored without calling
  the API.
- The writer and the judge come from different model families (Alibaba's Qwen and Google's Gemini),
  so neither grades its own family's output.

Measured on the development machine (Apple M5 Pro, 24 GB), with a retrieval prompt of about 1,950
tokens and thinking disabled:

| Build | GPU share | Output speed | Outcome |
|---|---|---|---|
| Official `qwen3.8:27b` | 82–87% | 0.4 tok/s | Failed on later requests: Metal out-of-memory |
| `qwen3.8-27b-iq4xs` | 100% | 16.8–17.0 tok/s | Stable, including at 16K context (about 9,900 prompt tokens) |

## Consequences

- Answers are reproducible from the pinned GGUF digest and the recorded generation settings. Each
  artefact records the model digest.
- Judge verdicts are reproducible from the cache even after the hosted model is retired, but a
  fresh judging run after retirement will not reproduce them exactly.
- Generation throughput limits how many setups receive answers and judgements. That is why the
  grid runs in two stages: every setup gets retrieval metrics, and only a subset gets generation.
- Timings depend on memory pressure from other processes. Generation runs are made with other
  applications closed, and artefacts record the hardware.
