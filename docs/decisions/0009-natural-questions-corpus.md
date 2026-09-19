# 0009. The evaluation corpus: Natural Questions

Date: 2026-09-19. Status: accepted. Supersedes decision 0004, which made BEIR SciFact provisional
plumbing.

## Context

The project claims to measure which retrieval setup answers questions correctly. That needs
questions with reference answers, and documents long and structured enough for chunking strategies
to differ at all. Most retrieval benchmarks supply passages of 46 to 132 words, shorter than a
single chunk, which would make the chunking comparison meaningless.

## Decision

Google's Natural Questions, validation split, pinned to one published revision of the dataset with
a SHA-256 digest for each of its seven files. Licence CC BY-SA 3.0. Real search questions over
whole Wikipedia pages: a median page carries 4,306 words and 25 section headings.

Four rules turn the raw dataset into the corpus, each chosen to keep the numbers honest.

- **Only questions with an agreed answer.** At least two of the five annotators must have marked
  both the paragraph containing the answer and the answer words inside it. That leaves 3,336 of
  7,830 questions.
- **One revision per article.** 402 articles appear as several revisions. Keeping them all would
  let a retriever find the right paragraph in the wrong revision and be scored as a miss. The
  fullest revision is kept, and answers from other revisions are relocated into it by matching
  their text: 362 relocated, 116 dropped because the paragraph no longer exists there. Every
  dropped question id is written to the build record.
- **End-of-page sections are left out** (References, External links, See also and similar): link
  and citation lists that crowd retrieval results. Measured first: no answer sits in them.
- **Pages are rebuilt from the dataset's token stream, not its HTML.** Answer positions are token
  positions, so this keeps the mapping exact. Checked against 1,404 real answers: paragraph and
  list answers map word for word, table answers to the answering cell.

## Consequences

- The built corpus holds 6,930 pages, 37.4M words and 3,220 questions.
- 93.1% of the source words reach the pages. The rest is the dropped sections, plus about 0.2% of
  stray text in malformed markup. The build record states both numbers, so the loss is visible
  rather than discovered later by someone counting.
- Relevance is judged at passage level: a chunk counts when it overlaps the annotated answer span.
- The corpus is derived, not redistributed. The build reads the pinned files and writes gzipped
  JSON Lines, and the record names the dataset revision, the digests and the dropped sections.
