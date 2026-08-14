# ADR 0001: Detect manipulation techniques, not truth

- **Status:** accepted
- **Date:** 2026-08-14

## Context

The original concept was a "lie detector" for news headlines. A model cannot
verify factual truth from headline text alone, and claiming to do so creates
credibility and defamation risk, and invites justified criticism from anyone
familiar with the field.

## Decision

ManipuLens detects **manipulation techniques** — rhetorical devices measurable
from text (curiosity gap, outrage bait, fear-mongering, false certainty,
emotional framing, sensational formatting) — and never claims to assess truth.
All user-facing output names techniques ("uses curiosity-gap and outrage
framing"), never verdicts ("this is a lie"). The closest we go to veracity is
the headline↔body *delivery gap* (does the headline overpromise relative to the
article?), which is an entailment claim about the article itself, not the world.

## Consequences

- Labels come from a written codebook with inter-annotator agreement, making the
  task well-posed and auditable.
- A political-neutrality audit becomes a flagship feature: we test that
  technique scores are invariant to entity swaps and balanced across outlet
  leanings.
- Marketing copy, API field names, model card, and docs must all use
  technique-detection language.
