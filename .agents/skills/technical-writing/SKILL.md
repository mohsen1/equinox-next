---
name: technical-writing
description: Write and edit clear technical prose for engineers. Use when creating or substantially editing documentation, implementation plans, design docs, research write-ups, runbooks, summaries, PR descriptions, tickets, or any repo prose where clarity, precision, and readability matter.
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Technical writing

Write engineering prose that helps readers act. The default reader is another engineer who knows the codebase shape but not the specific work.

Use this skill for documentation, plans, design docs, runbooks, research notes, tickets, PR descriptions, and substantial edits to existing prose. If another skill owns the structure, such as `plan`, keep that structure and use this skill as the final writing pass.

Do not add a separate interview step. Use the user's request, repo context, and source material. Ask only when a missing fact would change the outcome and cannot be discovered safely.

## Target outcome

The reader should be able to answer these questions quickly:

- what changed, or what decision is being proposed
- why it matters
- what is in scope and out of scope
- what the main tradeoffs, risks, and unknowns are
- what they should do next

## Writing rules

- Put the conclusion first. Lead with the decision, recommendation, status, or operational fact.
- Keep technical detail. Do not simplify away constraints, edge cases, failure modes, security notes, or data-loss risks.
- Prefer concrete nouns and verbs. Name the component, command, path, setting, owner, date, metric, or failure mode when it helps.
- Use active voice when the actor matters. Passive voice is fine when the actor is irrelevant.
- Keep paragraphs short. Split long sentences when they carry more than one idea.
- Cut filler: "it is important to note", "in order to", "going forward", "as mentioned above", "seamless", "robust", "leverage", "unlock".
- Keep necessary terms of art. Explain a term only when an engineer outside the immediate area may not know it.
- Be explicit about uncertainty. Use "unknown", "assumption", "risk", or "needs verification" instead of sounding certain.
- Do not turn engineering docs into marketing copy. Avoid hype, slogans, and vague benefits.

## Structure rules

- Match the existing document shape unless the structure blocks understanding.
- Use sentence case headings. Make headings descriptive enough to skim.
- Use bullets for parallel facts and numbered lists for ordered steps.
- Use tables for comparison, ownership, matrices, and file lists. Use prose or bullets for narrative.
- Keep code snippets short and purposeful. Use them for interfaces, commands, or examples that reduce ambiguity.
- Make links descriptive. The link text should say what the reader will open.
- Preserve exact conventions in code, logs, schemas, API fields, command output, direct quotes, and interface labels.

## Document-specific guidance

Plans and design docs should include:

- recommendation or decision
- current state and target state
- scope and non-goals
- implementation phases with dependencies
- tradeoffs and rejected options when they matter
- verification plan
- rollout, rollback, or recovery notes for risky changes
- open questions that block implementation or review

Runbooks should include:

- when to use the runbook
- prerequisites and safety checks
- exact steps in order
- expected signals after each risky step
- rollback or stop conditions

Research notes should include:

- answer first
- evidence and sources
- confidence level
- alternatives considered
- unresolved questions

PR descriptions and tickets should include:

- the user-visible or operational problem
- the concrete change
- the verification performed
- risks, follow-ups, or rollout notes

## Editing workflow

When editing existing prose:

1. Read enough surrounding context to understand the document's purpose.
2. Identify the action or decision the reader needs from the document.
3. Move the most important point up.
4. Remove duplication and throat-clearing.
5. Replace vague claims with concrete facts.
6. Tighten sentences without deleting technical meaning.
7. Keep unrelated sections and formatting stable.

## Common mistakes to fix

- external style-guide references that do not help the next agent write better
- rigid word bans that make accurate engineering language worse
- polished summaries that omit caveats, rollback, or verification
- large templates for small changes
- unexplained acronyms or local shorthand
- bold or italics used to compensate for weak structure
- questions added for style preference when context is already enough

## Final check

Before finishing, check:

- the most important point is first
- an engineer outside the immediate work can follow it on first read
- claims are specific enough to verify
- caveats and risks are still present
- the document tells the reader what to do next
- you removed words that do not change meaning
