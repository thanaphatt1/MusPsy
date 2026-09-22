# Data Quality Issue in MusPsy's Task 3 (Counseling Generation) Training Data

## Summary

MusPsy's publicly released `train/task3.json` — the Supervised Fine-Tuning data for
the Counseling Generation task — has a systematic extraction bug: the `input` and
`output` fields of each training record are almost never a genuine
(client's-line → counselor's-response) pair, despite the task's instruction framing
the model as "a psychotherapist" generating a response to the client. Instead, in
**99.9% of records, `input` and `output` belong to the same speaker** — both the
client's lines, or both the counselor's lines — because the extraction script that
built the dataset skips exactly one line when slicing the tail of each session
transcript.

This was discovered while running the fine-tuned MusPsy model in a live multi-session
inference loop: the counselor side of the conversation would periodically produce
client-voiced text (expressing gratitude, receiving advice, agreeing to homework)
while still nominally speaking as the therapist. Tracing this back to the training
data confirmed it is a direct, explainable consequence of this labeling bug, not an
artifact of the fine-tuning run or the serving setup.

## Background: the intended structure

Each record in `task3.json` has four fields:

- `instruction` — a fixed system-style prompt: *"You are a psychotherapist who has
  conducted multiple counseling sessions with the user... Goals: {goal}, User
  Background: {profile}, Last Memory: {memory}"*
- `history` — a list of `[client_line, counselor_line]` pairs: every prior exchange
  in the session, given as plain context
- `input` — intended to be the client's most recent line, the thing the counselor
  must respond to right now
- `output` — the counselor's response to `input`; this is the only field the model
  is actually trained against (the supervised target)

During fine-tuning (via LLaMA-Factory), these fields are flattened into one
continuous chat sequence, alternating `user`/`assistant` roles: every `history` pair
becomes a `user`→`assistant` turn, and the instruction is glued onto `input` to form
the final `user` turn, with `output` as the final `assistant` turn the loss is
computed against. The intended design is a clean training signal: *given the
conversation so far and the client's latest line, learn to produce the counselor's
reply.*

## What the data actually looks like

### Case A — `output` is a client line, not the counselor's response

`train/task3.json`, record index 0:

```json
{
  "input": "I'll do that.",
  "output": "Thank you. That means a lot coming from you. I'll see you next time!"
}
```

Cross-referenced against the same session's fully speaker-labeled transcript
(`train/task1.json`, record index 0 — the Memory Extraction task's `input` field,
which stores the raw dialogue with explicit `Consultant:`/`User:` prefixes):

```
Consultant: That could be a meaningful step. Meanwhile, for this week, try to
            focus on painting a little and consider jotting down any recurring
            thoughts related to stress as they emerge. This could help us in
            addressing them effectively.

User:       I'll do that.                                    <- task3 "input"

Consultant: Excellent, Liu. We can explore more about these thoughts in our
            next session. Remember, it's about progress, not perfection.
            You're doing great by simply being here and addressing these
            concerns.                                          <- SKIPPED, never appears in task3

User:       Thank you. That means a lot coming from you.
            I'll see you next time!                            <- task3 "output"
```

The genuine counselor response is silently dropped. `output` — the field the model
is actually trained to reproduce, under a "you are the psychotherapist" instruction
— is the **client's** closing line.

### Case B — `input` is a counselor line, not the client's query

`train/task3.json`, record index 500, cross-referenced the same way
(`train/task1.json`, record index 500, lines 24–28):

```
24  Consultant: Outdoor walks can be very rejuvenating... Anything else
                you'd like guidance on today?
25  User:       I think we've covered a lot today. I'll try out these
                strategies and see how it goes.
26  Consultant: Wonderful, Yang. Remember, progress is gradual...       <- task3 "input"
27  User:       Thank you for your support. I'll take these strategies
                seriously and look forward to our next talk.            <- SKIPPED, never appears in task3
28  Consultant: You're doing really well, Yang... see you next time!    <- task3 "output"
```

Here `output` happens to still be counselor-voiced, but `input` is *also* a
counselor line — the client's real line (27) that should have been `input` is
dropped entirely. The model's "query" in this example isn't a client utterance at
all.

## Methodology

`task3.json`'s own fields carry no speaker labels, so they can't be checked in
isolation. `task1.json` (Memory Extraction) covers the same sessions in the same
record order and stores each session's full transcript with explicit
`Consultant:`/`User:` prefixes on every line — this was used as ground truth.

For each record `i`:
1. Confirmed `task1[i]` and `task3[i]` refer to the same session (verified by
   matching `task3[i]`'s `history` text character-for-character inside
   `task1[i]`'s labeled transcript).
2. Took the literal text of `task3[i].input` and `task3[i].output`.
3. Located that text as a substring inside `task1[i]`'s labeled transcript.
4. Scanned backward from that position for the nearest preceding `Consultant:` or
   `User:` label to determine who actually spoke that line.

This is direct ground-truth text matching against explicit labels, not inference or
subjective judgment of "who sounds like the client."

## Results (full dataset, all 7,179 records)

| | Count | % |
|---|---:|---:|
| `input` = client line, `output` = client line ("both client") | 5,377 | 74.90% |
| `input` = counselor line, `output` = counselor line ("both counselor") | 1,796 | 25.02% |
| Correctly alternating (`input`=client → `output`=counselor) | 6 | 0.08% |
| Text not found in transcript | 0 | 0.00% |

**99.92% of Task 3's training records are same-speaker pairs. Only 6 out of 7,179
records (0.08%) show the intended client-query → counselor-response structure.**

## Why this matters for the fine-tuned model's behavior

Since `output` is the only field carrying a training gradient, and it is a client
line in 74.9% of examples, the model was — in the large majority of its Task 3
supervision — trained to complete "you are the psychotherapist, respond now" with
text that is actually the client's voice: expressions of gratitude, agreement,
receiving guidance, saying goodbye *to* the counselor. This directly explains
observed inference-time behavior such as:

```
[Counselor] Thanks. I'll try to keep that in mind. See you next time!
```

— a line structurally and tonally identical to what a client, not a counselor,
would say in this dataset's conventions. The effect was most visible toward the
later turns of longer test sessions, where the conversation's tone naturally
approaches something resembling a session's end — exactly the position this bug is
concentrated in, since every affected record is drawn from the tail of a session
transcript.

## A second, more severe form of the same bug: `history` pairing is also reversed

The `input`/`output` bug above corrupts exactly **one** turn per training
record — the final one. `history` — the list of prior-turn pairs given as plain
context, typically 8-10 pairs per record — has a separate, more consequential
problem: for roughly a quarter of records, **every single pair in `history` is
reversed, not just one of them.** So instead of "one bad turn out of ten," an
affected record has all ten turns backwards.

`history` was initially assumed to be reliable, since its pairs are drawn from
consecutive, unbroken transcript lines. That assumption is only half right: the
*content* of each pair is correct, but for a substantial share of records, the
*ordering* of each pair is reversed — `[counselor_line, client_line]` instead of
the intended `[client_line, counselor_line]`.

This happens because `history` pairs are built as simple consecutive line-pairs
from the transcript, without checking which role happens to speak first in that
particular session. A session that opens with the client ("Hi, I'm here today")
naturally alternates client/counselor/client/counselor, so consecutive pairing
produces `(client, counselor)` throughout. A session that opens with the
counselor ("Welcome back, how have you been?" — the typical opening for session
2 onward) produces `(counselor, client)` throughout instead. Verified directly:
for `task3.json` record index 10 (a session that opens with the counselor), all
10 of its `history` pairs are consistently reversed, not just the last one.

**This matters more than the `input`/`output` bug alone**, because LLaMA-Factory
always maps `pair[0] → user role, pair[1] → assistant role` regardless of who
actually spoke. For a reversed record, this means *every single turn in that
record* — not just the final supervised one — teaches the model to generate the
client's words under the "assistant" (counselor) role.

A dataset-wide check (locating each record's last `history` pair inside
`task1.json`'s labeled transcript, checking both possible orderings) found:

| | Count | % |
|---|---:|---:|
| Normal orientation (`client, counselor`) | 5,342 | 74.4% |
| Reversed orientation (`counselor, client`) | 1,798 | 25.0% |
| Empty history (session's first supervised turn) | 39 | 0.5% |

This closely tracks the `input`/`output` split above — both are the same
underlying phenomenon (which role opens the session), just observed from two
different angles.

## The fix

Since `task1.json` provides ground-truth speaker labels for the same sessions,
all three issues above can be corrected directly: for each `task3.json` record,
locate where its `history` ends inside `task1.json`'s labeled transcript
(checking both orderings to find the anchor), rebuild `history` as strictly
`(client, counselor)` pairs by walking the transcript from the start, and derive
`input`/`output` from the first genuine client→counselor exchange found from
that anchor point onward. `instruction` is left untouched.

One refinement worth calling out: an early version of this fix simply dropped
any unpaired opening counselor turn (i.e. when a session opens with the
counselor and there's no preceding client line to pair it with). Checking this
against `task1.json` directly showed that **96.4% of sessions genuinely open
with the counselor speaking first** (0% genuinely open with the client) — so
dropping that turn would have discarded real opening-greeting training signal
across nearly the whole dataset, not just the placeholder cases. The final
version instead reuses the original dataset's own workaround: pair the
unpaired opening counselor line with the same synthetic client-side placeholder
already used in 73.6% of the original records (`"Hi, I'm Here Today."`),
preserving the counselor's genuine, varied opening greeting as a trainable
target rather than discarding it. This isn't fabricating dialogue — it's
reusing the existing dataset convention for representing "the counselor speaks
first" inside a format that structurally requires client-first pairing.

Implemented in `fix_task3_labels.py`, writing `train/task3_fixed.json`.

**Result: 7,139 of 7,182 records (99.4%) successfully corrected.** The remaining
43 are legitimate edge cases — sessions too short to have a valid exchange left
after `history` — not script failures.

**Net effect on data volume**: with the opener-preservation fix, most records'
`history` length is now unchanged from the original (72.8%), only 1.8% got
shorter, and 25.4% actually gained pairs (recovering genuine opening greetings
that were missing or mislabeled in the reversed-orientation records). Across the
whole dataset: 92,452 → 95,095 total `history` pairs — a net **gain** of 2.86%,
not a loss.

Spot-checked against both examples above:
- Record 0: `output` is now *"Excellent, Liu. We can explore more about these
  thoughts in our next session... You're doing great by simply being here and
  addressing these concerns."* — the genuine counselor line, previously missing
  entirely. `history` is unchanged (9 pairs, same as original).
- Record 10 (reversed orientation): `input` is now *"Thank you for the
  encouragement. I'm eager to see the positive changes."* — verified as a
  genuine `User:`-labeled line in the transcript. `history` grew from 10 to 11
  pairs, now correctly including the genuine opening counselor greeting ("It's
  lovely to see you again! Have you had a chance to think about any
  activities...") that the original's mislabeled version never included at all.

## Scope note

This is a data construction issue in MusPsy's publicly released training data
(`train/task3.json`), not an issue with the fine-tuning run, the LoRA training
configuration, or the vLLM serving setup used in this reproduction. The training
process correctly learned exactly what the data taught it. A corrected version
of the dataset (`train/task3_fixed.json`) is available for re-training as a
direct before/after comparison.
