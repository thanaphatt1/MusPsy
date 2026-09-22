"""
fix_task3_labels.py

Corrects TWO related labeling bugs in MusPsy's train/task3.json, documented in
task3_data_quality_finding.md:

1. `input`/`output` are frequently a same-speaker pair (both client or both
   counselor) instead of a genuine client-query -> counselor-response pair,
   because the original extraction skips one line at the tail of each session
   transcript. (Affects ~99.9% of records, verified against task1.json.)

2. For ~25% of records, `history`'s pairing itself is reversed - every single
   pair is (counselor_line, client_line) instead of (client_line,
   counselor_line) - because history pairs are built as consecutive line-pairs
   from the transcript without regard to which role happens to speak first in
   that session. Since LLaMA-Factory always maps element[0]->user role,
   element[1]->assistant role, this means EVERY turn in an affected record
   (not just the final one) teaches the model to generate the client's voice
   under the "assistant" (counselor) role.

train/task1.json holds the same sessions with the full transcript explicitly
labeled ("Consultant:"/"User:" prefixes on every line) - used as ground truth
to rebuild both `history` and `input`/`output` directly from the transcript,
guaranteeing correct (client, counselor) pairing throughout. `instruction` is
left untouched.

Usage:
    python fix_task3_labels.py
    -> writes train/task3_fixed.json
"""

import json
from pathlib import Path

TRAIN_DIR = Path(__file__).parent / "train"
SNIPPET_LEN = 30


def parse_transcript(text: str) -> list[tuple[str, str]]:
    """Parse a Consultant:/User:-labeled transcript into (role, text) tuples."""
    segments = [s.strip() for s in text.strip().split("\n\n") if s.strip()]
    turns: list[tuple[str, str]] = []
    for seg in segments:
        if seg.startswith("Consultant:"):
            turns.append(("Consultant", seg[len("Consultant:"):].strip()))
        elif seg.startswith("User:"):
            turns.append(("User", seg[len("User:"):].strip()))
        elif turns:
            # Rare: a turn's text contains an embedded blank line, splitting it
            # across two segments with no new role prefix - glue it back onto
            # the previous turn rather than silently dropping it.
            role, prev_text = turns[-1]
            turns[-1] = (role, f"{prev_text}\n{seg}")
    return turns


def find_anchor(turns: list[tuple[str, str]], history: list) -> int | None:
    """Find the index in `turns` immediately after task3's history ends, by
    locating history's last pair inside the transcript - checking BOTH
    possible element orderings, since ~25% of records have history's pairing
    reversed (counselor, client) instead of (client, counselor)."""
    if not history:
        return 0
    a, b = history[-1]
    snip_a, snip_b = a.strip()[:SNIPPET_LEN], b.strip()[:SNIPPET_LEN]
    for i in range(len(turns) - 1):
        role_x, text_x = turns[i]
        role_y, text_y = turns[i + 1]
        # Normal orientation: history[-1] = (client, counselor)
        if role_x == "User" and text_x.startswith(snip_a) and role_y == "Consultant" and text_y.startswith(snip_b):
            return i + 2
        # Reversed orientation: history[-1] = (counselor, client)
        if role_x == "Consultant" and text_x.startswith(snip_a) and role_y == "User" and text_y.startswith(snip_b):
            return i + 2
    return None


SYNTHETIC_OPENER = "Hi, I'm Here Today."


def rebuild_history(turns: list[tuple[str, str]], end: int) -> list[list[str]]:
    """Rebuild a correctly-ordered (client, counselor) history from turns[0:end],
    regardless of which role happens to speak first in this session.

    96.4% of sessions genuinely open with the counselor speaking first (verified
    against task1.json), but LLaMA-Factory's alpaca history format strictly
    requires (client_line, counselor_line) pairs - there's no slot for an
    unpaired opening assistant turn. The original dataset's own workaround was
    a fixed placeholder client line ("Hi, I'm Here Today.", found verbatim in
    73.6% of all records) paired with the counselor's genuine opening greeting,
    letting that real, varied text still be trained on despite the format's
    constraint. Reused here rather than dropping the opening turn outright,
    which would discard real opening-greeting training signal."""
    pairs = []
    i = 0
    if end > 0 and turns[0][0] == "Consultant":
        pairs.append([SYNTHETIC_OPENER, turns[0][1]])
        i = 1
    while i < end - 1:
        role_a, text_a = turns[i]
        role_b, text_b = turns[i + 1]
        if role_a == "User" and role_b == "Consultant":
            pairs.append([text_a, text_b])
            i += 2
        else:
            # Misaligned - skip one turn to resync onto a User-starting pair.
            i += 1
    return pairs


def main():
    task1 = json.loads((TRAIN_DIR / "task1.json").read_text(encoding="utf-8"))
    task3 = json.loads((TRAIN_DIR / "task3.json").read_text(encoding="utf-8"))

    fixed = []
    skip_reasons = {"missing_task1": 0, "anchor_not_found": 0, "not_enough_lines": 0, "role_mismatch": 0}
    history_corrected = 0
    io_corrected = 0
    fully_unchanged = 0

    for i, rec3 in enumerate(task3):
        if i >= len(task1) or not task1[i].get("input"):
            skip_reasons["missing_task1"] += 1
            continue

        history = rec3.get("history", [])
        turns = parse_transcript(task1[i]["input"])
        anchor = find_anchor(turns, history)

        if anchor is None:
            skip_reasons["anchor_not_found"] += 1
            continue

        # Scan forward from the anchor for the first genuine (User, Consultant)
        # adjacent pair - for reversed-orientation sessions (transcript opens
        # with the counselor), the position right at `anchor` continues the
        # (Consultant, User) alternation, so the real client->counselor pair
        # sits one position later.
        pair_start = None
        for j in range(anchor, len(turns) - 1):
            if turns[j][0] == "User" and turns[j + 1][0] == "Consultant":
                pair_start = j
                break

        if pair_start is None:
            skip_reasons["not_enough_lines"] += 1
            continue

        role_in, text_in = turns[pair_start]
        role_out, text_out = turns[pair_start + 1]
        if role_in != "User" or role_out != "Consultant":
            skip_reasons["role_mismatch"] += 1
            continue

        new_history = rebuild_history(turns, pair_start)

        new_rec = dict(rec3)
        new_rec["history"] = new_history
        new_rec["input"] = text_in
        new_rec["output"] = text_out
        fixed.append(new_rec)

        hist_changed = new_history != history
        io_changed = (text_in != rec3["input"].strip()) or (text_out != rec3["output"].strip())
        if hist_changed:
            history_corrected += 1
        if io_changed:
            io_corrected += 1
        if not hist_changed and not io_changed:
            fully_unchanged += 1

    out_path = TRAIN_DIR / "task3_fixed.json"
    out_path.write_text(json.dumps(fixed, indent=4, ensure_ascii=False), encoding="utf-8")

    total_skipped = sum(skip_reasons.values())
    print(f"Total records in task3.json:            {len(task3)}")
    print(f"Written to task3_fixed.json:             {len(fixed)}")
    print(f"  - history corrected (reversed pairing): {history_corrected}")
    print(f"  - input/output corrected:               {io_corrected}")
    print(f"  - fully unchanged (already correct):    {fully_unchanged}")
    print(f"Skipped (couldn't resolve):              {total_skipped}")
    for reason, count in skip_reasons.items():
        print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
