"""
convert_for_wai_ctrs.py

Converts a transcript saved by muspsy_multisession_test.py (--save) into the
TheraMind-style JSON schema thesis-framework/evaluation/run_wai_ctrs.py already
knows how to parse (parse_theramind_json): a dict with a "sessions" key, each
session holding a flat "dialogs" list of "PATIENT: ..."/"DOCTOR: ..." lines.

The synthetic bootstrap opener muspsy_multisession_test.py injects to trigger the
counselor's first line ("Hi, I'm here for my first session." / "Hi, I'm here
again.") is dropped from the client side - it was never shown to SimulatedClient
and isn't a real client utterance. The counselor's genuine reply to it is kept,
so each session's dialogs simply start with the counselor speaking, same as
MusPsy's own training data convention (96.4% of real sessions open this way).

Usage:
    python convert_for_wai_ctrs.py fixed_model_test.json --profile-id muspsy_fixed_liu
    -> writes eval_input/muspsy_fixed_liu.json
"""

import argparse
import json
from pathlib import Path


def convert(sessions: list, profile_id: str) -> dict:
    sessions_export = {}
    for s in sessions:
        dialogs = []
        for i, turn in enumerate(s["history"]):
            client_text = turn["client"]
            counselor_text = turn["counselor"]
            if i == 0:
                # Drop the synthetic bootstrap opener - keep only the counselor's
                # genuine reply to it.
                dialogs.append(f"DOCTOR: {counselor_text}")
            else:
                dialogs.append(f"PATIENT: {client_text}")
                dialogs.append(f"DOCTOR: {counselor_text}")
        sessions_export[f"session_{s['session']}"] = {"dialogs": dialogs}

    return {"profile_id": profile_id, "sessions": sessions_export}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to a muspsy_multisession_test.py --save output JSON")
    parser.add_argument("--profile-id", required=True, help="Identifier for this run, becomes the output filename and profile_id field")
    parser.add_argument("--out-dir", default="eval_input", help="Output directory (default: eval_input)")
    args = parser.parse_args()

    sessions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    converted = convert(sessions, args.profile_id)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.profile_id}.json"
    out_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False), encoding="utf-8")

    total_dialogs = sum(len(s["dialogs"]) for s in converted["sessions"].values())
    print(f"Converted {len(converted['sessions'])} sessions, {total_dialogs} dialog lines -> {out_path}")


if __name__ == "__main__":
    main()
