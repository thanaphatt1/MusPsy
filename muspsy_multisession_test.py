"""
muspsy_multisession_test.py

Standalone multi-session inference test for the MusPsy fine-tuned model, following
the paper's three-task loop (Section 5, Figure 5):
    Task 1: Memory Extraction   - summarize a finished session into structured memory
    Task 2: Goal Planning       - propose the next session's goal from memory + profile
    Task 3: Counseling Generation - conduct the session, one counselor turn at a time

Each counselor turn is a single inference call against the "muspsy" model served by
muspsy_serve_vllm.ipynb, via thesis-framework's get_llm(). The instruction templates
below are copied verbatim from MusPsy/train/task{1,2,3}.json - the model was
fine-tuned on this exact wording/structure, so inference must match it exactly.

The simulated client reuses thesis-framework's own simulation.client.SimulatedClient
(same class run_session.py uses for its baseline agents) rather than a hand-rolled
prompt, so the client side is tested with the same, already-proven simulator - this
script only replaces the *counselor* side with MusPsy's own trained inference loop,
deliberately not routing through the framework's baseline architecture (CounselorSession
etc.), since that would test the framework's scaffolding around MusPsy's weights, not
MusPsy's actual designed behavior.

Usage:
    python muspsy_multisession_test.py --sessions 3 --turns 6
    python muspsy_multisession_test.py --sessions 2 --turns 4 --save transcript.json
"""

import argparse
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THESIS_FRAMEWORK_DIR = Path(r"C:\Users\61302262\Desktop\thesis-framework")
sys.path.insert(0, str(THESIS_FRAMEWORK_DIR))
from utils.llm_utils import get_llm  # noqa: E402
from simulation.client import SimulatedClient  # noqa: E402

COUNSELOR_MODEL = "vllm:muspsy"
TEMPERATURE = 0.7  # matches paper Appendix F: T=0.7 for counselor and client generation

# --- Task instruction templates, verbatim from MusPsy/train/task{1,2,3}.json ---

TASK1_INSTRUCTION = (
    "\nYou now need to help me summarize and consolidate the memory of this "
    "counseling session based on the given dialogue history. Note: All generated "
    "content must remain in English, and no Chinese characters should be output.\n"
    "The principles and content for summarizing the user's background are as "
    "follows:\n\nConseling Memories:\nEmotion and Cognitive State、Counselor "
    "Observations、Counseling Assignments、Session Goal.\n\ndialogue history:\n"
)

TASK2_INSTRUCTION = (
    "\nThe following is the memory from previous consultations and the user's "
    "background. Please generate an appropriate goal for the next consultation. "
    "When doing so, consider the user's individual characteristics, as outlined in "
    "their past sessions. These characteristics may include personality traits, "
    "interests, interpersonal relationships, and behavioral patterns that have "
    "emerged throughout the consultation process.\n\nIn your goal, aim to address "
    "the specific needs highlighted by the user’s personality traits, such as "
    "their emotional responses, communication preferences, coping mechanisms, and "
    "their tendency to engage with certain therapeutic techniques. Ensure that the "
    "goal takes into account their current state, challenges, and progress made in "
    "previous sessions.\n\nBy doing this, you will provide a goal that is "
    "personalized, actionable, and consistent with the principles of Cognitive "
    "Behavioral Therapy (CBT), tailored to the unique characteristics of the user.\n"
)

TASK3_INSTRUCTION_TEMPLATE = (
    "\nYou are a psychotherapist who has conducted multiple counseling sessions "
    "with the user. Below are the given historical session memories and the goal "
    "for this consultation. Your role is to engage in a conversation that builds "
    "upon past discussions while acknowledging the user's unique experiences and "
    "ongoing challenges.\n\nAt the start of the session, if the previous session "
    "included specific tasks or reflections(in [Consultation Assignments]) for the "
    "user, begin by checking in on their progress. \n\nThroughout the session, "
    "incorporate details about the user's personal background, such as "
    "relationships(in [Personal Characteristics]) with family and friends, hobbies, "
    "recurring emotional patterns, and coping mechanisms. Acknowledge their past "
    "reflections, struggles, and achievements to create a supportive and "
    "personalized therapeutic space.\n\nEncourage the user to explore their "
    "thoughts and emotions in depth, guiding them toward actionable strategies "
    "that align with their unique circumstances.\n\nGoals: {goal}  \n"
    "User Background: {profile}\n  \nLast Memory: {last_memory}  \n"
)

# Real client profile from MusPsy/train/task2.json, used as-is so the request
# format matches training distribution exactly (no synthetic/paraphrased profile).
# Also doubles as SimulatedClient's intake_form (plain text, no bracket parsing needed).
DEFAULT_PROFILE = (
    "[Personal Profile]\nLiu is a university student facing significant stress, "
    "particularly about upcoming exams and meeting family expectations, having "
    "grown up in a rural setting. Liu values academic achievement deeply and is "
    "concerned about disappointing their family. \n\n[Personal Characteristics]  "
    "\nHe also enjoys painting, which is sometimes used as a stress-relief "
    "activity.\n\n[Assessment and Diagnosis]\nLiu experiences anxiety and stress "
    "primarily triggered by academic pressures, yet there is no mention of a "
    "formal diagnosis of a mental health disorder."
)


def counselor_llm():
    return get_llm(COUNSELOR_MODEL, temperature=TEMPERATURE, max_tokens=512)


def run_task1_memory_extraction(dialogue_text: str) -> str:
    """Task 1: summarize a finished session's dialogue into structured memory."""
    llm = counselor_llm()
    # LLaMA-Factory's AlpacaDatasetConverter builds the final user turn as
    # "\n".join([instruction, input]) - since TASK1_INSTRUCTION already ends in
    # its own "\n", the real training format has a blank line here, not a
    # single newline. Must match exactly, not just concatenate.
    content = "\n".join([TASK1_INSTRUCTION, dialogue_text])
    resp = llm.invoke([{"role": "user", "content": content}])
    return resp.content


def run_task2_goal_planning(profile: str, memory: str) -> str:
    """Task 2: propose the next session's goal from profile + prior memory."""
    llm = counselor_llm()
    input_text = f"{profile}\nCounseling Memory:"
    if memory:
        input_text += f"\n\n{memory}"
    # See run_task1_memory_extraction's comment - match "\n".join([instruction, input])
    # exactly, not naive concatenation.
    content = "\n".join([TASK2_INSTRUCTION, input_text])
    resp = llm.invoke([{"role": "user", "content": content}])
    return resp.content


def run_task3_counselor_turn(profile: str, goal: str, last_memory: str, history: list) -> str:
    """Task 3: generate one counselor turn, given growing in-session history.
    history is a list of (client_turn, counselor_turn) pairs; the last entry's
    counselor_turn may be None (the turn currently being generated)."""
    llm = counselor_llm()
    instruction = TASK3_INSTRUCTION_TEMPLATE.format(
        goal=goal, profile=profile, last_memory=last_memory or "None"
    )
    messages = []
    for client_turn, counselor_turn in history[:-1]:
        messages.append({"role": "user", "content": client_turn})
        messages.append({"role": "assistant", "content": counselor_turn})
    last_client_turn = history[-1][0]
    # See run_task1_memory_extraction's comment - match "\n".join([instruction, input])
    # exactly, not naive concatenation.
    messages.append({"role": "user", "content": "\n".join([instruction, last_client_turn])})
    resp = llm.invoke(messages)
    return resp.content


def run_session(session_num: int, profile: str, goal: str, last_memory: str, client: SimulatedClient, num_turns: int) -> list:
    print(f"\n{'=' * 70}\nSession {session_num} | Goal: {goal.strip()[:200]}\n{'=' * 70}")
    history = []  # list of (client_turn, counselor_turn)

    # Bootstrap: a synthetic opener triggers the counselor's real first line,
    # matching Task 3's trained format (the client always speaks first in the
    # training data). This opener is internal only - it's never shown to
    # SimulatedClient, whose own history starts from the counselor's real first
    # substantive turn, matching how run_session.py drives its own baseline agents.
    opener = "Hi, I'm here again." if session_num > 1 else "Hi, I'm here for my first session."
    history.append((opener, None))
    counselor_turn = run_task3_counselor_turn(profile, goal, last_memory, history)
    history[-1] = (opener, counselor_turn)
    print(f"[Counselor] {counselor_turn}")

    for _ in range(num_turns):
        client_turn = client.generate_response(counselor_turn)
        has_end_token = "[/END]" in client_turn
        display = client_turn.replace("[/END]", "").strip()
        print(f"\n[Client]    {display}")
        if has_end_token:
            print("(session ended by client)")
            break

        history.append((client_turn, None))
        counselor_turn = run_task3_counselor_turn(profile, goal, last_memory, history)
        history[-1] = (client_turn, counselor_turn)
        print(f"[Counselor] {counselor_turn}")

    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", type=int, default=3, help="Number of counseling sessions to simulate")
    parser.add_argument("--turns", type=int, default=6, help="Turns (client+counselor exchanges) per session")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Client profile text (defaults to a real training example)")
    parser.add_argument("--attitude", default=None, help="Optional client attitude instruction (passed to SimulatedClient)")
    parser.add_argument("--save", default=None, help="Optional path to save the full transcript as JSON")
    args = parser.parse_args()

    profile = args.profile
    memory = ""
    goal = ""
    all_sessions = []

    for session_num in range(1, args.sessions + 1):
        print(f"\n>>> Task 2: Goal Planning for session {session_num}...")
        goal = run_task2_goal_planning(profile, memory)
        print(f"Goal: {goal}")

        client = SimulatedClient(
            intake_form=profile,
            attitude=args.attitude,
            session_num=session_num,
        )
        history = run_session(session_num, profile, goal, memory, client, args.turns)

        print(f"\n>>> Task 1: Memory Extraction for session {session_num}...")
        # Build dialogue text in the Consultant/User format Task 1 was trained on.
        dialogue_lines = []
        for client_turn, counselor_turn in history:
            dialogue_lines.append(f"Consultant: {counselor_turn}\n")
            dialogue_lines.append(f"User: {client_turn}\n")
        dialogue_text = "\n".join(dialogue_lines)
        memory = run_task1_memory_extraction(dialogue_text)
        print(f"Memory:\n{memory}")

        all_sessions.append({
            "session": session_num,
            "goal": goal,
            "history": [{"client": c, "counselor": co} for c, co in history],
            "memory": memory,
        })

    if args.save:
        Path(args.save).write_text(json.dumps(all_sessions, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved full transcript to {args.save}")


if __name__ == "__main__":
    main()
