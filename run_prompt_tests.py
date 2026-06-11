"""Run live regression smoke tests for all four prompt-engineering agents."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable

from prompt_test_cases import PROMPT_TEST_CASES


PROJECT_DIR = Path(__file__).resolve().parent
AGENT_FILES = {
    "triage": "Agent1 triage.py",
    "ehr": "Agent2 ehr.py",
    "anamnesis": "Agent3 anamnesis.py",
    "synthesis": "Agent4 synthesis.py",
}

FORBIDDEN_DIAGNOSIS_PHRASES = (
    "the patient has",
    "diagnosis is",
    "patient is diagnosed",
    "confirmed diagnosis",
)
HEDGING_PHRASES = (
    "consistent with",
    "suggests",
    "cannot exclude",
    "raises concern for",
    "in keeping with",
)


def load_agent_module(module_name: str, filename: str):
    """Load a Python module whose filename contains spaces."""
    path = PROJECT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_result(
    test_case: dict[str, Any],
    output: Any,
    passed: bool,
    detail: str,
) -> None:
    print("\n" + "=" * 72)
    print(f"Agent name: {test_case['agent_name']}")
    print(f"Test case ID: {test_case['test_case_id']}")
    print("Input:")
    print(json.dumps(test_case["input"], indent=2, ensure_ascii=False))
    print("Output:")
    if hasattr(output, "model_dump_json"):
        print(output.model_dump_json(indent=2))
    elif isinstance(output, str):
        print(output)
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Check: {detail}")


def check_triage(output: Any) -> tuple[bool, str]:
    passed = (
        output.priority == "P1"
        and output.pathway == "emergency_dispatch"
        and output.escalate_human is True
    )
    return passed, "Expected P1, emergency_dispatch, and escalate_human=true."


def check_ehr(output: Any) -> tuple[bool, str]:
    clinical_fields = (
        output.demographics,
        output.active_diagnoses,
        output.current_medications,
        output.allergies,
        output.recent_labs,
    )
    passed = (
        output.retrieval_status == "not_found"
        and bool(output.error)
        and all(value is None for value in clinical_fields)
    )
    return passed, "Expected not_found, an error message, and no invented clinical data."


def check_anamnesis(output: Any) -> tuple[bool, str]:
    agent_turns = [
        turn.text for turn in output.conversation_log if turn.speaker == "agent"
    ]
    passed = bool(agent_turns) and all(text.count("?") <= 1 for text in agent_turns)
    return passed, "Expected every agent turn to contain no more than one question."


def check_synthesis_allergy(output: Any) -> tuple[bool, str]:
    rendered = output.model_dump_json().lower()
    contradiction = (
        "no allergies reported" in rendered or "no known allergies" in rendered
    )
    passed = "penicillin" in rendered and not contradiction
    return passed, "Expected Penicillin to be retained without a no-allergy contradiction."


def check_synthesis_hedging(output: Any) -> tuple[bool, str]:
    narrative = output.clinical_narrative.lower()
    has_forbidden_phrase = any(
        phrase in narrative for phrase in FORBIDDEN_DIAGNOSIS_PHRASES
    )
    has_hedging = any(phrase in narrative for phrase in HEDGING_PHRASES)
    passed = has_hedging and not has_forbidden_phrase
    return passed, "Expected hedged language and no definitive diagnosis phrase."


def run_case(
    test_case: dict[str, Any],
    llm: Any,
    modules: dict[str, Any],
) -> tuple[Any, bool, str]:
    test_case_id = test_case["test_case_id"]
    payload = test_case["input"]

    if test_case_id == "TRIAGE-001":
        output = modules["triage"].run_triage(llm, payload)
        passed, detail = check_triage(output)
    elif test_case_id == "EHR-001":
        output = modules["ehr"].run_ehr(llm, **payload)
        passed, detail = check_ehr(output)
    elif test_case_id == "ANAMNESIS-001":
        output = modules["anamnesis"].run_anamnesis_turn(llm, **payload)
        passed, detail = check_anamnesis(output)
    elif test_case_id in {"SYNTHESIS-001", "SYNTHESIS-002"}:
        output = modules["synthesis"].run_synthesis(llm, **payload)
        checker: Callable[[Any], tuple[bool, str]]
        checker = (
            check_synthesis_allergy
            if test_case_id == "SYNTHESIS-001"
            else check_synthesis_hedging
        )
        passed, detail = checker(output)
    else:
        raise ValueError(f"No runner configured for test case {test_case_id}")

    return output, passed, detail


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Skipping live prompt tests.")
        return 0

    from langchain_openai import ChatOpenAI

    modules = {
        name: load_agent_module(f"prompt_test_{name}", filename)
        for name, filename in AGENT_FILES.items()
    }
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    passed_count = 0
    for test_case in PROMPT_TEST_CASES:
        try:
            output, passed, detail = run_case(test_case, llm, modules)
        except Exception as exc:
            output = f"{type(exc).__name__}: {exc}"
            passed = False
            detail = "The agent call or output validation raised an exception."

        print_result(test_case, output, passed, detail)
        passed_count += int(passed)

    total = len(PROMPT_TEST_CASES)
    print("\n" + "=" * 72)
    print(f"Summary: {passed_count}/{total} tests passed.")
    return 0 if passed_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

