import json
import os
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from orchestrator import ClinicalOrchestrator


def main():

    # ----------------------------
    # Load gold-standard scenarios
    # ----------------------------
    with open(
        "evaluation/gold_standard_outputs.json",
        "r",
        encoding="utf-8"
    ) as f:

        scenarios = json.load(f)

    # ----------------------------
    # Only evaluate ONE scenario
    # ----------------------------
    scenario = scenarios[0]

    print("=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)

    print("\nScenario ID:")
    print(scenario["scenario_id"])

    print("\nInput:")
    print(scenario["input"])

    # ----------------------------
    # Create LLM
    # ----------------------------
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    orch = ClinicalOrchestrator(llm)

    # ----------------------------
    # Run pipeline
    # ----------------------------
    result = orch.run_all(
        scenario["input"]
    )

    triage = result["triage"]
    synthesis = result["synthesis"]

    print("\n" + "=" * 70)
    print("EXPECTED vs ACTUAL")
    print("=" * 70)

    print()

    print(f"Expected Priority : {scenario['expected_priority']}")
    print(f"Actual Priority   : {triage['priority']}")

    print()

    print(f"Expected Risk     : {scenario['expected_risk']}")
    print(f"Actual Risk       : {synthesis['overall_risk']}")

    print()

    print("Expected Summary")
    print("----------------")
    print(scenario["expected_summary"])

    print()

    print("Actual Narrative")
    print("----------------")
    print(synthesis["clinical_narrative"])

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
