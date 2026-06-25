import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from orchestrator import ClinicalOrchestrator
from report_generator import generate_html_report


def main():

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    orch = ClinicalOrchestrator(llm)

    # test_input = "severe chest pain radiating to left arm and shortness of breath"
    test_input = "mild headache for two days"

    result = orch.run_all(test_input)

    print("\nGenerating report...")

    generate_html_report(result)

    print("Report opened in browser.")


if __name__ == "__main__":
    main()
