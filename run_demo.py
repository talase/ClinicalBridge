import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from orchestrator import ClinicalOrchestrator

def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    orch = ClinicalOrchestrator(llm)

    #test_input = "severe chest pain radiating to left arm and shortness of breath"
    test_input = "test_input = "mild headache for two days""

    result = orch.run_all(test_input)

    print("\n===== FINAL OUTPUT =====\n")
    print(result)

if __name__ == "__main__":
    main()
