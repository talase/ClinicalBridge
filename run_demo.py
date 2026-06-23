
from orchestrator import ClinicalOrchestrator
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

orch = ClinicalOrchestrator(llm)

cases = [
    "severe chest pain radiating to left arm",
    "headache fever stiff neck",
    "burning urination and mild pain"
]

for c in cases:
    print("\nINPUT:", c)
    result = orch.run_all(c)
    print(result)
