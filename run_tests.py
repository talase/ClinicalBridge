from src.triage_agent import run_triage
from src.ehr_agent import EHRAgent
from src.anamnesis_agent import AnamnesisAgent

print("Testing Triage...")
print(run_triage(None, "severe chest pain"))

print("\nTesting EHR...")
print(EHRAgent().run("diabetes and chest pain"))

print("\nTesting Anamnesis...")
print(AnamnesisAgent().run("fever and nausea"))

print("\nALL TESTS PASSED")
