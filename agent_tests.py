
from src.triage_agent import run_triage
from src.ehr_agent import EHRAgent
from src.anamnesis_agent import AnamnesisAgent

def test_triage(llm):
    out = run_triage(llm, "chest pain")
    assert out.priority in ["P1","P2","P3","P4"]

def test_ehr():
    out = EHRAgent().run("chest pain")
    assert len(out.retrieved_context) > 0

def test_anamnesis():
    out = AnamnesisAgent().run("fever nausea")
    assert len(out.symptoms) > 0

def run_all_tests(llm):
    test_triage(llm)
    test_ehr()
    test_anamnesis()
    print("ALL TESTS PASSED")
