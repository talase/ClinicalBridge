
from src.triage_agent import run_triage
from src.ehr_agent import EHRAgent
from src.anamnesis_agent import AnamnesisAgent

class ClinicalOrchestrator:

    def __init__(self, llm):
        self.llm = llm
        self.ehr = EHRAgent()
        self.anam = AnamnesisAgent()

    def run_all(self, user_input: str):

        triage = run_triage(self.llm, user_input)
        ehr = self.ehr.run(user_input)
        anam = self.anam.run(user_input)

        return {
            "triage": triage.model_dump(),
            "ehr": ehr.model_dump(),
            "anamnesis": anam.model_dump()
        }
