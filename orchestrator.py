from src.triage_agent import run_triage
from src.ehr_agent import EHRAgent
from src.anamnesis_agent import AnamnesisAgent


class ClinicalOrchestrator:

    def __init__(self, llm):
        self.llm = llm
        self.ehr = EHRAgent()
        self.anam = AnamnesisAgent()

    # -----------------------------
    # SAFE WRAPPER FUNCTION
    # -----------------------------
    def safe_call(self, func, fallback):
        try:
            return func()
        except Exception as e:
            return fallback(str(e))

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def run_all(self, user_input: str):

        # 1. TRIAGE (critical → must not fail)
        triage = self.safe_call(
            lambda: run_triage(self.llm, user_input),
            lambda err: {
                "priority": "P3",
                "priority_label": "LESS_URGENT",
                "pathway": "primary_care_scheduling",
                "red_flags": [],
                "escalate_human": False,
                "escalation_reason": f"triage fallback: {err}",
                "confidence": 0.5
            }
        )

        # 2. EHR (can fail safely)
        ehr = self.safe_call(
            lambda: self.ehr.run(user_input),
            lambda err: {
                "query": {"refined_query": user_input, "keywords": []},
                "retrieved_context": [{
                    "doc_id": "fallback",
                    "content": "EHR retrieval failed, using fallback context.",
                    "score": 0.0
                }],
                "clinical_reasoning": "Fallback reasoning due to retrieval failure.",
                "final_answer": "Manual review recommended."
            }
        )

        # 3. ANAMNESIS (safe extraction fallback)
        anam = self.safe_call(
            lambda: self.anam.run(user_input),
            lambda err: {
                "symptoms": [{"symptom": "unknown"}],
                "adherence_issues": [],
                "risk_flags": ["fallback"],
                "summary": "Anamnesis fallback triggered."
            }
        )

        return {
            "triage": triage.model_dump() if hasattr(triage, "model_dump") else triage,
            "ehr": ehr.model_dump() if hasattr(ehr, "model_dump") else ehr,
            "anamnesis": anam.model_dump() if hasattr(anam, "model_dump") else anam
        }
