
from schemas import AnamnesisOutput, SymptomItem

class AnamnesisAgent:

    def run(self, user_input: str) -> AnamnesisOutput:

        symptoms = []

        for s in ["pain", "fever", "nausea", "cough", "dizziness"]:
            if s in user_input.lower():
                symptoms.append(SymptomItem(symptom=s))

        return AnamnesisOutput(
            symptoms=symptoms,
            adherence_issues=[],
            risk_flags=["auto_extracted"],
            summary="Symptoms extracted from input."
        )
