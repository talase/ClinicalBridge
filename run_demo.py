import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from orchestrator import ClinicalOrchestrator


def print_report(result):
    triage = result.get("triage", {})
    ehr = result.get("ehr", {})
    rpm = result.get("rpm", {})
    anamnesis = result.get("anamnesis", {})
    synthesis = result.get("synthesis", {})

    print("\n" + "=" * 70)
    print("CLINICALBRIDGE CASE SUMMARY")
    print("=" * 70)

    # --------------------------------------------------
    # TRIAGE
    # --------------------------------------------------
    print("\nTRIAGE")
    print("-" * 70)
    print(f"Priority          : {triage.get('priority', 'N/A')}")
    print(f"Priority Label    : {triage.get('priority_label', 'N/A')}")
    print(f"Pathway           : {triage.get('pathway', 'N/A')}")
    print(f"Escalate Human    : {triage.get('escalate_human', 'N/A')}")
    print(f"Confidence        : {triage.get('confidence', 'N/A')}")

    red_flags = triage.get("red_flags", [])
    print(f"Red Flags         : {', '.join(red_flags) if red_flags else 'None'}")

    # --------------------------------------------------
    # EHR
    # --------------------------------------------------
    print("\nEHR")
    print("-" * 70)

    retrieved = ehr.get("retrieved_context", [])

    if retrieved:
        try:
            import json

            patient_data = json.loads(retrieved[0]["content"])

            print(f"Patient ID        : {patient_data.get('patient_id', 'N/A')}")
            print(f"Name              : {patient_data.get('name', 'N/A')}")
            print(f"Age               : {patient_data.get('age', 'N/A')}")
            print(f"Gender            : {patient_data.get('gender', 'N/A')}")

            diagnoses = patient_data.get("diagnoses", [])
            medications = patient_data.get("medications", [])
            allergies = patient_data.get("allergies", [])

            print(
                f"Diagnoses         : "
                f"{', '.join(diagnoses) if diagnoses else 'None'}"
            )

            print(
                f"Medications       : "
                f"{', '.join(medications) if medications else 'None'}"
            )

            print(
                f"Allergies         : "
                f"{', '.join(allergies) if allergies else 'None'}"
            )

        except Exception:
            print("Unable to parse EHR patient data.")

    # --------------------------------------------------
    # RPM
    # --------------------------------------------------
    print("\nRPM ALERTS")
    print("-" * 70)

    alerts = rpm.get("alerts", [])

    if alerts:
        for alert in alerts:
            print(
                f"- {alert.get('alert_type', 'Unknown')} "
                f"| Reading: {alert.get('reading', 'N/A')} "
                f"| Urgency: {alert.get('urgency', 'N/A')}"
            )
    else:
        print("No RPM alerts.")

    # --------------------------------------------------
    # ANAMNESIS
    # --------------------------------------------------
    print("\nANAMNESIS")
    print("-" * 70)

    symptoms = anamnesis.get("symptoms", [])

    if symptoms:
        for s in symptoms:
            print(f"- {s.get('symptom', 'Unknown')}")
    else:
        print("No symptoms extracted.")

    # --------------------------------------------------
    # SYNTHESIS
    # --------------------------------------------------
    print("\nSYNTHESIS")
    print("-" * 70)

    if isinstance(synthesis, dict):

        print(f"Case ID           : {synthesis.get('case_id', 'N/A')}")
        print(f"Overall Risk      : {synthesis.get('overall_risk', 'N/A')}")
        print(f"Physician Alert   : {synthesis.get('physician_alert', 'N/A')}")
        print(
            f"Confidence        : "
            f"{synthesis.get('synthesis_confidence', 'N/A')}"
        )

        print("\nClinical Narrative")
        print("-" * 70)
        print(synthesis.get("clinical_narrative", "N/A"))

        print("\nDifferential Diagnoses")
        print("-" * 70)

        for d in synthesis.get("differentials", []):
            print(
                f"{d.get('rank', '?')}. "
                f"{d.get('diagnosis', 'Unknown')} "
                f"({d.get('probability', 'N/A')})"
            )

        print("\nInformation Gaps")
        print("-" * 70)

        for gap in synthesis.get("information_gaps", []):
            print(f"- {gap}")

        print("\nRecommended Next Steps")
        print("-" * 70)

        for step in synthesis.get("recommended_next_steps", []):
            print(f"- {step}")

    else:
        print("Synthesis output unavailable.")

    print("\n" + "=" * 70)


def main():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    orch = ClinicalOrchestrator(llm)

    # test_input = "severe chest pain radiating to left arm and shortness of breath"
    test_input = "mild headache for two days"

    result = orch.run_all(test_input)

    print_report(result)


if __name__ == "__main__":
    main()
