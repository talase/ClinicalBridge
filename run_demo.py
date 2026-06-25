import os
import sys
import json
import webbrowser
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from orchestrator import ClinicalOrchestrator


def generate_html_report(result):

    triage = result.get("triage", {})
    ehr = result.get("ehr", {})
    rpm = result.get("rpm", {})
    anamnesis = result.get("anamnesis", {})
    synthesis = result.get("synthesis", {})

    patient = {}

    if ehr.get("retrieved_context"):
        try:
            patient = json.loads(
                ehr["retrieved_context"][0]["content"]
            )
        except Exception:
            patient = {}

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ClinicalBridge Report</title>

    <style>

        body {{
            font-family: Arial, sans-serif;
            background-color: white;
            margin: 40px;
            color: #222;
        }}

        h1 {{
            text-align: center;
            color: #0f4c81;
        }}

        h2 {{
            background-color: #f2f2f2;
            padding: 10px;
            border-left: 5px solid #0f4c81;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 25px;
        }}

        th {{
            background-color: #0f4c81;
            color: white;
            padding: 10px;
            text-align: left;
        }}

        td {{
            border: 1px solid #dddddd;
            padding: 10px;
        }}

        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        ul {{
            margin-top: 5px;
        }}

    </style>
</head>

<body>

<h1>ClinicalBridge Case Summary</h1>

<h2>Patient Information</h2>

<table>
    <tr>
        <th>Field</th>
        <th>Value</th>
    </tr>

    <tr>
        <td>Patient ID</td>
        <td>{patient.get("patient_id", "N/A")}</td>
    </tr>

    <tr>
        <td>Name</td>
        <td>{patient.get("name", "N/A")}</td>
    </tr>

    <tr>
        <td>Age</td>
        <td>{patient.get("age", "N/A")}</td>
    </tr>

    <tr>
        <td>Gender</td>
        <td>{patient.get("gender", "N/A")}</td>
    </tr>

</table>

<h2>Triage</h2>

<table>

    <tr>
        <th>Field</th>
        <th>Value</th>
    </tr>

    <tr>
        <td>Priority</td>
        <td>{triage.get("priority", "")}</td>
    </tr>

    <tr>
        <td>Priority Label</td>
        <td>{triage.get("priority_label", "")}</td>
    </tr>

    <tr>
        <td>Pathway</td>
        <td>{triage.get("pathway", "")}</td>
    </tr>

    <tr>
        <td>Escalate Human</td>
        <td>{triage.get("escalate_human", "")}</td>
    </tr>

    <tr>
        <td>Confidence</td>
        <td>{triage.get("confidence", "")}</td>
    </tr>

</table>

<h2>EHR Information</h2>

<table>

    <tr>
        <th>Field</th>
        <th>Value</th>
    </tr>

    <tr>
        <td>Diagnoses</td>
        <td>{", ".join(patient.get("diagnoses", [])) or "None"}</td>
    </tr>

    <tr>
        <td>Medications</td>
        <td>{", ".join(patient.get("medications", [])) or "None"}</td>
    </tr>

    <tr>
        <td>Allergies</td>
        <td>{", ".join(patient.get("allergies", [])) or "None"}</td>
    </tr>

</table>

<h2>RPM Alerts</h2>

<table>

    <tr>
        <th>Total Alerts</th>
    </tr>

    <tr>
        <td>{rpm.get("alerts_found", 0)}</td>
    </tr>

</table>

<h2>Anamnesis Summary</h2>

<p>
{anamnesis.get("summary", "No summary available")}
</p>

<h2>Synthesis</h2>

<table>

    <tr>
        <th>Field</th>
        <th>Value</th>
    </tr>

    <tr>
        <td>Case ID</td>
        <td>{synthesis.get("case_id", "")}</td>
    </tr>

    <tr>
        <td>Overall Risk</td>
        <td>{synthesis.get("overall_risk", "")}</td>
    </tr>

    <tr>
        <td>Physician Alert</td>
        <td>{synthesis.get("physician_alert", "")}</td>
    </tr>

    <tr>
        <td>Synthesis Confidence</td>
        <td>{synthesis.get("synthesis_confidence", "")}</td>
    </tr>

</table>

<h2>Clinical Narrative</h2>

<p>
{synthesis.get("clinical_narrative", "")}
</p>

<h2>Differential Diagnoses</h2>

<table>

<tr>
    <th>Rank</th>
    <th>Diagnosis</th>
    <th>ICD10</th>
    <th>Probability</th>
</tr>
"""

    for differential in synthesis.get("differentials", []):
        html += f"""
<tr>
    <td>{differential.get("rank", "")}</td>
    <td>{differential.get("diagnosis", "")}</td>
    <td>{differential.get("icd10", "")}</td>
    <td>{differential.get("probability", "")}</td>
</tr>
"""

    html += """
</table>

<h2>Information Gaps</h2>

<ul>
"""

    for gap in synthesis.get("information_gaps", []):
        html += f"<li>{gap}</li>"

    html += """
</ul>

<h2>Recommended Next Steps</h2>

<ul>
"""

    for step in synthesis.get("recommended_next_steps", []):
        html += f"<li>{step}</li>"

    html += """
</ul>

</body>
</html>
"""

    report_path = Path("clinicalbridge_report.html")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(html)

    webbrowser.open(report_path.resolve().as_uri())


def main():

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    orch = ClinicalOrchestrator(llm)

    # Example 1
    test_input = "severe chest pain radiating to left arm and shortness of breath"

    # Example 2
    #test_input = "mild headache for two days"

    #Example 3
    #test_input = "Severe chest pain radiating to the left arm, shortness of breath, sweating, and nausea."
    
    #Example 4
    #test_input = "Sudden facial drooping, slurred speech, and weakness in the right arm that started 20 minutes ago."
    
    #Example 5
    #test_input = "High fever of 39.2°C, productive cough, chest discomfort, and fatigue for three days."
    
    #Example 6
    #test_input = "Burning sensation while urinating and increased urinary frequency for two days."

    #Example 7
    #test_input = "Mild headache for two days with no fever, vision changes, or other symptoms."

    #Example 8
    #test_input = "I have been thinking about harming myself and I do not feel safe being alone."
    

    result = orch.run_all(test_input)

    print("\nGenerating HTML report...")

    generate_html_report(result)

    print("Report opened in browser.")


if __name__ == "__main__":
    main()
