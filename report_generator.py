import json
import webbrowser
from pathlib import Path


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

    priority = triage.get("priority", "P4")

    priority_colors = {
        "P1": "#dc3545",
        "P2": "#fd7e14",
        "P3": "#ffc107",
        "P4": "#28a745"
    }

    priority_color = priority_colors.get(priority, "#6c757d")

    risk = synthesis.get("overall_risk", "LOW")

    risk_colors = {
        "LOW": "#28a745",
        "MODERATE": "#ffc107",
        "HIGH": "#fd7e14",
        "CRITICAL": "#dc3545"
    }

    risk_color = risk_colors.get(risk, "#6c757d")

    html = f"""
<!DOCTYPE html>
<html>
<head>
<title>ClinicalBridge Dashboard</title>

<style>

body {{
    font-family: Arial, sans-serif;
    background-color: white;
    margin: 30px;
    color: #222;
}}

h1 {{
    text-align: center;
    color: #0f4c81;
}}

.section {{
    border: 1px solid #ddd;
    margin-bottom: 20px;
    padding: 15px;
    border-radius: 8px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th {{
    background-color: #f5f5f5;
    text-align: left;
    width: 30%;
}}

th, td {{
    border: 1px solid #ddd;
    padding: 10px;
}}

.badge {{
    color: white;
    padding: 6px 12px;
    border-radius: 5px;
    font-weight: bold;
}}

ul {{
    margin-top: 5px;
}}

</style>
</head>

<body>

<h1>ClinicalBridge Dashboard</h1>

<div class="section">
<h2>Triage</h2>

<table>
<tr>
<th>Priority</th>
<td>
<span class="badge" style="background:{priority_color}">
{triage.get("priority", "N/A")}
</span>
</td>
</tr>

<tr>
<th>Priority Label</th>
<td>{triage.get("priority_label", "N/A")}</td>
</tr>

<tr>
<th>Pathway</th>
<td>{triage.get("pathway", "N/A")}</td>
</tr>

<tr>
<th>Escalate Human</th>
<td>{triage.get("escalate_human", False)}</td>
</tr>

<tr>
<th>Confidence</th>
<td>{triage.get("confidence", "N/A")}</td>
</tr>

<tr>
<th>Red Flags</th>
<td>{", ".join(triage.get("red_flags", [])) or "None"}</td>
</tr>
</table>
</div>

<div class="section">
<h2>Patient Information</h2>

<table>
<tr>
<th>Patient ID</th>
<td>{patient.get("patient_id", "N/A")}</td>
</tr>

<tr>
<th>Name</th>
<td>{patient.get("name", "N/A")}</td>
</tr>

<tr>
<th>Age</th>
<td>{patient.get("age", "N/A")}</td>
</tr>

<tr>
<th>Gender</th>
<td>{patient.get("gender", "N/A")}</td>
</tr>

<tr>
<th>Diagnoses</th>
<td>{", ".join(patient.get("diagnoses", [])) or "None"}</td>
</tr>

<tr>
<th>Medications</th>
<td>{", ".join(patient.get("medications", [])) or "None"}</td>
</tr>

<tr>
<th>Allergies</th>
<td>{", ".join(patient.get("allergies", [])) or "None"}</td>
</tr>
</table>
</div>

<div class="section">
<h2>Synthesis</h2>

<table>
<tr>
<th>Case ID</th>
<td>{synthesis.get("case_id", "N/A")}</td>
</tr>

<tr>
<th>Overall Risk</th>
<td>
<span class="badge" style="background:{risk_color}">
{risk}
</span>
</td>
</tr>

<tr>
<th>Physician Alert</th>
<td>{synthesis.get("physician_alert", False)}</td>
</tr>

<tr>
<th>Confidence</th>
<td>{synthesis.get("synthesis_confidence", "N/A")}</td>
</tr>
</table>

<h3>Clinical Narrative</h3>
<p>{synthesis.get("clinical_narrative", "N/A")}</p>

<h3>Differential Diagnoses</h3>
<ul>
"""

    for diff in synthesis.get("differentials", []):
        html += f"""
<li>
<b>{diff.get("diagnosis", "Unknown")}</b>
({diff.get("probability", "N/A")})
</li>
"""

    html += """
</ul>

<h3>Information Gaps</h3>
<ul>
"""

    for gap in synthesis.get("information_gaps", []):
        html += f"<li>{gap}</li>"

    html += """
</ul>

<h3>Recommended Next Steps</h3>
<ul>
"""

    for step in synthesis.get("recommended_next_steps", []):
        html += f"<li>{step}</li>"

    html += """
</ul>
</div>

</body>
</html>
"""

    report_path = Path("ClinicalBridge_Report.html")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open(report_path.resolve().as_uri())

    print(f"\nHTML report generated: {report_path}")
