import json
import webbrowser
from pathlib import Path

def generate_html_report(result):

```
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
```

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

h2 {{
    margin-top: 0;
}}

.dashboard {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}}

.card {{
    border: 1px solid #d9d9d9;
    border-radius: 10px;
    padding: 15px;
    background: white;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.08);
}}

.full-width {{
    grid-column: 1 / span 2;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th {{
    background-color: #0f4c81;
    color: white;
    padding: 8px;
    text-align: left;
}}

td {{
    border: 1px solid #ddd;
    padding: 8px;
}}

tr:nth-child(even) {{
    background-color: #f8f8f8;
}}

.badge {{
    color: white;
    padding: 8px 15px;
    border-radius: 20px;
    font-weight: bold;
}}

ul {{
    margin: 0;
}}

</style>

</head>

<body>

<h1>ClinicalBridge Case Dashboard</h1>

<div class="dashboard">

<div class="card">
<h2>Patient Information</h2>

<table>
<tr><th>Field</th><th>Value</th></tr>
<tr><td>Patient ID</td><td>{patient.get("patient_id", "N/A")}</td></tr>
<tr><td>Name</td><td>{patient.get("name", "N/A")}</td></tr>
<tr><td>Age</td><td>{patient.get("age", "N/A")}</td></tr>
<tr><td>Gender</td><td>{patient.get("gender", "N/A")}</td></tr>
</table>

</div>

<div class="card">
<h2>Triage</h2>

<p>
<b>Priority:</b>
<span class="badge" style="background:{priority_color};">
{priority}
</span>
</p>

<table>
<tr><th>Field</th><th>Value</th></tr>
<tr><td>Priority Label</td><td>{triage.get("priority_label", "")}</td></tr>
<tr><td>Pathway</td><td>{triage.get("pathway", "")}</td></tr>
<tr><td>Escalate Human</td><td>{triage.get("escalate_human", "")}</td></tr>
<tr><td>Confidence</td><td>{triage.get("confidence", "")}</td></tr>
</table>

</div>

<div class="card">
<h2>EHR Information</h2>

<table>
<tr><th>Field</th><th>Value</th></tr>
<tr><td>Diagnoses</td><td>{", ".join(patient.get("diagnoses", [])) or "None"}</td></tr>
<tr><td>Medications</td><td>{", ".join(patient.get("medications", [])) or "None"}</td></tr>
<tr><td>Allergies</td><td>{", ".join(patient.get("allergies", [])) or "None"}</td></tr>
</table>

</div>

<div class="card">
<h2>RPM Alerts</h2>

<table>
<tr><th>Total Alerts</th></tr>
<tr><td>{rpm.get("alerts_found", 0)}</td></tr>
</table>

</div>

<div class="card full-width">

<h2>Synthesis</h2>

<p>
<b>Overall Risk:</b>
<span class="badge" style="background:{risk_color};">
{risk}
</span>
</p>

<table>
<tr><th>Field</th><th>Value</th></tr>
<tr><td>Case ID</td><td>{synthesis.get("case_id", "")}</td></tr>
<tr><td>Physician Alert</td><td>{synthesis.get("physician_alert", "")}</td></tr>
<tr><td>Confidence</td><td>{synthesis.get("synthesis_confidence", "")}</td></tr>
</table>

</div>

<div class="card full-width">

<h2>Clinical Narrative</h2>

<p>
{synthesis.get("clinical_narrative", "")}
</p>

</div>

<div class="card full-width">

<h2>Differential Diagnoses</h2>

<table>

<tr>
<th>Rank</th>
<th>Diagnosis</th>
<th>ICD10</th>
<th>Probability</th>
</tr>
"""

```
for differential in synthesis.get("differentials", []):
    html += f"""
```

<tr>
<td>{differential.get("rank", "")}</td>
<td>{differential.get("diagnosis", "")}</td>
<td>{differential.get("icd10", "")}</td>
<td>{differential.get("probability", "")}</td>
</tr>
"""

```
html += """
```

</table>

</div>

<div class="card">

<h2>Information Gaps</h2>

<ul>
"""

```
for gap in synthesis.get("information_gaps", []):
    html += f"<li>{gap}</li>"

html += """
```

</ul>

</div>

<div class="card">

<h2>Recommended Next Steps</h2>

<ul>
"""

```
for step in synthesis.get("recommended_next_steps", []):
    html += f"<li>{step}</li>"

html += """
```

</ul>

</div>

</div>

</body>
</html>
"""

```
report_file = Path("clinicalbridge_dashboard.html")

with open(report_file, "w", encoding="utf-8") as f:
    f.write(html)

webbrowser.open(report_file.resolve().as_uri())
```
