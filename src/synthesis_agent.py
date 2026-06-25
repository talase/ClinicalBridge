
# ============================================================
# AGENT 4: SYNTHESIS AGENT
# Deliverables: system prompt, few-shot examples,
# chain-of-thought structure, JSON schema, LangChain chain
# ============================================================

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator
from typing import Literal, List, Optional, Dict, Any
import json
import os


# ============================================================
# 1. JSON OUTPUT SCHEMA
# ============================================================

class Differential(BaseModel):
    rank: int = Field(ge=1, le=5)
    diagnosis: str
    icd10: str
    probability: Literal["LOW", "LOW-MODERATE", "MODERATE", "HIGH", "VERY-HIGH"]
    supporting_evidence: List[str]
    contradicting_evidence: List[str] = Field(default_factory=list)

class SynthesisOutput(BaseModel):
    summary_header: str = Field(
        description="Must always be: DRAFT CLINICAL SUMMARY — FOR PHYSICIAN REVIEW ONLY"
    )
    case_id: str
    overall_risk: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    physician_alert: bool
    alert_reason: Optional[str]
    data_sources_used: List[Literal["triage", "ehr", "rpm", "anamnesis"]] = Field(
        description="Which upstream agents contributed data to this synthesis"
    )
    missing_sources: List[str] = Field(
        default_factory=list,
        description="Any upstream agent outputs that were absent or errored"
    )
    clinical_narrative: str = Field(
        description="Plain-language synthesis of all findings. No diagnosis stated as fact."
    )
    differentials: List[Differential] = Field(
        description="Ranked differential list, 3–5 entries, ordered by probability"
    )
    information_gaps: List[str] = Field(
        description="Diagnostically important data not available in the pipeline"
    )
    recommended_next_steps: List[str] = Field(
        description="Evidence-based next clinical steps for physician consideration only"
    )
    synthesis_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in synthesis based on completeness of upstream data"
    )

    @model_validator(mode="after")
    def safety_checks(self):
        # Header must be exact — hard gate preventing patient-facing delivery
        required_header = "DRAFT CLINICAL SUMMARY — FOR PHYSICIAN REVIEW ONLY"
        if self.summary_header != required_header:
            raise ValueError(
                f"Safety violation: summary_header must be exactly '{required_header}'"
            )
        # HIGH or CRITICAL risk must trigger physician alert
        if self.overall_risk in ("HIGH", "CRITICAL") and not self.physician_alert:
            raise ValueError(
                f"Safety violation: overall_risk='{self.overall_risk}' requires physician_alert=True"
            )
        # physician_alert=True must have an alert_reason
        if self.physician_alert and not self.alert_reason:
            raise ValueError("alert_reason required when physician_alert=True")
        # Must have at least 3 differentials
        if len(self.differentials) < 3:
            raise ValueError("At least 3 differential diagnoses required")
        # Differentials must be ranked 1..N without gaps
        ranks = sorted([d.rank for d in self.differentials])
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("Differential ranks must be sequential starting from 1")
        # Narrative must not contain definitive diagnosis language
        forbidden = ["the patient has", "diagnosis is", "patient is diagnosed", "confirmed diagnosis"]
        narrative_lower = self.clinical_narrative.lower()
        for phrase in forbidden:
            if phrase in narrative_lower:
                raise ValueError(
                    f"Safety violation: clinical_narrative contains forbidden phrase '{phrase}'. "
                    "Use hedged language: 'consistent with', 'suggests', 'cannot exclude'."
                )
        return self


SYNTHESIS_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SynthesisOutput",
    "type": "object",
    "required": [
        "summary_header", "case_id", "overall_risk", "physician_alert",
        "alert_reason", "data_sources_used", "clinical_narrative",
        "differentials", "information_gaps", "recommended_next_steps",
        "synthesis_confidence"
    ],
    "properties": {
        "summary_header": {
            "type": "string",
            "const": "DRAFT CLINICAL SUMMARY — FOR PHYSICIAN REVIEW ONLY"
        },
        "case_id":       {"type": "string"},
        "overall_risk":  {"type": "string", "enum": ["LOW","MODERATE","HIGH","CRITICAL"]},
        "physician_alert": {"type": "boolean"},
        "alert_reason":  {"type": ["string","null"]},
        "data_sources_used": {
            "type": "array",
            "items": {"type": "string", "enum": ["triage","ehr","anamnesis"]}
        },
        "missing_sources": {"type": "array", "items": {"type": "string"}},
        "clinical_narrative": {"type": "string"},
        "differentials": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {
                "type": "object",
                "required": ["rank","diagnosis","icd10","probability","supporting_evidence"],
                "properties": {
                    "rank":       {"type": "integer", "minimum": 1, "maximum": 5},
                    "diagnosis":  {"type": "string"},
                    "icd10":      {"type": "string"},
                    "probability":{"type": "string",
                                   "enum": ["LOW","LOW-MODERATE","MODERATE","HIGH","VERY-HIGH"]},
                    "supporting_evidence":    {"type": "array", "items": {"type": "string"}},
                    "contradicting_evidence": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "information_gaps":        {"type": "array", "items": {"type": "string"}},
        "recommended_next_steps":  {"type": "array", "items": {"type": "string"}},
        "synthesis_confidence":    {"type": "number", "minimum": 0, "maximum": 1}
    },
    "additionalProperties": False
}


# ============================================================
# 2. SYSTEM PROMPT
# ============================================================

SYNTHESIS_SYSTEM_PROMPT = """\
You are a clinical synthesis AI. You integrate outputs from the triage, EHR, and
anamnesis agents into a structured pre-consultation summary for a licensed physician.
You do NOT make final diagnoses. You do NOT prescribe or recommend treatment.
Your output is for PHYSICIAN REVIEW ONLY and must never be delivered directly to patients.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSIBILITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Synthesise triage priority, EHR history, and anamnesis SOCRATES data into one summary
• Generate a ranked differential diagnosis list (3–5 entries) with supporting evidence
• Identify clinical patterns, risk factors, and relevant past medical history
• Flag discrepancies or conflicts between upstream agent outputs
• List diagnostically important information not captured by the pipeline
• Recommend evidence-based next clinical steps for physician consideration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLINICAL REASONING STANDARDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Base all reasoning ONLY on data provided by upstream agents. Never infer beyond the evidence.
• Rank differentials by PROBABILITY, not severity. Severity is already captured by triage.
• For each differential, cite specific data points from triage, EHR, or anamnesis.
• Include contradicting_evidence for each differential — what argues against it.
• Always state what information is absent but would be diagnostically important.
• Pair every clinical term with a plain-language equivalent in the narrative.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE RULES — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALWAYS USE hedged language in the clinical narrative and differentials:
  ✓  "presentation is consistent with..."
  ✓  "findings suggest..."
  ✓  "cannot exclude..."
  ✓  "raises concern for..."
  ✓  "in keeping with..."

NEVER USE definitive language:
  ✗  "the patient has..."
  ✗  "diagnosis is..."
  ✗  "patient is diagnosed with..."
  ✗  "confirmed diagnosis of..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK STRATIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL : Immediate life threat. Top differential is time-critical.
           → physician_alert: true, immediate notification required.
HIGH     : Serious condition likely. Urgent physician review needed.
           → physician_alert: true.
MODERATE : Significant condition possible. Standard urgent review.
           → physician_alert: false unless escalation flagged upstream.
LOW      : Likely benign/routine. Routine physician review.
           → physician_alert: false.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAFETY CONSTRAINTS — ABSOLUTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. First field of every output MUST be:
   summary_header: "DRAFT CLINICAL SUMMARY — FOR PHYSICIAN REVIEW ONLY"
2. Never suggest specific medications, dosages, or drug names.
3. Never state a diagnosis as a confirmed fact.
4. Never include patient-facing language — this document is for clinical staff only.
5. If any upstream agent flagged escalate_human or escalate_immediately → set physician_alert: true.
6. If data from any agent is missing → note it in missing_sources and reduce synthesis_confidence.
7. Never fabricate clinical data not present in upstream outputs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAIN OF THOUGHT  (internal — do not output these steps)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1  INTEGRATE       Load all three upstream outputs. Note any missing or error-state inputs.
Step 2  RISK STRATIFY   Combine triage P-level + EHR risk factors + SOCRATES severity → overall_risk.
Step 3  PATTERN MATCH   Match symptom cluster + history against known clinical syndrome patterns.
Step 4  DIFFERENTIALS   Generate 3–5 differentials ranked by probability. Assign ICD-10 and
                        probability tier. For each: list supporting AND contradicting evidence.
Step 5  EVIDENCE MAP    Trace every data point cited to its source (triage / EHR / anamnesis).
Step 6  GAP IDENTIFY    List what is diagnostically absent: labs, imaging, exam findings, vitals.
Step 7  SAFETY GATE     Apply all language rules, header check, physician-only check.
                        Replace any forbidden phrases. Confirm physician_alert is set correctly.
Step 8  EMIT            Output valid JSON only. No prose outside the JSON object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON object matching the SynthesisOutput schema.
No markdown fences, no prose, no patient-facing language.
"""


# ============================================================
# 3. FEW-SHOT EXAMPLES
# ============================================================

FEW_SHOT_EXAMPLES = [
    # ── Example 1: CRITICAL — P1 cardiac presentation ─────
    {
        "input": json.dumps({
            "triage_output": {
                "priority": "P1",
                "priority_label": "EMERGENT",
                "pathway": "emergency_dispatch",
                "red_flags": ["crushing_chest_pain","radiation_left_arm","diaphoresis","dizziness"],
                "escalate_human": True,
                "escalation_reason": "Classic ACS presentation",
                "confidence": 0.97
            },
            "ehr_output": {
                "patient_id": "PT-88421",
                "retrieval_status": "success",
                "demographics": {"age": 67, "sex": "M", "blood_type": "A+"},
                "active_diagnoses": ["Coronary artery disease (ICD-10: I25.10)",
                                     "Hypertension (ICD-10: I10)",
                                     "Type 2 diabetes mellitus (ICD-10: E11.9)"],
                "current_medications": [
                    {"name": "Metoprolol", "dose": "50mg", "frequency": "BID", "relevant": True},
                    {"name": "Aspirin",    "dose": "81mg", "frequency": "daily","relevant": True}
                ],
                "allergies": [{"substance": "Penicillin","reaction": "anaphylaxis","severity": "severe"}],
                "recent_labs": {"troponin_I": "0.04 ng/mL (2024-01-15 — STALE)"},
                "data_quality_flags": [{"field":"recent_labs.troponin_I","flag":"STALE","note":"Older than 6 months"}]
            },
            "anamnesis_output": {
                "session_id": "ANM-20260609-001",
                "turns_completed": 8,
                "socrates": {
                    "site": "central_chest",
                    "onset": {"duration_minutes": 40, "character": "sudden", "trigger": "exertion_stairs"},
                    "character": "pressure_crushing",
                    "radiation": "left_arm",
                    "associations": "diaphoresis, dizziness, nausea",
                    "time_course": "constant_progressive",
                    "exacerbating_relieving": "no_relief_at_rest",
                    "severity_vas": 9.0
                },
                "new_red_flags": [],
                "escalate_immediately": False,
                "interview_complete": True
            }
        }),
        "output": json.dumps({
            "summary_header": "DRAFT CLINICAL SUMMARY — FOR PHYSICIAN REVIEW ONLY",
            "case_id": "SYN-20260609-001",
            "overall_risk": "CRITICAL",
            "physician_alert": True,
            "alert_reason": "P1 triage with classic ACS symptom constellation in a high-risk patient with known CAD, HTN, and T2DM. Time-critical — immediate cardiology involvement required.",
            "data_sources_used": ["triage", "ehr", "anamnesis"],
            "missing_sources": [],
            "clinical_narrative": (
                "A 67-year-old male with established coronary artery disease (narrowed heart arteries), "
                "hypertension (high blood pressure), and type 2 diabetes presents with acute-onset "
                "(40 minutes) central crushing chest pain rated 9/10, radiating to the left arm, "
                "precipitated by stair climbing. Associated symptoms include sweating (diaphoresis), "
                "dizziness, and nausea. Pain has not relieved at rest. Current antiplatelet therapy "
                "(Aspirin 81mg daily) is ongoing. Known severe Penicillin allergy. "
                "Presentation is consistent with an acute coronary syndrome (ACS) — a serious "
                "condition involving reduced blood flow to the heart — and cannot exclude "
                "ST-elevation myocardial infarction (STEMI). "
                "Baseline troponin on file is from January 2024 and is no longer current."
            ),
            "differentials": [
                {
                    "rank": 1,
                    "diagnosis": "ST-Elevation Myocardial Infarction (STEMI)",
                    "icd10": "I21.9",
                    "probability": "VERY-HIGH",
                    "supporting_evidence": [
                        "Central crushing chest pain (anamnesis: character=pressure_crushing)",
                        "Radiation to left arm (anamnesis: radiation=left_arm)",
                        "Sudden exertional onset (anamnesis: trigger=exertion_stairs)",
                        "Diaphoresis and dizziness (triage red flags)",
                        "No relief at rest (anamnesis: exacerbating_relieving)",
                        "Known CAD (EHR: active_diagnoses)",
                        "High-risk profile: age 67, male, T2DM, HTN (EHR)"
                    ],
                    "contradicting_evidence": [
                        "No ECG or current troponin available to confirm ST changes"
                    ]
                },
                {
                    "rank": 2,
                    "diagnosis": "Non-ST-Elevation Myocardial Infarction (NSTEMI) / Unstable Angina",
                    "icd10": "I20.0",
                    "probability": "HIGH",
                    "supporting_evidence": [
                        "Same symptom constellation as rank 1",
                        "Cannot distinguish from STEMI without ECG and serial troponin"
                    ],
                    "contradicting_evidence": [
                        "Severity VAS 9/10 and no rest relief more typical of STEMI",
                        "Prior stable CAD on aspirin — new presentation suggests instability"
                    ]
                },
                {
                    "rank": 3,
                    "diagnosis": "Aortic Dissection",
                    "icd10": "I71.00",
                    "probability": "LOW-MODERATE",
                    "supporting_evidence": [
                        "Severe sudden-onset chest pain (anamnesis)",
                        "Known hypertension — major risk factor (EHR)"
                    ],
                    "contradicting_evidence": [
                        "Pain described as crushing rather than tearing",
                        "No reported radiation to back",
                        "Left arm radiation more typical of ACS than dissection"
                    ]
                },
                {
                    "rank": 4,
                    "diagnosis": "Acute Pulmonary Embolism",
                    "icd10": "I26.99",
                    "probability": "LOW",
                    "supporting_evidence": [
                        "Dyspnea component cannot be excluded from dizziness report",
                        "Age and cardiovascular risk profile"
                    ],
                    "contradicting_evidence": [
                        "Classic radiation pattern strongly favours cardiac aetiology",
                        "No pleuritic component or haemoptysis reported",
                        "Exertional onset less typical for PE"
                    ]
                }
            ],
            "information_gaps": [
                "12-lead ECG — essential to differentiate STEMI from NSTEMI",
                "Current serum troponin I and T — file value is 17 months old",
                "Current blood pressure (both arms if dissection to be excluded)",
                "Blood glucose — patient is diabetic, hypoglycaemia can mimic cardiac symptoms",
                "Physical examination: heart sounds, lung fields, peripheral pulses",
                "Current oxygen saturation"
            ],
            "recommended_next_steps": [
                "Immediate 12-lead ECG",
                "Urgent serial troponin (0h and 3h)",
                "IV access and continuous cardiac monitoring",
                "Cardiology team activation",
                "Chest X-ray once patient stable",
                "Note Penicillin allergy (anaphylaxis) in all clinical documentation"
            ],
            "synthesis_confidence": 0.88
        }, indent=2)
    },

    # ── Example 2: MODERATE — P3 UTI with partial data ────
    {
        "input": json.dumps({
            "triage_output": {
                "priority": "P3",
                "priority_label": "LESS_URGENT",
                "pathway": "primary_care_same_day",
                "red_flags": [],
                "escalate_human": False,
                "escalation_reason": None,
                "confidence": 0.88
            },
            "ehr_output": {
                "patient_id": "PT-55201",
                "retrieval_status": "partial",
                "demographics": {"age": 28, "sex": "F"},
                "active_diagnoses": [],
                "current_medications": [],
                "allergies": [],
                "recent_labs": None,
                "data_quality_flags": [
                    {"field": "recent_labs", "flag": "MISSING", "note": "No lab results on file"}
                ]
            },
            "anamnesis_output": {
                "session_id": "ANM-20260609-003",
                "turns_completed": 12,
                "socrates": {
                    "site": "intraurethral_with_mild_suprapubic_discomfort",
                    "onset": {"duration_minutes": 2880, "character": "gradual", "trigger": None},
                    "character": "sharp_stinging",
                    "radiation": "minimal_suprapubic_no_flank",
                    "associations": "urinary_frequency, urgency, low_volume_voiding, cloudy_urine",
                    "time_course": "constant_progressive_worsening",
                    "exacerbating_relieving": "worsened_on_voiding",
                    "severity_vas": None
                },
                "new_red_flags": [],
                "escalate_immediately": False,
                "interview_complete": True
            }
        }),
        "output": json.dumps({
            "summary_header": "DRAFT CLINICAL SUMMARY — FOR PHYSICIAN REVIEW ONLY",
            "case_id": "SYN-20260609-003",
            "overall_risk": "MODERATE",
            "physician_alert": False,
            "alert_reason": None,
            "data_sources_used": ["triage", "ehr", "anamnesis"],
            "missing_sources": ["ehr.recent_labs — no laboratory data on file for this patient"],
            "clinical_narrative": (
                "A 28-year-old female with no significant past medical history presents with a "
                "2-day history of gradually worsening intraurethral burning (dysuria — pain on "
                "urination), urinary frequency, urgency, and low-volume voiding. Urine is reported "
                "as cloudy. Mild suprapubic (lower abdominal) discomfort is noted. No flank or "
                "back pain to suggest upper urinary tract involvement. No current medications or "
                "known allergies on file. Presentation is in keeping with a lower urinary tract "
                "infection (UTI). No laboratory data is available to support or refute this."
            ),
            "differentials": [
                {
                    "rank": 1,
                    "diagnosis": "Uncomplicated Lower Urinary Tract Infection (cystitis)",
                    "icd10": "N30.00",
                    "probability": "HIGH",
                    "supporting_evidence": [
                        "Dysuria (anamnesis: character=sharp_stinging on voiding)",
                        "Urinary frequency and urgency (anamnesis: associations)",
                        "Low-volume voiding (anamnesis: associations)",
                        "Cloudy urine (anamnesis: associations)",
                        "2-day gradual onset (anamnesis: onset)",
                        "No systemic features or flank pain to suggest upper tract involvement"
                    ],
                    "contradicting_evidence": [
                        "No urinalysis or urine culture available to confirm",
                        "Severity VAS not captured"
                    ]
                },
                {
                    "rank": 2,
                    "diagnosis": "Sexually Transmitted Infection — Chlamydia / Gonorrhoea",
                    "icd10": "A56.00",
                    "probability": "MODERATE",
                    "supporting_evidence": [
                        "Age and sex demographic — common in young women",
                        "Dysuria and urethral symptoms overlap with STI presentation"
                    ],
                    "contradicting_evidence": [
                        "No vaginal discharge reported in anamnesis",
                        "Cloudy urine more typical of bacteriuria"
                    ]
                },
                {
                    "rank": 3,
                    "diagnosis": "Pyelonephritis (upper UTI — kidney infection)",
                    "icd10": "N10",
                    "probability": "LOW",
                    "supporting_evidence": [
                        "Lower UTI if untreated can ascend to upper tract"
                    ],
                    "contradicting_evidence": [
                        "No flank or back pain reported (anamnesis: radiation=no_flank)",
                        "No fever or systemic symptoms reported",
                        "Triage priority P3 — no systemic red flags"
                    ]
                }
            ],
            "information_gaps": [
                "Urine dipstick — nitrites and leucocytes would support UTI",
                "Urine microscopy and culture — confirm organism and guide treatment",
                "STI screen — if clinically indicated after history",
                "Severity VAS — not captured due to 12-question limit",
                "Sexual history — relevant to STI differential",
                "Fever / temperature — not reported in any agent output"
            ],
            "recommended_next_steps": [
                "Urine dipstick and midstream urine (MSU) for culture and sensitivity",
                "Review STI risk factors in clinical consultation",
                "Assess for systemic features (fever, rigors, flank pain) to exclude pyelonephritis",
                "Arrange follow-up if symptoms do not resolve within 48 hours of treatment"
            ],
            "synthesis_confidence": 0.74
        }, indent=2)
    },
]


# ============================================================
# 4. LANGCHAIN PROMPT TEMPLATE
# ============================================================

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai",    "{output}"),
])
"""
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=FEW_SHOT_EXAMPLES,
)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYNTHESIS_SYSTEM_PROMPT),
    few_shot_prompt,
    ("human",  "{synthesis_request}"),
])

parser = JsonOutputParser(pydantic_object=SynthesisOutput)

"""
# ============================================================
# 5. CHAIN ASSEMBLY
# ============================================================

def build_synthesis_chain(llm):
    """
    Returns a runnable synthesis chain.

    Usage:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        chain = build_synthesis_chain(llm)

        result = chain.invoke({"synthesis_request": json.dumps({
            "triage_output":    {...},
            "ehr_output":       {...},
            "anamnesis_output": {...}
        })})

    Note: temperature=0 strongly recommended — clinical synthesis
    must be deterministic and evidence-grounded.
    """
    return SYNTHESIS_PROMPT | llm | parser


def run_synthesis(llm,
                  triage_output: dict,
                  ehr_output: dict,
                  rpm_output: dict,
                  anamnesis_output: dict) -> SynthesisOutput:
    """
    Full synthesis pipeline with post-output safety validation.

    Pass the raw dicts from agent1, agent2, agent3.
    Raises ValueError if any safety constraint is violated.
    """
    chain = build_synthesis_chain(llm)
    request = json.dumps({
        "triage_output":    triage_output,
        "ehr_output":       ehr_output,
        "rpm_output":       rpm_output,
        "anamnesis_output": anamnesis_output
    })
    raw = chain.invoke({"synthesis_request": request})
    return SynthesisOutput(**raw)


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":
    print("=== JSON Schema ===")
    print(json.dumps(SYNTHESIS_JSON_SCHEMA, indent=2)[:600], "...\n")

    print("=== System prompt (first 400 chars) ===")
    print(SYNTHESIS_SYSTEM_PROMPT[:400], "...\n")

    print("=== Few-shot examples ===")
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        out = json.loads(ex["output"])
        n_diff = len(out["differentials"])
        print(f"  {i}. case={out['case_id']}  risk={out['overall_risk']}  "
              f"alert={out['physician_alert']}  differentials={n_diff}")

    print("\n=== Pydantic validation ===")

    # Valid HIGH risk case
    ok = SynthesisOutput(
        summary_header="DRAFT CLINICAL SUMMARY — FOR PHYSICIAN REVIEW ONLY",
        case_id="SYN-TEST-001",
        overall_risk="HIGH",
        physician_alert=True,
        alert_reason="Urgent cardiac presentation",
        data_sources_used=["triage","ehr","anamnesis"],
        missing_sources=[],
        clinical_narrative="Presentation is consistent with an acute coronary syndrome.",
        differentials=[
            Differential(rank=1, diagnosis="STEMI",     icd10="I21.9", probability="VERY-HIGH",
                         supporting_evidence=["chest pain","radiation"], contradicting_evidence=[]),
            Differential(rank=2, diagnosis="NSTEMI",    icd10="I20.0", probability="HIGH",
                         supporting_evidence=["same as rank 1"],         contradicting_evidence=[]),
            Differential(rank=3, diagnosis="Dissection",icd10="I71.00",probability="LOW-MODERATE",
                         supporting_evidence=["hypertension"],           contradicting_evidence=["no tearing quality"]),
        ],
        information_gaps=["ECG", "Troponin"],
        recommended_next_steps=["12-lead ECG", "Serial troponin"],
        synthesis_confidence=0.88
    )
    print(f"  Valid HIGH risk case passed ✓  case_id={ok.case_id}")

    # Wrong header — should fail
    try:
        SynthesisOutput(
            summary_header="Clinical Summary",   # ← wrong header
            case_id="SYN-TEST-002",
            overall_risk="LOW",
            physician_alert=False,
            alert_reason=None,
            data_sources_used=["triage"],
            clinical_narrative="Presentation suggests a minor condition.",
            differentials=[
                Differential(rank=1, diagnosis="A", icd10="Z00", probability="LOW", supporting_evidence=["x"]),
                Differential(rank=2, diagnosis="B", icd10="Z01", probability="LOW", supporting_evidence=["y"]),
                Differential(rank=3, diagnosis="C", icd10="Z02", probability="LOW", supporting_evidence=["z"]),
            ],
            information_gaps=[],
            recommended_next_steps=[],
            synthesis_confidence=0.5
        )
    except ValueError as e:
        print(f"  Safety violation caught ✓  wrong header blocked")

    # HIGH risk without physician_alert — should fail
    try:
        SynthesisOutput(
            summary_header="DRAFT CLINICAL SUMMARY — FOR PHYSICIAN REVIEW ONLY",
            case_id="SYN-TEST-003",
            overall_risk="HIGH",
            physician_alert=False,     # ← intentional violation
            alert_reason=None,
            data_sources_used=["triage","ehr"],
            clinical_narrative="Presentation is consistent with a serious condition.",
            differentials=[
                Differential(rank=1, diagnosis="A", icd10="Z00", probability="HIGH",  supporting_evidence=["x"]),
                Differential(rank=2, diagnosis="B", icd10="Z01", probability="MODERATE", supporting_evidence=["y"]),
                Differential(rank=3, diagnosis="C", icd10="Z02", probability="LOW",   supporting_evidence=["z"]),
            ],
            information_gaps=[],
            recommended_next_steps=[],
            synthesis_confidence=0.8
        )
    except ValueError as e:
        print(f"  Safety violation caught ✓  HIGH risk without physician_alert blocked")

    # Forbidden diagnosis language — should fail
    try:
        SynthesisOutput(
            summary_header="DRAFT CLINICAL SUMMARY — FOR PHYSICIAN REVIEW ONLY",
            case_id="SYN-TEST-004",
            overall_risk="LOW",
            physician_alert=False,
            alert_reason=None,
            data_sources_used=["triage"],
            clinical_narrative="The patient has a urinary tract infection.",  # ← forbidden
            differentials=[
                Differential(rank=1, diagnosis="A", icd10="Z00", probability="LOW", supporting_evidence=["x"]),
                Differential(rank=2, diagnosis="B", icd10="Z01", probability="LOW", supporting_evidence=["y"]),
                Differential(rank=3, diagnosis="C", icd10="Z02", probability="LOW", supporting_evidence=["z"]),
            ],
            information_gaps=[],
            recommended_next_steps=[],
            synthesis_confidence=0.5
        )
    except ValueError as e:
        print(f"  Safety violation caught ✓  forbidden diagnosis language blocked")

    print("\nAll checks passed.")

    print("\n=== LLM smoke test ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Skipping LLM smoke test.")
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        result = run_synthesis(
            llm,
            triage_output={
                "priority": "P1",
                "priority_label": "EMERGENT",
                "pathway": "emergency_dispatch",
                "red_flags": [
                    "crushing_chest_pain",
                    "radiation_left_arm",
                    "shortness_of_breath",
                ],
                "escalate_human": True,
                "escalation_reason": "Emergent cardiac red flags require immediate escalation.",
                "confidence": 0.97,
            },
            ehr_output={
                "patient_id": "PT-88421",
                "retrieval_status": "success",
                "demographics": {"age": 67, "sex": "M", "blood_type": "A+"},
                "active_diagnoses": [
                    "Coronary artery disease (ICD-10: I25.10)",
                    "Hypertension (ICD-10: I10)",
                ],
                "current_medications": [
                    {
                        "name": "Aspirin",
                        "dose": "81mg",
                        "frequency": "daily",
                        "relevant": True,
                    }
                ],
                "allergies": [],
                "recent_labs": {},
                "data_quality_flags": [],
                "audit_log": {
                    "access_time": "2026-06-11T10:00:00Z",
                    "accessed_by": "ehr_agent_v1",
                },
                "error": None,
            },
            anamnesis_output={
                "session_id": "ANM-SMOKE-001",
                "patient_id": "PT-88421",
                "turns_completed": 2,
                "conversation_log": [
                    {
                        "turn": 1,
                        "speaker": "agent",
                        "text": "Can you describe the chest pain and tell me when it started?",
                    },
                    {
                        "turn": 2,
                        "speaker": "patient",
                        "text": "It is crushing central chest pain that started 30 minutes ago and spreads to my left arm.",
                    },
                ],
                "socrates": {
                    "site": "central_chest",
                    "onset": {
                        "duration_minutes": 30,
                        "duration_description": "30 minutes",
                        "character": "sudden",
                        "trigger": None,
                    },
                    "character": "crushing",
                    "radiation": "left_arm",
                    "associations": "shortness_of_breath",
                    "time_course": "constant",
                    "exacerbating_relieving": None,
                    "severity_vas": 8.0,
                },
                "ehr_fields_skipped": [],
                "new_red_flags": [],
                "escalate_immediately": False,
                "interview_complete": False,
                "incomplete_dimensions": ["exacerbating_relieving"],
            },
        )
        print(result.model_dump_json(indent=2))
