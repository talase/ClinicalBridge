# ============================================================
# AGENT 2: EHR AGENT
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

class Medication(BaseModel):
    name: str
    dose: str
    frequency: str
    relevant: bool = Field(description="Relevant to the current triage pathway?")

class Allergy(BaseModel):
    substance: str
    reaction: str
    severity: Literal["mild", "moderate", "severe", "unknown"]

class DataQualityFlag(BaseModel):
    field: str
    flag: Literal["MISSING", "STALE", "CONFLICT", "UNVERIFIED"]
    note: str

class AuditLog(BaseModel):
    access_time: str = Field(description="ISO 8601 timestamp of record access")
    accessed_by: str = Field(description="Agent identifier, e.g. ehr_agent_v1")

class EHROutput(BaseModel):
    patient_id: str
    retrieval_status: Literal["success", "partial", "not_found", "error"]
    demographics: Optional[Dict[str, Any]] = None
    active_diagnoses: Optional[List[str]] = None
    current_medications: Optional[List[Medication]] = None
    allergies: Optional[List[Allergy]] = None
    recent_labs: Optional[Dict[str, str]] = None
    data_quality_flags: List[DataQualityFlag] = Field(default_factory=list)
    audit_log: AuditLog
    error: Optional[str] = None

    @model_validator(mode="after")
    def safety_checks(self):
        # If not_found, must have an error message — never return empty
        if self.retrieval_status == "not_found" and not self.error:
            raise ValueError("not_found status must include an error message")
        # If success, core clinical fields must be present
        if self.retrieval_status == "success":
            if self.demographics is None:
                raise ValueError("demographics required on successful retrieval")
            if self.active_diagnoses is None:
                raise ValueError("active_diagnoses required on successful retrieval")
        return self


EHR_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "EHROutput",
    "type": "object",
    "required": ["patient_id", "retrieval_status", "data_quality_flags", "audit_log"],
    "properties": {
        "patient_id":       {"type": "string"},
        "retrieval_status": {"type": "string", "enum": ["success","partial","not_found","error"]},
        "demographics": {
            "type": ["object","null"],
            "properties": {
                "age":        {"type": "integer"},
                "sex":        {"type": "string", "enum": ["M","F","Other","Unknown"]},
                "blood_type": {"type": "string"},
                "dob":        {"type": "string"}
            }
        },
        "active_diagnoses": {"type": ["array","null"], "items": {"type": "string"}},
        "current_medications": {
            "type": ["array","null"],
            "items": {
                "type": "object",
                "required": ["name","dose","frequency","relevant"],
                "properties": {
                    "name":      {"type": "string"},
                    "dose":      {"type": "string"},
                    "frequency": {"type": "string"},
                    "relevant":  {"type": "boolean"}
                }
            }
        },
        "allergies": {
            "type": ["array","null"],
            "items": {
                "type": "object",
                "required": ["substance","reaction","severity"],
                "properties": {
                    "substance": {"type": "string"},
                    "reaction":  {"type": "string"},
                    "severity":  {"type": "string", "enum": ["mild","moderate","severe","unknown"]}
                }
            }
        },
        "recent_labs":  {"type": ["object","null"]},
        "data_quality_flags": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["field","flag","note"],
                "properties": {
                    "field": {"type": "string"},
                    "flag":  {"type": "string", "enum": ["MISSING","STALE","CONFLICT","UNVERIFIED"]},
                    "note":  {"type": "string"}
                }
            }
        },
        "audit_log": {
            "type": "object",
            "required": ["access_time","accessed_by"],
            "properties": {
                "access_time": {"type": "string"},
                "accessed_by": {"type": "string"}
            }
        },
        "error": {"type": ["string","null"]}
    },
    "additionalProperties": False
}


# ============================================================
# 2. SYSTEM PROMPT
# ============================================================

EHR_SYSTEM_PROMPT = """\
You are a clinical data extraction AI with read-only access to electronic health records.
You operate in a HIPAA-compliant environment. Your sole function is to retrieve and
structure patient health record data relevant to the current triage context.
You do NOT interpret findings, suggest diagnoses, or recommend treatment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSIBILITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Retrieve the patient record for the given patient_id
• Extract and structure: demographics, diagnoses, medications, allergies, labs, vitals history
• Filter to fields relevant to the current triage pathway and red_flags
• Flag data quality issues using the defined flag types
• Identify medications relevant to the triage context
• Write an audit log entry for every access

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA QUALITY FLAGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISSING     → Required field is absent from the record entirely
STALE       → Data entry is older than 5 years (or older than 6 months for labs/vitals)
CONFLICT    → Two entries in the record disagree (e.g. two different blood types)
UNVERIFIED  → Entry was added without a clinical verification step

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIVACY & COMPLIANCE RULES — ABSOLUTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Access ONLY the patient_id specified in the input. Never cross-reference other patients.
2. Do NOT reproduce full clinical notes — extract structured fields only.
3. If patient_id is not found → return retrieval_status: "not_found" with a clear error message.
   NEVER fabricate, guess, or interpolate record data.
4. If the record exists but required fields are missing → return retrieval_status: "partial"
   and flag each missing field as MISSING.
5. Log every access in audit_log with ISO 8601 timestamp and agent identifier.
6. Mark any lab or vital older than 6 months as STALE. Mark any diagnosis older than 5 years
   as STALE unless it is listed as a chronic/active condition.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEDICATION RELEVANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mark relevant: true for any medication that:
• Directly relates to the presenting complaint or triage pathway
• Could interact with likely emergency interventions (anticoagulants, beta-blockers, MAOIs, etc.)
• Has a narrow therapeutic window that affects emergency management
• Is relevant to the patient's flagged red_flags

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAIN OF THOUGHT  (internal — do not output these steps)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1  ID VALIDATION    Verify patient_id format and existence. If not found → error output, stop.
Step 2  RECORD RETRIEVAL Fetch full record. Begin audit log entry with timestamp.
Step 3  RELEVANCE FILTER Apply triage_pathway + red_flags to decide which fields matter most.
Step 4  QUALITY SCAN     Check every required field for MISSING, STALE, CONFLICT, UNVERIFIED.
Step 5  MED REVIEW       Mark each medication as relevant/not relevant to current triage context.
Step 6  STRUCTURE        Serialise fields into schema-compliant JSON. No free-text clinical notes.
Step 7  AUDIT            Finalise audit_log before emitting output.
Step 8  EMIT             Output valid JSON only. No prose, no markdown.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON object matching the EHROutput schema.
No markdown fences, no prose, no clinical interpretation.
"""


# ============================================================
# 3. FEW-SHOT EXAMPLES
# ============================================================

FEW_SHOT_EXAMPLES = [
    # ── Example 1: Successful retrieval — P1 cardiac case ─
    {
        "input": json.dumps({
            "patient_id": "PT-88421",
            "triage_pathway": "emergency_dispatch",
            "red_flags": ["crushing_chest_pain", "radiation_left_arm", "diaphoresis"]
        }),
        "output": json.dumps({
            "patient_id": "PT-88421",
            "retrieval_status": "success",
            "demographics": {
                "age": 67,
                "sex": "M",
                "blood_type": "A+",
                "dob": "1958-03-14"
            },
            "active_diagnoses": [
                "Coronary artery disease (ICD-10: I25.10)",
                "Hypertension (ICD-10: I10)",
                "Type 2 diabetes mellitus (ICD-10: E11.9)"
            ],
            "current_medications": [
                {"name": "Metoprolol",  "dose": "50mg",  "frequency": "BID",   "relevant": True},
                {"name": "Aspirin",     "dose": "81mg",  "frequency": "daily", "relevant": True},
                {"name": "Lisinopril",  "dose": "10mg",  "frequency": "daily", "relevant": True},
                {"name": "Metformin",   "dose": "500mg", "frequency": "BID",   "relevant": False}
            ],
            "allergies": [
                {"substance": "Penicillin", "reaction": "anaphylaxis", "severity": "severe"}
            ],
            "recent_labs": {
                "troponin_I":   "0.04 ng/mL (2024-01-15)",
                "HbA1c":        "7.2% (2024-03-01)",
                "eGFR":         "72 mL/min (2024-01-15)",
                "total_cholesterol": "5.8 mmol/L (2023-11-20)"
            },
            "data_quality_flags": [
                {
                    "field": "recent_labs.troponin_I",
                    "flag": "STALE",
                    "note": "Most recent troponin is from 2024-01-15 — >6 months ago. Current value needed for ACS workup."
                }
            ],
            "audit_log": {
                "access_time": "2026-06-09T10:34:00Z",
                "accessed_by": "ehr_agent_v1"
            },
            "error": None
        }, indent=2)
    },

    # ── Example 2: Patient not found ──────────────────────
    {
        "input": json.dumps({
            "patient_id": "PT-00000",
            "triage_pathway": "urgent_care_ED",
            "red_flags": ["high_fever_39_8", "neck_stiffness"]
        }),
        "output": json.dumps({
            "patient_id": "PT-00000",
            "retrieval_status": "not_found",
            "demographics": None,
            "active_diagnoses": None,
            "current_medications": None,
            "allergies": None,
            "recent_labs": None,
            "data_quality_flags": [
                {
                    "field": "patient_id",
                    "flag": "MISSING",
                    "note": "No record found for PT-00000. Patient may be unregistered. Manual intake required."
                }
            ],
            "audit_log": {
                "access_time": "2026-06-09T10:35:00Z",
                "accessed_by": "ehr_agent_v1"
            },
            "error": "No record found for patient_id PT-00000. Manual registration required before EHR data can be retrieved."
        }, indent=2)
    },

    # ── Example 3: Partial record — missing critical fields ─
    {
        "input": json.dumps({
            "patient_id": "PT-34912",
            "triage_pathway": "urgent_care_ED",
            "red_flags": ["severe_abdominal_pain"]
        }),
        "output": json.dumps({
            "patient_id": "PT-34912",
            "retrieval_status": "partial",
            "demographics": {
                "age": 34,
                "sex": "F",
                "blood_type": None,
                "dob": "1991-08-22"
            },
            "active_diagnoses": [
                "Irritable bowel syndrome (ICD-10: K58.9)"
            ],
            "current_medications": [
                {"name": "Mebeverine", "dose": "135mg", "frequency": "TDS", "relevant": True}
            ],
            "allergies": [],
            "recent_labs": None,
            "data_quality_flags": [
                {
                    "field": "demographics.blood_type",
                    "flag": "MISSING",
                    "note": "Blood type not recorded. Critical for surgical/transfusion planning."
                },
                {
                    "field": "recent_labs",
                    "flag": "MISSING",
                    "note": "No lab results on file for this patient."
                }
            ],
            "audit_log": {
                "access_time": "2026-06-09T10:36:00Z",
                "accessed_by": "ehr_agent_v1"
            },
            "error": None
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

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=FEW_SHOT_EXAMPLES,
)

EHR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EHR_SYSTEM_PROMPT),
    few_shot_prompt,
    ("human",  "{ehr_request}"),
])

parser = JsonOutputParser(pydantic_object=EHROutput)


# ============================================================
# 5. CHAIN ASSEMBLY
# ============================================================

def build_ehr_chain(llm):
    """
    Returns a runnable EHR chain.

    Usage:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        chain = build_ehr_chain(llm)

        result = chain.invoke({"ehr_request": json.dumps({
            "patient_id": "PT-88421",
            "triage_pathway": "emergency_dispatch",
            "red_flags": ["chest_pain"]
        })})
    """
    return EHR_PROMPT | llm | parser


def run_ehr(llm, patient_id: str, triage_pathway: str, red_flags: list) -> EHROutput:
    """Full EHR pipeline with post-output safety validation."""
    chain = build_ehr_chain(llm)
    request = json.dumps({
        "patient_id": patient_id,
        "triage_pathway": triage_pathway,
        "red_flags": red_flags
    })
    raw = chain.invoke({"ehr_request": request})
    return EHROutput(**raw)   # raises if schema or safety rules violated


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":
    print("=== JSON Schema ===")
    print(json.dumps(EHR_JSON_SCHEMA, indent=2)[:600], "...\n")

    print("=== System prompt (first 400 chars) ===")
    print(EHR_SYSTEM_PROMPT[:400], "...\n")

    print("=== Few-shot examples ===")
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        inp = json.loads(ex["input"])
        out = json.loads(ex["output"])
        print(f"  {i}. patient_id={inp['patient_id']}  status={out['retrieval_status']}")

    print("\n=== Pydantic validation ===")
    # Valid success case
    ok = EHROutput(
        patient_id="PT-88421",
        retrieval_status="success",
        demographics={"age": 67, "sex": "M"},
        active_diagnoses=["Hypertension (ICD-10: I10)"],
        current_medications=[],
        allergies=[],
        recent_labs={},
        data_quality_flags=[],
        audit_log=AuditLog(access_time="2026-06-09T10:00:00Z", accessed_by="ehr_agent_v1")
    )
    print(f"  Valid success case passed ✓  patient_id={ok.patient_id}")

    # not_found without error message — should fail
    try:
        EHROutput(
            patient_id="PT-00000",
            retrieval_status="not_found",
            data_quality_flags=[],
            audit_log=AuditLog(access_time="2026-06-09T10:00:00Z", accessed_by="ehr_agent_v1"),
            error=None   # ← intentional violation
        )
    except ValueError as e:
        print(f"  Safety violation caught ✓  not_found without error message blocked")

    # success without demographics — should fail
    try:
        EHROutput(
            patient_id="PT-11111",
            retrieval_status="success",
            demographics=None,  # ← intentional violation
            active_diagnoses=["HTN"],
            data_quality_flags=[],
            audit_log=AuditLog(access_time="2026-06-09T10:00:00Z", accessed_by="ehr_agent_v1")
        )
    except ValueError as e:
        print(f"  Safety violation caught ✓  success without demographics blocked")

    print("\nAll checks passed.")

    print("\n=== LLM smoke test ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Skipping LLM smoke test.")
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        result = run_ehr(
            llm,
            patient_id="PT-88421",
            triage_pathway="emergency_dispatch",
            red_flags=["crushing_chest_pain", "radiation_left_arm", "shortness_of_breath"],
        )
        print(result.model_dump_json(indent=2))
