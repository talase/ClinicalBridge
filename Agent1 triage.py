# ============================================================
# AGENT 1: TRIAGE AGENT
# Deliverables: system prompt, few-shot examples,
# chain-of-thought structure, JSON schema, LangChain chain
# ============================================================

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, List, Optional
import json
import os
# ============================================================
# 1. JSON OUTPUT SCHEMA
# ============================================================

class TriageOutput(BaseModel):
    priority: Literal["P1", "P2", "P3", "P4"] = Field(
        description="Urgency level: P1=Emergent, P2=Urgent, P3=Less Urgent, P4=Non-Urgent"
    )
    priority_label: Literal["EMERGENT", "URGENT", "LESS_URGENT", "NON_URGENT"] = Field(
        description="Human-readable label matching the priority level"
    )
    pathway: str = Field(description="Clinical routing pathway identifier")
    red_flags: List[str] = Field(description="Red-flag symptoms identified in patient input")
    escalate_human: bool = Field(description="Must this case be escalated to a human clinician?")
    escalation_reason: Optional[str] = Field(description="Reason for escalation, null if not escalating")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence (0.0–1.0)")

    @model_validator(mode="after")
    def safety_checks(self):
        # P1 must always escalate
        if self.priority == "P1" and not self.escalate_human:
            raise ValueError("Safety violation: P1 triage MUST set escalate_human=True")
        # Escalation reason required when escalating
        if self.escalate_human and not self.escalation_reason:
            raise ValueError("escalation_reason required when escalate_human=True")
        # Label must match priority
        mapping = {"P1": "EMERGENT", "P2": "URGENT", "P3": "LESS_URGENT", "P4": "NON_URGENT"}
        if mapping[self.priority] != self.priority_label:
            raise ValueError(f"priority_label '{self.priority_label}' does not match priority '{self.priority}'")
        return self


TRIAGE_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TriageOutput",
    "type": "object",
    "required": ["priority","priority_label","pathway","red_flags","escalate_human","escalation_reason","confidence"],
    "properties": {
        "priority":          {"type": "string", "enum": ["P1","P2","P3","P4"]},
        "priority_label":    {"type": "string", "enum": ["EMERGENT","URGENT","LESS_URGENT","NON_URGENT"]},
        "pathway":           {"type": "string"},
        "red_flags":         {"type": "array", "items": {"type": "string"}},
        "escalate_human":    {"type": "boolean"},
        "escalation_reason": {"type": ["string","null"]},
        "confidence":        {"type": "number", "minimum": 0, "maximum": 1}
    },
    "additionalProperties": False
}


# ============================================================
# 2. SYSTEM PROMPT
# ============================================================
# Note: curly braces in the schema are escaped as {{ }} so
# LangChain does not treat them as template variables.

TRIAGE_SYSTEM_PROMPT = """\
You are a medical triage AI assistant in a clinical decision-support system.
You evaluate patient-reported symptoms and assign urgency priority.
You do NOT diagnose, treat, or advise.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIORITY LEVELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P1 — EMERGENT    : Life-threatening. Immediate intervention required (<15 min).
P2 — URGENT      : Serious. Prompt attention needed (<2 hours).
P3 — LESS URGENT : Needs care, not immediately life-threatening (<24 hours).
P4 — NON-URGENT  : Routine care. Can be scheduled.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RED-FLAG SYMPTOMS  (auto-escalate P1 or P2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Chest pain, pressure, tightness, or crushing sensation
• Pain radiating to arm, jaw, neck, or back
• Difficulty breathing / shortness of breath
• Sudden severe headache ("worst of my life")
• Stroke signs — FAST: face drooping, arm weakness, speech difficulty
• Anaphylaxis signs: throat swelling, hives, sudden BP drop
• Uncontrolled or severe bleeding
• Loss of consciousness or altered mental status
• High fever (>39.5 °C) with neck stiffness or non-blanching rash
• Suicidal ideation, self-harm intent, or any abuse disclosure
• Severe sudden-onset abdominal pain
• Diabetic emergency: extreme confusion, fruity breath, no response
• Atypical MI: nausea, jaw/back pain, fatigue + dyspnea (women, elderly, diabetics)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLINICAL PATHWAYS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
emergency_dispatch       → P1: call ambulance/911 immediately
urgent_care_ED           → P2: same-day emergency department visit
primary_care_same_day    → P3: same-day GP appointment
primary_care_scheduling  → P4: routine scheduling
mental_health_crisis     → self-harm / suicidal ideation
safeguarding_review      → abuse disclosure

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAFETY RULES — ABSOLUTE, NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER provide a diagnosis or differential.
2. NEVER recommend a treatment or medication.
3. When uncertain between two levels → assign the HIGHER level.
4. Any red-flag symptom detected → escalate to P1 or P2 immediately.
5. Self-harm / suicide / abuse → mental_health_crisis or safeguarding_review, escalate_human: true.
6. Patient age <18 stated → escalate one level higher if any doubt.
7. Confidence < 0.75 → escalate one level higher automatically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAIN OF THOUGHT  (internal — do not output these steps)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 PARSE         Extract symptom terms, duration, severity, emotional cues.
Step 2 RED-FLAG SCAN Cross-reference with the red-flag list above.
Step 3 CLUSTER       Group by organ system (cardiac, neuro, respiratory, GI…).
Step 4 SCORE         Assign preliminary P-level from cluster severity + red-flag count.
Step 5 UNCERTAINTY   If confidence < 0.75 or ambiguous → escalate one level.
Step 6 SAFETY        Check for self-harm, abuse, safeguarding language.
Step 7 ROUTE         Map final P-level to the correct pathway.
Step 8 EMIT          Output valid JSON only. No prose outside the JSON object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON object. No markdown fences, no explanation.
Required fields: priority, priority_label, pathway, red_flags,
                 escalate_human, escalation_reason, confidence
"""


# ============================================================
# 3. FEW-SHOT EXAMPLES
# ============================================================

FEW_SHOT_EXAMPLES = [
    # P1 — Classic STEMI presentation
    {
        "input": (
            "I have crushing chest pain in the centre of my chest radiating to my left arm. "
            "I'm sweating heavily, feel very dizzy, and it started 40 minutes ago walking upstairs. "
            "The pain is 9 out of 10."
        ),
        "output": json.dumps({
            "priority": "P1",
            "priority_label": "EMERGENT",
            "pathway": "emergency_dispatch",
            "red_flags": ["crushing_chest_pain","radiation_left_arm","diaphoresis","dizziness","exertional_onset"],
            "escalate_human": True,
            "escalation_reason": "Classic ACS/STEMI presentation: central crushing pain with left arm radiation, diaphoresis, dizziness on exertion. Immediate emergency services required.",
            "confidence": 0.97
        }, indent=2)
    },
    # P2 — Possible meningitis
    {
        "input": (
            "I've had a fever of 39.8 degrees for two days, a really bad headache, "
            "my neck feels very stiff, and I'm sensitive to bright light."
        ),
        "output": json.dumps({
            "priority": "P2",
            "priority_label": "URGENT",
            "pathway": "urgent_care_ED",
            "red_flags": ["high_fever_39_8","severe_headache","neck_stiffness","photophobia"],
            "escalate_human": True,
            "escalation_reason": "Classic meningitis triad: high fever, severe headache, neck stiffness with photophobia. Bacterial meningitis cannot be excluded without urgent evaluation.",
            "confidence": 0.91
        }, indent=2)
    },
    # P3 — UTI
    {
        "input": (
            "Burning when I urinate for two days, frequent urge to go but little comes out. "
            "Mild lower abdominal discomfort. No fever."
        ),
        "output": json.dumps({
            "priority": "P3",
            "priority_label": "LESS_URGENT",
            "pathway": "primary_care_same_day",
            "red_flags": [],
            "escalate_human": False,
            "escalation_reason": None,
            "confidence": 0.88
        }, indent=2)
    },
    # P4 — Medication refill
    {
        "input": (
            "I need a refill on my blood pressure medication, lisinopril. "
            "Been on it three years. Running out. Feel completely fine."
        ),
        "output": json.dumps({
            "priority": "P4",
            "priority_label": "NON_URGENT",
            "pathway": "primary_care_scheduling",
            "red_flags": [],
            "escalate_human": False,
            "escalation_reason": None,
            "confidence": 0.99
        }, indent=2)
    },
    # P1 — Self-harm disclosure
    {
        "input": (
            "I've been feeling really low and I've been thinking about hurting myself. "
            "I don't know what to do anymore."
        ),
        "output": json.dumps({
            "priority": "P1",
            "priority_label": "EMERGENT",
            "pathway": "mental_health_crisis",
            "red_flags": ["suicidal_ideation","self_harm_intent"],
            "escalate_human": True,
            "escalation_reason": "Patient has disclosed thoughts of self-harm. Immediate mental health crisis intervention required.",
            "confidence": 0.99
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

TRIAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",  TRIAGE_SYSTEM_PROMPT),
    few_shot_prompt,
    ("human",   "{patient_input}"),
])

parser = JsonOutputParser(pydantic_object=TriageOutput)


# ============================================================
# 5. CHAIN ASSEMBLY
# ============================================================

def build_triage_chain(llm):
    """
    Returns a runnable triage chain.

    Usage:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        chain = build_triage_chain(llm)
        result = chain.invoke({"patient_input": "I have chest pain..."})
    """
    return TRIAGE_PROMPT | llm | parser


def run_triage(llm, patient_input: str) -> TriageOutput:
    """Full pipeline with post-output safety validation."""
    chain = build_triage_chain(llm)
    raw = chain.invoke({"patient_input": patient_input})
    return TriageOutput(**raw)   # raises if any safety rule violated


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":
    print("=== JSON Schema ===")
    print(json.dumps(TRIAGE_JSON_SCHEMA, indent=2))

    print("\n=== System prompt (first 400 chars) ===")
    print(TRIAGE_SYSTEM_PROMPT[:400], "...\n")

    print("=== Few-shot examples ===")
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        p = json.loads(ex["output"])
        print(f"  {i}. {p['priority']} {p['priority_label']:12} → {p['pathway']}")

    print("\n=== Pydantic validation ===")
    ok = TriageOutput(
        priority="P1", priority_label="EMERGENT",
        pathway="emergency_dispatch", red_flags=["chest_pain"],
        escalate_human=True, escalation_reason="Test", confidence=0.95
    )
    print(f"  Valid P1 passed ✓  priority={ok.priority}")

    try:
        TriageOutput(
            priority="P1", priority_label="EMERGENT",
            pathway="emergency_dispatch", red_flags=["chest_pain"],
            escalate_human=False,       # ← intentional violation
            escalation_reason=None, confidence=0.95
        )
    except ValueError as e:
        print(f"  Safety violation caught ✓  {e}")

    print("\nAll checks passed.")

    print("\n=== LLM smoke test ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Skipping LLM smoke test.")
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        result = run_triage(llm, "I have chest pain and shortness of breath.")
        print(result.model_dump_json(indent=2))
