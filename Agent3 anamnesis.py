# ============================================================
# AGENT 3: ANAMNESIS AGENT
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

class Onset(BaseModel):
    duration_minutes: Optional[int] = None
    duration_description: Optional[str] = None   # e.g. "two days"
    character: Optional[Literal["sudden", "gradual", "unknown"]] = None
    trigger: Optional[str] = None                # e.g. "exertion_stairs"

class SOCRATES(BaseModel):
    site:                   Optional[str] = Field(None, description="Location of symptom")
    onset:                  Optional[Onset] = Field(None, description="When and how it started")
    character:              Optional[str] = Field(None, description="Quality of symptom e.g. crushing, burning")
    radiation:              Optional[str] = Field(None, description="Does it spread anywhere?")
    associations:           Optional[str] = Field(None, description="Associated symptoms")
    time_course:            Optional[str] = Field(None, description="Constant, intermittent, progressive")
    exacerbating_relieving: Optional[str] = Field(None, description="What makes it better or worse")
    severity_vas:           Optional[float] = Field(None, ge=0.0, le=10.0,
                                description="Pain severity 0–10 visual analogue scale")

class ConversationTurn(BaseModel):
    turn: int
    speaker: Literal["agent", "patient"]
    text: str

class AnamnesisOutput(BaseModel):
    session_id: str
    patient_id: str
    turns_completed: int = Field(ge=0, le=12)
    conversation_log: List[ConversationTurn]
    socrates: SOCRATES
    ehr_fields_skipped: List[str] = Field(
        default_factory=list,
        description="SOCRATES dimensions already answered by EHR — not re-asked"
    )
    new_red_flags: List[str] = Field(
        default_factory=list,
        description="New red-flag symptoms surfaced during interview not in original triage"
    )
    escalate_immediately: bool = Field(
        description="True if a new red flag was found — pause interview, escalate now"
    )
    interview_complete: bool = Field(
        description="True if all obtainable SOCRATES dimensions are filled or 12-question limit reached"
    )
    incomplete_dimensions: List[str] = Field(
        default_factory=list,
        description="SOCRATES dimensions not obtained, with reason"
    )

    @model_validator(mode="after")
    def safety_checks(self):
        # If new red flags found, must escalate immediately
        if self.new_red_flags and not self.escalate_immediately:
            raise ValueError(
                "Safety violation: new_red_flags present but escalate_immediately is False"
            )
        # turns_completed cannot exceed 12
        if self.turns_completed > 12:
            raise ValueError("turns_completed exceeds maximum of 12")
        # conversation_log must match turns_completed
        if len(self.conversation_log) < self.turns_completed:
            raise ValueError("conversation_log has fewer entries than turns_completed")
        return self


ANAMNESIS_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AnamnesisOutput",
    "type": "object",
    "required": [
        "session_id", "patient_id", "turns_completed",
        "conversation_log", "socrates",
        "new_red_flags", "escalate_immediately", "interview_complete"
    ],
    "properties": {
        "session_id":     {"type": "string"},
        "patient_id":     {"type": "string"},
        "turns_completed":{"type": "integer", "minimum": 0, "maximum": 12},
        "conversation_log": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["turn", "speaker", "text"],
                "properties": {
                    "turn":    {"type": "integer"},
                    "speaker": {"type": "string", "enum": ["agent","patient"]},
                    "text":    {"type": "string"}
                }
            }
        },
        "socrates": {
            "type": "object",
            "properties": {
                "site":                   {"type": ["string","null"]},
                "onset":                  {"type": ["object","null"]},
                "character":              {"type": ["string","null"]},
                "radiation":              {"type": ["string","null"]},
                "associations":           {"type": ["string","null"]},
                "time_course":            {"type": ["string","null"]},
                "exacerbating_relieving": {"type": ["string","null"]},
                "severity_vas":           {"type": ["number","null"], "minimum": 0, "maximum": 10}
            }
        },
        "ehr_fields_skipped":    {"type": "array", "items": {"type": "string"}},
        "new_red_flags":         {"type": "array", "items": {"type": "string"}},
        "escalate_immediately":  {"type": "boolean"},
        "interview_complete":    {"type": "boolean"},
        "incomplete_dimensions": {"type": "array", "items": {"type": "string"}}
    },
    "additionalProperties": False
}


# ============================================================
# 2. SYSTEM PROMPT
# ============================================================

ANAMNESIS_SYSTEM_PROMPT = """\
You are a clinical history-taking AI. You conduct structured medical history interviews
using the SOCRATES framework. You work alongside a licensed physician — you gather
information only; you do NOT diagnose, interpret findings, or suggest treatment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOCRATES FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each letter represents one dimension to capture for the chief complaint:

  S — Site           Where exactly is the symptom? Does it stay in one place?
  O — Onset          When did it start? Did it come on suddenly or gradually?
                     What were you doing when it started?
  C — Character      How would you describe it? (sharp, dull, crushing, burning, aching)
  R — Radiation      Does it spread or move anywhere else?
  A — Associations   Any other symptoms at the same time? (nausea, sweating, dizziness)
  T — Time course    Is it constant or does it come and go? Getting better or worse?
  E — Exac/Relieving What makes it worse? What makes it better?
  S — Severity       On a scale of 0 to 10 — 0 being no pain, 10 being the worst
                     pain you have ever felt — where would you rate it?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERVIEW RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Ask ONE question per turn. Never combine two questions in one message.
2. Use plain language — maximum Grade 8 reading level. No medical jargon.
3. Begin each response with a brief empathetic acknowledgement of the patient's answer.
4. Do NOT re-ask any dimension already answered in the EHR summary.
5. Hard stop at 12 questions — mark any remaining dimensions as not_obtained.
6. NEVER ask about or discuss diagnoses, differentials, or treatment options.
7. NEVER comment on what the symptoms might mean.
8. Maintain a calm, warm, and professional tone throughout.
9. If the patient gives an alarming answer (new red flag) → set escalate_immediately: true
   and stop the interview. Output the partial SOCRATES collected so far.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW RED FLAGS TO WATCH FOR DURING INTERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the patient mentions any of the following mid-interview, stop and escalate:
• Chest tightness, pressure, or new radiating pain
• Sudden severe headache or vision changes
• Difficulty breathing or throat tightening
• Loss of consciousness or confusion
• Thoughts of self-harm or suicide
• Uncontrolled bleeding or severe sudden-onset pain (VAS 9–10)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAIN OF THOUGHT  (internal — do not output these steps)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1  CONTEXT LOAD     Load chief complaint, EHR summary, triage priority, session history.
Step 2  EHR GAP ANALYSIS Check which SOCRATES dimensions are already answered by EHR data.
                         Add those to ehr_fields_skipped. Do not ask about them.
Step 3  QUESTION SELECT  Identify the highest-priority unanswered SOCRATES dimension.
                         Draft a single open-ended question in plain language.
Step 4  RESPONSE PARSE   Parse the patient's answer into the relevant SOCRATES field.
                         Extract any symptom keywords mentioned.
Step 5  RED-FLAG MONITOR After each patient response, check against the red-flag list above.
                         If any match → set escalate_immediately: true, stop interview.
Step 6  EMPATHY FRAME    Prepend a brief empathetic acknowledgement before the next question.
                         Adjust warmth level to match patient's distress level.
Step 7  COMPLETION CHECK If all obtainable dimensions are filled OR turns_completed = 12
                         → set interview_complete: true.
Step 8  EMIT             Output the full AnamnesisOutput JSON. No prose outside the JSON.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON object matching the AnamnesisOutput schema.
Include the full conversation_log up to the current turn.
No markdown fences, no prose, no clinical interpretation.
"""


# ============================================================
# 3. FEW-SHOT EXAMPLES
# ============================================================

FEW_SHOT_EXAMPLES = [
    # ── Example 1: Ongoing session — chest pain, 3 turns ──
    # EHR already told us: site=central_chest, onset trigger=exertion
    # So we skip S and part of O, start on character
    {
        "input": json.dumps({
            "session_id": "ANM-20260609-001",
            "patient_id": "PT-88421",
            "triage_priority": "P1",
            "chief_complaint": "crushing chest pain radiating to left arm",
            "ehr_summary": {
                "age": 67, "sex": "M",
                "active_diagnoses": ["CAD","HTN","T2DM"],
                "note": "EHR confirms: pain is central chest, started on exertion (stairs), 40 min ago"
            },
            "session_history": [
                {"turn": 1, "speaker": "agent",
                 "text": "I understand you're in a lot of discomfort right now. I'm going to ask you a few quick questions to help the care team. How would you describe the pain — for example, is it sharp, dull, crushing, or burning?"},
                {"turn": 2, "speaker": "patient",
                 "text": "It feels like a heavy weight pressing down on my chest, like something is crushing it."},
                {"turn": 3, "speaker": "agent",
                 "text": "Thank you for describing that. Does the pain or that crushing feeling move or spread anywhere — for example to your arm, jaw, neck, or back?"},
                {"turn": 4, "speaker": "patient",
                 "text": "Yes, it goes down my left arm. It started there too."}
            ],
            "current_turn": 4
        }),
        "output": json.dumps({
            "session_id": "ANM-20260609-001",
            "patient_id": "PT-88421",
            "turns_completed": 4,
            "conversation_log": [
                {"turn": 1, "speaker": "agent",
                 "text": "I understand you're in a lot of discomfort right now. I'm going to ask you a few quick questions to help the care team. How would you describe the pain — for example, is it sharp, dull, crushing, or burning?"},
                {"turn": 2, "speaker": "patient",
                 "text": "It feels like a heavy weight pressing down on my chest, like something is crushing it."},
                {"turn": 3, "speaker": "agent",
                 "text": "Thank you for describing that. Does the pain or that crushing feeling move or spread anywhere — for example to your arm, jaw, neck, or back?"},
                {"turn": 4, "speaker": "patient",
                 "text": "Yes, it goes down my left arm. It started there too."}
            ],
            "socrates": {
                "site": "central_chest",
                "onset": {
                    "duration_minutes": 40,
                    "duration_description": "40 minutes",
                    "character": "sudden",
                    "trigger": "exertion_stairs"
                },
                "character": "pressure_crushing_heavy_weight",
                "radiation": "left_arm_bilateral_onset",
                "associations": None,
                "time_course": None,
                "exacerbating_relieving": None,
                "severity_vas": None
            },
            "ehr_fields_skipped": ["site", "onset"],
            "new_red_flags": [],
            "escalate_immediately": False,
            "interview_complete": False,
            "incomplete_dimensions": ["associations","time_course","exacerbating_relieving","severity_vas"]
        }, indent=2)
    },

    # ── Example 2: New red flag surfaces mid-interview ─────
    {
        "input": json.dumps({
            "session_id": "ANM-20260609-002",
            "patient_id": "PT-34912",
            "triage_priority": "P3",
            "chief_complaint": "stomach ache for two days",
            "ehr_summary": {
                "age": 34, "sex": "F",
                "active_diagnoses": ["IBS"]
            },
            "session_history": [
                {"turn": 1, "speaker": "agent",
                 "text": "I'm sorry to hear you've been in pain. Can you point to exactly where in your stomach the pain is?"},
                {"turn": 2, "speaker": "patient",
                 "text": "It started around my belly button but now it's moved to the lower right side and it's much worse."},
                {"turn": 3, "speaker": "agent",
                 "text": "Thank you for letting me know. On a scale of 0 to 10, where 0 is no pain and 10 is the worst pain you've ever felt, how would you rate it right now?"},
                {"turn": 4, "speaker": "patient",
                 "text": "It's a 9 out of 10. It came on really suddenly in the last hour and I feel sick and hot."}
            ],
            "current_turn": 4
        }),
        "output": json.dumps({
            "session_id": "ANM-20260609-002",
            "patient_id": "PT-34912",
            "turns_completed": 4,
            "conversation_log": [
                {"turn": 1, "speaker": "agent",
                 "text": "I'm sorry to hear you've been in pain. Can you point to exactly where in your stomach the pain is?"},
                {"turn": 2, "speaker": "patient",
                 "text": "It started around my belly button but now it's moved to the lower right side and it's much worse."},
                {"turn": 3, "speaker": "agent",
                 "text": "Thank you for letting me know. On a scale of 0 to 10, where 0 is no pain and 10 is the worst pain you've ever felt, how would you rate it right now?"},
                {"turn": 4, "speaker": "patient",
                 "text": "It's a 9 out of 10. It came on really suddenly in the last hour and I feel sick and hot."}
            ],
            "socrates": {
                "site": "periumbilical_migrating_to_RLQ",
                "onset": {
                    "duration_minutes": 60,
                    "duration_description": "sudden onset in last hour, preceded by 2 days milder pain",
                    "character": "sudden",
                    "trigger": None
                },
                "character": None,
                "radiation": None,
                "associations": "nausea, fever",
                "time_course": "progressive_worsening_with_migration",
                "exacerbating_relieving": None,
                "severity_vas": 9.0
            },
            "ehr_fields_skipped": [],
            "new_red_flags": [
                "severe_pain_vas_9",
                "RLQ_migration_possible_appendicitis",
                "fever_with_acute_abdomen"
            ],
            "escalate_immediately": True,
            "interview_complete": False,
            "incomplete_dimensions": [
                "character — interview halted due to escalation",
                "radiation — interview halted due to escalation",
                "exacerbating_relieving — interview halted due to escalation"
            ]
        }, indent=2)
    },

    # ── Example 3: Complete interview — all 8 dimensions ──
    {
        "input": json.dumps({
            "session_id": "ANM-20260609-003",
            "patient_id": "PT-55201",
            "triage_priority": "P3",
            "chief_complaint": "burning sensation when urinating",
            "ehr_summary": {"age": 28, "sex": "F", "active_diagnoses": []},
            "session_history": [
                {"turn": 1, "speaker": "agent",   "text": "Where exactly do you feel the burning — inside when you urinate, or more around the outside?"},
                {"turn": 2, "speaker": "patient",  "text": "Inside, when the urine is actually coming out."},
                {"turn": 3, "speaker": "agent",   "text": "Thank you. When did this burning start, and did it come on suddenly?"},
                {"turn": 4, "speaker": "patient",  "text": "It started about two days ago, gradually got worse through the day."},
                {"turn": 5, "speaker": "agent",   "text": "I see. How would you describe the burning — is it a sharp sting, a dull ache, or something else?"},
                {"turn": 6, "speaker": "patient",  "text": "A sharp sting each time I go."},
                {"turn": 7, "speaker": "agent",   "text": "Does the pain or discomfort move or spread to your back, sides, or abdomen?"},
                {"turn": 8, "speaker": "patient",  "text": "Not really, just stays there. Maybe a little lower tummy ache."},
                {"turn": 9, "speaker": "agent",   "text": "Have you noticed any other symptoms — for example, needing to go more often than usual, or any unusual colour or smell?"},
                {"turn": 10, "speaker": "patient", "text": "Yes, I need to go very often but barely anything comes out. The urine looks a bit cloudy."},
                {"turn": 11, "speaker": "agent",  "text": "Is it there all the time or does it come and go?"},
                {"turn": 12, "speaker": "patient", "text": "Every time I try to urinate. It's been getting steadily worse since it started."}
            ],
            "current_turn": 12
        }),
        "output": json.dumps({
            "session_id": "ANM-20260609-003",
            "patient_id": "PT-55201",
            "turns_completed": 12,
            "conversation_log": [
                {"turn": 1,  "speaker": "agent",   "text": "Where exactly do you feel the burning — inside when you urinate, or more around the outside?"},
                {"turn": 2,  "speaker": "patient",  "text": "Inside, when the urine is actually coming out."},
                {"turn": 3,  "speaker": "agent",   "text": "Thank you. When did this burning start, and did it come on suddenly?"},
                {"turn": 4,  "speaker": "patient",  "text": "It started about two days ago, gradually got worse through the day."},
                {"turn": 5,  "speaker": "agent",   "text": "I see. How would you describe the burning — is it a sharp sting, a dull ache, or something else?"},
                {"turn": 6,  "speaker": "patient",  "text": "A sharp sting each time I go."},
                {"turn": 7,  "speaker": "agent",   "text": "Does the pain or discomfort move or spread to your back, sides, or abdomen?"},
                {"turn": 8,  "speaker": "patient",  "text": "Not really, just stays there. Maybe a little lower tummy ache."},
                {"turn": 9,  "speaker": "agent",   "text": "Have you noticed any other symptoms — for example, needing to go more often than usual, or any unusual colour or smell?"},
                {"turn": 10, "speaker": "patient", "text": "Yes, I need to go very often but barely anything comes out. The urine looks a bit cloudy."},
                {"turn": 11, "speaker": "agent",  "text": "Is it there all the time or does it come and go?"},
                {"turn": 12, "speaker": "patient", "text": "Every time I try to urinate. It's been getting steadily worse since it started."}
            ],
            "socrates": {
                "site": "intraurethral_with_mild_suprapubic_discomfort",
                "onset": {
                    "duration_minutes": 2880,
                    "duration_description": "2 days",
                    "character": "gradual",
                    "trigger": None
                },
                "character": "sharp_stinging",
                "radiation": "minimal_suprapubic_ache_no_flank_radiation",
                "associations": "urinary_frequency, urinary_urgency, low_volume_voiding, cloudy_urine",
                "time_course": "constant_progressive_worsening",
                "exacerbating_relieving": "worsened_on_voiding",
                "severity_vas": None
            },
            "ehr_fields_skipped": [],
            "new_red_flags": [],
            "escalate_immediately": False,
            "interview_complete": True,
            "incomplete_dimensions": [
                "severity_vas — 12-question limit reached before this dimension could be asked"
            ]
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

ANAMNESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANAMNESIS_SYSTEM_PROMPT),
    few_shot_prompt,
    ("human",  "{anamnesis_request}"),
])

parser = JsonOutputParser(pydantic_object=AnamnesisOutput)


# ============================================================
# 5. CHAIN ASSEMBLY
# ============================================================

def build_anamnesis_chain(llm):
    """
    Returns a runnable anamnesis chain.

    Usage:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        chain = build_anamnesis_chain(llm)

        result = chain.invoke({"anamnesis_request": json.dumps({
            "session_id": "ANM-001",
            "patient_id": "PT-88421",
            "triage_priority": "P1",
            "chief_complaint": "chest pain",
            "ehr_summary": {...},
            "session_history": [...],
            "current_turn": 1
        })})

    Note: temperature=0.2 recommended — allows natural conversational variation
    while keeping clinical structure consistent.
    """
    return ANAMNESIS_PROMPT | llm | parser


def run_anamnesis_turn(llm, session_id: str, patient_id: str,
                       triage_priority: str, chief_complaint: str,
                       ehr_summary: dict, session_history: list,
                       current_turn: int) -> AnamnesisOutput:
    """
    Run a single anamnesis turn and return validated output.
    Call this in a loop, passing session_history from previous output each time.
    Stop when output.interview_complete=True or output.escalate_immediately=True.
    """
    chain = build_anamnesis_chain(llm)
    request = json.dumps({
        "session_id": session_id,
        "patient_id": patient_id,
        "triage_priority": triage_priority,
        "chief_complaint": chief_complaint,
        "ehr_summary": ehr_summary,
        "session_history": session_history,
        "current_turn": current_turn
    })
    raw = chain.invoke({"anamnesis_request": request})
    return AnamnesisOutput(**raw)


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":
    print("=== JSON Schema ===")
    print(json.dumps(ANAMNESIS_JSON_SCHEMA, indent=2)[:600], "...\n")

    print("=== System prompt (first 400 chars) ===")
    print(ANAMNESIS_SYSTEM_PROMPT[:400], "...\n")

    print("=== Few-shot examples ===")
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        inp = json.loads(ex["input"])
        out = json.loads(ex["output"])
        status = "ESCALATE" if out["escalate_immediately"] else (
                 "COMPLETE" if out["interview_complete"] else "IN_PROGRESS")
        print(f"  {i}. session={inp['session_id']}  turns={out['turns_completed']}  status={status}")

    print("\n=== Pydantic validation ===")

    # Valid in-progress session
    ok = AnamnesisOutput(
        session_id="ANM-001",
        patient_id="PT-88421",
        turns_completed=3,
        conversation_log=[
            ConversationTurn(turn=1, speaker="agent", text="Where is the pain?"),
            ConversationTurn(turn=2, speaker="patient", text="In my chest."),
            ConversationTurn(turn=3, speaker="agent", text="How would you describe it?"),
        ],
        socrates=SOCRATES(site="central_chest"),
        new_red_flags=[],
        escalate_immediately=False,
        interview_complete=False,
        incomplete_dimensions=["onset","character","radiation","associations","time_course","exacerbating_relieving","severity_vas"]
    )
    print(f"  Valid in-progress session passed ✓  turns={ok.turns_completed}")

    # Red flag without escalation — should fail
    try:
        AnamnesisOutput(
            session_id="ANM-002",
            patient_id="PT-11111",
            turns_completed=2,
            conversation_log=[
                ConversationTurn(turn=1, speaker="agent", text="How are you feeling?"),
                ConversationTurn(turn=2, speaker="patient", text="I want to hurt myself."),
            ],
            socrates=SOCRATES(),
            new_red_flags=["self_harm_intent"],   # red flag present
            escalate_immediately=False,            # ← intentional violation
            interview_complete=False,
            incomplete_dimensions=[]
        )
    except ValueError as e:
        print(f"  Safety violation caught ✓  red flag without escalation blocked")

    print("\nAll checks passed.")

    print("\n=== LLM smoke test ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Skipping LLM smoke test.")
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        result = run_anamnesis_turn(
            llm,
            session_id="ANM-SMOKE-001",
            patient_id="PT-88421",
            triage_priority="P1",
            chief_complaint="central chest pain with shortness of breath",
            ehr_summary={
                "age": 67,
                "sex": "M",
                "active_diagnoses": ["Coronary artery disease", "Hypertension"],
            },
            session_history=[
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
            current_turn=2,
        )
        print(result.model_dump_json(indent=2))
