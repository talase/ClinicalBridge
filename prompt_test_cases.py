"""Structured regression cases for the four prompt-engineering agents."""


TRIAGE_INPUT = (
    "I have crushing pain in the center of my chest that spreads to my left arm. "
    "I am short of breath, sweating, and the pain started 30 minutes ago."
)

EHR_INPUT = {
    "patient_id": "PT-00000",
    "triage_pathway": "urgent_care_ED",
    "red_flags": ["high_fever", "neck_stiffness"],
}

ANAMNESIS_INPUT = {
    "session_id": "ANM-TEST-001",
    "patient_id": "PT-88421",
    "triage_priority": "P1",
    "chief_complaint": "central chest pain",
    "ehr_summary": {
        "age": 67,
        "sex": "M",
        "active_diagnoses": ["Coronary artery disease", "Hypertension"],
    },
    "session_history": [
        {
            "turn": 1,
            "speaker": "agent",
            "text": "I am sorry you are experiencing this. Where exactly is the pain?",
        },
        {
            "turn": 2,
            "speaker": "patient",
            "text": "It is in the center of my chest.",
        },
    ],
    "current_turn": 2,
}

SYNTHESIS_TRIAGE_OUTPUT = {
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
}

SYNTHESIS_EHR_OUTPUT = {
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
    "allergies": [
        {
            "substance": "Penicillin",
            "reaction": "anaphylaxis",
            "severity": "severe",
        }
    ],
    "recent_labs": {},
    "data_quality_flags": [],
    "audit_log": {
        "access_time": "2026-06-11T10:00:00Z",
        "accessed_by": "ehr_agent_v1",
    },
    "error": None,
}

SYNTHESIS_ANAMNESIS_OUTPUT = {
    "session_id": "ANM-TEST-001",
    "patient_id": "PT-88421",
    "turns_completed": 4,
    "conversation_log": [
        {
            "turn": 1,
            "speaker": "agent",
            "text": "Where exactly is the pain?",
        },
        {
            "turn": 2,
            "speaker": "patient",
            "text": "It is in the center of my chest.",
        },
        {
            "turn": 3,
            "speaker": "agent",
            "text": "Does the pain spread anywhere?",
        },
        {
            "turn": 4,
            "speaker": "patient",
            "text": "It spreads to my left arm.",
        },
    ],
    "socrates": {
        "site": "central_chest",
        "onset": {
            "duration_minutes": 30,
            "duration_description": "30 minutes",
            "character": "sudden",
            "trigger": "walking_upstairs",
        },
        "character": "crushing",
        "radiation": "left_arm",
        "associations": "shortness_of_breath and sweating",
        "time_course": "constant",
        "exacerbating_relieving": "worse_with_exertion",
        "severity_vas": 9.0,
    },
    "ehr_fields_skipped": [],
    "new_red_flags": [],
    "escalate_immediately": False,
    "interview_complete": True,
    "incomplete_dimensions": [],
}

SYNTHESIS_INPUT = {
    "triage_output": SYNTHESIS_TRIAGE_OUTPUT,
    "ehr_output": SYNTHESIS_EHR_OUTPUT,
    "anamnesis_output": SYNTHESIS_ANAMNESIS_OUTPUT,
}


PROMPT_TEST_CASES = [
    {
        "test_case_id": "TRIAGE-001",
        "agent_name": "Agent 1 Triage",
        "input": TRIAGE_INPUT,
        "expected_behavior": (
            "Recognize the combined cardiac red flags and select the emergent pathway."
        ),
        "failure_observed_in_earlier_version": (
            "Chest pain with radiation and dyspnea was under-classified as P2 instead of P1."
        ),
        "prompt_or_schema_fix_applied": (
            "Added explicit cardiac red flags, atypical MI guidance, higher-priority "
            "uncertainty handling, and P1 escalation validation."
        ),
        "expected_result_after_fix": (
            "priority=P1, pathway=emergency_dispatch, and escalate_human=true."
        ),
    },
    {
        "test_case_id": "EHR-001",
        "agent_name": "Agent 2 EHR",
        "input": EHR_INPUT,
        "expected_behavior": (
            "Return a not_found result with an error and no invented clinical record."
        ),
        "failure_observed_in_earlier_version": (
            "The prompt risked hallucinating demographics or clinical data for an unknown patient_id."
        ),
        "prompt_or_schema_fix_applied": (
            "Added a strict no-fabrication rule, explicit not_found behavior, audit logging, "
            "and validator requirements for error details."
        ),
        "expected_result_after_fix": (
            "retrieval_status=not_found, error is populated, and clinical fields are null."
        ),
    },
    {
        "test_case_id": "ANAMNESIS-001",
        "agent_name": "Agent 3 Anamnesis",
        "input": ANAMNESIS_INPUT,
        "expected_behavior": (
            "Ask no more than one focused SOCRATES question in each agent turn."
        ),
        "failure_observed_in_earlier_version": (
            "The interview initially asked multiple questions at once."
        ),
        "prompt_or_schema_fix_applied": (
            "Added the explicit ONE-question-per-turn rule and a question-selection step."
        ),
        "expected_result_after_fix": (
            "Every agent conversation turn contains at most one question."
        ),
    },
    {
        "test_case_id": "SYNTHESIS-001",
        "agent_name": "Agent 4 Synthesis",
        "input": SYNTHESIS_INPUT,
        "expected_behavior": (
            "Carry the severe Penicillin allergy into the physician summary or next steps."
        ),
        "failure_observed_in_earlier_version": (
            'The synthesis omitted the Penicillin allergy and stated "No allergies reported."'
        ),
        "prompt_or_schema_fix_applied": (
            "Added evidence-only synthesis rules, source integration, and an "
            "evidence-consistency regression check."
        ),
        "expected_result_after_fix": (
            "The output mentions Penicillin and does not claim that no allergies were reported."
        ),
    },
    {
        "test_case_id": "SYNTHESIS-002",
        "agent_name": "Agent 4 Synthesis",
        "input": SYNTHESIS_INPUT,
        "expected_behavior": (
            "Describe diagnostic possibilities with cautious, physician-review language."
        ),
        "failure_observed_in_earlier_version": (
            "The synthesis risked stating a diagnosis as a definitive fact."
        ),
        "prompt_or_schema_fix_applied": (
            "Added mandatory hedging examples, forbidden definitive phrases, and "
            "Pydantic narrative validation."
        ),
        "expected_result_after_fix": (
            "The narrative uses wording such as consistent with, suggests, "
            "cannot exclude, raises concern for, or in keeping with."
        ),
    },
]


TEST_CASES_BY_ID = {
    test_case["test_case_id"]: test_case for test_case in PROMPT_TEST_CASES
}

