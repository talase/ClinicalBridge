# Prompt Iteration Log

This log records how observed prompt failures were converted into prompt rules,
schema constraints, validators, and repeatable regression tests.

## Agent 1 Triage

| Version | Problem found | Prompt/schema change | Reason for change | Result after retesting |
|---|---|---|---|---|
| v1: basic prompt | Urgency decisions were inconsistent. | Defined P1-P4 priorities and routing pathways. | Establish a shared triage scale. | Routine and urgent cases were separated more consistently. |
| v2: JSON-only output | Free-text responses were difficult to consume downstream. | Required JSON-only output with named triage fields. | Make outputs deterministic and machine-readable. | Outputs could be parsed by the pipeline. |
| v3: safety guardrails | Chest pain could be under-classified as P2. | Added cardiac red flags, higher-priority uncertainty handling, and mandatory escalation rules. | Prevent delay for life-threatening presentations. | The chest-pain regression case routes to P1 emergency dispatch. |
| v4: Pydantic validation | Prompt rules alone could still produce contradictory fields. | Added priority-label matching, P1 escalation, and escalation-reason validators. | Enforce safety after generation. | Invalid P1 outputs are rejected. |
| v5: LLM smoke tests and evidence-consistency checks | Changes needed repeatable end-to-end verification. | Added an optional LLM smoke test and `TRIAGE-001`. | Detect prompt regressions against a known cardiac case. | Local validation passes; live testing runs when an API key is available. |

## Agent 2 EHR

| Version | Problem found | Prompt/schema change | Reason for change | Result after retesting |
|---|---|---|---|---|
| v1: basic prompt | Record extraction behavior was loosely defined. | Defined relevant EHR fields and retrieval responsibilities. | Standardize the clinical context returned downstream. | Known-record examples became structured. |
| v2: JSON-only output | Narrative EHR summaries were difficult to audit. | Required JSON-only structured extraction. | Support reliable parsing and field-level review. | Outputs could be consumed by the synthesis agent. |
| v3: safety guardrails | Unknown patient IDs risked fabricated data. | Added no-fabrication, not-found, privacy, and audit-log rules. | Protect patient safety and data integrity. | Unknown records return an explicit failure state. |
| v4: Pydantic validation | A not-found result could omit its error explanation. | Required errors for `not_found` and core fields for successful retrieval. | Prevent empty or internally inconsistent responses. | Invalid retrieval states are rejected. |
| v5: LLM smoke tests and evidence-consistency checks | Non-hallucination behavior needed regression coverage. | Added an optional LLM smoke test and `EHR-001`. | Verify that absent records remain absent. | The test checks for null clinical fields and a populated error. |

## Agent 3 Anamnesis

| Version | Problem found | Prompt/schema change | Reason for change | Result after retesting |
|---|---|---|---|---|
| v1: basic prompt | Interviews lacked a consistent history-taking structure. | Introduced the SOCRATES framework. | Ensure clinically useful symptom coverage. | Interviews captured standard symptom dimensions. |
| v2: JSON-only output | Conversation state could not be reliably resumed. | Required structured conversation logs and SOCRATES fields. | Preserve state across turns. | Session data became machine-readable. |
| v3: safety guardrails | The agent asked several questions in one turn. | Added one-question-per-turn, red-flag monitoring, and a 12-question limit. | Reduce patient burden and support safe escalation. | Retesting shows focused single-question turns. |
| v4: Pydantic validation | Red flags and turn counts could conflict with session state. | Added escalation and conversation-length validators. | Reject unsafe or malformed interview outputs. | Red flags without escalation are blocked. |
| v5: LLM smoke tests and evidence-consistency checks | Interview style needed repeatable checks. | Added an optional LLM smoke test and `ANAMNESIS-001`. | Detect regression to multi-question turns. | The runner counts question marks in each agent turn. |

## Agent 4 Synthesis

| Version | Problem found | Prompt/schema change | Reason for change | Result after retesting |
|---|---|---|---|---|
| v1: basic prompt | Summaries varied in structure and clinical caution. | Defined physician-review synthesis responsibilities. | Create a consistent pre-consultation summary. | Outputs became more clinically organized. |
| v2: JSON-only output | Narrative-only summaries were hard to validate. | Required structured risk, differential, gap, and next-step fields. | Enable automated downstream checks. | Synthesis output became parseable. |
| v3: safety guardrails | Allergies could be omitted and diagnoses could sound definitive. | Added evidence-only integration, source tracing, allergy preservation expectations, and mandatory hedging. | Prevent contradictions and overstatement. | Penicillin is retained and diagnostic language is cautious. |
| v4: Pydantic validation | High-risk alerts and narrative wording could violate prompt rules. | Added exact-header, alert, differential-rank, and forbidden-language validators. | Enforce critical constraints after generation. | Unsafe synthesis outputs are rejected. |
| v5: LLM smoke tests and evidence-consistency checks | Cross-agent evidence loss needed direct regression coverage. | Added an optional LLM smoke test plus `SYNTHESIS-001` and `SYNTHESIS-002`. | Verify allergy consistency and hedged language end to end. | The runner checks both evidence preservation and narrative wording. |

## Portfolio Evidence

Prompt iterations are documented through four linked artifacts:

- **Test cases:** structured regression inputs and expected behavior for every agent.
- **Observed failures:** concrete examples of earlier under-classification, hallucination risk,
  interview design issues, evidence omission, and overconfident language.
- **Prompt/schema fixes:** the prompt rules, JSON contracts, and validators introduced in response.
- **Successful retesting:** local validators, optional live LLM smoke tests, and simple
  evidence-consistency assertions recorded as PASS or FAIL.

Together, these artifacts show an iterative prompt-engineering process rather than a
single final prompt: identify a failure, apply a targeted control, and retest the behavior.

