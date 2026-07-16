# Cloud Documentation Assistant

A RAG-based assistant that generates production-ready **CloudFormation (YAML)** and **Terraform (JSON)** templates from natural-language requests, validates them against real schema/deployment rules, and iteratively repairs common LLM hallucinations before returning output.

Built on **Amazon Bedrock (Nova Lite)**, **FAISS**, and **Streamlit**.

---

## Why this exists

LLMs are good at producing CloudFormation/Terraform that *looks* correct but frequently contains subtle, schema-valid-yet-wrong mistakes — properties that don't exist on a resource type, `Ref`/`GetAtt` confusion, circular dependencies, or account-level policy violations that no linter can see. This project wraps template generation in a validation + repair pipeline specifically designed around the recurring failure patterns of a single LLM backend, rather than trusting the model's first output.

---

## Features

- **Dual format support** — generates CloudFormation YAML or Terraform JSON, auto-detected from the request or inferred from conversation history
- **RAG retrieval** — pulls relevant reference templates from a FAISS index (S3-backed) to ground generation in real patterns
- **cfn-lint validation** — every CloudFormation template is validated in-process before being returned
- **Static analysis beyond linting** — custom checks catch issues that pass schema validation but fail at deploy time:
  - IAM role/trust-policy service mismatches (e.g. a `pipes.amazonaws.com` role reused as a Rule target's `RoleArn`)
  - Account-level SCP violations (instance-size ceilings, root-principal grants) — including a parameter-indirection bypass where the violation is hidden inside a `Parameters.*.Default`
  - Dead Letter Queue redrive-policy direction and `Ref`-vs-`GetAtt` correctness
  - Fabricated AWS-managed policy ARNs
  - Missing/misconfigured `AWS::Events::ApiDestination` ↔ `AWS::Events::Connection` pairs
  - Invalid `Parameters.*.Default` values (intrinsic functions where only literals are allowed)
  - Invalid `AWS::Logs::LogGroup` `RetentionInDays` enum values
- **Deterministic auto-fixes** — a library of regex-based corrections for the LLM's most common, confirmed-repeating mistakes (see below), applied before validation runs at all, rather than burning repair rounds on fixable syntax issues
- **Capped LLM repair loop** — up to 2 rounds of automated repair, feeding cfn-lint errors *and* static-check warnings back to the model, with automatic detection of a repair round that made no changes (to avoid wasted retries)
- **Post-generation improvement suggestions** — a second LLM pass reviews the final template for genuine issues, with heavy anti-hallucination filtering (see below)
- **Conversation memory** — supports "fix this," "update the template," etc. without repasting the full YAML/JSON

---

## Architecture

```
generation/template_chain.py   # core generation, validation, repair, and review pipeline
retrieval/retriever.py         # FAISS-backed retrieval of reference templates
app/streamlit_app.py           # Streamlit UI
config/config.py               # AWS profiles, model IDs, region config
```

**LLM backend:** Amazon Nova Lite (`amazon.nova-lite-v1:0`) via Bedrock — a hard, locked-in constraint of this project.

**AWS profiles:** two explicit profiles are used —`S3_PROFILE` (`eu-north-1`, for the FAISS index) and `BEDROCK_PROFILE` (`us-east-1`, for model inference).

---

## Deterministic auto-fixes

These are applied automatically, in order, before cfn-lint or any static check runs — each one exists because the underlying mistake was **confirmed to repeat** across multiple generations, not a one-off:

| Fix | What it corrects |
|---|---|
| `_quote_colon_in_descriptions` | Unquoted `Description`/`ConstraintDescription` values containing `": "`, which break YAML parsing entirely |
| `_fix_duplicate_parameter_descriptions` | Duplicate `Description:` keys inside the same Parameter block (invalid YAML) |
| `_fix_redrive_policy_casing` | PascalCase `RedrivePolicy` keys (`DeadLetterTargetArn`/`MaxReceiveCount`) that must be camelCase per the raw SQS JSON schema |
| `_fix_pipe_sqs_source_parameters` | `SqsQueue` → `SqsQueueParameters`, `BatchingWindow` → `MaximumBatchingWindowInSeconds` |
| `_fix_managed_policy_getatt_arn` | `!GetAtt <ManagedPolicyId>.Arn` (invalid — ManagedPolicy has no `Arn` attribute in `Fn::GetAtt`; rewritten to `!Ref`) |
| `_strip_unneeded_sub` | Harmless-but-lint-triggering `!Sub` wrapping a string with no `${...}` inside |
| Missing-space regexes | `Key:!Sub`, `Key:'value'` and similar malformed YAML the model emits without a space after the colon |

---

## Improvement-suggestion review

After a valid template is produced, a second LLM pass looks for further issues. Because this step has proven to hallucinate frequently, it's filtered hard before anything reaches the user:

- Every suggestion must include a **verbatim quoted line** from the actual template as evidence — fabricated evidence is discarded
- Suggestions referencing a logical ID that doesn't exist in the template are discarded
- Suggestions proposing something CloudFormation itself forbids (e.g. an intrinsic function as a Logical ID) are discarded
- Suggestions contradicting known-correct schema rules (e.g. proposing invalid `!Ref <Id>.Arn` syntax, or restructuring an EventBridge Connection incorrectly) are discarded
- Suggestions claiming a property is "missing" when it's actually present in that resource's block are discarded
- Suggestions proposing a circular dependency are discarded
- Suggestions duplicating the user's own request, or repeating a prior turn's already-addressed feedback, are discarded

This review layer is intentionally conservative — it currently has a high discard rate by design, since a missed hallucination reaching the user is worse than an over-cautious filter.

---

## Known limitations

- Improvement suggestions are surfaced for manual yes/no confirmation — nothing is auto-applied
- cfn-lint validation only covers CloudFormation; Terraform output is checked for valid JSON only, not `terraform validate`
- Some legacy encoding artifacts (mojibake in older comments) exist from a prior character-encoding mismatch and are cosmetic, not functional

---

## Requirements

- Python 3.11
- AWS credentials configured for both the S3 and Bedrock profiles referenced in `config/config.py`
- `cfn-lint` installed in the same interpreter running Streamlit

```bash
pip install -r requirements.txt --break-system-packages
```

## Running

```bash
streamlit run app/streamlit_app.py
```
