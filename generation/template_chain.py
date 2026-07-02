import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from retrieval.retriever import retrieve_and_format
from config.config import OPENROUTER_API_KEY, LLM_MODEL


CLOUDFORMATION_RULES = """
You are a CloudFormation expert assistant. When generating any CloudFormation template always follow these rules without being told:

1. IAM — always use AWS::IAM::ManagedPolicy resources, never inline policies. Attach via ManagedPolicyArns on the Role directly. Never use AWS::IAM::PolicyAttachment, it is not a valid resource type.
2. Names — never hardcode resource names. Always use !Sub "${AWS::StackName}-..." for all resource names.
3. Endpoints — any URL parameter must have AllowedPattern: "^https://.*" and ConstraintDescription.
4. Parameters — every parameter must have a Description field.
5. Secrets — always use NoEcho: true for API keys, passwords, tokens.
6. SQS — AWS::SQS::Queue does not support Description property. Use Tags instead.
7. Pipes — AWS::Pipes::Pipe Source and Target must be plain !GetAtt ARN strings, not nested objects.
8. ApiDestination — use InvocationEndpoint property, not Endpoint or DestinationArn. Always include HttpMethod and InvocationRateLimitPerSecond.
9. Connection — always use ApiKeyAuthParameters structure exactly:
   AuthorizationType: API_KEY
   AuthParameters:
     ApiKeyAuthParameters:
       ApiKeyName: <ref>
       ApiKeyValue: <ref>
10. Pipes IAM — scope permissions to specific resource ARNs, never use Resource: "*". Always include secretsmanager:GetSecretValue and secretsmanager:DescribeSecret on arn:aws:secretsmanager:*:*:secret:events!connection/*
11. Scheduler — always include FlexibleTimeWindow with Mode as a quoted string "OFF":
    FlexibleTimeWindow:
      Mode: "OFF"
    IMPORTANT: Never write Mode: OFF without quotes. YAML will parse OFF as boolean false and CloudFormation will reject it.
12. Lambda — use python3.12 and boto3 by default. Read all config from os.environ.
13. Outputs — always export every output with Export: Name: !Sub "${AWS::StackName}-<name>". Include an output for every major resource created.
14. DLQ — always add a Dead Letter Queue for SQS source queues with maxReceiveCount: 3
15. Return only valid YAML — no explanations before or after the template.
16. YAML boolean trap — always quote these values when used as strings: "OFF", "ON", "YES", "NO", "TRUE", "FALSE". Without quotes YAML parses them as booleans, causing CloudFormation validation errors.
17. Required parameters — for any Parameter without a Default value, the Description must clearly state it is required and include an example. Example: Description: "Required. The endpoint URL for the Unit API. Example: https://api.example.com/notify"
18. Conversation memory — if the user refers to "the template", "this template", "the one above", "it", or asks to "fix" or "update" something without repasting the full YAML, look at the most recent template in the conversation history and modify that one. Always return the FULL corrected template, not just the changed parts.
"""


TERRAFORM_RULES = """
You are a Terraform expert assistant. When generating Terraform configuration in JSON syntax (.tf.json), always follow these rules without being told:

1. Structure — top-level keys are "terraform", "provider", "resource", "variable", "output" as needed. Output must be a single valid JSON object.
2. Provider block — always include:
   "terraform": {"required_providers": {"aws": {"source": "hashicorp/aws", "version": "~> 5.0"}}},
   "provider": {"aws": {"region": "eu-north-1"}}
3. Resource types — use correct AWS provider naming, e.g. aws_sqs_queue, aws_lambda_function, aws_iam_role, aws_iam_policy, aws_iam_role_policy_attachment, aws_scheduler_schedule, aws_pipes_pipe, aws_cloudwatch_event_api_destination, aws_cloudwatch_event_connection.
4. Naming — do not hardcode resource names; use variables where CloudFormation would use !Sub "${AWS::StackName}-...". Define a "stack_name" variable and reference it as "${var.stack_name}-..." inside "name" fields.
5. References — reference other resources using Terraform interpolation syntax as JSON strings, e.g. "${aws_sqs_queue.my_queue.arn}", never bare ARNs.
6. IAM — use aws_iam_role with a JSON-encoded assume role policy, plus separate aws_iam_policy and aws_iam_role_policy_attachment resources. Never inline broad "*" permissions; scope to specific resource ARNs.
7. Secrets/API keys — mark sensitive variables with "sensitive": true in the variable block.
8. Scheduler — aws_scheduler_schedule requires a "flexible_time_window" block with "mode": "OFF" (as a plain JSON string, not boolean).
9. Outputs — define an "output" block for every major resource created (ARNs, URLs, names).
10. DLQ — always add a dead-letter aws_sqs_queue for source queues, and configure "redrive_policy" with maxReceiveCount: 3 on the source queue.
11. Connection auth — aws_cloudwatch_event_connection auth_parameters must nest api key fields one level deeper under an "api_key" block: "auth_parameters": {"api_key": {"key": "...", "value": "..."}}. Never put "api_key_name"/"api_key_value" directly under auth_parameters.
12. API Destination — aws_cloudwatch_event_api_destination uses flat top-level attributes: "invocation_endpoint", "http_method", "invocation_rate_limit_per_second", "connection_arn". Never nest these inside a "destination_config" block; that block does not exist in this resource's schema.
13. Return only valid JSON — no explanations, no markdown fences, no comments (JSON does not support comments).
14. Conversation memory — if the user refers to "the template", "this template", "the one above", "it", or asks to "fix"/"update" without repasting it, modify the most recent Terraform JSON template in conversation history and return the FULL corrected JSON, not a partial diff.
"""


def detect_format(query, chat_history=None):
    """Detect whether the user wants Terraform JSON or CloudFormation YAML.
    Falls back to the format of the last generated template if the request
    doesn't mention a format explicitly (e.g. 'fix this')."""
    q = query.lower()
    if "terraform" in q or "tf.json" in q:
        return "terraform_json"
    if "cloudformation" in q or "cfn" in q or "yaml" in q or "yml" in q:
        return "cloudformation_yaml"

    if chat_history:
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant" and msg.get("type") == "template":
                return msg.get("format", "cloudformation_yaml")

    return "cloudformation_yaml"


def get_llm():
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=4096,
        # Discourages the model from repeating the same token over and
        # over — the main cause of the "aquifers aquifers aquifers..."
        # style degeneration seen on some free-tier models.
        model_kwargs={"frequency_penalty": 0.4}
    )
    print(f"LLM initialized: {LLM_MODEL}")
    return llm


def is_template_request(query):
    """Detect if the user is asking for a template"""
    keywords = [
        "generate", "create", "template", "cloudformation",
        "yaml", "json", "infrastructure", "iac", "stack",
        "write a", "give me a", "make a", "build a",
        "fix", "update", "modify", "change", "correct", "edit",
        "terraform"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)


def get_last_template(chat_history, fmt=None):
    """Find the most recently generated template in chat history.
    If fmt is given, only return a template that matches that format,
    so a 'fix this' request doesn't inject the wrong format's template."""
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant" and msg.get("type") == "template" and "template" in msg:
            msg_fmt = msg.get("format", "cloudformation_yaml")
            if fmt is None or msg_fmt == fmt:
                return msg["template"]
    return None


def _looks_degenerate(text):
    """Detect common free-model failure signatures: literal pad tokens,
    or the same word repeated many times in a row (a repetition loop)."""
    if not text:
        return True
    if "<pad>" in text:
        return True
    words = text.split()
    if len(words) > 20:
        run = 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                run += 1
                if run >= 5:
                    return True
            else:
                run = 1
    return False


def _strip_fences(content):
    """Strip markdown code fences regardless of language tag."""
    cleaned = content.strip()
    for fence in ("```yaml", "```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
            break
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _generate_with_retry(llm, messages, fmt, max_retries=2):
    """Call the LLM, detect degenerate output (pad tokens, repetition
    loops, invalid JSON for the terraform_json format), and retry cleanly
    instead of returning garbage to the user."""
    last_content = ""
    for attempt in range(max_retries + 1):
        response = llm.invoke(messages)
        content = response.content
        print(f"Attempt {attempt + 1} — raw response: {content[:300]}")

        if _looks_degenerate(content):
            print(f"WARNING: degenerate output detected on attempt {attempt + 1}, retrying...")
            last_content = content
            continue

        if fmt == "terraform_json":
            cleaned = _strip_fences(content)
            try:
                json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"WARNING: invalid JSON on attempt {attempt + 1}, retrying...")
                last_content = content
                continue

        return content

    print("All retries produced bad output — returning last attempt as-is.")
    return last_content


def generate_template(request, embeddings, chat_history=None):
    """Main template generation function. Supports both CloudFormation YAML
    and Terraform JSON output, auto-detected from the request or from the
    format of the last template in the conversation. Also supports
    conversation memory so users can say 'fix this' or 'update the template'
    without repasting it, and automatically retries if the model produces
    degenerate output (common on free-tier models under load)."""
    try:
        print(f"\nTemplate request: {request}")

        if chat_history is None:
            chat_history = []

        fmt = detect_format(request, chat_history)
        print(f"Detected format: {fmt}")

        # Retrieve relevant sample templates
        print("Retrieving relevant sample templates...")
        chunks, context = retrieve_and_format(request, embeddings)

        # Initialize LLM
        llm = get_llm()

        # Build messages: system rules first (format-specific ruleset)
        rules = TERRAFORM_RULES if fmt == "terraform_json" else CLOUDFORMATION_RULES
        messages = [SystemMessage(content=rules)]

        # Add recent conversation history so the model has context
        # (skip giant template code blocks except the most recent one, to save tokens)
        last_template = get_last_template(chat_history, fmt=fmt)

        for msg in chat_history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # For template messages, only include a short marker instead of
                # repeating the full template every turn (we inject the latest one separately below)
                if msg.get("type") == "template":
                    msg_fmt_label = "Terraform JSON" if msg.get("format") == "terraform_json" else "CloudFormation"
                    messages.append(AIMessage(content=f"[Generated a {msg_fmt_label} template - see below for latest version]"))
                else:
                    messages.append(AIMessage(content=msg.get("content", "")))

        fence_lang = "json" if fmt == "terraform_json" else "yaml"
        format_label = "Terraform JSON (.tf.json)" if fmt == "terraform_json" else "CloudFormation"

        if fmt == "terraform_json":
            critical_reminders = """- flexible_time_window.mode must be the plain string "OFF", not a boolean
- All resource references must use "${resource_type.name.attribute}" interpolation syntax
- Sensitive variables (API keys, tokens) must have "sensitive": true
- aws_cloudwatch_event_connection auth_parameters must nest under "api_key": {"key": ..., "value": ...}
- aws_cloudwatch_event_api_destination uses flat invocation_endpoint/http_method/invocation_rate_limit_per_second — no destination_config block
- Output must be a single valid JSON object with no trailing commas or comments"""
        else:
            critical_reminders = """- FlexibleTimeWindow Mode must be written as Mode: "OFF" with quotes, never Mode: OFF
- All YAML reserved words (OFF, ON, YES, NO, TRUE, FALSE) must be quoted when used as strings
- All parameters without defaults must have descriptions with examples
- ApiKeyValue must never be empty
- Never use AWS::IAM::PolicyAttachment"""

        # If there's a previous template, explicitly inject it as the working context
        if last_template:
            messages.append(HumanMessage(content=f"""For reference, here is the most recently generated/discussed {format_label} template:

```{fence_lang}
{last_template}
```

Sample templates for additional reference:
{context}

New request: {request}

If this request asks to fix, update, or modify "the template", apply the changes to the template shown above and return the COMPLETE corrected template, not a partial diff.

Critical reminders before you output:
{critical_reminders}

Output only valid {fence_lang.upper()}:"""))
        else:
            messages.append(HumanMessage(content=f"""Generate a {format_label} template for the following request.
Use the sample templates below as reference for patterns and structure.

Sample templates for reference:
{context}

Request: {request}

Critical reminders before you output:
{critical_reminders}

Output only valid {fence_lang.upper()}:"""))

        # Generate template, with automatic retry if the model degenerates
        print("Generating template...")
        raw_content = _generate_with_retry(llm, messages, fmt, max_retries=1)
        print(f"Final raw response: {raw_content[:500]}")

        # Clean up output (strip markdown fences regardless of language tag)
        template = _strip_fences(raw_content)

        if fmt == "terraform_json":
            # Validate it's actually parseable JSON; log a warning if not (don't crash the app)
            try:
                json.loads(template)
            except json.JSONDecodeError as je:
                print(f"WARNING: generated Terraform JSON did not parse cleanly: {je}")
        else:
            # Post-processing fix: replace unquoted OFF/ON/YES/NO in Mode fields
            template = re.sub(r'(Mode:\s+)(OFF|ON|YES|NO)(\s*$)', r'\1"\2"\3', template, flags=re.MULTILINE)

        # Extract sources
        sources = list(set([
            chunk.metadata.get("source_file", "unknown")
            for chunk in chunks
        ]))

        print("Template generated successfully")
        print(f"Sources used: {sources}")

        return {
            "request": request,
            "template": template,
            "format": fmt,
            "sources": sources,
            "chunks": chunks
        }

    except Exception as e:
        print(f"Error generating template: {e}")
        raise