import re
import json
import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config.config import BEDROCK_MODEL_ID, AWS_REGION, BEDROCK_PROFILE
from retrieval.retriever import retrieve_and_format

KNOWN_GOOD_PATTERNS = """
Reference patterns Ã¢â‚¬â€ copy this exact syntax when these resources appear:

# EventBridge Connection (no InvocationType property exists)
MyConnection:
  Type: AWS::Events::Connection
  Properties:
    Name: my-connection
    AuthorizationType: API_KEY
    AuthParameters:
      ApiKeyAuthParameters:
        ApiKeyName: key1
        ApiKeyValue: !Ref ApiKeyParam

# EventBus with logging (property is LogConfig, not LoggingConfig; IncludeDetail is a STRING with allowed values FULL/NONE, never a boolean; no LogGroupArn property exists)
MyEventBus:
  Type: AWS::Events::EventBus
  Properties:
    Name: my-bus
    KmsKeyIdentifier: alias/aws/events
    LogConfig:
      Level: ERROR
      IncludeDetail: "FULL"

# Pipe logging (property is LogConfiguration, NOT LogConfig Ã¢â‚¬â€ that's the EventBus property name, do not reuse it here. Nests under CloudwatchLogsLogDestination.LogGroupArn; IncludeExecutionData is a LIST of strings like ["ALL"], not a boolean; Level is a string. Like any other target writing to CloudWatch Logs, the log group needs a companion ResourcePolicy or logs silently never arrive.)
MyPipeLogGroup:
  Type: AWS::Logs::LogGroup
  Properties:
    LogGroupName: /aws/vendedlogs/pipes/my-pipe
MyPipeLogGroupPolicy:
  Type: AWS::Logs::ResourcePolicy
  Properties:
    PolicyName: !Sub "${AWS::StackName}-pipe-logs-policy"
    PolicyDocument: !Sub |
      {"Version":"2012-10-17","Statement":[{"Sid":"TrustPipesToStoreLogEvent","Effect":"Allow","Principal":{"Service":"pipes.amazonaws.com"},"Action":["logs:CreateLogStream","logs:PutLogEvents"],"Resource":"arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/vendedlogs/pipes/my-pipe:*"}]}
MyPipeWithLogging:
  Type: AWS::Pipes::Pipe
  DependsOn: MyPipeLogGroupPolicy
  Properties:
    DesiredState: RUNNING
    LogConfiguration:
      Level: ERROR
      IncludeExecutionData:
        - "ALL"
      CloudwatchLogsLogDestination:
        LogGroupArn: !GetAtt MyPipeLogGroup.Arn

# Any string containing ${AWS::...} pseudo parameters MUST be wrapped in !Sub, or CloudFormation leaves the literal text "${AWS::Region}" unresolved instead of substituting it
CorrectPseudoParamUsage: !Sub "arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:/example:*"

# Rule targeting CloudWatch Logs Ã¢â‚¬â€ Target has no TargetType field, and needs a companion ResourcePolicy or logs silently never arrive
MyLogGroup:
  Type: AWS::Logs::LogGroup
  Properties:
    LogGroupName: /aws/events/my-rule
MyLogGroupPolicy:
  Type: AWS::Logs::ResourcePolicy
  Properties:
    PolicyName: !Sub "${AWS::StackName}-eventbridge-logs-policy"
    PolicyDocument: !Sub |
      {"Version":"2012-10-17","Statement":[{"Sid":"TrustEventsToStoreLogEvent","Effect":"Allow","Principal":{"Service":["events.amazonaws.com","delivery.logs.amazonaws.com"]},"Action":["logs:CreateLogStream","logs:PutLogEvents"],"Resource":"arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/events/my-rule:*"}]}
MyRule:
  Type: AWS::Events::Rule
  DependsOn: MyLogGroupPolicy
  Properties:
    Targets:
      - Id: LogsTarget
        Arn: !GetAtt MyLogGroup.Arn

# Referencing a ManagedPolicy (Ref returns the ARN Ã¢â‚¬â€ GetAtt has no Arn attribute)
ManagedPolicyArns:
  - !Ref MyManagedPolicy
  # Never invent an AWS-managed policy ARN name (e.g. AmazonEventBridgeApiDestinationsRolePolicy, AWSEventsPipesFullAccess) unless you are certain it exists in the real AWS provider. If uncertain, always create a new AWS::IAM::ManagedPolicy resource scoped to only the specific actions/resources needed, rather than guessing a managed policy name.

# Pipe activation state (property is DesiredState, values RUNNING/STOPPED Ã¢â‚¬â€ not PipeState/ENABLED)
MyPipe:
  Type: AWS::Pipes::Pipe
  Properties:
    DesiredState: RUNNING
# AWS::Pipes::Pipe has NO FlexibleTimeWindow property Ã¢â‚¬â€ that belongs only to AWS::Scheduler::Schedule. Never add FlexibleTimeWindow to a Pipe resource under any circumstances.

# IAM action for invoking an API Destination (not events:PutEvents)
MyApiDestinationPolicy:
  Type: AWS::IAM::ManagedPolicy
  Properties:
    PolicyDocument:
      Version: "2012-10-17"
      Statement:
        - Effect: Allow
          Action:
            - events:InvokeApiDestination
          Resource: !GetAtt MyApiDestination.Arn

# AWS::Pipes::Pipe HttpParameters Ã¢â‚¬â€ all three sub-fields are PascalCase.
# QueryStringParameters is commonly miswritten as queryStringParameters
# (lowercase q) Ã¢â‚¬â€ this fails cfn-lint (E3002) even though the other two
# fields are usually cased correctly.
TargetParameters:
  HttpParameters:
    PathParameterValues: []
    HeaderParameters: {}
    QueryStringParameters: {}
  InputTemplate: "{\"body\": <$.body>}"

# Rule target RoleArn vs Pipe RoleArn Ã¢â‚¬â€ DO NOT reuse. A role's trust policy
# Principal must match the service that will assume it. A role trusted by
# pipes.amazonaws.com cannot be used as a Rule target's RoleArn (which needs
# events.amazonaws.com), and vice versa. Each needs its own role + policy.
MyRuleRole:
  Type: AWS::IAM::Role
  Properties:
    AssumeRolePolicyDocument:
      Version: "2012-10-17"
      Statement:
        - Effect: Allow
          Principal:
            Service: events.amazonaws.com
          Action: sts:AssumeRole
    ManagedPolicyArns:
      - !Ref MyRulePolicy

# Ref on AWS::Events::Connection returns the FULL CONNECTION ARN, never the
# connection name. Never interpolate !Ref <Connection> into a secret ARN
# string expecting a bare name Ã¢â‚¬â€ use the literal Name string instead, with
# a wildcard for the auto-generated suffix (the suffix cannot be known at
# template-authoring time):
Resource: !Sub "arn:aws:secretsmanager:${AWS::Region}:${AWS::AccountId}:secret:events!connection/MyConnectionName/*"

# Rule 32 Ã¢â‚¬â€ Ref vs GetAtt for ARNs on Rule/Pipe outputs. Ref on
# AWS::Events::Rule and AWS::Pipes::Pipe returns the resource NAME, not
# the ARN. This deploys fine and passes cfn-lint (it's not a schema
# error), but silently exports the wrong value to anyone consuming it
# via Fn::ImportValue expecting an ARN. Always use !GetAtt <LogicalId>.Arn
# for these two resource types when an ARN is needed, e.g. in Outputs:
Outputs:
  RuleArn:
    Value: !GetAtt MTMSCourseInstallationRule.Arn
  PipeArn:
    Value: !GetAtt MTMSCourseInstallationPipe.Arn
"""


CLOUDFORMATION_RULES_ONLY = """
When generating any CloudFormation template always follow these rules without being told:

1. IAM Ã¢â‚¬â€ always use AWS::IAM::ManagedPolicy resources, never inline policies. Attach via ManagedPolicyArns on the Role directly. Never use AWS::IAM::PolicyAttachment, it is not a valid resource type.
2. Names Ã¢â‚¬â€ never hardcode resource names. Always use !Sub "${AWS::StackName}-..." for all resource names.
3. Endpoints Ã¢â‚¬â€ any URL parameter must have AllowedPattern: "^https://.*" and ConstraintDescription.
4. Parameters Ã¢â‚¬â€ every parameter must have a Description field.
5. Secrets Ã¢â‚¬â€ always use NoEcho: true for API keys, passwords, tokens.
6. SQS Ã¢â‚¬â€ AWS::SQS::Queue does not support Description property. Use Tags instead.
7. Pipes Ã¢â‚¬â€ AWS::Pipes::Pipe Source and Target must be plain !GetAtt ARN strings, not nested objects.
8. ApiDestination Ã¢â‚¬â€ use InvocationEndpoint property, not Endpoint or DestinationArn. Always include HttpMethod and InvocationRateLimitPerSecond.
9. Connection Ã¢â‚¬â€ always use ApiKeyAuthParameters structure exactly:
   AuthorizationType: API_KEY
   AuthParameters:
     ApiKeyAuthParameters:
       ApiKeyName: <ref>
       ApiKeyValue: <ref>
10. Pipes IAM Ã¢â‚¬â€ scope permissions to specific resource ARNs, never use Resource: "*". Always include secretsmanager:GetSecretValue and secretsmanager:DescribeSecret on arn:aws:secretsmanager:*:*:secret:events!connection/*
11. Scheduler Ã¢â‚¬â€ always include FlexibleTimeWindow with Mode as a quoted string "OFF":
    FlexibleTimeWindow:
      Mode: "OFF"
    IMPORTANT: Never write Mode: OFF without quotes. YAML will parse OFF as boolean false and CloudFormation will reject it.
12. Lambda Ã¢â‚¬â€ use python3.12 and boto3 by default. Read all config from os.environ.
12b. Lambda Code Ã¢â‚¬â€ AWS::Lambda::Function always requires a Code property. If the user's request is infrastructure-only with no actual function logic provided, use a minimal inline placeholder:
     Code:
       ZipFile: |
         def handler(event, context):
             print(event)
             return {"statusCode": 200}
     Never omit Code entirely Ã¢â‚¬â€ cfn-lint rejects it as a required property (E3003).
12c. IAM Role policies referencing their own resource's log group Ã¢â‚¬â€ when writing a Lambda execution role's CloudWatch Logs policy, never use !Sub with ${LogicalId} where LogicalId is the Lambda function that has Role: !GetAtt <ThisRole>.Arn Ã¢â‚¬â€ that creates a circular dependency (E3004). Instead, hardcode the log group path using the same !Sub "${AWS::StackName}-..." pattern already used for the function's FunctionName, e.g. arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/lambda/${AWS::StackName}-<function-name-suffix>:* Ã¢â‚¬â€ never interpolate the function's logical ID itself.
13. Outputs Ã¢â‚¬â€ always export every output with Export: Name: !Sub "${AWS::StackName}-<name>". Include an output for every major resource created.
14. DLQ Ã¢â‚¬â€ always add a Dead Letter Queue for SQS source queues with maxReceiveCount: 3
15. Return only valid YAML Ã¢â‚¬â€ no explanations before or after the template. WRONG: "Here's the CloudFormation template you requested:\n\nAWSTemplateFormatVersion: ...". CORRECT: response starts directly with "AWSTemplateFormatVersion: '2010-09-09'".
16. YAML boolean trap Ã¢â‚¬â€ always quote these values when used as strings: "OFF", "ON", "YES", "NO", "TRUE", "FALSE". Without quotes YAML parses them as booleans, causing CloudFormation validation errors.
17. Required parameters Ã¢â‚¬â€ for any Parameter without a Default value, the Description must clearly state it is required and include an example. Example: Description: "Required. The endpoint URL for the Unit API. Example: https://api.example.com/notify"
18. Conversation memory Ã¢â‚¬â€ if the user refers to "the template", "this template", "the one above", "it", or asks to "fix" or "update" something without repasting the full YAML, look at the most recent template in the conversation history and modify that one. Always return the FULL corrected template, not just the changed parts.
19. EventBridge Ã¢â€ â€™ CloudWatch Logs targets Ã¢â‚¬â€ a Rule or Pipe target pointing at a CloudWatch Logs log group requires a companion AWS::Logs::ResourcePolicy granting events.amazonaws.com and delivery.logs.amazonaws.com permission to CreateLogStream/PutLogEvents on that log group ARN. CloudFormation cannot create this automatically (only the console does it silently) Ã¢â‚¬â€ omitting it means the rule deploys successfully but silently never delivers any logs. Never put a TargetType field on a Target object; it does not exist in the schema Ã¢â‚¬â€ a CloudWatch Logs target is identified purely by its Arn.
20. AWS::IAM::ManagedPolicy references Ã¢â‚¬â€ ManagedPolicy has no "Arn" attribute in Fn::GetAtt. Always reference it with !Ref <LogicalId> to get its ARN, never !GetAtt <LogicalId>.Arn.
21. AWS::Events::Connection has no InvocationType property Ã¢â‚¬â€ never include it. AWS::Events::EventBus encryption/logging uses KmsKeyIdentifier directly (no EncryptionType wrapper) and LogConfig with fields Level (string: OFF/ERROR/INFO/TRACE) and IncludeDetail (string: FULL/NONE Ã¢â‚¬â€ never a boolean). There is no LogGroupArn property on EventBus Ã¢â‚¬â€ EventBridge manages the log destination itself via LogConfig.
22. API Destination invocation permission Ã¢â‚¬â€ the IAM action required to invoke an API Destination is events:InvokeApiDestination. Never use events:PutEvents for this purpose.
23. AWS::Pipes::Pipe logging Ã¢â‚¬â€ the property is LogConfiguration, never LogConfig (LogConfig is the EventBus property name, do not reuse it on a Pipe). It nests under CloudwatchLogsLogDestination.LogGroupArn, IncludeExecutionData (a LIST of strings, e.g. ["ALL"], not a boolean), and Level (string). Exactly like a Rule target writing to CloudWatch Logs, the pipe's log group needs a companion AWS::Logs::ResourcePolicy granting pipes.amazonaws.com permission to CreateLogStream/PutLogEvents on that log group's ARN Ã¢â‚¬â€ never emit a ResourcePolicy with an empty or incomplete PolicyDocument (missing Principal/Action/Resource); it must always contain the full, valid statement.
24. Pseudo parameters inside string literals Ã¢â‚¬â€ any string containing ${AWS::Region}, ${AWS::AccountId}, ${AWS::StackName}, or similar must be wrapped in !Sub. A bare string like Resource: "arn:...:${AWS::Region}:..." without !Sub is invalid; CloudFormation will not substitute the variable and it will deploy with the literal, unresolved text.
25. Logical IDs Ã¢â‚¬â€ every resource's logical ID (the YAML key naming the resource, e.g. MyBucket: under Resources:) must be alphanumeric only Ã¢â‚¬â€ no underscores, no hyphens. CloudFormation logical IDs must match ^[a-zA-Z0-9]+$. Use PascalCase instead of underscores: write MTMSCourseInstallationConnection, never MTMS_Course_Installation_Connection. This applies only to the logical ID itself Ã¢â‚¬â€ the Name: property inside Properties (e.g. Name: MTMS-Course-Installation) may still contain hyphens and is unaffected.
26. Colons inside plain string values Ã¢â‚¬â€ any Description, ConstraintDescription, or other string value that contains a colon followed by a space (": "), such as "Example: https://..." or "Note: ...", must be wrapped in double quotes. YAML interprets an unquoted ": " as the start of a new mapping key even mid-sentence, which causes a template-wide parsing error.
27. AWS::Pipes::Pipe SQS source Ã¢â‚¬â€ the property is SqsQueueParameters (lowercase "qs"), never SQSQueueParameters.
28. AWS::Pipes::Pipe API Destination target Ã¢â‚¬â€ HttpParameters and InputTemplate go directly under TargetParameters with no wrapper property. There is no EventBridgeApiDestinationParameters property; it does not exist in the schema. If the user asks to pass "only the body" or similar, always include InputTemplate: "{\"body\": <$.body>}" under TargetParameters Ã¢â‚¬â€ omitting it forwards the entire event, not just the body.
29. SQS queue type Ã¢â‚¬â€ never create a FIFO queue (no .fifo suffix, no FifoQueue: true, no ContentBasedDeduplication) unless the user explicitly says "FIFO". A request for a "Standard queue" or unspecified queue type must produce a plain AWS::SQS::Queue with no FIFO properties.
30. AWS::Events::Connection Ref Ã¢â‚¬â€ Ref returns the FULL ARN, not the connection name. Never build a secret ARN string like "events!connection!${MyConnection}" expecting a name Ã¢â‚¬â€ that inserts an entire ARN into the string, producing an invalid path. Use the literal connection Name string plus a wildcard suffix instead.
31. RoleArn/Principal matching Ã¢â‚¬â€ every IAM Role handed to a resource as a RoleArn (Rule target, Pipe RoleArn, Scheduler role, etc.) must have an AssumeRolePolicyDocument Principal matching the exact service that will assume it. Never reuse one role across two different target types (e.g. a role trusted by pipes.amazonaws.com must never be assigned as a Rule target's RoleArn, which requires events.amazonaws.com). Each distinct assuming service gets its own Role + ManagedPolicy pair.
32. Ref return-value differences per resource type Ã¢â‚¬â€ ... Never assume Ref returns an ARN without checking this list first; when in doubt, use !GetAtt <LogicalId>.Arn.
33. AWS::Pipes::Pipe HttpParameters casing Ã¢â‚¬â€ under TargetParameters.HttpParameters, all three sub-fields must be exact PascalCase: PathParameterValues, HeaderParameters, QueryStringParameters. The model has repeatedly emitted "queryStringParameters" with a lowercase q Ã¢â‚¬â€ this is invalid and fails cfn-lint (E3002). Always write QueryStringParameters with a capital Q, matching PathParameterValues and HeaderParameters exactly.
34. Self-referential RedrivePolicy Ã¢â‚¬â€ a queue's RedrivePolicy.deadLetterTargetArn must never reference its own logical ID (e.g. MTMSCourseInstallationDLQ pointing RedrivePolicy at !GetAtt MTMSCourseInstallationDLQ.Arn). This creates an unresolvable circular dependency. A Dead Letter Queue itself does not need a RedrivePolicy unless the user explicitly requests a second-tier DLQ chain (DLQ -> DLQ2) Ã¢â‚¬â€ do not add one to a DLQ just because a source queue nearby has one.
35. AWS::SQS::Queue message retention property name Ã¢â‚¬â€ the property is MessageRetentionPeriod, never RetentionPeriod. "RetentionPeriod" does not exist on this resource type and fails cfn-lint (E3002). This applies to both source queues and their Dead Letter Queues.
36. Account-level SCP constraints Ã¢â‚¬â€ this account enforces two Service Control Policies that cfn-lint cannot see, so violating them produces a template that validates fine but is DENIED at deploy time:
    a) Instance sizing ceiling Ã¢â‚¬â€ AWS::EC2::Instance/LaunchConfiguration/LaunchTemplate InstanceType must be one of t*.micro/t*.small/t*.medium/t*.large only (e.g. t3.micro, t3a.small Ã¢â‚¬â€ never m5.large, t3.xlarge, or anything outside micro/small/medium/large). AWS::RDS::DBInstance DBInstanceClass must be db.t*.micro/small/medium/large only. AWS::ElastiCache::CacheCluster/ReplicationGroup CacheNodeType must be cache.t*.micro only Ã¢â‚¬â€ no other cache size is permitted.
    b) No root principal Ã¢â‚¬â€ never write a Principal that references the account root ARN (arn:aws:iam::*:root) in any trust policy, resource policy, or bucket/queue policy. This is denied unconditionally regardless of the intended permissions Ã¢â‚¬â€ always scope Principal to a specific service (e.g. events.amazonaws.com) or a specific IAM role/user ARN instead.
"""


CLOUDFORMATION_RULES = KNOWN_GOOD_PATTERNS + """
You are a CloudFormation expert assistant.

OUTPUT FORMAT Ã¢â‚¬â€ READ THIS FIRST: Your entire response must be the raw YAML template and nothing else. Do not include any preamble, explanation, summary, or closing remarks Ã¢â‚¬â€ not even a single line like "Here is the template:" or "Let me know if you need changes." The first character of your response must be the YAML content itself (e.g. "AWSTemplateFormatVersion:"), and the last character must be the final line of the template.
""" + CLOUDFORMATION_RULES_ONLY


TERRAFORM_RULES_ONLY = """
When generating Terraform configuration in JSON syntax (.tf.json), always follow these rules without being told:

1. Structure Ã¢â‚¬â€ top-level keys are "terraform", "provider", "resource", "variable", "output" as needed. Output must be a single valid JSON object.
2. Provider block Ã¢â‚¬â€ always include:
   "terraform": {"required_providers": {"aws": {"source": "hashicorp/aws", "version": "~> 5.0"}}},
   "provider": {"aws": {"region": "eu-north-1"}}
3. Resource types Ã¢â‚¬â€ use correct AWS provider naming, e.g. aws_sqs_queue, aws_lambda_function, aws_iam_role, aws_iam_policy, aws_iam_role_policy_attachment, aws_scheduler_schedule, aws_pipes_pipe, aws_cloudwatch_event_api_destination, aws_cloudwatch_event_connection.
4. Naming Ã¢â‚¬â€ do not hardcode resource names; use variables where CloudFormation would use !Sub "${AWS::StackName}-...". Define a "stack_name" variable and reference it as "${var.stack_name}-..." inside "name" fields.
5. References Ã¢â‚¬â€ reference other resources using Terraform interpolation syntax as JSON strings, e.g. "${aws_sqs_queue.my_queue.arn}", never bare ARNs.
6. IAM Ã¢â‚¬â€ use aws_iam_role with a JSON-encoded assume role policy, plus separate aws_iam_policy and aws_iam_role_policy_attachment resources. Never inline broad "*" permissions; scope to specific resource ARNs.
7. Secrets/API keys Ã¢â‚¬â€ mark sensitive variables with "sensitive": true in the variable block.
8. Scheduler Ã¢â‚¬â€ aws_scheduler_schedule requires a "flexible_time_window" block with "mode": "OFF" (as a plain JSON string, not boolean).
9. Outputs Ã¢â‚¬â€ define an "output" block for every major resource created (ARNs, URLs, names).
10. DLQ Ã¢â‚¬â€ always add a dead-letter aws_sqs_queue for source queues, and configure "redrive_policy" with maxReceiveCount: 3 on the source queue.
11. Connection auth Ã¢â‚¬â€ aws_cloudwatch_event_connection auth_parameters must nest api key fields one level deeper under an "api_key" block: "auth_parameters": {"api_key": {"key": "...", "value": "..."}}. Never put "api_key_name"/"api_key_value" directly under auth_parameters.
12. API Destination Ã¢â‚¬â€ aws_cloudwatch_event_api_destination uses flat top-level attributes: "invocation_endpoint", "http_method", "invocation_rate_limit_per_second", "connection_arn". Never nest these inside a "destination_config" block; that block does not exist in this resource's schema.
13. Return only valid JSON Ã¢â‚¬â€ no explanations, no markdown fences, no comments. WRONG: "Here's the Terraform config:\n\n```json\n{...}\n```". CORRECT: response starts directly with "{" and ends with "}".
14. Conversation memory Ã¢â‚¬â€ if the user refers to "the template", "this template", "the one above", "it", or asks to "fix"/"update" without repasting it, modify the most recent Terraform JSON template in conversation history and return the FULL corrected JSON, not a partial diff.
15. EventBridge resource types Ã¢â‚¬â€ the correct resource types are aws_cloudwatch_event_bus and aws_cloudwatch_event_rule. There is no aws_eventbridge_bus or aws_eventbridge_rule resource type in the AWS provider; using either will fail terraform validate entirely. aws_cloudwatch_event_bus does not have a "logging_config" or "cloudwatch_logs_group_arn" attribute Ã¢â‚¬â€ event bus logging in Terraform is configured via a separate aws_cloudwatch_event_bus resource with no built-in logging block; to deliver bus logs to CloudWatch, create the log group and a companion resource policy exactly as in rule 19 below.
16. jsonencode() closing syntax Ã¢â‚¬â€ every ${jsonencode({...})} interpolation must close with three characters in this exact order: the closing brace of the JSON object }, then the closing paren of jsonencode ), then the closing brace of the interpolation }. The full closing sequence is "})}" immediately before the final closing quote. A string ending in ")\"" instead of "})}\"" is missing the interpolation's closing brace and is invalid HCL/JSON Ã¢â‚¬â€ this is a common and easy mistake, so double-check every jsonencode() call in the file before returning it, not just the first one.
17. SQS queue type Ã¢â‚¬â€ Standard queues by default, ALWAYS, for BOTH the source queue AND its DLQ Ã¢â‚¬â€ a redrive relationship requires matching queue types, so if one is Standard the other must be too. Only use "fifo_queue": true and a ".fifo" name suffix on either queue if the user's request contains the literal word "FIFO" or "fifo". Before finalizing output, check every aws_sqs_queue block in the file, including DLQs: if "fifo_queue" appears anywhere and the user's original request did not contain the word "FIFO", remove it and the ".fifo" suffix from that resource.
18. API Destination invocation permission Ã¢â‚¬â€ the IAM action required to invoke an API Destination is events:InvokeApiDestination, scoped to that specific API Destination's ARN. Never use events:PutEvents for this purpose, and never use "Resource": ["*"] on any IAM policy statement in this file Ã¢â‚¬â€ every action must be scoped to the specific ARN of the resource it applies to, matching the "least-privilege" requirement whenever the user requests minimum permissions.
19. CloudWatch Logs resource policy Ã¢â‚¬â€ whenever a Rule or Pipe target writes to a CloudWatch Logs log group, or an Event Bus is configured with logging, include a companion aws_cloudwatch_log_resource_policy resource granting the correct service principal (events.amazonaws.com AND delivery.logs.amazonaws.com, as a list, for Rules/Event Buses; pipes.amazonaws.com for Pipes) permission for logs:CreateLogStream and logs:PutLogEvents on that specific log group's ARN. Follow this exact pattern, copying the bracket structure precisely:
"policy_document": "${jsonencode({Version=\"2012-10-17\",Statement=[{Effect=\"Allow\",Principal={Service=[\"events.amazonaws.com\",\"delivery.logs.amazonaws.com\"]},Action=[\"logs:CreateLogStream\",\"logs:PutLogEvents\"],Resource=[\"${aws_cloudwatch_log_group.EXAMPLE.arn}\"]}]})}"
Count the closing characters from right to left before returning: the string must end in exactly `]})}"` Ã¢â‚¬â€ one closing bracket for Resource's list, one closing brace for the Statement object, one closing bracket for the Statement list, one closing brace for the jsonencode() object, one closing paren for jsonencode(), one closing brace for the interpolation, then the quote. Omitting this resource policy means logs are silently never delivered even though the resource deploys successfully.
20. aws_cloudwatch_event_bus attributes Ã¢â‚¬â€ the KMS attribute is kms_key_identifier (not kms_key_id Ã¢â‚¬â€ that name belongs to aws_cloudwatch_log_group, a different resource). The logging block is named log_config (not logging_config), containing level and include_execution_data as direct children.
21. aws_pipes_pipe logging Ã¢â‚¬â€ the top-level attribute is log_configuration (not logging_config), containing level and include_execution_data as direct children, PLUS a nested cloudwatch_logs_log_destination block containing log_group_arn. Never place log_group_arn as a flat sibling of level/include_execution_data Ã¢â‚¬â€ it must be nested one level deeper inside cloudwatch_logs_log_destination. Follow this exact shape: "log_configuration": {"level": "ERROR", "include_execution_data": true, "cloudwatch_logs_log_destination": {"log_group_arn": "${aws_cloudwatch_log_group.EXAMPLE.arn}"}}
22. Interpolation syntax Ã¢â‚¬â€ ALWAYS use ${...} (dollar sign + curly braces). NEVER use $(...) (dollar sign + parentheses) anywhere Ã¢â‚¬â€ that is shell syntax, not Terraform, and produces a broken literal string instead of a reference. Scan every string for the exact sequence "$(" and correct it to "${" before returning, especially inside jsonencode() Resource/ARN lists.
23. aws_cloudwatch_event_api_destination Ã¢â‚¬â€ connection_arn is a REQUIRED argument. Every api_destination resource must include "connection_arn": "${aws_cloudwatch_event_connection.<name>.arn}" pointing at a connection defined in the same file. Never create an aws_cloudwatch_event_connection that nothing references.
24. EventBridge Rule with an SQS queue target Ã¢â‚¬â€ the target queue must have a companion aws_sqs_queue_policy resource granting events.amazonaws.com sqs:SendMessage permission, scoped to that queue's ARN Ã¢â‚¬â€ parallel to the CloudWatch Logs resource-policy pattern in rule 19. Omitting it means the rule fires but delivery is silently denied at runtime.
"""


TERRAFORM_RULES = """
You are a Terraform expert assistant.

OUTPUT FORMAT Ã¢â‚¬â€ READ THIS FIRST: Your entire response must be a single raw JSON object and nothing else. No preamble, no explanation, no markdown fences, no trailing commentary. The first character of your response must be "{" and the last character must be "}".
""" + TERRAFORM_RULES_ONLY


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


from langchain_aws import ChatBedrock

def get_llm():
    session = boto3.Session(profile_name=BEDROCK_PROFILE)
    llm = ChatBedrock(
        model_id=BEDROCK_MODEL_ID,
        region_name=AWS_REGION,
        client=session.client("bedrock-runtime", region_name=AWS_REGION),
        model_kwargs={"temperature": 0.4, "max_tokens": 4096},
    )
    print(f"LLM initialized: Bedrock {BEDROCK_MODEL_ID}")
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


def _extract_text(content):
    """response.content is usually a str, but some providers/LangChain
    versions return a list of content blocks instead, e.g.
    [{"type": "text", "text": "..."}], sometimes with a separate
    reasoning block. Normalize to plain string before any string
    method touches it."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in ("reasoning", "thinking"):
                    continue
                if "text" in block:
                    parts.append(block["text"])
        return "".join(parts)
    return str(content) if content is not None else ""


def _strip_think_block(content):
    content = _extract_text(content)
    if not content:
        return content
    if "<think>" in content:
        if "</think>" in content:
            content = content.split("</think>", 1)[1]
        else:
            print("WARNING: <think> block never closed Ã¢â‚¬â€ model likely hit max_tokens while reasoning")
            content = content.split("<think>", 1)[0]
    return content.strip()


def _strip_fences(content):
    """Strip a leading Qwen <think> block, then markdown code fences
    regardless of language tag, then fall back to trimming any stray
    preamble sentence Qwen adds despite instructions not to."""
    cleaned = _strip_think_block(content)

    for fence in ("```yaml", "```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
            break
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Fallback: if there's still a stray preamble line before the real
    # content (Qwen occasionally adds one despite instructions), drop
    # everything before the first line that looks like real YAML/JSON.
    lines = cleaned.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("{") or re.match(r"^[A-Za-z][\w:.\-]*:", stripped):
            if i > 0:
                print(f"WARNING: stripped {i} preamble line(s) before real content")
            cleaned = "\n".join(lines[i:])
            break

    return cleaned.strip()


def _quote_colon_in_descriptions(template):
    """Nova Lite frequently emits unquoted Description/ConstraintDescription
    values containing 'Example: ...' or 'Note: ...' (rule 26), which breaks
    YAML parsing entirely ('mapping values are not allowed here') before
    cfn-lint or any static check ever runs. Auto-quote any such line that
    isn't already quoted, mirroring the other auto-correction regexes below
    rather than relying solely on the model to follow rule 26."""
    def fix_line(m):
        indent, key, value = m.group(1), m.group(2), m.group(3)
        value = value.rstrip()
        if not value:
            return m.group(0)
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return m.group(0)
        if ": " not in value:
            return m.group(0)
        escaped = value.replace('"', '\\"')
        return f'{indent}{key}: "{escaped}"'

    return re.sub(r'^(\s*)(Description|ConstraintDescription):[ \t]+(.+)$', fix_line, template, flags=re.MULTILINE)


def _fix_redrive_policy_casing(template):
    """RedrivePolicy is one of the few CloudFormation properties that's a raw
    JSON blob mirroring the underlying SQS API directly, rather than a
    normal PascalCase-modeled property. Nova Lite consistently emits
    PascalCase keys (DeadLetterTargetArn, MaxReceiveCount) here, which
    cfn-lint rejects (E3002) even though PascalCase is the norm everywhere
    else in the template. Deterministically lowercase these two specific
    keys rather than relying on the model to get this inconsistency right."""
    template = re.sub(r'\bDeadLetterTargetArn\b', 'deadLetterTargetArn', template)
    template = re.sub(r'\bMaxReceiveCount\b', 'maxReceiveCount', template)
    return template


def _fix_pipe_sqs_source_parameters(template):
    """AWS::Pipes::Pipe SourceParameters for an SQS source must use the key
    SqsQueueParameters, never SqsQueue (rule 27's cousin bug Ã¢â‚¬â€ the model
    reliably gets the property name wrong here). The batching-window field
    inside it is MaximumBatchingWindowInSeconds, never BatchingWindow.
    Fix both deterministically rather than depending on repair rounds,
    which have proven unreliable for this exact pair of errors."""
    template = re.sub(r'^(\s*)SqsQueue:\s*$', r'\1SqsQueueParameters:', template, flags=re.MULTILINE)
    template = re.sub(r'^(\s*)BatchingWindow(\s*:)', r'\1MaximumBatchingWindowInSeconds\2', template, flags=re.MULTILINE)
    return template


def _strip_unneeded_sub(template):
    """Fn::Sub with no ${...} inside is valid CloudFormation but triggers
    cfn-lint W1020. Deterministically strip it instead of burning an LLM
    repair round on it Ã¢â‚¬â€ this is exactly the harmless case seen when a
    static JSON payload (e.g. a Scheduler Target's Input) got wrapped in
    !Sub unnecessarily. Only touches multi-line block scalars (!Sub |),
    since that's the pattern this has actually occurred in."""
    def fix_block(m):
        indent, content = m.group(1), m.group(2)
        if "${" in content:
            return m.group(0)
        return f"{indent}|\n{content}"
    return re.sub(r'(\s*)!Sub \|\n((?:\1  .*\n?)+)', fix_block, template)


def _validate_cloudformation(template):
    """Run cfn-lint against generated YAML. Returns (is_valid, error_text, lint_ran).

    Uses cfnlint's Python API (cfnlint.api.lint) directly in-process rather than
    shelling out via subprocess. This guarantees it runs against the exact same
    interpreter/venv that's running Streamlit, with no dependence on PATH or on
    which venv happens to be "active" in the shell that launched the app Ã¢â‚¬â€ that
    mismatch (two different venv311 folders resolving inconsistently) was the
    root cause of validation being silently skipped before.
    """
    try:
        import cfnlint.api as cfnlint_api
    except ImportError:
        print("WARNING: cfnlint not installed in this interpreter Ã¢â‚¬â€ skipping validation. Run: pip install cfn-lint")
        return True, "", False

    try:
        matches = cfnlint_api.lint(template)
    except Exception as e:
        print(f"WARNING: cfn-lint raised an unexpected error Ã¢â‚¬â€ skipping validation: {e}")
        return True, "", False

    if not matches:
        return True, "", True

    error_text = "\n".join(str(m) for m in matches)
    return False, error_text, True

def _strip_cfn_intrinsic_tags(template):
    """yaml.safe_load can't parse short-form CloudFormation intrinsic tags
    (!Ref, !GetAtt, !Sub, !Join, !If, !Select, !Split, etc.) without a custom
    constructor. Rather than hardcoding an incomplete list Ã¢â‚¬â€ which silently
    breaks parsing (and therefore skips ALL downstream static checks: role
    mismatch, SCP compliance) whenever Nova Lite uses a tag not in that list Ã¢â‚¬â€
    strip any bare '!Word' tag generically with a regex."""
    return re.sub(r'!\w+\b', '', template)


def _extract_logical_id(ref_str):
    """Pull the logical ID out of a stripped '!GetAtt Foo.Arn' -> 'Foo.Arn' string
    after the yaml.safe_load strip in _check_role_principal_mismatch turned it
    into a plain string."""
    if isinstance(ref_str, str):
        return ref_str.split(".")[0].strip()
    return None


def _check_role_principal_mismatch(template):
    """Static check: for every 'RoleArn: !GetAtt X.Arn' assignment, find the
    role X's AssumeRolePolicyDocument Principal.Service and flag if the
    role is reused across resources that imply different assuming services
    (Rule targets need events.amazonaws.com; Pipes need pipes.amazonaws.com;
    Scheduler needs scheduler.amazonaws.com). Returns a list of warning
    strings (empty if none found) to feed into the repair prompt alongside
    cfn-lint errors Ã¢â‚¬â€ this catches cross-service role reuse that cfn-lint's
    schema validation can't see because it's a semantic/deployment-time
    error, not a schema violation."""
    import yaml as pyyaml
    try:
        doc = pyyaml.safe_load(_strip_cfn_intrinsic_tags(template))
    except Exception as e:
        print(f"WARNING: role-principal check skipped, template did not parse after tag-stripping: {e}")
        return []  # don't block on parse issues; cfn-lint already covers syntax

    if not isinstance(doc, dict):
        return []

    resources = doc.get("Resources", {}) or {}
    role_services = {}
    for name, res in resources.items():
        if not isinstance(res, dict):
            continue
        if res.get("Type") == "AWS::IAM::Role":
            stmts = res.get("Properties", {}).get("AssumeRolePolicyDocument", {}).get("Statement", [])
            if isinstance(stmts, dict):
                stmts = [stmts]
            for s in stmts:
                if not isinstance(s, dict):
                    continue
                svc = s.get("Principal", {}).get("Service") if isinstance(s.get("Principal"), dict) else None
                if svc:
                    role_services[name] = svc if isinstance(svc, list) else [svc]

    expected_service = {
        "AWS::Pipes::Pipe": "pipes.amazonaws.com",
        "AWS::Scheduler::Schedule": "scheduler.amazonaws.com",
    }
    warnings = []
    for name, res in resources.items():
        if not isinstance(res, dict):
            continue
        rtype = res.get("Type")
        props = res.get("Properties", {}) or {}

        # Direct RoleArn on Pipe/Scheduler
        if rtype in expected_service and "RoleArn" in props:
            role_name = _extract_logical_id(props["RoleArn"])
            if role_name in role_services and expected_service[rtype] not in role_services[role_name]:
                warnings.append(
                    f"{name} ({rtype}) uses RoleArn pointing to {role_name}, whose trust policy allows "
                    f"{role_services[role_name]}, not {expected_service[rtype]}. Create a separate Role "
                    f"trusted by {expected_service[rtype]} instead of reusing {role_name}."
                )

        # Rule targets
        if rtype == "AWS::Events::Rule":
            targets = props.get("Targets", [])
            if isinstance(targets, list):
                for t in targets:
                    if not isinstance(t, dict):
                        continue
                    if "RoleArn" in t:
                        role_name = _extract_logical_id(t["RoleArn"])
                        if role_name in role_services and "events.amazonaws.com" not in role_services[role_name]:
                            warnings.append(
                                f"{name} target {t.get('Id')} uses RoleArn pointing to {role_name}, whose trust "
                                f"policy allows {role_services[role_name]}, not events.amazonaws.com. Create a "
                                f"separate Role trusted by events.amazonaws.com instead of reusing {role_name}."
                            )
    return warnings


def _check_scp_compliance(template):
    """Static check mirroring the two account-level SCPs enforced on this
    account: (1) an EC2/RDS/ElastiCache instance-size ceiling Ã¢â‚¬â€ only
    t*.micro/small/medium/large (db.t*... for RDS, cache.t*.micro for
    ElastiCache) are allowed; anything else passes cfn-lint but is denied
    at the account level. (2) no Principal may ever be an account root ARN
    (arn:aws:iam::*:root) Ã¢â‚¬â€ also denied account-wide regardless of what the
    template's own IAM logic intends. Returns a list of warning strings
    (empty if none found), fed into the same repair loop as
    _check_role_principal_mismatch."""
    import yaml as pyyaml
    try:
        doc = pyyaml.safe_load(_strip_cfn_intrinsic_tags(template))
    except Exception as e:
        print(f"WARNING: SCP compliance check skipped, template did not parse after tag-stripping: {e}")
        return []

    if not isinstance(doc, dict):
        return []

    resources = doc.get("Resources", {}) or {}
    warnings = []

    EC2_ALLOWED = ["t*.micro", "t*.small", "t*.medium", "t*.large"]
    RDS_ALLOWED = ["db.t*.micro", "db.t*.small", "db.t*.medium", "db.t*.large"]
    CACHE_ALLOWED = ["cache.t*.micro"]

    def wildcard_match(pattern, value):
        if not isinstance(value, str):
            return False
        regex = "^" + re.escape(pattern).replace(r'\*', '.*') + "$"
        return bool(re.match(regex, value, re.IGNORECASE))

    def any_match(patterns, value):
        return any(wildcard_match(p, value) for p in patterns)

    def _principal_is_unsafe(principal_val):
        """A Principal is unsafe if it's an account root ARN, or a bare
        wildcard ('*' or ['*']) granting public access. Handles the
        'AWS' sub-key format (Principal: {AWS: ...}) as well as a bare
        Principal: '*' string."""
        if isinstance(principal_val, dict):
            aws_val = principal_val.get("AWS")
            values = aws_val if isinstance(aws_val, list) else [aws_val]
        else:
            values = [principal_val]

        for v in values:
            if not isinstance(v, str):
                continue
            if v.strip() == "*":
                return True
            if "root" in v.lower() and "arn:aws:iam" in v.lower():
                return True
        return False

    def _principal_ref_looks_unsafe(ref_name):
        """A Principal: {AWS: !Ref X} where X wasn't caught by the direct
        ARN/wildcard check might still be a parameterized root-grant bypass Ã¢â‚¬â€
        X's own name or Description strongly implying it's meant to hold a
        root ARN at deploy time. Check the Parameter's metadata, not just its
        Default (a required parameter has no Default to inspect)."""
        if not isinstance(ref_name, str):
            return False
        param = (doc.get("Parameters", {}) or {}).get(ref_name)
        if not isinstance(param, dict):
            return False
        haystack = (ref_name + " " + str(param.get("Description", ""))).lower()
        return "root" in haystack and ("arn:aws:iam" in haystack or "account root" in haystack)

    def find_root_principal(obj):
        found = False
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "Principal":
                    if _principal_is_unsafe(v):
                        found = True
                    aws_val = v.get("AWS") if isinstance(v, dict) else v
                    ref_candidates = aws_val if isinstance(aws_val, list) else [aws_val]
                    for rc in ref_candidates:
                        if _principal_ref_looks_unsafe(rc):
                            found = True
                if find_root_principal(v):
                    found = True
        elif isinstance(obj, list):
            for item in obj:
                if find_root_principal(item):
                    found = True
        return found

    def find_root_principal_in_parameter_defaults(res_name, res):
        """Catches the parameter-indirection bypass: a PolicyDocument
        that isn't inline but is instead '!Ref SomeParameter', where
        that parameter's Default value is a JSON string containing an
        unsafe Principal. After tag-stripping, '!Ref SomeParameter'
        becomes the bare string 'SomeParameter' in the parsed doc."""
        props = res.get("Properties", {}) or {}
        policy_doc = props.get("PolicyDocument")
        if not isinstance(policy_doc, str):
            return None

        param = (doc.get("Parameters", {}) or {}).get(policy_doc)
        if not isinstance(param, dict):
            return None

        default_val = param.get("Default")
        if not isinstance(default_val, str):
            return None

        try:
            parsed = json.loads(default_val)
        except (json.JSONDecodeError, TypeError):
            return None

        if find_root_principal(parsed):
            return (
                f"{res_name} references PolicyDocument: !Ref {policy_doc}, and that parameter's "
                f"Default value contains a policy with an unsafe Principal (root ARN or wildcard '*'). "
                f"Moving the policy into a parameter does not exempt it from the account SCP Ã¢â‚¬â€ inline "
                f"the PolicyDocument directly on the resource with a scoped, non-wildcard Principal instead."
            )
        return None

    def _resolve_param_values(value):
        """After tag-stripping, '!Ref X' becomes the bare string 'X'. If X is
        actually a Parameter's logical ID, resolve to what that parameter can
        really be (its AllowedValues, or its Default) instead of treating the
        parameter's own name as the value Ã¢â‚¬â€ otherwise every !Ref'd
        InstanceType/DBInstanceClass/CacheNodeType false-positives here.
        Returns None if the parameter is required with no Default/AllowedValues
        (can't be verified statically Ã¢â‚¬â€ better to skip than false-positive)."""
        if not isinstance(value, str):
            return [value]
        params = doc.get("Parameters", {}) or {}
        param = params.get(value)
        if not isinstance(param, dict):
            return [value]
        allowed = param.get("AllowedValues")
        if isinstance(allowed, list) and allowed:
            return allowed
        default = param.get("Default")
        if default is not None:
            return [default]
        return None

    for name, res in resources.items():
        if not isinstance(res, dict):
            continue
        rtype = res.get("Type")
        props = res.get("Properties", {}) or {}

        if rtype in ("AWS::EC2::Instance", "AWS::AutoScaling::LaunchConfiguration", "AWS::EC2::LaunchTemplate"):
            itype = props.get("InstanceType")
            if itype is None and rtype == "AWS::EC2::LaunchTemplate":
                itype = props.get("LaunchTemplateData", {}).get("InstanceType")
            if itype:
                candidates = _resolve_param_values(itype)
                if candidates is not None:
                    bad = [v for v in candidates if not any_match(EC2_ALLOWED, v)]
                    if bad:
                        warnings.append(
                            f"{name} ({rtype}) InstanceType resolves to {bad}, which violates the account SCP "
                            f"restricting EC2 instances to t*.micro/small/medium/large only. This passes cfn-lint "
                            f"but is DENIED at deploy time Ã¢â‚¬â€ use one of the allowed sizes instead."
                        )

        if rtype == "AWS::RDS::DBInstance":
            dclass = props.get("DBInstanceClass")
            if dclass:
                candidates = _resolve_param_values(dclass)
                if candidates is not None:
                    bad = [v for v in candidates if not any_match(RDS_ALLOWED, v)]
                    if bad:
                        warnings.append(
                            f"{name} (AWS::RDS::DBInstance) DBInstanceClass resolves to {bad}, which violates the "
                            f"account SCP restricting RDS instances to db.t*.micro/small/medium/large only. This "
                            f"passes cfn-lint but is DENIED at deploy time Ã¢â‚¬â€ use one of the allowed sizes instead."
                        )

        if rtype in ("AWS::ElastiCache::CacheCluster", "AWS::ElastiCache::ReplicationGroup"):
            ntype = props.get("CacheNodeType")
            if ntype:
                candidates = _resolve_param_values(ntype)
                if candidates is not None:
                    bad = [v for v in candidates if not any_match(CACHE_ALLOWED, v)]
                    if bad:
                        warnings.append(
                            f"{name} ({rtype}) CacheNodeType resolves to {bad}, which violates the account SCP "
                            f"restricting ElastiCache nodes to cache.t*.micro only. This passes cfn-lint but is "
                            f"DENIED at deploy time Ã¢â‚¬â€ use cache.t*.micro instead."
                        )

        if find_root_principal(res):
            warnings.append(
                f"{name} ({rtype}) has a Principal referencing the account root ARN (arn:aws:iam::*:root) "
                f"or a public wildcard ('*'). The account SCP denies unrestricted root/public access "
                f"unconditionally Ã¢â‚¬â€ remove this and use a specific service or IAM role/user principal instead."
            )

        param_warning = find_root_principal_in_parameter_defaults(name, res)
        if param_warning:
            warnings.append(param_warning)

    return warnings


def _check_dlq_redrive_direction(template):
    """Static check for a recurring, confirmed-twice bug: Nova Lite puts
    RedrivePolicy on the DLQ itself (pointing back at the main queue) instead
    of on the main/source queue (pointing at the DLQ) Ã¢â‚¬â€ backwards. Also
    catches the companion bug: deadLetterTargetArn using !Ref (returns the
    queue URL) instead of !GetAtt <Id>.Arn (returns the ARN) Ã¢â‚¬â€ schema-valid,
    so cfn-lint never flags it, but semantically wrong at deploy time.
    A queue is inferred to be a DLQ if its logical ID or QueueName contains
    'DLQ' or 'DeadLetter' (case-insensitive) Ã¢â‚¬â€ same heuristic a human reader
    would use, since CloudFormation has no formal DLQ resource type."""
    import yaml as pyyaml
    try:
        doc = pyyaml.safe_load(_strip_cfn_intrinsic_tags(template))
    except Exception as e:
        print(f"WARNING: DLQ redrive-direction check skipped, template did not parse: {e}")
        return []

    if not isinstance(doc, dict):
        return []

    resources = doc.get("Resources", {}) or {}
    warnings = []

    def looks_like_dlq(logical_id, res):
        name_field = str((res.get("Properties", {}) or {}).get("QueueName", ""))
        haystack = (logical_id + " " + name_field).lower()
        return "dlq" in haystack or "deadletter" in haystack or "dead-letter" in haystack

    def extract_target_id(deadletter_val, raw_template):
        """After tag-stripping, both !Ref X and !GetAtt X.Arn collapse to
        bare strings Ã¢â‚¬â€ 'X' and 'X.Arn' respectively. Re-check the RAW
        (untouched) template text to tell which tag was actually used,
        since that distinction is exactly what we need to flag."""
        if not isinstance(deadletter_val, str):
            return None, None
        target_id = deadletter_val.split(".")[0].strip()
        ref_pattern = re.search(rf'!Ref\s+{re.escape(target_id)}\b', raw_template)
        getatt_arn_pattern = re.search(rf'!GetAtt\s+{re.escape(target_id)}\.Arn\b', raw_template)
        if getatt_arn_pattern:
            return target_id, "GetAtt.Arn"
        if ref_pattern:
            return target_id, "Ref"
        return target_id, "unknown"

    for name, res in resources.items():
        if not isinstance(res, dict) or res.get("Type") != "AWS::SQS::Queue":
            continue
        props = res.get("Properties", {}) or {}
        redrive = props.get("RedrivePolicy")
        if not isinstance(redrive, dict):
            continue

        target_id, ref_style = extract_target_id(redrive.get("deadLetterTargetArn"), template)

        if looks_like_dlq(name, res):
            warnings.append(
                f"{name} appears to be a Dead Letter Queue (based on its name), but it has its own "
                f"RedrivePolicy pointing at {target_id or 'another queue'}. RedrivePolicy belongs on the "
                f"SOURCE/main queue (pointing at the DLQ), not on the DLQ itself. Remove RedrivePolicy from "
                f"{name} entirely, and instead add it to the main queue that should redrive INTO {name}."
            )
        elif ref_style == "Ref":
            warnings.append(
                f"{name}'s RedrivePolicy.deadLetterTargetArn uses !Ref {target_id}, which returns the queue "
                f"URL, not its ARN. deadLetterTargetArn requires an ARN Ã¢â‚¬â€ change this to "
                f"!GetAtt {target_id}.Arn instead. This is schema-valid so cfn-lint will not catch it, but it "
                f"silently misconfigures the redrive at deploy time."
            )

    return warnings

def _check_fabricated_managed_policy_arns(template):
    """Static check for a recurring hallucination: Nova Lite building an
    AWS-managed policy ARN using !Sub "${AWS::AccountId}" instead of the
    literal 'aws' account field. AWS-managed policies (AWS*/Amazon*-prefixed,
    e.g. AWSLambdaBasicExecutionRole) always live under
    arn:aws:iam::aws:policy/..., never under the deploying account's ID.
    Using AccountId here produces an ARN that will never resolve at deploy
    time (IAM will reject it as a policy that doesn't exist in this
    account), but it's syntactically valid YAML so cfn-lint's schema
    validation never catches it."""
    warnings = []
    pattern = re.compile(r'arn:aws:iam::\$\{AWS::AccountId\}:policy/(?:service-role/)?((?:AWS|Amazon)[A-Za-z0-9]+)')
    for m in pattern.finditer(template):
        policy_name = m.group(1)
        warnings.append(
            f"Found a ManagedPolicyArns entry referencing '{policy_name}' via "
            f"arn:aws:iam::${{AWS::AccountId}}:policy/... Ã¢â‚¬â€ this is an AWS-managed "
            f"policy (name starts with AWS/Amazon), which always lives under "
            f"arn:aws:iam::aws:policy/..., not the deploying account's ID. Using "
            f"AccountId here produces an ARN that does not exist and will fail at "
            f"deploy time, even though it passes cfn-lint. Change the account "
            f"segment from ${{AWS::AccountId}} to the literal string 'aws', e.g. "
            f"arn:aws:iam::aws:policy/service-role/{policy_name}."
        )
    return warnings

def _check_api_destination_missing_connection(template):
    """Static check for a confirmed generation failure: Nova Lite emitting an
    AWS::Events::ApiDestination with ApiKeyAuthParameters directly on it (a
    property that does not exist on this resource type -- it belongs on
    AWS::Events::Connection per KNOWN_GOOD_PATTERNS) and/or omitting the
    REQUIRED ConnectionArn property entirely, with no companion
    AWS::Events::Connection resource anywhere in the template."""
    import yaml as pyyaml
    try:
        doc = pyyaml.safe_load(_strip_cfn_intrinsic_tags(template))
    except Exception as e:
        print(f"WARNING: ApiDestination-connection check skipped, template did not parse after tag-stripping: {e}")
        return []

    if not isinstance(doc, dict):
        return []

    resources = doc.get("Resources", {}) or {}
    connection_logical_ids = {
        name for name, res in resources.items()
        if isinstance(res, dict) and res.get("Type") == "AWS::Events::Connection"
    }

    warnings = []
    for name, res in resources.items():
        if not isinstance(res, dict) or res.get("Type") != "AWS::Events::ApiDestination":
            continue
        props = res.get("Properties", {}) or {}

        if "ApiKeyAuthParameters" in props or "AuthorizationType" in props:
            warnings.append(
                f"{name} (AWS::Events::ApiDestination) has ApiKeyAuthParameters/AuthorizationType "
                f"directly on it. These properties do not exist on AWS::Events::ApiDestination -- they "
                f"belong on a separate AWS::Events::Connection resource. Create an AWS::Events::Connection "
                f"with AuthorizationType: API_KEY and AuthParameters.ApiKeyAuthParameters, then reference "
                f"it from {name} via ConnectionArn: !Ref <ConnectionLogicalId>. Remove the auth properties "
                f"from {name} itself."
            )

        connection_arn = props.get("ConnectionArn")
        if not connection_arn:
            warnings.append(
                f"{name} (AWS::Events::ApiDestination) is missing the REQUIRED ConnectionArn property. "
                f"Every ApiDestination must reference an AWS::Events::Connection resource via "
                f"ConnectionArn: !Ref <ConnectionLogicalId>. Create the Connection resource if none exists "
                f"and add this property -- cfn-lint rejects an ApiDestination without it (E3003)."
            )
        elif not connection_logical_ids:
            warnings.append(
                f"{name} (AWS::Events::ApiDestination) references ConnectionArn: {connection_arn}, but no "
                f"AWS::Events::Connection resource exists anywhere in this template. Create the missing "
                f"Connection resource with the appropriate AuthorizationType and AuthParameters."
            )

    return warnings


def _check_invalid_parameter_defaults(template):
    """CloudFormation Parameters.*.Default must be a literal scalar Ã¢â‚¬â€ no
    intrinsic functions (!Sub, !Ref, !GetAtt, etc.) allowed. Nova Lite has
    emitted Default: !Sub "${AWS::StackName}-bucket" (fails cfn-lint E2001
    + two W1030s) Ã¢â‚¬â€ catch it on raw template text before tag-stripping so
    it feeds into the same repair loop as SCP/role checks."""
    warnings = []
    in_parameters = False
    current_param = None
    for line in template.split("\n"):
        if re.match(r'^Parameters:\s*$', line):
            in_parameters = True
            continue
        if in_parameters and re.match(r'^[A-Za-z]', line):
            in_parameters = False
        if not in_parameters:
            continue
        m_param = re.match(r'^  ([A-Za-z0-9]+):\s*$', line)
        if m_param:
            current_param = m_param.group(1)
            continue
        m_default = re.match(r'^\s*Default:\s*(!\w+\b.*)$', line)
        if m_default and current_param:
            warnings.append(
                f"Parameter {current_param} has Default: {m_default.group(1).strip()}, which uses an "
                f"intrinsic function. Parameters.Default only accepts a literal string/number Ã¢â‚¬â€ never "
                f"!Sub, !Ref, !GetAtt, or any other tag. Remove the Default line (or use a literal value) "
                f"and apply the intrinsic function directly on the resource property that needs it instead."
            )
    return warnings

def _suggestion_claims_missing_but_present(text, logical_id, template):
    """Reject suggestions that claim something is absent ('does not have',
    'is missing', 'lacks', etc.) when it's actually present in that
    resource's own block. Nova Lite has repeatedly asserted a property is
    missing when it's plainly in the template text (e.g. claiming
    RequestParameters lacks an Authorization entry when it's right there,
    or that an IAM role has no CloudWatch Logs permission when it already
    has AWSLambdaBasicExecutionRole attached). Scope the check to the
    claimed resource's own block rather than the whole template, so a
    property with the same name on a DIFFERENT resource doesn't cause a
    false negative here."""
    lower = text.lower()
    negative_markers = [
        "does not have", "doesn't have", "does not include", "doesn't include",
        "is missing", "lacks", "missing the", "is empty", "block is empty",
    ]
    if not any(m in lower for m in negative_markers):
        return False

    block_match = re.search(
        rf'^  {re.escape(logical_id)}:\s*\n((?:(?!^  [A-Za-z0-9]+:\s*$).)*)',
        template, flags=re.MULTILINE | re.DOTALL
    )
    resource_block = block_match.group(0) if block_match else template

    # If the suggestion quotes a specific property/section name and that
    # exact name is already present in this resource's own block, the
    # "missing" claim is directly contradicted.
    quoted = re.findall(r'`([^`]+)`', text) + re.findall(r'"([^"]+)"', text)
    for q in quoted:
        if q and not q.startswith("$") and q in resource_block:
            return True

    # NEW: also catch unquoted property names, e.g. "is missing the
    # StatusCode" or "should include BurstLimit and RateLimit" Ã¢â‚¬â€ Nova Lite
    # frequently names the property directly without quoting it. Extract
    # CamelCase-style tokens (AWS property names are always PascalCase)
    # mentioned in the suggestion and check each against the resource
    # block directly.
    camel_tokens = re.findall(r'\b[A-Z][a-zA-Z]{2,}[A-Z][a-zA-Z]*\b', text)
    for token in camel_tokens:
        if re.search(rf'{re.escape(token)}\s*:', resource_block):
            return True

    # Special case: claims that an IAM role lacks CloudWatch Logs
    # permission, when it already has the AWS-managed basic-execution
    # policy (which grants exactly that) or an inline logs: statement.
    if "cloudwatch" in lower and "log" in lower and ("polic" in lower or "permission" in lower):
        if "LambdaBasicExecutionRole" in resource_block or re.search(r'logs:(CreateLogGroup|CreateLogStream|PutLogEvents)', resource_block):
            return True

    return False


def _suggestion_evidence_contradicts_issue(evidence, desc):
    """Catch the case where the model quotes a REAL line from the template
    (passes the verbatim EVIDENCE check) but then asserts something false
    about it â€” e.g. calling a !Ref/!Sub-based value 'hardcoded', or saying
    a property 'should specify X' when the quoted evidence already contains
    X. This is a self-contradiction detectable purely from the evidence and
    issue text themselves, no template lookup needed."""
    desc_lower = desc.lower()
    evidence_lower = evidence.lower()

    if "hardcod" in desc_lower and ("!ref" in evidence_lower or "!sub" in evidence_lower):
        return True

    should_specify = re.search(r'should (?:specify|use|include|reference)\s+["\']?([\w.:/-]{3,})["\']?', desc, re.IGNORECASE)
    if should_specify:
        claimed_value = should_specify.group(1).strip('."\'')
        if claimed_value and claimed_value.lower() in evidence_lower:
            return True

    return False

def _suggestion_evidence_contradicts_issue(evidence, desc):
    """Catch the case where the model quotes a REAL line from the template
    (passes the verbatim EVIDENCE check) but then asserts something false
    about it â€” e.g. calling a !Ref/!Sub-based value 'hardcoded', or saying
    a property 'should specify X' when the quoted evidence already contains
    X. This is a self-contradiction detectable purely from the evidence and
    issue text themselves, no template lookup needed."""
    desc_lower = desc.lower()
    evidence_lower = evidence.lower()

    if "hardcod" in desc_lower and ("!ref" in evidence_lower or "!sub" in evidence_lower):
        return True

    should_specify = re.search(r'should (?:specify|use|include|reference)\s+["\']?([\w.:/-]{3,})["\']?', desc, re.IGNORECASE)
    if should_specify:
        claimed_value = should_specify.group(1).strip('."\'')
        if claimed_value and claimed_value.lower() in evidence_lower:
            return True

    return False


def _suggestion_proposes_circular_dependency(logical_id, desc, template):
    """Reject suggestions that propose making `logical_id` depend on another
    resource, when that other resource ALREADY depends on `logical_id` (via
    DependsOn, !Ref, or !GetAtt) in the current template. Applying such a
    suggestion â€” e.g. 'ApiGatewayDeployment should depend on ApiGatewayStage'
    when Stage already depends on Deployment via DeploymentId: !Ref
    ApiGatewayDeployment â€” creates a circular dependency, which
    CloudFormation rejects outright at deploy time. This is a deterministic
    graph check on the suggestion's own proposal, not an evidence-quote
    check, since nothing here needs to be 'fabricated' to be dangerous."""
    if "depend" not in desc.lower():
        return False

    resources_section_match = re.search(r'^Resources:\s*\n(.*?)(?=^Outputs:|\Z)', template, flags=re.MULTILINE | re.DOTALL)
    resources_text = resources_section_match.group(1) if resources_section_match else template
    all_ids = set(re.findall(r'^  ([A-Za-z0-9]+):\s*$', resources_text, flags=re.MULTILINE))

    mentioned_ids = [rid for rid in all_ids if rid != logical_id and re.search(rf'\b{re.escape(rid)}\b', desc)]

    for other_id in mentioned_ids:
        block_match = re.search(
            rf'^  {re.escape(other_id)}:\s*\n((?:(?!^  [A-Za-z0-9]+:\s*$).)*)',
            resources_text, flags=re.MULTILINE | re.DOTALL
        )
        other_block = block_match.group(0) if block_match else ""
        if re.search(rf'DependsOn:.*{re.escape(logical_id)}', other_block, flags=re.DOTALL) or \
           re.search(rf'!(Ref|GetAtt)\s+{re.escape(logical_id)}\b', other_block):
            return True
    return False

def _suggestion_violates_hard_rules(text):
    """Reject suggestions that themselves propose something CloudFormation
    forbids â€” e.g. Nova Lite confusing a resource's Logical ID (the YAML
    mapping key, which Rule 25 requires to be a static alphanumeric string)
    with its Name property (which correctly uses !Sub). A suggestion like
    'the logical ID should follow !Sub "${AWS::StackName}-X"' is invalid on
    its face: applying it produces an intrinsic function as a mapping key,
    which cannot parse. Catch this class before it ever reaches the user or
    a repair prompt, rather than relying on repair rounds to un-break it."""
    lower = text.lower()
    if "logical id" in lower and re.search(r'!(Sub|Ref|GetAtt|Join|Select|Split|If)\b', text):
        return True
    return False

def _suggestion_proposes_fabricated_arn_format(text):
    """Reject suggestions proposing arn:aws:iam::aws:service/... — this ARN
    namespace does not exist. IAM Principal.Service values are bare service
    principal strings (e.g. 'events.amazonaws.com'), never ARNs. Nova Lite
    has hallucinated this exact pattern when 'improving' a trust policy."""
    if re.search(r'arn:aws:iam::aws:service/', text):
        return True
    return False


def _review_template_for_improvements(llm, template, request, fmt, rules_only, prior_suggestions=None, current_scp_warnings=None):
    """..."""
    fence_lang = "json" if fmt == "terraform_json" else "yaml"
    prior_suggestions = prior_suggestions or []
    current_scp_warnings = current_scp_warnings or []
    scp_flagged_ids = set()
    for w in current_scp_warnings:
        m = re.match(r"^([A-Za-z0-9]+)\s*\(", w)
        if m:
            scp_flagged_ids.add(m.group(1))

    prior_block = ""
    if prior_suggestions:
        prior_list = "\n".join(f"- {s}" for s in prior_suggestions)
        prior_block = f"""
The PREVIOUS version of this template (before this turn's fix) had these flagged issues:
{prior_list}

If this turn's request addressed any of these, they are now RESOLVED Ã¢â‚¬â€ do not re-flag them unless you can point to the exact line in the CURRENT template below proving the issue still exists.
"""

    review_prompt = f"""You are reviewing a {("Terraform JSON" if fmt == "terraform_json" else "CloudFormation")} template you just generated, looking for genuine, verifiable issues â€” not restating the request, not inventing problems that aren't there.

{prior_block}

Original request:
{request}

Generated template:
```{fence_lang}
{template}
```

Reference schema rules:
{rules_only}

HARD RULES for what counts as a valid suggestion:
- Every suggestion must point to something ACTUALLY WRONG in the template shown above, not a stylistic preference, not something already correct, and not something you recall from a different template.
- Never claim a property or section is "missing," "absent," or "empty" without having checked the CURRENT template text above for it first. If it's there, don't flag it.
- Never propose a fix that itself violates the reference schema rules above (e.g. never suggest an intrinsic function like !Sub as a resource's Logical ID â€” Logical IDs must be a static alphanumeric string; that is a different thing from a resource's Name property).
- Do not repeat or rephrase anything already in the "Original request" above â€” if the user already asked for it, it's not a new suggestion.

EVIDENCE REQUIRED: for every suggestion, you must quote the EXACT line (copied character-for-character from the template above) that demonstrates the problem. If you cannot find a real line proving the issue exists, do not include the suggestion â€” do not paraphrase, summarize, or invent a line that looks plausible.

Rules for your answer:
- If you find nothing worth flagging, respond with exactly: NONE
- Otherwise, respond with a bullet list of AT MOST 5 items. Each item must be in this EXACT format, on one line:
  - [LogicalId] EVIDENCE: "exact line copied from the template above" ISSUE: description of the problem (at least 10 words)
- Output nothing except NONE or the bullet list."""

    try:
        response = llm.invoke([HumanMessage(content=review_prompt)], temperature=0.1)
        content = _strip_think_block(response.content)
    except Exception as e:
        print(f"WARNING: improvement review call failed, skipping: {e}")
        return []

    content = content.strip()
    if _looks_degenerate(content):
        print("WARNING: degenerate output detected in improvement review, skipping suggestions.")
        return []
    if not content or content.upper().startswith("NONE"):
        return []

    suggestions = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            text = line[2:].strip()
        elif line.startswith("-"):
            text = line[1:].strip()
            text = text.lstrip("- ").strip()  # handle double-dash bullets like "-- [Id] ..."
        else:
            continue

        text = re.sub(r'^[-\s]+', '', text)  # strips any leftover leading dash(es)/space(s), e.g. from "- - [Id] ..." model output

        if len(text.split()) < 8:
            print(f"WARNING: discarding suspiciously short 'suggestion': {text!r}")
            continue
        if text in template:
            print(f"WARNING: discarding suggestion that's verbatim template text: {text!r}")
            continue
        quoted_spans = re.findall(r'`([^`]+)`', text)
        intrinsic_spans = re.findall(r'(![A-Za-z]+(?:\s+[\w\.]+)+)', text)
        all_spans = quoted_spans + intrinsic_spans
        if len(all_spans) >= 2 and len(set(all_spans)) < len(all_spans):
            print(f"WARNING: discarding suggestion with duplicate identical spans: {text!r}")
            continue

        # NEW: extract [LogicalId] tag and verify it actually exists in the template
        m = re.match(r"^\[([A-Za-z0-9]+)\]\s*(.+)$", text)
        if not m:
            print(f"WARNING: discarding untagged suggestion (missing [LogicalId]): {text!r}")
            continue
        logical_id, rest = m.group(1), m.group(2)
        resources_section_match = re.search(r'^Resources:\s*\n(.*?)(?=^Outputs:|\Z)', template, flags=re.MULTILINE | re.DOTALL)
        resources_text = resources_section_match.group(1) if resources_section_match else template
        resource_ids = set(re.findall(r'^  ([A-Za-z0-9]+):\s*$', resources_text, flags=re.MULTILINE))
        if not resource_ids:
            resource_ids = set(re.findall(r'^\s{2}([A-Za-z0-9]+):\n\s{4}Type:', resources_text, flags=re.MULTILINE))
        if not resource_ids:
            resource_ids = set(re.findall(r'^\s{2}([A-Za-z0-9]+):\n\s{4}Type:', resources_text, flags=re.MULTILINE))
        if logical_id not in resource_ids:
            print(f"WARNING: discarding suggestion referencing '{logical_id}', which is not an actual Resources logical ID in this template: {text!r}")
            continue

        # NEW: require a quoted EVIDENCE line, copied verbatim from the
        # template, verified as a real substring before trusting the ISSUE
        # text at all. Closes the whole class of "asserts something false
        # about the template" hallucinations at once, instead of chasing
        # each new phrasing (missing/absent, should-instead-of, etc.) with
        # a separate keyword filter.
        evidence_match = re.match(r'^EVIDENCE:\s*"(.+?)"\s*ISSUE:\s*(.+)$', rest, flags=re.DOTALL)
        if not evidence_match:
            print(f"WARNING: discarding suggestion missing EVIDENCE/ISSUE tags: {text!r}")
            continue
        evidence, desc = evidence_match.group(1).strip(), evidence_match.group(2).strip()

        def _normalize_ws(s):
            return re.sub(r'\s+', ' ', s.strip())

        if _normalize_ws(evidence) not in _normalize_ws(template):
            print(f"WARNING: discarding suggestion with fabricated EVIDENCE not found verbatim in template: {text!r} | evidence={evidence!r}")
            continue

        if _suggestion_evidence_contradicts_issue(evidence, desc):
            print(f"WARNING: discarding suggestion whose own evidence contradicts its claim: {text!r} | evidence={evidence!r}")
            continue

        if _suggestion_proposes_circular_dependency(logical_id, desc, template):
            print(f"WARNING: discarding suggestion proposing a circular dependency for {logical_id}: {text!r}")
            continue

        text = desc # drop the tag/evidence for display

        # NEW: reject suggestions that propose something CloudFormation
        # structurally forbids (e.g. an intrinsic function as a Logical ID)
        if _suggestion_violates_hard_rules(text):
            print(f"WARNING: discarding suggestion that violates a hard CloudFormation rule: {text!r}")
            continue
        if _suggestion_proposes_fabricated_arn_format(text):
            print(f"WARNING: discarding suggestion proposing a fabricated ARN format: {text!r}")
            continue


        # NEW: reject "missing/absent" claims contradicted by the resource's
        # own block in the current template
        if _suggestion_claims_missing_but_present(text, logical_id, template):
            print(f"WARNING: discarding suggestion claiming {logical_id} is missing something that's actually present: {text!r}")
            continue

        # NEW: verify "hardcoded name" claims against the actual template
        # text. Nova Lite repeatedly conflates a resource's Logical ID (the
        # YAML key, e.g. "MyQueue:") with an actual Name-type property
        # (Name, QueueName, FunctionName, etc.) Ã¢â‚¬â€ flagging the Logical ID
        # itself as a "hardcoded resource name" even when no such property
        # exists on the resource at all. Only trust the suggestion if the
        # claimed string genuinely appears as a Name-type property value
        # somewhere in the template.
        if "hardcoded" in text.lower() and "name" in text.lower():
            quoted_values = [q for q in re.findall(r'"([^"]+)"', text) if "AWS::StackName" not in q]
            if quoted_values:
                name_prop_lines = re.findall(
                    r'^\s*(?:Name|QueueName|FunctionName|BucketName|RuleName|TopicName|TableName|RoleName|StreamName|ClusterName)\s*:\s*.*$',
                    template, flags=re.MULTILINE
                )
                verified = any(claimed in line for claimed in quoted_values for line in name_prop_lines)
                if not verified:
                    print(f"WARNING: discarding unverified 'hardcoded name' suggestion for {logical_id} Ã¢â‚¬â€ no matching Name-type property found in template: {text!r}")
                    continue
        # NEW: cross-check root/wildcard-principal claims against the actual
        # static SCP check Ã¢â‚¬â€ Nova Lite has flagged stale template Description
        # text as if it were the current Principal value. Trust the static
        # check over the model's own claim.
        if "root" in text.lower() and logical_id not in scp_flagged_ids:
            print(f"WARNING: discarding suggestion claiming root-principal issue for {logical_id}, which the static SCP check found compliant: {text!r}")
            continue

        # NEW: programmatic dedup against prior turn's suggestions
        if _is_stale_suggestion(text, prior_suggestions):
            print(f"WARNING: discarding stale/repeated suggestion: {text!r}")
            continue

        # NEW: discard suggestions that just echo the user's own fix request Ã¢â‚¬â€
        # a sign the model is repeating the problem statement instead of
        # verifying whether the fix actually landed
        if _is_stale_suggestion(text, [request]):
            print(f"WARNING: discarding suggestion that echoes the user's own fix request: {text!r}")
            continue

        suggestions.append(text)

    return suggestions[:5]


def _is_stale_suggestion(new_text, prior_suggestions, threshold=0.6):
    new_words = set(re.sub(r'[^\w\s]', '', new_text.lower()).split())
    for prior in prior_suggestions:
        prior_words = set(re.sub(r'[^\w\s]', '', prior.lower()).split())
        if not new_words:
            continue
        overlap = len(new_words & prior_words) / len(new_words)
        if overlap > threshold:
            return True
    return False

def _generate_with_retry(llm, messages, fmt, max_retries=2):
    """Call the LLM, detect degenerate output (pad tokens, repetition
    loops, invalid JSON for the terraform_json format), and retry cleanly
    instead of returning garbage to the user."""
    last_content = ""
    for attempt in range(max_retries + 1):
        response = llm.invoke(messages)
        # Strip the <think> block before any downstream checks so we're
        # scanning the actual answer, not the model's reasoning trace.
        content = _strip_think_block(response.content)
        print(f"Attempt {attempt + 1} Ã¢â‚¬â€ raw response: {content[:300]}")

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

    print("All retries produced bad output Ã¢â‚¬â€ returning last attempt as-is.")
    return last_content


def generate_template(request, embeddings, chat_history=None):
    """Main template generation function. Supports both CloudFormation YAML
    and Terraform JSON output, auto-detected from the request or from the
    format of the last template in the conversation. Also supports
    conversation memory so users can say 'fix this' or 'update the template'
    without repasting it, and automatically retries if the model produces
    degenerate output (common on free-tier models under load). CloudFormation
    output additionally goes through a cfn-lint validation pass, static
    role/trust-policy, SCP, DLQ-direction, and Parameter-default checks, and
    an LLM repair pass (up to 2 rounds) to catch hallucinated/invalid
    resource properties, cross-service role reuse, and account-SCP
    violations before returning.
    """
    try:
        print(f"\nTemplate request: {request}")
        print(f"DEBUG: template_chain.py loaded from {__file__}")

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
        rules_only = TERRAFORM_RULES_ONLY if fmt == "terraform_json" else CLOUDFORMATION_RULES_ONLY
        messages = [SystemMessage(content=rules)]

        # Add recent conversation history so the model has context
        # (skip giant template code blocks except the most recent one, to save tokens)
        last_template = get_last_template(chat_history, fmt=fmt)

        for msg in chat_history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
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
- aws_cloudwatch_event_api_destination uses flat invocation_endpoint/http_method/invocation_rate_limit_per_second Ã¢â‚¬â€ no destination_config block
- Output must be a single valid JSON object with no trailing commas or comments"""
        else:
            critical_reminders = """- FlexibleTimeWindow Mode must be written as Mode: "OFF" with quotes, never Mode: OFF
- All YAML reserved words (OFF, ON, YES, NO, TRUE, FALSE) must be quoted when used as strings
- All parameters without defaults must have descriptions with examples
- ApiKeyValue must never be empty
- Never use AWS::IAM::PolicyAttachment
- Never reuse one IAM Role's RoleArn across two resources that need different trust-policy principals (e.g. a pipes.amazonaws.com-trusted role must never be used as a Rule target's RoleArn)"""

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

Output ONLY the {fence_lang.upper()} content. Do not write "Here is the template" or any other sentence before or after it. Begin your response with the first line of the {fence_lang.upper()} itself:"""))
        else:
            messages.append(HumanMessage(content=f"""Generate a {format_label} template for the following request.
Use the sample templates below as reference for patterns and structure.

Sample templates for reference:
{context}

Request: {request}

Critical reminders before you output:
{critical_reminders}

Output ONLY the {fence_lang.upper()} content. Do not write "Here is the template" or any other sentence before or after it. Begin your response with the first line of the {fence_lang.upper()} itself:"""))

        print("Generating template...")
        raw_content = _generate_with_retry(llm, messages, fmt, max_retries=1)
        print(f"Final raw response: {raw_content[:500]}")

        template = _strip_fences(raw_content)

        if fmt == "terraform_json":
            try:
                json.loads(template)
            except json.JSONDecodeError as je:
                print(f"WARNING: generated Terraform JSON did not parse cleanly: {je}")
        else:
            # --- deterministic auto-fixes, applied once, in order ---
            template = re.sub(r'(Mode:\s+)(OFF|ON|YES|NO)(\s*$)', r'\1"\2"\3', template, flags=re.MULTILINE)
            template = re.sub(r'^(\s*)queryStringParameters(\s*:)', r'\1QueryStringParameters\2', template, flags=re.MULTILINE | re.IGNORECASE)
            template = _quote_colon_in_descriptions(template)
            template = _strip_unneeded_sub(template)
            template = _fix_redrive_policy_casing(template)
            template = _fix_pipe_sqs_source_parameters(template)
            # Nova Lite also sometimes omits the space between a key's colon and
            # a quoted scalar value entirely, e.g. ConstraintDescription:'must be...'
            # or KeyName:'my-key-pair' Ã¢â‚¬â€ invalid YAML, breaks parsing before any
            # validation can run. Insert the missing space.
            template = re.sub(r'^(\s*[A-Za-z][\w]*):([\'"])', r'\1: \2', template, flags=re.MULTILINE)
            # Nova Lite consistently omits the space between ":" and an intrinsic
            # function tag (e.g. "Key:!Sub" instead of "Key: !Sub"), which is
            # invalid YAML and breaks cfn-lint parsing entirely.
            template = re.sub(r'(:)(!(?:Sub|GetAtt|Ref|Join|Select|Split|If|Not|Equals|Condition))', r'\1 \2', template)
            template = re.sub(r'^(\s*-)(!(?:Sub|GetAtt|Ref|Join|Select|Split))', r'\1 \2', template, flags=re.MULTILINE)
            print(f"DEBUG: after colon-fix, sample line check: {[l for l in template.split(chr(10)) if ':!' in l][:3]}")

            # --- validation + static checks, run once ---
            is_valid, lint_errors, lint_ran = _validate_cloudformation(template)
            role_warnings = _check_role_principal_mismatch(template)
            scp_warnings = _check_scp_compliance(template)
            param_default_warnings = _check_invalid_parameter_defaults(template)
            dlq_warnings = _check_dlq_redrive_direction(template)
            arn_warnings = _check_fabricated_managed_policy_arns(template)
            connection_warnings = _check_api_destination_missing_connection(template)

            if not lint_ran:
                print("cfn-lint validation SKIPPED (module not found in this interpreter) Ã¢â‚¬â€ output is unverified.")
            if role_warnings:
                print(f"Role/trust-policy mismatch(es) found:\n" + "\n".join(role_warnings))
            if scp_warnings:
                print(f"SCP compliance violation(s) found:\n" + "\n".join(scp_warnings))
            if dlq_warnings:
                print(f"DLQ redrive-direction issue(s) found:\n" + "\n".join(dlq_warnings))
            if arn_warnings:
                print(f"Fabricated managed-policy ARN(s) found:\n" + "\n".join(arn_warnings))
            if connection_warnings:
                print(f"ApiDestination/Connection issue(s) found:\n" + "\n".join(connection_warnings))

            needs_repair = (
                (lint_ran and not is_valid)
                or bool(role_warnings)
                or bool(scp_warnings)
                or bool(param_default_warnings)
                or bool(dlq_warnings)
                or bool(arn_warnings)
                or bool(connection_warnings)
            )
            repair_status = "not_needed"

            if needs_repair:
                current_lint_errors = lint_errors if (lint_ran and not is_valid) else ""
                current_role_warnings = role_warnings
                current_scp_warnings = scp_warnings
                current_param_warnings = param_default_warnings
                current_dlq_warnings = dlq_warnings
                current_arn_warnings = arn_warnings
                current_connection_warnings = connection_warnings
                max_repair_rounds = 2

                for attempt in range(max_repair_rounds):
                    combined_errors = current_lint_errors
                    if current_role_warnings:
                        combined_errors += (
                            "\n\nAdditional issues found by static analysis (role/trust-policy mismatches Ã¢â‚¬â€ "
                            "these will NOT show up in cfn-lint but WILL fail at deploy/runtime):\n"
                            + "\n".join(f"- {w}" for w in current_role_warnings)
                        )
                    if current_scp_warnings:
                        combined_errors += (
                            "\n\nAdditional issues found by static analysis (account SCP violations Ã¢â‚¬â€ "
                            "these will NOT show up in cfn-lint but WILL be DENIED at deploy time):\n"
                            + "\n".join(f"- {w}" for w in current_scp_warnings)
                        )
                    if current_param_warnings:
                        combined_errors += (
                            "\n\nAdditional issues found by static analysis (invalid Parameter.Default values Ã¢â‚¬â€ "
                            "cfn-lint will reject these, not just a deploy-time denial):\n"
                            + "\n".join(f"- {w}" for w in current_param_warnings)
                        )
                    if current_dlq_warnings:
                        combined_errors += (
                            "\n\nAdditional issues found by static analysis (DLQ RedrivePolicy misconfiguration Ã¢â‚¬â€ "
                            "will NOT show up in cfn-lint but silently breaks message redrive at runtime):\n"
                            + "\n".join(f"- {w}" for w in current_dlq_warnings)
                        )
                    if current_arn_warnings:
                        combined_errors += (
                            "\n\nAdditional issues found by static analysis (fabricated AWS-managed policy ARNs Ã¢â‚¬â€ "
                            "will NOT show up in cfn-lint but will fail at deploy time with a policy-not-found error):\n"
                            + "\n".join(f"- {w}" for w in current_arn_warnings)
                        )
                    if current_connection_warnings:
                        combined_errors += (
                            "\n\nAdditional issues found by static analysis (ApiDestination/Connection issues -- "
                            "these WILL show up in cfn-lint as a missing required property, but the correct fix "
                            "(adding a Connection resource) may not be obvious from the cfn-lint error alone):\n"
                            + "\n".join(f"- {w}" for w in current_connection_warnings)
                        )

                    print(f"Repair attempt {attempt + 1}/{max_repair_rounds} for the following issues:\n{combined_errors}")
                    repair_messages = messages + [
                        AIMessage(content=template),
                        HumanMessage(content=f"""The template above has the following issues:

{combined_errors}

Return the COMPLETE corrected template with these issues fixed. Output ONLY the YAML content, no explanation.""")
                    ]
                    repaired = _generate_with_retry(llm, repair_messages, fmt, max_retries=1)
                    repaired = _strip_fences(repaired)
                    repaired = _quote_colon_in_descriptions(repaired)
                    repaired = _strip_unneeded_sub(repaired)
                    repaired = _fix_redrive_policy_casing(repaired)
                    repaired = _fix_pipe_sqs_source_parameters(repaired)
                    repaired = re.sub(r'^(\s*[A-Za-z][\w]*):([\'"])', r'\1: \2', repaired, flags=re.MULTILINE)
                    repaired = re.sub(r'(:)(!(?:Sub|GetAtt|Ref|Join|Select|Split|If|Not|Equals|Condition))', r'\1 \2', repaired)
                    repaired = re.sub(r'^(\s*-)(!(?:Sub|GetAtt|Ref|Join|Select|Split))', r'\1 \2', repaired, flags=re.MULTILINE)

                    repair_unchanged = repaired.strip() == template.strip()
                    if repair_unchanged:
                        print(f"WARNING: repair attempt {attempt + 1} returned an unchanged template Ã¢â‚¬â€ the model did not apply any fix.")

                    template = repaired

                    still_valid, remaining_errors, repair_lint_ran = _validate_cloudformation(template)
                    remaining_role_warnings = _check_role_principal_mismatch(template)
                    remaining_scp_warnings = _check_scp_compliance(template)
                    remaining_param_warnings = _check_invalid_parameter_defaults(template)
                    remaining_dlq_warnings = _check_dlq_redrive_direction(template)
                    remaining_arn_warnings = _check_fabricated_managed_policy_arns(template)
                    remaining_connection_warnings = _check_api_destination_missing_connection(template)

                    if not repair_lint_ran:
                        print(f"cfn-lint validation SKIPPED on repair attempt {attempt + 1} Ã¢â‚¬â€ treating as unverified.")
                        repair_status = "lint_skipped"
                        break

                    if (
                        still_valid
                        and not remaining_role_warnings
                        and not remaining_scp_warnings
                        and not remaining_param_warnings
                        and not remaining_dlq_warnings
                        and not remaining_arn_warnings
                        and not remaining_connection_warnings
                    ):
                        print(f"Repair attempt {attempt + 1} succeeded Ã¢â‚¬â€ cfn-lint clean, no static-check warnings.")
                        repair_status = "unchanged_after_repair" if repair_unchanged else "repaired_clean"
                        break

                    current_lint_errors = remaining_errors if not still_valid else ""
                    current_role_warnings = remaining_role_warnings
                    current_scp_warnings = remaining_scp_warnings
                    current_param_warnings = remaining_param_warnings
                    current_dlq_warnings = remaining_dlq_warnings
                    current_arn_warnings = remaining_arn_warnings
                    current_connection_warnings = remaining_connection_warnings

                    if repair_unchanged:
                        print("Repair made no changes Ã¢â‚¬â€ further rounds unlikely to help, stopping early.")
                        repair_status = "repair_failed"
                        break

                    if attempt == max_repair_rounds - 1:
                        if not still_valid:
                            print(f"WARNING: repair pass still has cfn-lint errors after {max_repair_rounds} attempts, returning best attempt:\n{remaining_errors}")
                        if remaining_role_warnings:
                            print("WARNING: still has role/trust-policy mismatches, returning best attempt:\n" + "\n".join(remaining_role_warnings))
                        if remaining_scp_warnings:
                            print("WARNING: still has SCP violations, returning best attempt:\n" + "\n".join(remaining_scp_warnings))
                        if remaining_param_warnings:
                            print("WARNING: still has invalid Parameter.Default values, returning best attempt:\n" + "\n".join(remaining_param_warnings))
                        if remaining_dlq_warnings:
                            print("WARNING: still has DLQ redrive-direction issues, returning best attempt:\n" + "\n".join(remaining_dlq_warnings))
                        if remaining_arn_warnings:
                            print("WARNING: still has fabricated managed-policy ARNs, returning best attempt:\n" + "\n".join(remaining_arn_warnings))
                        if remaining_connection_warnings:
                            print("WARNING: still has ApiDestination/Connection issues, returning best attempt:\n" + "\n".join(remaining_connection_warnings))
                        repair_status = "repair_failed"
            else:
                print("cfn-lint passed, no role/trust-policy mismatches, SCP-compliant Ã¢â‚¬â€ no repair needed.")

        sources = list(set([
            chunk.metadata.get("source_file", "unknown")
            for chunk in chunks
        ]))

        prior_suggestions = []
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant" and msg.get("type") == "template":
                prior_suggestions = msg.get("suggestions", [])
                break

        final_scp_warnings = _check_scp_compliance(template)
        suggestions = _review_template_for_improvements(llm, template, request, fmt, rules_only, prior_suggestions, final_scp_warnings)
        if suggestions:
            print(f"Improvement suggestions found: {suggestions}")
        else:
            print("No improvement suggestions Ã¢â‚¬â€ template looks complete.")

        print("Template generated successfully")
        print(f"Sources used: {sources}")

        return {
            "request": request,
            "template": template,
            "format": fmt,
            "sources": sources,
            "chunks": chunks,
            "suggestions": suggestions,
            "repair_status": repair_status
        }

    except Exception as e:
        print(f"Error generating template: {e}")
        raise