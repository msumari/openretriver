import json
import os
from typing import Callable

GenerateFn = Callable[[str, str], str]

DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
DEFAULT_REGION = "us-east-1"


def bedrock_provider(
    model_id: str | None = None,
    region: str | None = None,
) -> GenerateFn:
    import boto3

    _region = region or os.environ.get("BEDROCK_REGION", DEFAULT_REGION)
    _model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
    client = boto3.client("bedrock-runtime", region_name=_region)

    def generate(system: str, user_message: str) -> str:
        response = client.invoke_model(
            modelId=_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "system": system,
                    "messages": [{"role": "user", "content": user_message}],
                }
            ),
        )
        body = json.loads(response["body"].read())
        return body["content"][0]["text"]

    return generate


def get_provider() -> GenerateFn:
    provider_name = os.environ.get("MODEL_PROVIDER", "bedrock")
    providers = {
        "bedrock": bedrock_provider,
    }
    factory = providers.get(provider_name)
    if factory is None:
        raise ValueError(
            f"Unknown model provider: {provider_name!r}. Available: {list(providers.keys())}"
        )
    return factory()
