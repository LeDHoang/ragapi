# rag_core/llm_bedrock.py
import os, json
from typing import Optional, Any, Iterable
import boto3
import botocore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _as_text(s: Any) -> str:
    if s is None:
        return ""
    return s if isinstance(s, str) else str(s)

def _iter_blocks(v: Any) -> Iterable[dict]:
    if not v:
        return []
    if isinstance(v, list):
        return [b for b in v if isinstance(b, dict)]
    return [v] if isinstance(v, dict) else []

class BedrockLLM:
    """
    Anthropic (Claude) via AWS Bedrock (non-streaming).
    Auth requires the standard temporary credential triplet:
      AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_SESSION_TOKEN

    We accept AWS_BEARER_TOKEN_BEDROCK as an alias for AWS_SESSION_TOKEN.
    """

    def __init__(
        self,
        region: Optional[str] = None,
        model_id: Optional[str] = None,
        max_tokens: int = 800
    ):
        self.region = (
            region
            or os.getenv("BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or "ap-southeast-1"
        )
        self.model_id = (
            model_id
            or os.getenv("BEDROCK_LLM_MODEL_ID")
            or os.getenv("BEDROCK_MODEL_ID")
            or "anthropic.claude-3-haiku-20240307-v1:0"
        )
        self.max_tokens = max_tokens

        # Store original bearer token for potential refresh
        self._bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

        # Bridge bearer → session token if only the custom var is set
        bearer = self._bearer_token
        if bearer and not os.getenv("AWS_SESSION_TOKEN"):
            os.environ["AWS_SESSION_TOKEN"] = bearer

        self._refresh_credentials()
        self._create_client()

    def _refresh_credentials(self):
        """Refresh AWS credentials from environment"""
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN")

        # If using bearer token, bridge it to session token
        if self._bearer_token and not session_token:
            session_token = self._bearer_token

        # Validate presence and fail fast with a helpful message
        missing = [n for n, v in [
            ("AWS_ACCESS_KEY_ID", access_key),
            ("AWS_SECRET_ACCESS_KEY", secret_key),
            ("AWS_SESSION_TOKEN", session_token),
        ] if not v]
        if missing:
            raise RuntimeError(
                "Missing AWS temporary credentials for Bedrock: "
                + ", ".join(missing)
                + ". Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
                  "and either AWS_SESSION_TOKEN or AWS_BEARER_TOKEN_BEDROCK."
            )

        self._access_key = access_key
        self._secret_key = secret_key
        self._session_token = session_token

    def _create_client(self):
        """Create a new boto3 session and client"""
        session = boto3.Session(
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            aws_session_token=self._session_token,
            region_name=self.region,
        )
        self.client = session.client("bedrock-runtime")

    def _refresh_client_if_expired(self, error_msg: str):
        """Refresh the boto3 client if credentials appear to be expired"""
        if ("invalid" in error_msg.lower() or "expired" in error_msg.lower() or
            "security token" in error_msg.lower() or "unauthorized" in error_msg.lower()):

            # Reload environment variables (in case they were updated)
            import importlib
            importlib.reload(os)

            # Refresh credentials from environment
            self._refresh_credentials()
            self._create_client()
            return True
        return False

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "system": _as_text(system_prompt),
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": _as_text(user_prompt)}]}
            ],
        }

        max_retries = 2
        for attempt in range(max_retries):
            try:
                resp = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body).encode("utf-8"),
                    contentType="application/json",
                    accept="application/json",
                )
                payload = json.loads(resp["body"].read().decode("utf-8"))
                break  # Success, exit retry loop

            except botocore.exceptions.ClientError as e:
                code = e.response.get("Error", {}).get("Code", "ClientError")
                msg = e.response.get("Error", {}).get("Message", str(e))

                # Try to refresh credentials on first failure
                if attempt == 0 and self._refresh_client_if_expired(msg):
                    continue  # Retry with refreshed credentials

                # If it's a credential issue, provide helpful guidance
                if "invalid" in msg.lower() or "expired" in msg.lower() or "security token" in msg.lower():
                    raise RuntimeError(
                        f"Bedrock invoke_model failed: {code}: {msg}\n\n"
                        "This usually means your AWS credentials have expired. "
                        "Please update your .env file with fresh credentials:\n"
                        "1. Get new temporary AWS credentials\n"
                        "2. Update AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN\n"
                        "   (or update AWS_BEARER_TOKEN_BEDROCK)\n"
                        "3. Restart your FastAPI server"
                    ) from e
                else:
                    raise RuntimeError(f"Bedrock invoke_model failed: {code}: {msg}") from e
            except Exception as e:
                raise RuntimeError(f"Bedrock call failed: {e}") from e
        else:
            # All retries failed
            raise RuntimeError("Failed to call Bedrock API after multiple attempts")

        out_parts = []
        for block in _iter_blocks(payload.get("content")):
            if block.get("type") == "text":
                out_parts.append(_as_text(block.get("text")))
        return "".join(_as_text(p) for p in out_parts).strip()
