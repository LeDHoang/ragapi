import os, json, base64
from typing import Optional, Any, List, Dict
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

class UnifiedLLM(ABC):
    """Unified interface for both Bedrock and OpenAI LLMs with vision support"""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """Generate text response"""
        pass

    @abstractmethod
    def generate_with_image(self, system_prompt: str, user_prompt: str, image_path: str, temperature: float = 0.2) -> str:
        """Generate text response with image input"""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        pass

class BedrockLLM(UnifiedLLM):
    """Bedrock LLM implementation with vision support"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._init_bedrock()

    def _init_bedrock(self):
        import boto3
        import botocore

        # Bridge bearer token if needed
        bearer = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        if bearer and not os.getenv("AWS_SESSION_TOKEN"):
            os.environ["AWS_SESSION_TOKEN"] = bearer

        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN")

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

        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=self.cfg.bedrock_region,
        )
        self.client = session.client("bedrock-runtime")
        self.embedding_client = session.client("bedrock-runtime")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 800,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ],
        }

        try:
            resp = self.client.invoke_model(
                modelId=self.cfg.bedrock_llm_model_id,
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read().decode("utf-8"))
            content = payload.get("content", [])
            return "".join(block.get("text", "") for block in content if block.get("type") == "text")
        except Exception as e:
            raise RuntimeError(f"Bedrock LLM call failed: {e}")

    def generate_with_image(self, system_prompt: str, user_prompt: str, image_path: str, temperature: float = 0.2) -> str:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 800,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",  # Adjust based on actual image type
                                "data": image_data
                            }
                        }
                    ]
                }
            ],
        }

        try:
            resp = self.client.invoke_model(
                modelId=self.cfg.bedrock_vision_model_id,
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read().decode("utf-8"))
            content = payload.get("content", [])
            return "".join(block.get("text", "") for block in content if block.get("type") == "text")
        except Exception as e:
            raise RuntimeError(f"Bedrock vision call failed: {e}")

    def embed_text(self, text: str) -> List[float]:
        # Use Amazon Titan embedding model
        body = {
            "inputText": text,
            "dimensions": self.cfg.bedrock_embedding_dim,
            "normalize": True
        }

        try:
            resp = self.embedding_client.invoke_model(
                modelId=self.cfg.bedrock_embedding_model_id,
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read().decode("utf-8"))
            return payload.get("embedding", [])
        except Exception as e:
            raise RuntimeError(f"Bedrock embedding call failed: {e}")

class OpenAILLM(UnifiedLLM):
    """OpenAI LLM implementation with vision support"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._init_openai()

    def _init_openai(self):
        from openai import OpenAI

        if not self.cfg.openai_api_key:
            raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY in .env")

        self.client = OpenAI(
            api_key=self.cfg.openai_api_key,
            base_url=self.cfg.openai_base_url
        )

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.openai_llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=800
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI LLM call failed: {e}")

    def generate_with_image(self, system_prompt: str, user_prompt: str, image_path: str, temperature: float = 0.2) -> str:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine image type
        ext = image_path.lower().split('.')[-1]
        if ext in ['jpg', 'jpeg']:
            media_type = "image/jpeg"
        elif ext == 'png':
            media_type = "image/png"
        else:
            media_type = "image/jpeg"  # fallback

        try:
            response = self.client.chat.completions.create(
                model=self.cfg.openai_vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                temperature=temperature,
                max_tokens=800
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI vision call failed: {e}")

    def embed_text(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=self.cfg.openai_embedding_model,
                input=text,
                dimensions=self.cfg.openai_embedding_dim if "3-" in self.cfg.openai_embedding_model else None
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding call failed: {e}")

class MockLLM(UnifiedLLM):
    """Mock LLM for testing when no real credentials are available"""

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        return f"[MOCK LLM] System: {system_prompt[:50]}... User: {user_prompt[:50]}..."

    def generate_with_image(self, system_prompt: str, user_prompt: str, image_path: str, temperature: float = 0.2) -> str:
        return f"[MOCK VISION] Image: {image_path}, Query: {user_prompt[:50]}..."

    def embed_text(self, text: str) -> List[float]:
        # Return a simple hash-based embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        # Generate a vector of size based on config
        size = 1024  # Default embedding size
        return [(hash_int >> (i * 8)) & 0xFF for i in range(size)]

def create_llm(cfg) -> UnifiedLLM:
    """Factory function to create the appropriate LLM based on configuration"""
    provider = cfg.llm_provider.lower()

    # Check if we have credentials for the requested provider
    aws_access = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session = os.getenv("AWS_SESSION_TOKEN")
    aws_bearer = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    openai_key = os.getenv("OPENAI_API_KEY")

    aws_creds = bool(aws_access and aws_secret and (aws_session or aws_bearer))
    openai_creds = bool(openai_key)

    # Check for actual credentials and use appropriate LLM
    if provider == "bedrock" and aws_creds:
        print(f"Using Bedrock LLM: {cfg.bedrock_llm_model_id}")
        return BedrockLLM(cfg)
    elif provider == "openai" and openai_creds:
        print(f"Using OpenAI LLM: {cfg.openai_llm_model}")
        return OpenAILLM(cfg)
    else:
        # Fallback to mock LLM if no credentials or unsupported provider
        print("Using mock LLM (no valid credentials configured)")
        return MockLLM()
