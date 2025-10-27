from typing import Optional, Dict, Any, List
import logging
import json
import boto3
import openai
from .config import config

logger = logging.getLogger(__name__)

class UnifiedLLM:
    def __init__(self):
        self.config = config
        self.openai_client = self._init_openai()
        self.bedrock_client = self._init_bedrock()
        
    def _init_openai(self):
        """Initialize OpenAI client"""
        if self.config.OPENAI_API_KEY:
            return openai.Client(
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_BASE_URL
            )
        return None
    
    def _init_bedrock(self):
        """Initialize AWS Bedrock client"""
        if self.config.AWS_ACCESS_KEY_ID and self.config.AWS_SECRET_ACCESS_KEY:
            return boto3.client(
                service_name='bedrock-runtime',
                region_name=self.config.AWS_REGION,
                aws_access_key_id=self.config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=self.config.AWS_SECRET_ACCESS_KEY
            )
        return None
    
    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings using configured provider"""
        
        model = model or self.config.EMBEDDING_MODEL
        
        try:
            if "text-embedding" in model:  # OpenAI
                return await self._get_openai_embeddings(texts, model)
            else:  # AWS Bedrock
                return await self._get_bedrock_embeddings(texts, model)
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    async def _get_openai_embeddings(
        self,
        texts: List[str],
        model: str
    ) -> List[List[float]]:
        """Get embeddings from OpenAI"""
        
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            response = self.openai_client.embeddings.create(
                model=model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            raise
    
    async def _get_bedrock_embeddings(
        self,
        texts: List[str],
        model: str
    ) -> List[List[float]]:
        """Get embeddings from AWS Bedrock"""
        
        if not self.bedrock_client:
            raise ValueError("Bedrock client not initialized")
        
        try:
            embeddings = []
            for text in texts:
                body = json.dumps({
                    "inputText": text
                })
                
                response = self.bedrock_client.invoke_model(
                    modelId=model,
                    body=body
                )
                
                response_body = json.loads(response['body'].read())
                embeddings.append(response_body['embedding'])
            
            return embeddings
        except Exception as e:
            logger.error(f"Bedrock embedding failed: {str(e)}")
            raise
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using configured provider"""
        
        model = model or self.config.LLM_MODEL
        
        try:
            if "gpt" in model:  # OpenAI
                return await self._generate_openai_text(
                    prompt, model, system_prompt, temperature, max_tokens
                )
            else:  # AWS Bedrock
                return await self._generate_bedrock_text(
                    prompt, model, system_prompt, temperature, max_tokens
                )
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise
    
    async def _generate_openai_text(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int]
    ) -> str:
        """Generate text using OpenAI"""
        
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI text generation failed: {str(e)}")
            raise
    
    async def _generate_bedrock_text(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int]
    ) -> str:
        """Generate text using AWS Bedrock"""
        
        if not self.bedrock_client:
            raise ValueError("Bedrock client not initialized")
        
        try:
            # Format prompt based on model
            if "claude" in model.lower():
                formatted_prompt = f"Human: {prompt}\n\nAssistant:"
                if system_prompt:
                    formatted_prompt = f"System: {system_prompt}\n\n{formatted_prompt}"
            else:
                formatted_prompt = prompt
            
            body = json.dumps({
                "prompt": formatted_prompt,
                "temperature": temperature,
                "maxTokens": max_tokens or 2048
            })
            
            response = self.bedrock_client.invoke_model(
                modelId=model,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            return response_body.get('completion') or response_body.get('generated_text', '')
        except Exception as e:
            logger.error(f"Bedrock text generation failed: {str(e)}")
            raise
    
    async def analyze_image(
        self,
        image_data: str,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Analyze image using vision model"""
        
        model = model or self.config.VISION_MODEL
        
        try:
            if "gpt" in model:  # OpenAI Vision
                return await self._analyze_image_openai(
                    image_data, prompt, model, system_prompt
                )
            else:  # AWS Bedrock Vision
                return await self._analyze_image_bedrock(
                    image_data, prompt, model, system_prompt
                )
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            raise
    
    async def _analyze_image_openai(
        self,
        image_data: str,
        prompt: str,
        model: str,
        system_prompt: Optional[str]
    ) -> str:
        """Analyze image using OpenAI Vision"""
        
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        })
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI vision analysis failed: {str(e)}")
            raise
    
    async def _analyze_image_bedrock(
        self,
        image_data: str,
        prompt: str,
        model: str,
        system_prompt: Optional[str]
    ) -> str:
        """Analyze image using AWS Bedrock Vision"""
        
        if not self.bedrock_client:
            raise ValueError("Bedrock client not initialized")
        
        try:
            # Format prompt
            formatted_prompt = prompt
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\n{prompt}"
            
            body = json.dumps({
                "prompt": formatted_prompt,
                "image": image_data,
                "maxTokens": 1000
            })
            
            response = self.bedrock_client.invoke_model(
                modelId=model,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            return response_body.get('generated_text', '')
        except Exception as e:
            logger.error(f"Bedrock vision analysis failed: {str(e)}")
            raise