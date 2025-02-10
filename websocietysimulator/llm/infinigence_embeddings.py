from typing import Any, List
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import logging

logger = logging.getLogger("websocietysimulator")


class InfinigenceEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "bge-m3",
        infinity_api_url: str = "https://cloud.infini-ai.com/maas/v1"
    ):
        self.api_key = api_key
        self.model = model
        self.infinity_api_url = infinity_api_url
        
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=60),  # 等待时间从10秒开始，指数增长，最长60秒
        stop=stop_after_attempt(5)  # 最多重试5次
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "input": texts
            }
            
            response = requests.post(
                f"{self.infinity_api_url}/embeddings",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                return [data["embedding"] for data in response.json()["data"]]
            else:
                raise ValueError(f"API call failed: {response.text}")
        except Exception as e:
            logger.warning(f"InfinigenceEmbeddings API call failed: {e}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text into a vector"""
        embeddings = self.embed_documents([text])
        return embeddings[0] 
    
class QwenEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v3",
        api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=60),  # 等待时间从10秒开始，指数增长，最长60秒
        stop=stop_after_attempt(5)  # 最多重试5次
    )
    def embed_documents(self, text: str) -> List[float]:
        """Embed a list of documents into vectors"""
        try:
            completion = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=1024,
                encoding_format="float"
            )
            return completion.data[0].embedding
        except Exception as e:
            logger.warning(f"InfinigenceEmbeddings API call failed: {e}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text into a vector"""
        embeddings = self.embed_documents([text])
        return embeddings