"""
LLM 服务模块
封装大语言模型调用，支持 OpenAI 兼容接口
"""
import json
import re
from openai import OpenAI
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIM
)


class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.embed_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """通用对话接口"""
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

    def chat_with_history(self, messages: list, temperature: float = 0.3) -> str:
        """带历史消息的对话"""
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=temperature,
            messages=messages
        )
        return response.choices[0].message.content

    def get_embedding(self, text: str) -> list:
        """获取文本的向量表示"""
        response = self.embed_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: list) -> list:
        """批量获取文本向量"""
        response = self.embed_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]

    def extract_json(self, text: str) -> list | dict | None:
        """从LLM响应中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试从markdown代码块中提取
        patterns = [
            r'```json\s*\n?(.*?)\n?\s*```',
            r'```\s*\n?(.*?)\n?\s*```',
            r'\[.*\]',
            r'\{.*\}'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if '```' in pattern else match.group(0))
                except (json.JSONDecodeError, IndexError):
                    continue
        return None
