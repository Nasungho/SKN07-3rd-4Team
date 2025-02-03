import time
import numpy as np

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

from pymilvus import connections, Collection, FieldSchema, CollectionSchema
from pymilvus import MilvusClient, DataType

def embed_and_store_message(message_text: str):
    # OpenAI 임베딩 사용 (자신의 API 키와 환경에 맞게 설정 필요)
    embeddings = OpenAIEmbeddings()
    embedding = embeddings.embed([message_text])[0]

    # 메시지를 Milvus에 저장
    timestamp = int(time.time())
    data = [
        [embedding],  # 벡터
        [message_text],  # 메시지
        [timestamp]  # 타임스탬프
    ]
    
    collection.insert(data)

def search_similar_messages(query: str, top_k: int = 5):
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed([query])[0]

    # Milvus에서 유사한 메시지 검색
    search_params = {
        "metric_type": "IP",  # 내적 유사도 (cosine similarity와 비슷한 방식)
        "params": {"nprobe": 10}
    }

    results = collection.search([query_embedding], "embedding", search_params, limit=top_k)
    
    # 결과 출력
    similar_messages = []
    for result in results[0]:
        message = result.entity.get("message")
        similar_messages.append(message)
    
    return similar_messages
