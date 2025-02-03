from api import api_keys

import os
import numpy as np
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.docstore.document import Document

import chainlit as cl
from chainlit.types import AskFileResponse

from pymilvus import Collection, connections, utility

# Milvus 설정
milvus_host = os.environ.get("MILVUS_HOST", "localhost")
milvus_port = os.environ.get("MILVUS_PORT", "19530")

# 연결 Milvus 서버에
connections.connect(alias="default", host=milvus_host, port=milvus_port)

index_name = "langchain-demo"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

namespaces = set()

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

# Milvus 컬렉션 이름
def get_milvus_collection(namespace: str):
    collection_name = f"{index_name}_{namespace}"
    
    # 컬렉션이 없다면 새로 생성
    if not utility.has_collection(collection_name):
        schema = [
            # 각 문서의 벡터 데이터를 저장할 필드
            {"name": "embedding", "type": "FLOAT_VECTOR", "params": {"dim": 1536}},
            # 문서의 메타데이터 저장
            {"name": "metadata", "type": "JSON"},
        ]
        
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "params": {"nlist": 100}})
    else:
        collection = Collection(name=collection_name)
        
    return collection

def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    loader = Loader(file.path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs

def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file
    namespace = file.id

    if namespace in namespaces:
        # 이미 존재하는 Milvus 컬렉션에서 검색
        docsearch = get_milvus_search(namespace)
    else:
        # Milvus에 새로 벡터 저장
        docsearch = store_in_milvus(docs, namespace)
        namespaces.add(namespace)

    return docsearch

def store_in_milvus(docs: List[Document], namespace: str):
    collection = get_milvus_collection(namespace)
    embeddings_vectors = []
    metadata = []
    
    for doc in docs:
        # 문서를 벡터로 변환
        doc_vector = embeddings.embed_text(doc.page_content)
        embeddings_vectors.append(doc_vector)
        metadata.append(doc.metadata)
    
    # Milvus에 벡터 및 메타데이터 저장
    collection.insert([embeddings_vectors, metadata])
    return "Data stored in Milvus"

def get_milvus_search(namespace: str):
    collection = get_milvus_collection(namespace)
    # 컬렉션에서 모든 벡터를 검색하는 예시
    search_results = collection.query(
        expr="*",  # 모든 문서 검색
        output_fields=["embedding", "metadata"],
        limit=10  # 최대 10개까지 검색
    )
    return search_results

@cl.on_chat_start
async def start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    # Milvus에서 문서 검색을 위한 설정
    docsearch = await cl.make_async(get_docsearch)(file)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch,
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()




if __name__ == "__main__":
    ...