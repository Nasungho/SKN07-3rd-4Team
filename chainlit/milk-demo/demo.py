from operator import itemgetter

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

from chainlit.types import ThreadDict
import chainlit as cl

from data_processing import db_connection, get_sentence_embedding, change_float_vector
import os

db_uri = os.environ.get("_DB_URI")
db_token = os.environ.get("_DB_ROOT_TOKEN")
db_name = os.environ.get("_DB_NAME")

client = db_connection(db_uri, db_token, db_name).get_client()

def get_review_content(question, client):
    # 4. Single vector search
    query_vector = change_float_vector(get_sentence_embedding(question))[0]
    res = client.search(
        collection_name="review_data",
        anns_field="review_vector6",
        data=[query_vector],
        limit=3, # 최대 3개
        #search_params={"metric_type": "IP"},
        output_fields = ['review_varchar6']
    )
    # 리뷰
    answer = []
    for hits in res:
        for hit in hits:
            print(hit)
            answer.append(hit['entity']['review_varchar6'])

    return '\n'.join(answer)


# 화장품 리뷰 쿼리
def get_review_content_price(question, client):
    # 4. Single vector search
    query_vector = change_float_vector(get_sentence_embedding(question))[0]
    res = client.search(
        collection_name="review_data2",
        anns_field="review_vector3",
        data=[query_vector],
        limit=5, # 최대 3개
        #search_params={"metric_type": "IP"},
        output_fields = ['review_varchar3', 'review_product3']
    )
    # 리뷰
    answer = []
    for hits in res:
        for hit in hits:
            print(hit)
            answer.append(f"상품명 : {hit['entity']['review_product3']} 리뷰 : {hit['entity']['review_varchar3']}")

    return '\n'.join(answer)

def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "너는 사용자가 질문을 하면 아래의 내용을 바탕으로 대답해주는 챗봇이야. {content}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


@cl.password_auth_callback
def auth():
    return cl.User(identifier="test")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")

    query_content = get_review_content(message.content, client)

    async for chunk in runnable.astream(
        {"question": message.content, "content" : query_content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)

if __name__ == "__main__":
    ...
