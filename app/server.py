from app.core import AgentState
from app.prompts.prompt01 import CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT
from fastapi import FastAPI
from langserve import add_routes
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import setup_logging
import os

# 로깅 설정
logger = setup_logging()

# FastAPI 앱 생성
app = FastAPI (
    title="AIFRED RAG Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces with LangGraph",
)


def load_vector_database() -> VectorStoreRetriever:
    '''Vector database 로드'''
    FAISS_INDEX_PATH = os.environ["FAISS_INDEX_PATH"]
    embeddings = OpenAIEmbeddings()
    
    try:
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"----- FAISS index loaded from: {FAISS_INDEX_PATH}")
    except ValueError as e:
        logger.error(f"Error loading FAISS index: {e}")
        logger.info("Attempting to load index without embeddings...")
        db = FAISS.load_local(FAISS_INDEX_PATH, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded without embeddings. Applying embeddings now.")
        db.embeddings = embeddings

    retriever = db.as_retriever(search_kwargs={"k": 1})
    logger.info(f"----- FAISS retriever created")
    logger.info(f"----- Number of items in FAISS index: {len(db.index_to_docstore_id)}")

    return retriever


def get_context(agentState: AgentState) -> AgentState:
    '''컨텍스트를 조회하는 함수'''
    query = agentState['question']

    logger.info(f"####### query = {query}")

    retriever = load_vector_database()
    docs = retriever.get_relevant_documents(query)
    logger.info("####### docs = " +str(docs))
    
    for doc in docs:
        agentState["context"] += doc.page_content + "\n"

    return agentState



def generate_response(state: AgentState) -> AgentState:
    claude_model = ChatAnthropic(model="claude-3-sonnet-20240229")

    prompt = ANSWER_PROMPT.format(context=state['context'], question=state['question'])
    state['response'] = claude_model.invoke(prompt)
    return state


workflow = StateGraph(AgentState)
workflow.add_node("get_context", get_context)
workflow.add_node("generate_response", generate_response)
workflow.set_entry_point("get_context")
workflow.add_edge("get_context", "generate_response")
workflow.add_edge("generate_response", END)

chain = workflow.compile()
# chain.with_types(input_type=ChatHistory)


# 라우트 추가
add_routes(app, chain, path="/prompt", enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)