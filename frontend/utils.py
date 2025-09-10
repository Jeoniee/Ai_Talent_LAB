import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")
DEPLOY_EMBED = os.getenv("AOAI_DEPLOY_EMBED_3_SMALL")
VDB_DIR = "./vectordb"

# 문서 로드 & 임베딩 & Chroma 저장
def load_and_ingest():
    loader = TextLoader("data/death_penalty_guide.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    splits = splitter.split_documents(docs)

    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL"),
        azure_endpoint=os.getenv("AOAI_ENDPOINT"),
        api_key=os.getenv("AOAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-05-01-preview"),
    )

    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VDB_DIR,
    )
    print("💖 벡터 DB 생성 완료")
    return splits

# Retriever 반환
def get_retriever():
    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL"),
        azure_endpoint=os.getenv("AOAI_ENDPOINT"),
        api_key=os.getenv("AOAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-05-01-preview"),
    )

    db = Chroma(
        persist_directory=VDB_DIR,
        embedding_function=embeddings,
    )
    return db.as_retriever(search_kwargs={"k": 3})
