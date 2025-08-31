# 지금까지 배운 RAG 기법을 자유롭게 활용하여 나만의 에이전트를 만들어보세요
# 정보보안기사 요약집 RAG Agent

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# pdf 로드
try :
    loader = PyMuPDFLoader("정보보안기사 필기 요약(합본).pdf")
    docs = loader.load()
except Exception as e:
    print(f"pdf 로드 중 오류 발생: {e}")
    exit(1)

# pdf 텍스트가 잘 뽑히는지 먼저 확인
# print(len(docs), "pages loaded")
# print(docs[0].page_content[:300])

# 텍스트 splitter
text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # 한 덩어리 최대 길이
                    chunk_overlap=100 # 덩어리 간 중복
                )

# Splitter 실행
splits = text_splitter.split_documents(docs)
# print(f"총 {len(splits)} 개의 청크로 분할")
# print(splits[0].page_content[:300])


# Indexing (Vector DB)
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT")
)

vectorstore = FAISS.from_documents(splits, embeddings)

# Retriever
# retriever = vectorstore.as_retriever()

# 검색 파라미터 최적화 MMR
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.7})

# LLM + QA 체인
llm = AzureChatOpenAI(
    model="gpt-4o-mini",    # or os.getenv("AOAI_DEPLOY_GPT4O_MINI")
    api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    api_version="2024-02-01"
)
qa = RetrievalQA.from_chain_type(llm, retriever=retriever)

# test
# query = input("대칭키와 비대칭키의 차이를 설명해줘")
query = input("질문을 입력하세요: ")

result = qa.run(query)
print("💖 ---질문--- 💖 :", query)
print("🫧 ---답변--- 🫧 :", result)