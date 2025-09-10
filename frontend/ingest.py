# ingest.py
# 목적:
#   data/ 폴더 안의 PDF/TXT 문서를 읽어 텍스트 청크로 나눈 뒤
#   Azure OpenAI 임베딩으로 벡터화하여 ChromaDB에 저장.
#   (--faiss 옵션을 쓰면 FAISS 백업 인덱스도 생성)

import argparse     # 커맨드라인 옵션 파싱
import os
import shutil
from typing import List, Tuple

# LangChain + OpenAI 관련 모듈
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# utils.py에서 공통 설정/함수 불러오기
from utils import load_documents, VDB_DIR, AOAI_ENDPOINT, AOAI_API_KEY, DEPLOY_EMBED

# -----------------------------
# 전역 설정값
# -----------------------------
CHUNK_SIZE = 1000          # 청크 크기 (문서 잘라내는 단위)
CHUNK_OVERLAP = 150        # 청크 간 오버랩 (앞뒤 겹치게 해서 맥락 유지)

# -----------------------------
# 환경변수 체크 & 임베딩 준비
# -----------------------------
def _require_env():
    """환경변수(AOAI_ENDPOINT/API_KEY) 설정 여부 확인"""
    if not AOAI_ENDPOINT or not AOAI_API_KEY:
        raise RuntimeError("AOAI_ENDPOINT 또는 AOAI_API_KEY가 비어 있습니다. .env 파일을 확인하세요.")

def get_splitter() -> RecursiveCharacterTextSplitter:
    """텍스트 분할기 생성 (RecursiveCharacterTextSplitter)"""
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len,)

def get_embeddings() -> AzureOpenAIEmbeddings:
    """Azure OpenAI 임베딩 객체 생성"""
    _require_env()
    return AzureOpenAIEmbeddings(
        azure_deployment=DEPLOY_EMBED,  # .env에서 설정된 임베딩 모델명
        api_key=AOAI_API_KEY,
        azure_endpoint=AOAI_ENDPOINT,
        api_version="2024-06-01",
    )

# -----------------------------
# 문서 → 청크 → texts/metadatas
# -----------------------------
def build_payload() -> Tuple[List[str], List[dict]]:
    """
    data/ 폴더에서 문서를 불러와 청크 단위로 분할하고
    텍스트 리스트(texts)와 메타데이터 리스트(metadatas)를 반환.
    """
    docs = load_documents()  # utils.load_documents: [{"path":..., "content":...}]
    if not docs:
        raise SystemExit("data/ 폴더에 TXT/PDF 문서를 넣어주세요.")

    splitter = get_splitter()
    texts, metas = [], []

    for d in docs:
        chunks = splitter.split_text(d["content"])
        for ch in chunks:
            if not ch.strip():
                continue
            texts.append(ch)
            metas.append({"source": d["path"]})  # 각 청크의 출처 경로 기록

    if not texts:
        raise SystemExit("문서는 있었지만 유효한 청크를 만들지 못했습니다.")
    return texts, metas


# -----------------------------
# Chroma / FAISS 빌더
# -----------------------------
def build_chroma(texts: List[str], metas: List[dict], persist_dir: str):
    """ChromaDB 인덱스 생성 및 저장"""
    os.makedirs(persist_dir, exist_ok=True)
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=get_embeddings(),
        metadatas=metas,
        persist_directory=persist_dir,
    )
    vectordb.persist()


def build_faiss(texts: List[str], metas: List[dict], faiss_dir: str):
    """FAISS 인덱스 생성 및 저장 (백업용)"""
    os.makedirs(faiss_dir, exist_ok=True)
    index = FAISS.from_texts(
        texts=texts,
        embedding=get_embeddings(),
        metadatas=metas,
    )
    index.save_local(os.path.join(faiss_dir, "faiss_index"))


# -----------------------------
# 엔트리포인트 (main)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG 인덱스 생성 스크립트")
    parser.add_argument("--rebuild", action="store_true", help="기존 인덱스 삭제 후 재생성")
    parser.add_argument("--faiss", action="store_true", help="FAISS 백업 인덱스도 생성")
    args = parser.parse_args()

    chroma_dir = VDB_DIR  # vectordb/ (utils.py에서 불러옴)
    faiss_dir = os.path.join(VDB_DIR, "faiss_backup")

    # --rebuild 옵션: 기존 인덱스 폴더 삭제
    if args.rebuild:
        if os.path.isdir(chroma_dir):
            shutil.rmtree(chroma_dir)
        if os.path.isdir(faiss_dir):
            shutil.rmtree(faiss_dir)
        print(" 기존 인덱스 디렉토리 삭제 완료.")

    # 문서 로딩 → 청크 생성
    print(" 문서 로딩 및 청크 분할 중…")
    texts, metas = build_payload()
    print(f" 청크 {len(texts)}개 준비 완료.")

    # Chroma 인덱싱
    print(" Chroma 인덱싱 중…")
    build_chroma(texts, metas, chroma_dir)
    print(f" Chroma 인덱스 완료 → {chroma_dir}")

    # --faiss 옵션: FAISS 백업도 함께
    if args.faiss:
        print(" FAISS 백업 인덱스 생성 중…")
        build_faiss(texts, metas, faiss_dir)
        print(f" FAISS 인덱스 저장 → {faiss_dir}")



if __name__ == "__main__":
    main()