# utils.py
# 프로젝트 전체에서 공통으로 쓰이는 함수와 환경설정을 모아두는 파일

import os
from typing import List, Dict
from dotenv import load_dotenv  # .env 파일(환경변수) 불러오는 라이브러리
from pypdf import PdfReader     # PDF 문서를 파싱해서 텍스트 추출하는 라이브러리

load_dotenv()

AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")
DEPLOY_GPT4O_MINI = os.getenv("AOAI_DEPLOY_GPT4O_MINI", "gpt-4o-mini")
DEPLOY_EMBED = os.getenv("AOAI_DEPLOY_EMBED_3_SMALL", "text-embedding-3-small")

DATA_DIR = os.getenv("DATA_DIR", "data")
VDB_DIR = os.getenv("VDB_DIR", "vectordb")
SUPPORTED_EXT = {".pdf", ".txt"}                 # 지원 파일 확장자

# PDF 파일 읽어서 텍스트 추출하는 함수
def read_pdf(path:str) -> str:
    try:
        reader = PdfReader(path)
        return "\n".join([p.extract_text() for p in reader.pages])
    except Exception as e:
        return f"Error: {e}"

# txt 파일 읽기
def read_txt(path:str) -> str:
    # 파일 열기 (UTF-8 인코딩, 오류 무시)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# 폴더 안의 PDF/TXT 문서들을 모두 로드
def load_documents(folder:str= DATA_DIR) -> List[Dict[str, str]]:
    """
    폴더 아래의 PDF/TXT 파일을 모두 읽어
    [{ 'path': <절대경로>, 'content': <텍스트> }, ...] 형태로 반환
    """
    docs: List[Dict[str, str]] = []
    for root, _, files in os.walk(folder):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in SUPPORTED_EXT:
                continue
            full = os.path.join(root, fn)
            text = read_pdf(full) if ext == ".pdf" else read_txt(full)
            if text and text.strip():
                docs.append({"path": os.path.abspath(full), "content": text})
    return docs


if __name__ == "__main__":
    docs = load_documents()
    print(f"[문서 개수] {len(docs)}")
    if docs:
        print(f"[첫 문서 경로] {docs[0]['path']}")
        print(f"[미리보기]\n{docs[0]['content'][:300]}")
