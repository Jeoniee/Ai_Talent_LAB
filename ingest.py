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