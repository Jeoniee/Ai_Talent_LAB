# ⚖️ AI Legal Debate Agent (검사·변호사·판사 MultiAgent) — README

모의 법정 주제를 주면 **검사·변호사·판사**가 라운드별로 토론하고, **판사**가 근거와 함께 요약·권고를 내리는 **LangGraph + RAG + Streamlit** 기반 데모입니다.

* **핵심**: 멀티 에이전트(역할 분리) · RAG(법률/판례 스니펫 인용) · Streamlit 실시간 UI
* **목표**: 교육/토론 훈련/정책 토론용 시뮬레이터 — 근거 중심, 균형 잡힌 시각 제시
* **적용 도메인**: 로스쿨/법학 교육, 시민교육, 조직 내 정책 토론 툴, 에듀테크

---

## ✨ Features

* **역할 분리 멀티에이전트**: 검사(Prosecution) · 변호사(Defense) · 판사(Judge)
* **라운드 토론**: 개시 → 반박 ↔ 재반박 → 판사 요약/권고
* **RAG 근거 인용**: 토론마다 관련 법/판례 스니펫 및 **출처 표기**
* **Few-shot + CoT 프롬프트**: 논거 구조화(원칙→판례→정책효과→반례) 강제
* **Streamlit UI**: 주제 입력 → 토론 보기(라운드 타임라인) → 결과 저장
* **FastAPI 백엔드, Docker 배포, Redis 메모리(멀티턴 유지)

---

## 🏗️ Architecture

```
Streamlit(UI)
   └─(prompt/topic)→  LangGraph(StateGraph)
                         ├─ Planner (주제→플로우 계획)
                         ├─ Retriever (RAG, k=4~5)
                         ├─ Prosecution Agent (검사)
                         ├─ Defense Agent (변호사)
                         ├─ Judge Agent (판사)
                         └─ Writer/Reporter (요약·권고·출처)
Vector DB (Chroma / FAISS)  ←  법률/판례/가이드 문서 임베딩
LLM (Azure OpenAI GPT-4o-mini) / Embeddings (text-embedding-3-small)
```

---

## 📂 Project Structure

```
project/
├── data/                  # 법률/판례/가이드 텍스트 또는 PDF
├── vectordb/              # RAG 인덱스 저장 경로(자동 생성)
├── utils.py               # .env, 파일 로더, 공통 유틸
├── ingest.py              # 임베딩/인덱싱 스크립트 (Chroma/FAISS)
├── prompts.py             # 역할/샷 프롬프트 정의
├── graph.py               # LangGraph 멀티에이전트 플로우
├── streamlit_app.py       # Streamlit UI
├── requirements.txt       # 의존성
└── .env                   # AOAI 키/엔드포인트 (커밋 금지)
```

---

## 🔧 Setup

### 1) Python 가상환경

```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows PowerShell
# .venv\Scripts\Activate.ps1
```

### 2) Install

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Environment (.env)

```
AOAI_ENDPOINT=https://<your-azure-openai-endpoint>.openai.azure.com/
AOAI_API_KEY=<your-api-key>
AOAI_DEPLOY_GPT4O_MINI=gpt-4o-mini
AOAI_DEPLOY_EMBED_3_SMALL=text-embedding-3-small
DATA_DIR=data
VDB_DIR=vectordb
```

> ⚠️ `.env`는 절대 커밋하지 말 것. `.gitignore`에 추가.

---

## 🗂️ Data Preparation

`data/` 폴더에 다음 유형 문서를 넣으세요. (PDF/TXT 권장)

* 헌법/형법 요약, 형사소송법 기본서 일부
* 대법원 판례 요지 모음(연습용 요약 텍스트)
* 국제 인권 규약 요지(ICCPR 등)
* 법철학/정책 보고서 일부(형벌의 목적·억지효과 논쟁 등)

> 스캔 PDF는 텍스트 추출 실패 가능 → 가급적 텍스트 기반 문서 사용 또는 TXT로 변환.

---

## 🧬 Build Vector Index (RAG)

```bash
python ingest.py
```

* `data/` 내 문서를 1000자/150 오버랩으로 청크 분할 → **Chroma**에 저장.
* (옵션) `ingest.py`에서 **FAISS**도 병행 저장 가능.

---

## ▶️ Run (Streamlit)

```bash
streamlit run streamlit_app.py
```

* 주제 예시: `사형제도 유지 vs 폐지`, `혐오표현 규제 vs 표현의 자유`, `영장주의 예외 확대 여부`
* **실행 흐름**: 주제 입력 → Planner → (Retriever) → 검사 → 변호사 → 재반박 → 판사 요약/권고 → 출처 표시

---

## 🧠 Prompt Engineering (요지)

* **Planner (System)**:

  * 입력 주제를 토대로 토론 라운드 계획(JSON) 산출
  * `{ "rounds": ["prosecution_opening", "defense_rebuttal", "cross", "judge_summary"], "need_retrieval": true }`
* **Prosecution/Defense (System)**:

  * 역할·톤·논리틀(CoT) 강제: `[법률원칙→판례→정책효과→반례 대응]` 순서로 작성
  * “반대측의 핵심 주장 1\~2개를 명시적으로 요약·반박하라”
* **Judge (System)**:

  * 양측 주요 논거를 **균형 있고 검증 가능**하게 요약
  * 판례/규정 우선순위에 따라 권고 결론 + 한계·추가 검토사항 제시
* **Few-shot**:

  * 토론 샘플 2개 제공(사형제, 혐오표현 규제) — 형식·톤을 학습

> 모든 발언 블록은 **“출처” 섹션**을 하단에 필수 포함하도록 지시.

---

## 🧩 LangGraph Flow

```
[entry] → Planner
          ├─ need_retrieval? → Retriever(k=4~5, with sources)
          └─ (skip)
        → Prosecution(opener)
        → Defense(rebuttal)
        → Prosecution(rebuttal)
        → Defense(rebuttal)
        → Judge(summary+recommendation)
        → Writer/Reporter(formatting with citations) → [END]
```

* **Memory(선택)**: 앞선 발언을 다음 라운드로 전달하여 반박 맥락 유지
* **Tool(ReAct, 선택)**: `stat_lookup`, `case_finder`, `term_define` 등 보조 도구 호출

---

## 🗃️ RAG Strategy

* **Splitter**: `RecursiveCharacterTextSplitter(chunk=1000, overlap=150)`
* **Vector DB**: Chroma(`vectordb/`) — 영구 폴더, FAISS 백업(옵션)
* **검색**: similarity\_search k=4~~5, 스니펫 400~~800자 요약 적용
* **정합성 가드**: Writer에 “출처 미존재 주장 금지, 모호할 경우 ‘근거 불충분’으로 표기” 규칙 삽입

---

## 🧪 Demo Prompts

* *"사형제도 유지 vs 폐지에 대해 토론해줘."*
* *"혐오표현 규제 법제화의 정당성과 한계를 토론해줘."*
* *"영장주의 예외 확대의 필요성과 위험을 토론해줘."*

---

## 🖥️ Streamlit UX Tips

* 토론 타임라인(라운드별 카드) + 역할 아바타(⚖️, 🧑‍⚖️, 👨‍💼, 👩‍💼)
* “근거 보기” 토글로 스니펫 펼치기
* 결과 **Markdown/HTML 저장** 또는 **TXT 다운로드** 버튼

---

## (선택) FastAPI Endpoints

```http
POST /index        # 문서 인덱싱
POST /debate       # {"topic": "..."} → 토론 결과 반환
POST /explain      # 특정 발언의 근거/판례 확장 설명
```

* Streamlit은 프론트, FastAPI는 백엔드 API로 분리하면 확장 용이.

---

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

```bash
docker build -t legal-debate-agent .
docker run -p 8501:8501 --env-file .env -v $(pwd)/vectordb:/app/vectordb legal-debate-agent
```

---

## ✅ Compliance Checklist (과제 평가 대응)

* **Prompt Engineering**: 역할/CoT/Few-shot/출력 강제(JSON·출처)
* **LangChain & LangGraph**: 다중 에이전트 플로우 + (선택)Memory/Tool(ReAct)
* **RAG**: 전처리·임베딩·k검색·스니펫 인용·출처 표기
* **서비스화**: Streamlit UI (+ 선택: FastAPI/Docker)
* **운영 품질**: .env 키 관리, 모듈화, 로그 마스킹(선택)

---

## 🧯 Troubleshooting

* **No documents found**: `data/` 폴더에 TXT/PDF 넣었는지 확인 → `python ingest.py`
* **스캔 PDF 텍스트 누락**: OCR된 TXT로 변환 후 사용 권장
* **키/엔드포인트 오류**: `.env` 값 재확인, 네트워크(사내망) 확인
* **메모리 누락(멀티턴 맥락 깨짐)**: LangGraph MemorySaver 활성화 여부 확인

---

## 🗺️ Roadmap

* 판례 메타데이터(사건번호/연도/법원) 정교화
* 실시간 스트리밍 토론(토큰 단위 업데이트)
* 역할 추가(배심원, 학자, 정책분석가)
* 채점 기준 루브릭(논리/근거/정확성/공정성) 도입

---

## 🔐 Security / Ethics

* 법률 자문 대체 아님(교육/토론용) 고지
* 편향/오정보 방지: 출처 강제, 반대 주장 요약·반박 의무화
* 민감 주제 토론 시 중립성과 책임 있는 서술 유지

---

## 📝 License

* 교육/연구 목적으로 자유 사용(사내 규정 따름). 상용 전환 시 별도 검토 권장.

---

## 🙏 Acknowledgements

* Azure OpenAI, LangChain/LangGraph, Chroma/FAISS 커뮤니티
* (선택) 공개 법률 요약/학술 자료 제공 기관
