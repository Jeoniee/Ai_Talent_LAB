import streamlit as st
import requests

st.set_page_config(page_title="AI 토론 에이전트", layout="wide")
st.title("⚖️ AI 토론 에이전트 (FastAPI 버전)")

topic = st.text_input("토론 주제를 입력하세요:", "사형제도 유지 vs 폐지")

if st.button("토론 시작"):
    with st.spinner("토론 진행 중..."):
        res = requests.post("http://localhost:8001/debate", json={"topic": topic})
        if res.status_code == 200:
            result = res.json()
            st.success("토론 완료")
            st.subheader("최종 보고서")
            st.write(result["final_report"])
        else:
            st.error(f"API 호출 실패: {res.status_code}")
