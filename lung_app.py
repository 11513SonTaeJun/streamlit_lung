import streamlit as st
import pandas as pd
import joblib
import platform
from pathlib import Path
import os
import requests

# -----------------------------------
# 페이지 설정
# -----------------------------------
st.set_page_config(
    page_title="폐 건강 군집 예측 시스템",
    page_icon="🫁",
    layout="wide"
)

# -----------------------------------
# 커스텀 CSS (모던 UI)
# -----------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #4F8BF9, #6DD5FA);
    color: white;
    border-radius: 12px;
    border: none;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #3a7be0, #55c6f0);
    color: white;
}

.metric-card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.title-text {
    font-size: 42px;
    font-weight: 800;
    color: #222;
}

.sub-text {
    font-size: 18px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# matplotlib 한글 폰트 설정
# -----------------------------------
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import os

# -----------------------------------
# 군집 이름
# -----------------------------------
cluster_names = {
    0: "매우 건강군",
    1: "폐암 위험군",
    2: "전반적 건강군",
    3: "강한 폐암 위험군"
}

# -----------------------------------
# 데이터 로드
# -----------------------------------
df = pd.read_csv("lung_cancer_examples.csv")
model = joblib.load("lung_model.pkl")
scaler = joblib.load("lung_scaler.pkl")

# -----------------------------------
# 제목 영역
# -----------------------------------
st.markdown("""
<div class="metric-card">
    <div class="title-text">🫁 폐 건강 군집 예측 시스템 2026</div>
    <div class="sub-text">
        입력된 정보를 기반으로 폐 건강 상태 군집을 예측합니다.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# 입력 방식 선택
# -----------------------------------
input_mode = st.radio(
    "입력 방식을 선택하세요",
    ["게이지바 조절", "직접 입력"],
    horizontal=True
)

# -----------------------------------
# 입력 UI
# -----------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)

    if input_mode == "게이지바 조절":
        age = st.slider("나이", 0, 120, 30)
    else:
        age = st.number_input(
            "나이 입력",
            min_value=0,
            max_value=120,
            value=30
        )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)

    if input_mode == "게이지바 조절":
        smokes = st.slider(
            "흡연 여부",
            0.0,
            1.0,
            0.0,
            step=0.1
        )
    else:
        smokes = st.number_input(
            "흡연 여부 입력",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )

    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)

    if input_mode == "게이지바 조절":
        areaQ = st.slider(
            "공기질",
            0,
            100,
            50
        )
    else:
        areaQ = st.number_input(
            "공기질 입력",
            min_value=0,
            max_value=100,
            value=50
        )

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# 예측 버튼
# -----------------------------------
if st.button("🔍 군집 예측 시작"):

    # 새 환자 데이터
    new_patient = pd.DataFrame(
        [[age, smokes, areaQ]],
        columns=['나이', '흡연 여부', '공기질']
    )

    # 스케일링
    new_patient_scaled = scaler.transform(new_patient)

    # 예측
    pred_cluster = model.predict(new_patient_scaled)

    cluster_num = pred_cluster[0]
    cluster_result = cluster_names.get(cluster_num, "알 수 없음")

    # 결과 표시
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="color:#4F8BF9;">
            예측 결과: {cluster_num}번 군집
        </h2>
        <h3 style="color:#222222;">
    🩺 {cluster_result}
</h3>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------
    # 시각화
    # -----------------------------------
    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 7))

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

# -----------------------------------
# Plotly 시각화
# -----------------------------------
import plotly.express as px

# 기존 데이터 시각화
fig = px.scatter(
    df,
    x='나이',
    y='흡연 여부',
    color=df['cluster'].astype(str),
    color_discrete_sequence=px.colors.qualitative.Vivid,
    title='폐 건강 군집 시각화',
    labels={
        '나이': '나이',
        '흡연 여부': '흡연 여부',
        'color': '군집'
    }
)

# 새 환자 추가
fig.add_scatter(
    x=[age],
    y=[smokes],
    mode='markers',
    marker=dict(
        size=20,
        color='red',
        symbol='x'
    ),
    name='새 환자'
)

# UI 스타일
fig.update_layout(
    template='plotly_white',
    height=700,
    title_font_size=24,
    font=dict(
        family='Arial',
        size=14
    )
)

# 출력
st.plotly_chart(fig, use_container_width=True)
