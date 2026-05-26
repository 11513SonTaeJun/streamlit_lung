import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# -----------------------------------
# 한글 폰트 설정
# -----------------------------------
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux
    plt.rc('font', family='Malgun Gothic')

plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------
# 군집 이름 매핑
# -----------------------------------
cluster_names = {
    0: "매우 건강군",
    1: "폐암 위험군",
    2: "전반적 건강군",
    3: "강한 폐암 위험군"
}

df = pd.read_csv("lung_cancer_examples.csv")
model = joblib.load('lung_model.pkl')
scaler = joblib.load('lung_scaler.pkl')

# -----------------------------------
# Streamlit 제목
# -----------------------------------
st.title("폐 건강 군집 예측 시스템 2026")

st.write("귀하의 정보를 입력하면 폐 건강도를 예측합니다.")

# -----------------------------------
# 사용자 입력
# -----------------------------------
age = st.number_input("나이 입력", min_value=0.0, max_value=120.0, value=30.0)
smokes = st.number_input("흡연 여부 입력 (빈도에 기반)", min_value=0.0, max_value=1.0, value=0.0)
areaQ = st.number_input("공기질 입력 (숫자가 클수록 좋은 공기질)", value=50.0)

# -----------------------------------
# 예측 버튼
# -----------------------------------
if st.button("군집 예측"):

    # 새로운 환자 데이터 생성
    new_patient = pd.DataFrame(
        [[age, smokes, areaQ]],
        columns=['나이', '흡연 여부', '공기질']
    )

    # 스케일링
    new_patient_scaled = scaler.transform(new_patient)

    # 군집 예측
    pred_cluster = model.predict(new_patient_scaled)

    cluster_num = pred_cluster[0]
    cluster_result = cluster_names.get(cluster_num, "알 수 없음")

    # 결과 출력
    st.success(f"이 환자는 {cluster_num}번 군집 ({cluster_result})에 속합니다.")

    # -----------------------------------
    # 그래프 출력
    # -----------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        df['나이'],
        df['흡연 여부'],
        c=df['cluster'],
        alpha=0.5,
        cmap='viridis'
    )

    # 새 환자 표시
    ax.scatter(
        age,
        smokes,
        c='black',
        s=300,
        marker='X',
        label='새 환자'
    )

    ax.set_xlabel('나이')
    ax.set_ylabel('흡연 여부')
    ax.set_title('환자 군집 시각화')

    ax.legend()

    st.pyplot(fig)