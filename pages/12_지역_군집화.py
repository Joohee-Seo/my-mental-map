import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================================================================
# 열두 번째 페이지: 정답 없이(비지도학습) 253개 지역을, 질환
# 구성 패턴이 비슷한 것끼리 K-Means로 자동으로 묶어봐요.
# ================================================================

st.set_page_config(page_title="지역 군집화 (K-Means)", page_icon="🧩")

st.title("🧩 비슷한 질환 패턴끼리 지역 묶어보기")
st.write(
    "이번엔 정답표 없이(비지도학습), 질환 구성 패턴이 서로 비슷한 지역들을 "
    "**K-Means**라는 방법으로 자동으로 그룹 지어봐요."
)
st.info(
    "💡 **비지도학습이란?** 지도학습(11번 페이지)은 '정답(질환명)'을 알려주고 "
    "배우게 했지만, 여기서는 정답 없이 '이 지역들은 서로 비슷하게 생겼다'는 "
    "패턴만 보고 컴퓨터가 스스로 그룹을 나눠요."
)

질환목록 = ["ADHD", "불면증", "우울증", "조울증", "불안장애", "조현병"]


@st.cache_data
def 데이터_불러오기():
    df = pd.read_csv(
        "data/시군구별_성별_주요_정신질환_통계_2024.csv",
        encoding="cp949",
    )
    최신연도 = df["진료년도"].max()
    return df[df["진료년도"] == 최신연도]


df = 데이터_불러오기()

# 부천시/군위군처럼 행정구역 변경으로 남은, 전부 0인 옛 표기 지역은 제외
지역별_합계 = df.groupby(["시도", "시군구"])["환자수"].transform("sum")
df = df[지역별_합계 > 0]

# -----------------------------------------------------------
# 지역x질환 환자수를 펼치고, "비중(%)"으로 바꿔줘요.
#    - 그냥 환자수로 묶으면 인구 많은 대도시끼리만 묶여버려요
#      (규모가 비슷해서). 그래서 "이 지역 환자 중 질환별 비중이
#      몇 %인지"로 바꿔서, 지역 크기와 상관없이 '패턴 모양'
#      자체가 비슷한 곳끼리 묶이게 해요.
# -----------------------------------------------------------
지역별_질환 = df.groupby(["시도", "시군구", "상병구분"], as_index=False)["환자수"].sum()
가로형 = 지역별_질환.pivot_table(
    index=["시도", "시군구"], columns="상병구분", values="환자수", fill_value=0
)[질환목록]

비중 = 가로형.div(가로형.sum(axis=1), axis=0) * 100  # 행마다 합이 100이 되도록

# -----------------------------------------------------------
# 표준화 (StandardScaler)
#    - 질환마다 비중의 들쭉날쭉한 정도가 달라서, 평균 0·표준편차
#      1로 맞춰줘야 K-Means가 특정 질환에 치우치지 않아요.
# -----------------------------------------------------------
scaler = StandardScaler()
표준화_배열 = scaler.fit_transform(비중)

# -----------------------------------------------------------
# 몇 개로 묶을지(k) 고르기 + 엘보우(elbow) 그래프로 참고하기
# -----------------------------------------------------------
st.subheader("1️⃣ 몇 개의 그룹으로 나눌까요?")


@st.cache_data
def 엘보우_계산(_배열):
    결과 = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(_배열)
        결과.append({"k": k, "inertia": km.inertia_})
    return pd.DataFrame(결과)


엘보우 = 엘보우_계산(표준화_배열)
fig_elbow = px.line(엘보우, x="k", y="inertia", markers=True)
fig_elbow.update_layout(
    xaxis_title="그룹 수 (k)",
    yaxis_title="inertia (그룹 안 퍼짐 정도, 낮을수록 촘촘함)",
    height=300,
)
st.plotly_chart(fig_elbow, use_container_width=True)
st.caption(
    "💡 선이 꺾이면서 완만해지는 지점(팔꿈치 모양)이 보통 적당한 그룹 수예요. "
    "그래프를 참고해서 아래에서 그룹 수를 직접 골라보세요."
)

k = st.slider("그룹 수 (k)", min_value=2, max_value=8, value=4)

# -----------------------------------------------------------
# 실제 K-Means 학습
# -----------------------------------------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
군집번호 = kmeans.fit_predict(표준화_배열)

비중_결과 = 비중.reset_index()
비중_결과["군집"] = ("그룹 " + pd.Series(군집번호 + 1).astype(str)).values

# -----------------------------------------------------------
# 2차원으로 압축(PCA)해서 산점도로 보여주기
#    - 실제로는 6개 질환(6차원)으로 그룹을 나눴지만, 사람 눈으로
#      보려면 2차원으로 압축해야 해요. PCA가 그 역할을 해줘요.
# -----------------------------------------------------------
st.subheader("2️⃣ 그룹이 실제로 어떻게 나뉘었는지 (산점도)")
st.write(
    "**이 그래프가 필요한 이유**: 실제로는 질환 6개를 기준으로 그룹을 "
    "나눴지만, 우리 눈에는 2차원 평면으로 압축해서 보여줘야 그룹이 잘 "
    "나뉘었는지 확인할 수 있어요."
)

pca = PCA(n_components=2, random_state=42)
좌표 = pca.fit_transform(표준화_배열)
비중_결과["PCA1"] = 좌표[:, 0]
비중_결과["PCA2"] = 좌표[:, 1]

fig_scatter = px.scatter(
    비중_결과,
    x="PCA1",
    y="PCA2",
    color="군집",
    hover_name="시군구",
    hover_data={"시도": True, "PCA1": False, "PCA2": False},
    category_orders={"군집": [f"그룹 {i+1}" for i in range(k)]},
)
fig_scatter.update_layout(height=520)
st.plotly_chart(fig_scatter, use_container_width=True)

설명분산 = pca.explained_variance_ratio_.sum() * 100
st.caption(
    f"💡 이 2차원 그림은 원래 정보(6개 질환 비중)의 약 {설명분산:.0f}%를 "
    "담고 있어요. 100%가 아니라서 실제 그룹 경계와 살짝 다르게 보일 수 있어요."
)

# -----------------------------------------------------------
# 각 그룹이 어떤 특징을 가졌는지 표로 보여주기 (해석)
# -----------------------------------------------------------
st.subheader("3️⃣ 각 그룹은 어떤 특징이 있을까요?")
st.write(
    "그룹별로 6개 질환의 **평균 비중(%)**을 보여줘요. 다른 그룹보다 눈에 "
    "띄게 높은 질환이 그 그룹의 특징이라고 볼 수 있어요."
)

그룹별_평균 = 비중_결과.groupby("군집")[질환목록].mean().round(1)
그룹별_평균 = 그룹별_평균.reindex([f"그룹 {i+1}" for i in range(k)])

fig_heat = px.imshow(
    그룹별_평균,
    text_auto=True,
    color_continuous_scale="Blues",
    labels=dict(color="평균 비중(%)"),
    aspect="auto",
)
fig_heat.update_layout(height=80 + k * 60)
st.plotly_chart(fig_heat, use_container_width=True)

st.subheader("4️⃣ 각 그룹에는 어떤 지역이 있을까요?")
선택_그룹 = st.selectbox("그룹 선택", [f"그룹 {i+1}" for i in range(k)])
해당_지역 = 비중_결과[비중_결과["군집"] == 선택_그룹][["시도", "시군구"]].reset_index(drop=True)
st.write(f"**{선택_그룹}** — 총 {len(해당_지역)}개 지역")
st.dataframe(해당_지역, use_container_width=True, height=300)

st.caption(
    "⚠️ 그룹 이름(그룹 1, 2, 3...)에는 정해진 의미가 없어요 — 컴퓨터가 "
    "패턴만 보고 나눈 것이라, 그룹 번호 자체보다는 위 표(3번)에서 어떤 "
    "질환 비중이 높은 그룹인지를 보고 해석하는 게 중요해요. 그룹 수(k)를 "
    "바꾸면 결과도 달라질 수 있어요."
)
