import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# ================================================================
# 열네 번째 페이지: 6개 질환 패턴이 "다른 지역들과 확연히 다른"
# 이례적인 지역을, Isolation Forest가 자동으로 찾아줘요.
# 5번 페이지(상관관계)에서 사람 눈으로 찾던 걸, 이번엔 모델이
# 대신 해주는 셈이에요.
# ================================================================

st.set_page_config(page_title="이상치 탐지", page_icon="🚨")

st.title("🚨 유난히 튀는 지역, 자동으로 찾기")
st.write(
    "질환 6개의 구성 패턴이 다른 지역들과 확연히 다른 곳을, "
    "**Isolation Forest**라는 방법으로 자동으로 찾아봐요."
)
st.info(
    "💡 **Isolation Forest란?** 데이터를 이것저것 무작위로 나눠보는데, "
    "'평범한' 지역은 다른 지역들과 뒤섞여 있어서 나누기 어렵고, "
    "'이례적인' 지역은 몇 번만 나눠도 금방 혼자 떨어져 나와요. 그 "
    "'금방 떨어져 나오는 정도'로 이상치를 판단하는 방법이에요."
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
# 지역x질환 환자수를 "비중(%)"으로 바꿔줘요 (12번 페이지와 같은
# 이유예요 — 인구 많은 대도시가 아니라 '패턴 모양' 자체가 다른
# 지역을 찾고 싶어서예요).
# -----------------------------------------------------------
지역별_질환 = df.groupby(["시도", "시군구", "상병구분"], as_index=False)["환자수"].sum()
가로형 = 지역별_질환.pivot_table(
    index=["시도", "시군구"], columns="상병구분", values="환자수", fill_value=0
)[질환목록]
비중 = 가로형.div(가로형.sum(axis=1), axis=0) * 100

# -----------------------------------------------------------
# 이상치로 볼 비율 정하기
# -----------------------------------------------------------
비율 = st.slider("이상치로 볼 지역 비율(%)", min_value=1, max_value=20, value=5)

모델 = IsolationForest(contamination=비율 / 100, random_state=42, n_estimators=200)
이상치_라벨 = 모델.fit_predict(비중)  # -1이면 이상치, 1이면 정상
이상치_점수 = -모델.score_samples(비중)  # 점수가 높을수록 더 이례적

결과 = 비중.reset_index()
결과["구분"] = ["이상치" if x == -1 else "정상" for x in 이상치_라벨]
결과["이상치_점수"] = 이상치_점수.round(3)

st.caption(f"전체 {len(결과)}개 지역 중 **{(결과['구분']=='이상치').sum()}개**를 이상치로 찾았어요.")

# -----------------------------------------------------------
# PCA로 2차원 압축해서, 이상치가 어디 있는지 산점도로 보여주기
# -----------------------------------------------------------
st.subheader("1️⃣ 이상치가 어디쯤 있는지 (산점도)")
st.write(
    "**이 그래프가 필요한 이유**: 6개 질환 비중을 한눈에 보기 어려우니, "
    "2차원으로 압축해서 이상치(빨간색)가 정상 지역들(회색) 무리에서 "
    "얼마나 떨어져 있는지 보여줘요."
)

pca = PCA(n_components=2, random_state=42)
좌표 = pca.fit_transform(비중)
결과["PCA1"] = 좌표[:, 0]
결과["PCA2"] = 좌표[:, 1]

fig_scatter = px.scatter(
    결과,
    x="PCA1",
    y="PCA2",
    color="구분",
    color_discrete_map={"정상": "lightgray", "이상치": "crimson"},
    hover_name="시군구",
    hover_data={"시도": True, "이상치_점수": True, "PCA1": False, "PCA2": False},
    category_orders={"구분": ["정상", "이상치"]},
)
fig_scatter.update_layout(height=520)
st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------------------------------
# 이상치 목록 표
# -----------------------------------------------------------
st.subheader("2️⃣ 어떤 지역들이 이상치로 나왔을까요?")
이상치_목록 = (
    결과[결과["구분"] == "이상치"]
    .sort_values("이상치_점수", ascending=False)[
        ["시도", "시군구", "이상치_점수"] + 질환목록
    ]
    .reset_index(drop=True)
)
st.dataframe(
    이상치_목록.style.format({col: "{:.1f}" for col in 질환목록}),
    use_container_width=True,
)

# -----------------------------------------------------------
# 이상치 하나 골라서, 왜 이상치인지(전국 평균과 비교) 보여주기
# -----------------------------------------------------------
if len(이상치_목록) > 0:
    st.subheader("3️⃣ 왜 이 지역이 이상치일까요?")
    이상치_이름 = (이상치_목록["시도"] + " " + 이상치_목록["시군구"]).tolist()
    선택 = st.selectbox("이상치 지역 선택", 이상치_이름)
    선택_시도, 선택_시군구 = 선택.split(" ", 1)

    해당지역_비중 = 비중.loc[(선택_시도, 선택_시군구)]
    전국평균_비중 = 비중.mean()

    비교표 = pd.DataFrame(
        {"이 지역 비중(%)": 해당지역_비중, "전국 평균 비중(%)": 전국평균_비중}
    ).round(1)

    fig_compare = go.Figure()
    fig_compare.add_trace(
        go.Bar(x=질환목록, y=비교표["이 지역 비중(%)"], name=선택, marker_color="crimson")
    )
    fig_compare.add_trace(
        go.Bar(
            x=질환목록, y=비교표["전국 평균 비중(%)"], name="전국 평균",
            marker_color="lightgray",
        )
    )
    fig_compare.update_layout(
        barmode="group", yaxis_title="비중(%)", legend_title_text="", height=400
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    st.caption("💡 빨간 막대와 회색 막대의 차이가 클수록, 그 질환이 이 지역을 이상치로 만든 이유예요.")

st.caption(
    "⚠️ 이상치라고 해서 '나쁘다'는 뜻은 아니에요 — 그냥 '전형적인 패턴과 "
    "다르다'는 통계적인 의미예요. 위쪽 슬라이더로 비율을 바꾸면 어떤 "
    "지역이 이상치로 잡히는지도 달라져요."
)
