import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------
# 두 번째 페이지: 지역(시도)마다 어떤 정신질환이 더 많은지,
# 그리고 두 질환끼리는 어떤 관계가 있는지 살펴보는 페이지예요.
# -----------------------------------------------------------

st.set_page_config(page_title="지역별 질환 비교", page_icon="🗺️")

st.title("🗺️ 지역마다 어떤 정신질환이 더 많을까요?")
st.write(
    "같은 나라 안에서도 지역마다 정신질환의 종류·크기가 다를 수 있어요. "
    "시도별 전체 규모 차이와, 두 질환 사이의 관계를 그래프로 살펴봐요."
)

질환목록 = ["ADHD", "불면증", "우울증", "조울증", "불안장애", "조현병"]


# -----------------------------------------------------------
# 데이터 불러오기 (main.py와 똑같은 방식)
# -----------------------------------------------------------
@st.cache_data
def 데이터_불러오기():
    df = pd.read_csv(
        "data/시군구별_성별_주요_정신질환_통계_2024.csv",
        encoding="cp949",
    )
    최신연도 = df["진료년도"].max()
    return df[df["진료년도"] == 최신연도]


df = 데이터_불러오기()

# =============================================================
# 그래프 1. 시도별 환자수를 질환별로 색칠한 누적 막대그래프
# =============================================================
st.subheader("1️⃣ 시도별 환자수 (질환별 색으로 구분)")
st.write(
    "**이 그래프가 필요한 이유**: 지역마다 전체 환자수 규모가 얼마나 다른지"
    "(수도권 vs 지방), 그리고 그 안에서 어떤 질환이 큰 비중을 차지하는지를 "
    "한 번에 보여줘요."
)

시도_질환별 = df.groupby(["시도", "상병구분"], as_index=False)["환자수"].sum()

# 시도 총합이 큰 순서대로 막대를 나열하기 위한 순서 계산
시도_순서 = (
    시도_질환별.groupby("시도")["환자수"]
    .sum()
    .sort_values(ascending=False)
    .index.tolist()
)

fig_bar = px.bar(
    시도_질환별,
    x="시도",
    y="환자수",
    color="상병구분",
    category_orders={"시도": 시도_순서, "상병구분": 질환목록},
    labels={"환자수": "환자수 (명)", "시도": "시도", "상병구분": "질환"},
)
fig_bar.update_layout(
    xaxis_title="시도 (전체 환자수가 큰 순서)",
    yaxis_title="환자수 (명)",
    legend_title_text="질환",
)
st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================
# 그래프 2. 시군구마다 두 질환을 x축·y축으로 놓은 산점도
# =============================================================
st.subheader("2️⃣ 두 질환, 지역마다 어떻게 다를까요?")
st.write(
    "**이 그래프가 필요한 이유**: 표에는 숫자가 너무 많아서 눈에 잘 안 들어오는 "
    "'이 지역은 A 질환은 많은데 B 질환은 적네?' 같은 편차를, "
    "점 하나(=지역 하나)로 표현하면 한눈에 알아볼 수 있어요."
)

col1, col2 = st.columns(2)
with col1:
    질환x = st.selectbox("↔️ x축 질환", 질환목록, index=질환목록.index("우울증"))
with col2:
    질환y = st.selectbox("↕️ y축 질환", 질환목록, index=질환목록.index("조현병"))

# 시군구 단위로 (시도, 시군구) 조합마다 질환별 환자수를 옆으로 펼쳐줘요.
지역별_질환 = df.groupby(["시도", "시군구", "상병구분"], as_index=False)["환자수"].sum()
가로형 = 지역별_질환.pivot_table(
    index=["시도", "시군구"], columns="상병구분", values="환자수", fill_value=0
).reset_index()

fig_scatter = px.scatter(
    가로형,
    x=질환x,
    y=질환y,
    hover_name="시군구",
    hover_data={"시도": True},
    color_discrete_sequence=["#7C5CFC"],
)
fig_scatter.update_layout(
    xaxis_title=f"{질환x} 환자수 (명)",
    yaxis_title=f"{질환y} 환자수 (명)",
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.info(
    "💡 오른쪽 아래나 왼쪽 위처럼 대각선에서 멀리 떨어진 점일수록, "
    "그 지역은 두 질환의 비중이 유난히 다르다는 뜻이에요."
)
