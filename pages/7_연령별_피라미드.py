import glob

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------
# 일곱 번째 페이지: 질환마다 몇 살 때 환자가 제일 많은지,
# 남녀를 좌우로 나눠서 인구 피라미드 모양으로 살펴봐요.
# -----------------------------------------------------------

st.set_page_config(page_title="연령대별 분포", page_icon="🎂")

st.title("🎂 이 질환, 몇 살 때 가장 많을까요?")
st.write(
    "질환마다 환자가 몰리는 나이대가 달라요. 왼쪽은 여자, 오른쪽은 남자로 "
    "나눠서, 나이대별 환자수를 피라미드 모양으로 살펴봐요."
)

질환목록 = ["ADHD", "불면증", "우울증", "조울증", "불안장애", "조현병"]
연령_순서 = [
    "0~9세", "10~19세", "20~29세", "30~39세", "40~49세", "50~59세",
    "60~69세", "70~79세", "80~89세", "90~99세", "100세이상",
]


# -----------------------------------------------------------
# 데이터 불러오기
#    - 파일 이름을 하드코딩하지 않고, data 폴더 안에서 이름에
#      '연령'이 들어간 CSV를 자동으로 찾아요. 이렇게 하면 파일
#      이름이 조금 달라도(원본 그대로 올렸어도) 잘 동작해요.
# -----------------------------------------------------------
@st.cache_data
def 데이터_불러오기():
    후보파일 = [f for f in glob.glob("data/*.csv") if "연령" in f]
    if not 후보파일:
        st.error(
            "❌ data 폴더 안에서 '연령'이라는 글자가 들어간 CSV 파일을 "
            "찾지 못했어요. 시군구·성별·연령별 통계 CSV가 data 폴더 "
            "안에 올라가 있는지 확인해주세요."
        )
        st.stop()

    파일경로 = 후보파일[0]
    try:
        df = pd.read_csv(파일경로, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(파일경로, encoding="utf-8")

    최신연도 = df["진료년도"].max()
    df = df[df["진료년도"] == 최신연도]
    # 파일마다 질환 컬럼 이름이 조금 달라서(상병구분/상별구분) 통일해줘요.
    df = df.rename(columns={"상별구분": "상병구분"})
    return df


df = 데이터_불러오기()

col1, col2 = st.columns(2)
with col1:
    질환 = st.selectbox("🔍 질환 선택", 질환목록, index=질환목록.index("우울증"))
with col2:
    지역범위 = st.selectbox("📍 범위", ["전국"] + sorted(df["시도"].unique().tolist()))

st.write(
    f"**이 그래프가 필요한 이유**: '{질환}'이 어느 나이대에 몰려 있는지는 표 "
    "숫자만 봐서는 잘 안 들어와요. 피라미드 모양으로 그리면 나이대별 쏠림이 "
    "바로 보여요."
)

# -----------------------------------------------------------
# 선택한 질환·범위로 걸러서, 나이대x성별로 환자수 합치기
# -----------------------------------------------------------
d = df[df["상병구분"] == 질환]
if 지역범위 != "전국":
    d = d[d["시도"] == 지역범위]

나이별 = d.groupby(["연령구분", "성별"], as_index=False)["환자수"].sum()
나이별["연령구분"] = pd.Categorical(나이별["연령구분"], categories=연령_순서, ordered=True)
나이별 = 나이별.sort_values("연령구분")

여성 = 나이별[나이별["성별"] == "여"].set_index("연령구분").reindex(연령_순서)["환자수"].fillna(0)
남성 = 나이별[나이별["성별"] == "남"].set_index("연령구분").reindex(연령_순서)["환자수"].fillna(0)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=연령_순서,
        x=-여성,  # 왼쪽으로 그리기 위해 음수로
        name="여자",
        orientation="h",
        marker_color="#FF6B9D",
        customdata=여성,
        hovertemplate="%{y} 여자: %{customdata:,.0f}명<extra></extra>",
    )
)
fig.add_trace(
    go.Bar(
        y=연령_순서,
        x=남성,
        name="남자",
        orientation="h",
        marker_color="#4C9AFF",
        hovertemplate="%{y} 남자: %{x:,.0f}명<extra></extra>",
    )
)

최댓값 = max(여성.max(), 남성.max())
fig.update_layout(
    barmode="relative",
    bargap=0.1,
    xaxis=dict(
        title="환자수 (명) — 왼쪽 여자 · 오른쪽 남자",
        tickvals=[-최댓값, -최댓값 / 2, 0, 최댓값 / 2, 최댓값],
        ticktext=[
            f"{최댓값:,.0f}", f"{최댓값/2:,.0f}", "0",
            f"{최댓값/2:,.0f}", f"{최댓값:,.0f}",
        ],
    ),
    yaxis_title="연령대",
    height=560,
    legend_title_text="성별",
)
st.plotly_chart(fig, use_container_width=True)

# 환자수가 가장 많은 나이대를 자동으로 찾아서 알려주기
전체_나이별 = 나이별.groupby("연령구분")["환자수"].sum()
최다연령 = 전체_나이별.idxmax()
st.info(f"💡 '{질환}'({지역범위})은 **{최다연령}**에서 환자수가 가장 많아요.")

st.caption(
    "⚠️ 나이대가 세분화되어 있어 10명 미만 마스킹(=0 표기)의 영향을 더 많이 "
    "받을 수 있어요. 막대가 짧다고 반드시 실제로 적은 건 아닐 수 있어요."
)
