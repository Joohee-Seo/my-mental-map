import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# ================================================================
# 열한 번째 페이지: 머신러닝으로 "지역 · 나이대 · 성별"을 알려주면,
# 그 조건에서 6개 질환 중 어떤 질환의 비중이 높았는지 예측해봐요.
# ================================================================

st.set_page_config(page_title="질환 예측기 (머신러닝)", page_icon="🤖")

st.title("🤖 머신러닝으로 질환 경향 예측해보기")

# ----------------------------------------------------------------
# ⚠️ 아주 중요한 안내부터 먼저 드릴게요.
#    이 페이지가 실제로 무엇을 예측하는 건지 정확히 알아야
#    오해 없이 쓸 수 있어요.
# ----------------------------------------------------------------
st.warning(
    "**이 예측기가 실제로 하는 일**\n\n"
    "이 자료에는 '전체 인구 중 몇 명이 병에 걸렸는지'를 알 수 있는 "
    "**인구수 정보가 없어요**. 병원에서 진료받은 **환자수**만 있어요.\n\n"
    "그래서 이 페이지는 '이 지역에 살면 병에 걸릴 확률'을 예측하는 게 "
    "**아니에요**. 대신 **'이미 이 6개 질환 중 하나로 진료받은 환자들 "
    "중에서, 지역·나이대·성별이 비슷한 사람은 어떤 질환으로 진료받은 "
    "경우가 많았는지'**를 예측해요. 즉, '걸릴 확률'이 아니라 "
    "**'진료받는다면 어떤 질환일 가능성이 높은지(질환 간 상대적 비중)'**"
    "예요.\n\n"
    "또한 2024년 한 해 자료만 있어서 '앞으로'의 추세를 학습한 것도 "
    "아니에요. **의학적 진단이나 실제 발병 확률로는 절대 쓰시면 안 "
    "되고, 머신러닝 학습 연습용으로만 봐주세요.**"
)

질환목록 = ["ADHD", "불면증", "우울증", "조울증", "불안장애", "조현병"]
연령_순서 = [
    "0~9세", "10~19세", "20~29세", "30~39세", "40~49세", "50~59세",
    "60~69세", "70~79세", "80~89세", "90~99세", "100세이상",
]


# ----------------------------------------------------------------
# 데이터 불러오기 (7번 페이지와 같은 연령별 CSV를 사용해요)
# ----------------------------------------------------------------
@st.cache_data
def 데이터_불러오기():
    df = pd.read_csv(
        "data/시군구별_성별_연령별_주요_정신질환_통계_2024.csv",
        encoding="cp949",
    )
    최신연도 = df["진료년도"].max()
    df = df[df["진료년도"] == 최신연도]
    df = df.rename(columns={"상별구분": "상병구분"})
    return df


원본 = 데이터_불러오기()

# 시군구는 빼고 시도 단위로 합쳐요(지역을 너무 잘게 쪼개면 자료가
# 부족해져서 모델이 잘 못 배워요). 마스킹된(환자수=0) 조합도 빼요.
학습자료 = (
    원본.groupby(["시도", "연령구분", "성별", "상병구분"], as_index=False)["환자수"]
    .sum()
)
학습자료 = 학습자료[학습자료["환자수"] > 0]


# ----------------------------------------------------------------
# 모델 학습 (환자수를 '가중치'로 써서, 환자가 많은 조합을 모델이
# 더 중요하게 배우도록 해요). 앱이 켜질 때 한 번만 학습하도록
# 캐시에 저장해둬요.
# ----------------------------------------------------------------
@st.cache_resource
def 모델_학습(학습자료: pd.DataFrame):
    X = pd.get_dummies(학습자료[["시도", "연령구분", "성별"]])
    y = 학습자료["상병구분"]
    가중치 = 학습자료["환자수"]

    모델 = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42
    )
    모델.fit(X, y, sample_weight=가중치)
    return 모델, X.columns.tolist()


모델, 학습컬럼 = 모델_학습(학습자료)

st.divider()
st.subheader("🔍 조건을 넣고 예측해보기")

col1, col2, col3 = st.columns(3)
with col1:
    선택_시도 = st.selectbox("시도", sorted(학습자료["시도"].unique().tolist()))
with col2:
    선택_연령 = st.selectbox("연령대", 연령_순서)
with col3:
    선택_성별 = st.radio("성별", ["남", "여"], horizontal=True)

if st.button("🤖 예측하기", type="primary"):
    # 사용자가 고른 조건을, 모델이 배울 때 썼던 것과 똑같은 형태
    # (원-핫 인코딩)로 만들어줘요.
    입력 = pd.DataFrame(
        [{"시도": 선택_시도, "연령구분": 선택_연령, "성별": 선택_성별}]
    )
    입력_인코딩 = pd.get_dummies(입력)
    입력_인코딩 = 입력_인코딩.reindex(columns=학습컬럼, fill_value=0)

    확률 = 모델.predict_proba(입력_인코딩)[0]
    결과 = pd.DataFrame({"질환": 모델.classes_, "예측비중(%)": (확률 * 100).round(1)})
    결과 = 결과.sort_values("예측비중(%)", ascending=False)

    st.write(
        f"**{선택_시도} · {선택_연령} · {선택_성별}** 조건에서, "
        "진료받는다면 어떤 질환일 가능성이 높은지의 상대적 비중이에요 "
        "(6개 질환 비중의 합 = 100%)."
    )

    fig = px.bar(
        결과,
        x="예측비중(%)",
        y="질환",
        orientation="h",
        color="예측비중(%)",
        color_continuous_scale="Blues",
        category_orders={"질환": 결과["질환"].tolist()},
        text="예측비중(%)",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="예측 비중 (%)",
        coloraxis_showscale=False,
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    최상위 = 결과.iloc[0]
    st.info(f"💡 이 조건에서는 **{최상위['질환']}**의 비중이 {최상위['예측비중(%)']}%로 가장 높게 나왔어요.")

st.divider()

# ----------------------------------------------------------------
# 모델이 예측할 때 무엇을 가장 많이 참고했는지 보여주기
#    - 원-핫 인코딩된 컬럼들을 원래 특성(시도/연령대/성별)별로
#      합쳐서, "어떤 요인이 가장 중요했는지" 간단히 보여줘요.
# ----------------------------------------------------------------
st.subheader("📊 모델이 예측할 때 무엇을 가장 중요하게 봤을까요?")

중요도 = pd.Series(모델.feature_importances_, index=학습컬럼)


def 원래_특성(컬럼명: str) -> str:
    if 컬럼명.startswith("시도_"):
        return "시도(지역)"
    if 컬럼명.startswith("연령구분_"):
        return "연령대"
    if 컬럼명.startswith("성별_"):
        return "성별"
    return 컬럼명


특성별_중요도 = 중요도.groupby(원래_특성).sum().sort_values(ascending=False)

fig_importance = px.bar(
    x=특성별_중요도.values,
    y=특성별_중요도.index,
    orientation="h",
    labels={"x": "중요도", "y": ""},
    color=특성별_중요도.values,
    color_continuous_scale="Purples",
)
fig_importance.update_layout(coloraxis_showscale=False, height=300)
st.plotly_chart(fig_importance, use_container_width=True)

st.caption(
    "💡 막대가 길수록, 모델이 질환을 구분할 때 그 특성(지역/연령대/성별)을 "
    "더 많이 참고했다는 뜻이에요. 보통 '연령대'가 가장 크게 나오는 경우가 "
    "많아요 — 질환마다 주로 나타나는 나이대 차이가 크기 때문이에요."
)
