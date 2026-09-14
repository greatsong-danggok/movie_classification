# main.py — 수집된 누적 관객 수가 기준을 넘는지 분류한다
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

MOVIES = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"


@st.cache_data
def load_data():
    return pd.read_csv(MOVIES, dtype={"movieCd": str}).sort_values("movieCd").reset_index(drop=True)


df = load_data()
기준 = st.selectbox("성공이라고 부를 기준 (누적 관객)",
                  [100_000, 300_000, 500_000, 1_000_000, 3_000_000], index=3,
                  format_func=lambda v: f"{v // 10_000}만 명 이상 ({v:,}명)")
df["성공"] = (df["total_audi"] >= 기준).astype(int)
st.caption(f"누적 관객 {기준:,}명 이상을 성공으로 봅니다 · "
           f"전체 {len(df):,}편 가운데 성공 {df['성공'].sum():,}편")
st.info("수집된 누적 관객 기록의 분류 연습입니다. 개봉 전 예측 성능을 뜻하지 않습니다.")

is_test = pd.Series(df.index % 10 < 3, index=df.index)   # 열 편 중 앞 세 편을 테스트용으로
X = df[["first_scrn", "first_show", "peak"]].rename(
    columns={"first_scrn": "스크린 수", "first_show": "상영 횟수", "peak": "성수기"})
y = df["성공"]

scaler = StandardScaler().fit(X[~is_test])
logi = LogisticRegression(max_iter=2000).fit(scaler.transform(X[~is_test]), y[~is_test])

prob = logi.predict_proba(scaler.transform(X[is_test]))[:, 1]
확률표 = pd.DataFrame({"영화": df.loc[is_test, "movieNm"].values,
                     "추정 확률": prob.round(3),
                     "실제": ["성공" if v else "기준 미달" for v in y[is_test]],
                     "스크린 수": df.loc[is_test, "first_scrn"].values,
                     "누적 관객": df.loc[is_test, "total_audi"].values,
                     "개봉일": pd.to_datetime(df.loc[is_test, "openDt"], format="%Y%m%d",
                                           errors="coerce").dt.strftime("%Y.%m.%d").values})
확률표 = 확률표.sort_values("추정 확률", ascending=False)
fig = px.scatter(확률표, x="스크린 수", y="추정 확률", color="실제", hover_name="영화",
                 log_x=True, color_discrete_map={"성공": "#d64545", "기준 미달": "#2b7fd6"})
fig.add_hline(y=0.5, line_dash="dash", annotation_text="문턱값 0.5")
fig.update_traces(marker=dict(size=11, opacity=0.8))
fig.update_layout(height=420, xaxis_title="스크린 수(개) · 로그 눈금", yaxis_title="추정 성공 확률")
st.plotly_chart(fig, width="stretch")
st.caption("점 하나가 테스트용 영화 한 편입니다. 색은 실제 레이블이고, "
           "가로 점선 위가 성공으로 예측한 영화입니다. 점에 마우스를 올리면 제목이 보입니다.")
st.dataframe(확률표, width="stretch", hide_index=True, column_config={
    "추정 확률": st.column_config.ProgressColumn("추정 확률", min_value=0.0, max_value=1.0, format="%.2f"),
    "스크린 수": st.column_config.NumberColumn("스크린 수", format="%,d개"),
    "누적 관객": st.column_config.NumberColumn("누적 관객", format="%,d명"),
})

choices = df.loc[is_test].copy()
selected = st.selectbox("테스트용 영화", list(range(len(choices))),
                        format_func=lambda i: choices.iloc[i]["movieNm"])
p = float(prob[selected])
st.write(f"추정 성공 확률: {p:.1%}")
st.write("예측:", "성공" if p >= 0.5 else "기준 미달")
st.write("실제 레이블:", "성공" if int(y[is_test].iloc[selected]) else "기준 미달")

# 의사결정트리를 학습하고, 트리가 던지는 질문을 한글로 읽는다
import math
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# 트리는 값의 크기에 영향받지 않으므로 표준화하지 않은 X를 그대로 사용한다
# 끝마디에 영화가 다섯 편 미만이면 더 나누지 않는다. 한두 편으로 만든 규칙을 막는다
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0).fit(X[~is_test], y[~is_test])

st.subheader("두 모델의 정확도")
c1, c2 = st.columns(2)
c1.metric("로지스틱 회귀 정확도", f"{accuracy_score(y[is_test], (prob >= 0.5).astype(int)):.3f}")
c2.metric("의사결정트리 정확도", f"{accuracy_score(y[is_test], tree.predict(X[is_test])):.3f}")

단위 = {"스크린 수": "개", "상영 횟수": "회"}
t = tree.tree_
열이름 = list(X.columns)


def 질문(마디):
    """분기 조건을 한글 질문으로 바꾼다. '예'가 왼쪽 가지다."""
    속성 = 열이름[t.feature[마디]]
    경계 = float(t.threshold[마디])
    if 속성 == "성수기":
        return "성수기에 개봉하지 않았는가?"
    return f"{속성}가 {math.floor(경계):,}{단위.get(속성, '')} 이하인가?"


def 마디설명(마디):
    편수 = int(t.n_node_samples[마디])
    성공 = int(round(t.value[마디][0][1] * 편수))
    return f"영화 {편수}편 중 성공 {성공}편"


def 답(마디):
    편수 = int(t.n_node_samples[마디])
    성공 = int(round(t.value[마디][0][1] * 편수))
    return "성공" if 성공 * 2 > 편수 else "기준 미달"


def 트리읽기(마디=0, 들여=""):
    """트리를 질문과 예·아니오로 읽어 내려간다."""
    if t.feature[마디] < 0:
        return [f"판정: {답(마디)}  ({마디설명(마디)})"]
    줄 = [f"{질문(마디)}  ({마디설명(마디)})"]
    for 답변, 자식 in (("예", t.children_left[마디]), ("아니오", t.children_right[마디])):
        아래 = 트리읽기(자식, 들여)
        줄.append(f"{들여}   {답변} → {아래[0]}")
        줄.extend(f"{들여}        {x}" for x in 아래[1:])
    return 줄


st.subheader("트리는 어떤 질문을 던지나")
st.code("\n".join(트리읽기()), language=None)
st.caption("위에서부터 질문에 답하며 내려가면 마지막 줄의 판정에 닿습니다. "
           "괄호 안은 그 마디에 있는 훈련용 영화의 편수입니다. "
           "끝마디의 판정은 그 안에서 편수가 많은 쪽입니다. "
           "성공과 기준 미달이 같은 수이면 기준 미달로 판정합니다.")

with st.expander("트리 그림으로 보기"):
    st.graphviz_chart(export_graphviz(tree, out_file=None, feature_names=열이름,
                                      class_names=["기준 미달", "성공"], filled=True, rounded=True))
