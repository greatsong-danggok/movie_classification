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
기준 = st.selectbox("성공 기준 (총 관객)", [100_000, 300_000, 500_000, 1_000_000, 3_000_000], index=3)
df["성공"] = (df["total_audi"] >= 기준).astype(int)
st.caption(f"성공 {df['성공'].sum()}편 / {len(df)}편")
st.info("수집된 누적 관객 기록의 분류 연습입니다. 개봉 전 예측 성능을 뜻하지 않습니다.")

is_test = pd.Series(df.index % 10 < 3, index=df.index)   # 열 편 중 앞 세 편을 테스트용으로
X = df[["first_scrn", "first_show", "peak"]]
y = df["성공"]
st.caption(f"훈련용 {(~is_test).sum()}편 · 테스트용 {is_test.sum()}편 · 전체 {len(df)}편")

scaler = StandardScaler().fit(X[~is_test])
logi = LogisticRegression(max_iter=2000).fit(scaler.transform(X[~is_test]), y[~is_test])

prob = logi.predict_proba(scaler.transform(X[is_test]))[:, 1]
st.metric("로지스틱 회귀 정확도", f"{accuracy_score(y[is_test], (prob >= 0.5).astype(int)):.3f}")
확률표 = pd.DataFrame({"영화": df.loc[is_test, "movieNm"].values,
                     "추정 확률": prob,
                     "실제": ["성공" if v else "기준 미달" for v in y[is_test]],
                     "스크린 수": df.loc[is_test, "first_scrn"].values,
                     "개봉일": df.loc[is_test, "openDt"].values})
확률표 = 확률표.sort_values("추정 확률", ascending=False)
fig = px.bar(확률표, x="영화", y="추정 확률", color="실제",
             color_discrete_map={"성공": "#b07a00", "기준 미달": "#b9b3a5"})
fig.add_hline(y=0.5, line_dash="dash", annotation_text="문턱값 0.5")
fig.update_layout(xaxis_tickangle=-60, height=460, xaxis_title=None)
st.plotly_chart(fig, width="stretch")
st.caption("막대는 테스트용 영화의 추정 확률입니다. 색은 실제 레이블입니다.")
st.dataframe(확률표, width="stretch")

choices = df.loc[is_test].copy()
selected = st.selectbox("테스트용 영화", list(range(len(choices))),
                        format_func=lambda i: choices.iloc[i]["movieNm"])
p = float(prob[selected])
st.write(f"추정 성공 확률: {p:.1%}")
st.write("예측:", "성공" if p >= 0.5 else "기준 미달")
st.write("실제 레이블:", "성공" if int(y[is_test].iloc[selected]) else "기준 미달")

# 의사결정트리를 학습하고 두 모델의 정확도를 나란히 본다
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz

# 트리는 값의 크기에 영향받지 않으므로 표준화하지 않은 X를 그대로 사용한다
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[~is_test], y[~is_test])

st.subheader("두 모델의 정확도")
c1, c2 = st.columns(2)
c1.metric("로지스틱 회귀 정확도", f"{accuracy_score(y[is_test], (prob >= 0.5).astype(int)):.3f}")
c2.metric("의사결정트리 정확도", f"{accuracy_score(y[is_test], tree.predict(X[is_test])):.3f}")

st.subheader("트리는 어떤 질문을 먼저 던지나")
st.graphviz_chart(export_graphviz(tree, out_file=None, feature_names=list(X.columns),
                                  class_names=["기준 미달", "성공"], filled=True))
st.code(export_text(tree, feature_names=list(X.columns), max_depth=3))

# 도전 — 모델은 무엇을 보고 맞혔나 (앞 코드 끝에 이어 붙일 부분)
st.subheader("도전 — 모델은 무엇을 보고 맞혔나")
중요도 = pd.Series(tree.feature_importances_, index=X.columns)
중요도 = 중요도[중요도 > 0].sort_values(ascending=False).round(3).reset_index()
중요도.columns = ["속성", "중요도"]
st.dataframe(중요도, hide_index=True)
st.caption("중요도는 이 모델이 무엇을 단서로 삼았는지를 보여 줄 뿐, 그것이 원인이라는 뜻은 아닙니다.")

# 로지스틱 회귀는 부호가 있는 가중치로 답한다
가중치 = pd.Series(logi.coef_[0], index=X.columns).round(3)
가중치 = 가중치.reindex(가중치.abs().sort_values(ascending=False).index).reset_index()
가중치.columns = ["속성", "가중치"]
st.dataframe(가중치.head(10), hide_index=True)
st.caption(f"절편 {logi.intercept_[0]:.3f} · 가중치는 표준화한 값에 곱하는 수입니다. "
           "양수면 성공 쪽으로, 음수면 기준 미달 쪽으로 밉니다.")
