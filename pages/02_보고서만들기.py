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

# 도전 — 모델은 무엇을 보고 맞혔나
st.subheader("모델은 무엇을 보고 맞혔나")
중요도 = pd.Series(tree.feature_importances_, index=X.columns)
중요도 = 중요도[중요도 > 0].sort_values(ascending=False).round(3).reset_index()
중요도.columns = ["속성", "중요도"]
st.dataframe(중요도, hide_index=True)
중요도그림 = px.pie(중요도, names="속성", values="중요도", hole=0.35)
중요도그림.update_traces(textinfo="label+percent", sort=False)
중요도그림.update_layout(height=300, showlegend=False)
st.plotly_chart(중요도그림, width="stretch")
st.caption("모두 더하면 1이 되므로 조각의 크기가 그대로 기여한 비율입니다.")
가중치 = pd.Series(logi.coef_[0], index=X.columns).round(3)
가중치 = 가중치.reindex(가중치.abs().sort_values(ascending=False).index).reset_index()
가중치.columns = ["속성", "가중치"]
st.dataframe(가중치, hide_index=True)
st.caption(f"절편 {logi.intercept_[0]:.3f} · 가중치는 표준화한 값에 곱하는 수입니다. "
           "양수면 성공 쪽으로, 음수면 기준 미달 쪽으로 밉니다.")

# ④ 팀장에게 보고하기 — 최근 개봉작은 어떻게 될까 (앞 코드 끝에 이어 붙일 부분)
import datetime as dt

st.subheader("팀장에게 보고하기 — 최근 개봉작 예측")
일수 = st.slider("최근 며칠 안에 개봉한 영화를 볼까요", 7, 90, 60)

개봉일 = pd.to_datetime(df["openDt"], format="%Y%m%d", errors="coerce")
최근 = (개봉일 >= pd.Timestamp(dt.date.today()) - pd.Timedelta(days=일수)).fillna(False)

if not 최근.any():
    st.info("이 기간에 개봉한 영화가 없습니다. 기간을 늘려 보세요.")
else:
    확률 = logi.predict_proba(scaler.transform(X[최근]))[:, 1]
    보고서 = pd.DataFrame({
        "영화": df.loc[최근, "movieNm"].values,
        "개봉일": 개봉일[최근].dt.strftime("%Y-%m-%d").values,
        "스크린 수": df.loc[최근, "first_scrn"].values,
        "지금까지 관객": df.loc[최근, "total_audi"].values,
        "성공 확률": 확률.round(3),
        "로지스틱 회귀": ["성공" if p >= 0.5 else "기준 미달" for p in 확률],
        "의사결정트리": ["성공" if v else "기준 미달" for v in tree.predict(X[최근])],
        "실제": ["이미 넘음" if a >= 기준 else "아직" for a in df.loc[최근, "total_audi"]],
        "학습에 사용": ["아니오" if t else "예" for t in pd.Series(is_test, index=df.index)[최근]],
    }).sort_values("성공 확률", ascending=False)
    st.dataframe(보고서, hide_index=True, width="stretch")
    갈림 = int((보고서["로지스틱 회귀"] != 보고서["의사결정트리"]).sum())
    st.caption(f"최근 {일수}일 안에 개봉한 {len(보고서)}편입니다. "
               f"두 모델의 판단이 갈린 영화는 {갈림}편입니다. "
               "'아직'인 영화는 정답이 나오지 않았습니다. "
               "학습에 사용한 영화를 잘 맞히는 것은 당연하므로 그 줄은 성능의 근거가 되지 않습니다.")

import re

st.subheader("이 영화를 트리는 어떻게 판단했나")
고른영화 = st.selectbox("보고서의 영화 중 하나", list(보고서["영화"]))
행 = df.index[df["movieNm"] == 고른영화][0]
경로 = tree.decision_path(X.loc[[행]]).indices

단계 = []
for 순서, 마디 in enumerate(경로, start=1):
    if t.feature[마디] < 0:
        단계.append({"단계": "끝", "트리가 던진 질문": "더 나누지 않는다",
                     "이 영화의 값": "", "답": f"판정: {답(마디)}",
                     "이 마디의 훈련용 영화": 마디설명(마디)})
    else:
        속성 = X.columns[t.feature[마디]]
        값 = float(X.loc[행, 속성])
        단계.append({"단계": f"{순서}번째", "트리가 던진 질문": 질문(마디),
                     "이 영화의 값": f"{속성} {값:g}",
                     "답": "예 → 왼쪽" if 값 <= float(t.threshold[마디]) else "아니오 → 오른쪽",
                     "이 마디의 훈련용 영화": 마디설명(마디)})
st.dataframe(pd.DataFrame(단계), hide_index=True, width="stretch")
st.caption(f"{고른영화}이(가) 트리를 내려간 길입니다. 마지막 줄의 답이 표에 적힌 의사결정트리 판정입니다.")

# 그 길을 트리 그림에 붉은 테두리로 표시한다
그림 = export_graphviz(tree, out_file=None, feature_names=list(X.columns),
                     class_names=["기준 미달", "성공"], filled=True, rounded=True)
for 마디 in 경로:
    그림 = re.sub(rf'(?m)^({마디}) \[label=', r'\1 [penwidth=5, color="#d64545", label=', 그림)
st.graphviz_chart(그림)
st.caption("붉은 테두리가 이 영화가 지나간 마디입니다.")
