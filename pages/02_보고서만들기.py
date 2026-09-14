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

# 입력으로 사용할 속성을 고른다. 이 아래 전부가 여기서 고른 속성을 따른다
PEOPLE = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_people.csv"


@st.cache_data
def load_people():
    return pd.read_csv(PEOPLE, encoding="utf-8-sig", dtype={"movieCd": str})


확장 = df.merge(load_people()[["movieCd", "actors_n", "lead_n", "showTm", "watchGrade", "company"]],
              on="movieCd", how="left").reset_index(drop=True)
확장["테스트용"] = 확장.index % 10 < 3        # 속성을 바꿔도 훈련용·테스트용 구분은 그대로 둔다
확장["청소년관람불가"] = (확장["watchGrade"] == "청소년관람불가").astype(float)
큰곳 = 확장.loc[~확장["테스트용"], "company"].value_counts().head(5).index   # 훈련용에서만 정한다
확장["대형 배급사"] = 확장["company"].isin(큰곳).astype(float)
확장 = 확장.rename(columns={"first_scrn": "스크린 수", "first_show": "상영 횟수", "peak": "성수기",
                         "actors_n": "배우 수", "lead_n": "주연 수", "showTm": "상영시간"})
후보 = ["스크린 수", "상영 횟수", "성수기", "배우 수", "주연 수", "상영시간", "청소년관람불가", "대형 배급사"]
고른속성 = st.multiselect("입력으로 사용할 속성", 후보, default=["스크린 수", "상영 횟수", "성수기"])
if not 고른속성:
    st.warning("속성을 하나 이상 골라 주세요.")
    st.stop()

쓸수있음 = 확장[고른속성].notna().all(axis=1)   # 고른 속성의 값이 빠진 영화는 이번 학습에서 뺀다
df = 확장.loc[쓸수있음].reset_index(drop=True)
is_test = df.pop("테스트용")
X, y = df[고른속성], df["성공"]
if is_test.sum() < 5 or (~is_test).sum() < 20:
    st.warning("고른 속성으로 사용할 수 있는 영화가 너무 적습니다. 속성을 바꿔 보세요.")
    st.stop()
st.caption(f"{len(고른속성)}가지 입력 · 값이 모두 있는 영화 {len(df):,}편으로 학습합니다"
           f"(전체 {len(확장):,}편). "
           "배우 수·주연 수·상영시간·관람등급·배급사는 7차시 실험실에서 사용한 인물 표에서 가져옵니다. "
           "속성을 바꾸면 아래 화면이 모두 다시 계산됩니다.")

scaler = StandardScaler().fit(X[~is_test])
logi = LogisticRegression(max_iter=2000).fit(scaler.transform(X[~is_test]), y[~is_test])

prob = logi.predict_proba(scaler.transform(X[is_test]))[:, 1]
확률표 = pd.DataFrame({"영화": df.loc[is_test, "movieNm"].values,
                     "추정 확률": prob.round(3),
                     "실제": ["성공" if v else "기준 미달" for v in y[is_test]],
                     "스크린 수": df.loc[is_test, "스크린 수"].values,
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

단위 = {"스크린 수": "개", "상영 횟수": "회", "배우 수": "명", "주연 수": "명", "상영시간": "분"}
예아니오 = {"성수기": "성수기에 개봉하지 않았는가?",
          "청소년관람불가": "청소년관람불가 등급이 아닌가?",
          "대형 배급사": "대형 배급사가 아닌가?"}
t = tree.tree_
열이름 = list(X.columns)


def 조사(말):
    """받침이 있으면 '이', 없으면 '가'를 붙인다."""
    끝 = ord(말[-1])
    받침 = (끝 - 0xAC00) % 28 if 0xAC00 <= 끝 <= 0xD7A3 else 0
    return 말 + ("이" if 받침 else "가")


def 질문(마디):
    """분기 조건을 한글 질문으로 바꾼다. '예'가 왼쪽 가지다."""
    속성 = 열이름[t.feature[마디]]
    경계 = float(t.threshold[마디])
    if 속성 in 예아니오:                       # 0과 1뿐인 속성은 경계가 0.5다
        return 예아니오[속성]
    return f"{조사(속성)} {math.floor(경계):,}{단위.get(속성, '')} 이하인가?"


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

# 도전 — 두 모델은 무엇을 보고 판단했나
st.subheader("의사결정트리가 본 것 — 속성 중요도")
중요도 = pd.Series(tree.feature_importances_, index=X.columns)
중요도 = 중요도[중요도 > 0].sort_values(ascending=False).round(3).reset_index()
중요도.columns = ["속성", "중요도"]
중요도그림 = px.pie(중요도, names="속성", values="중요도", hole=0.35)
중요도그림.update_traces(textinfo="label+percent", sort=False)
중요도그림.update_layout(height=300, showlegend=False)
st.plotly_chart(중요도그림, width="stretch")
st.caption("중요도는 트리가 그 속성으로 얼마나 많이, 얼마나 크게 나눴는지를 나타냅니다. "
           "모두 더하면 1이고, 0 이상이라 확률을 높이는 쪽인지 낮추는 쪽인지는 알 수 없습니다. "
           "이 값은 트리에만 있습니다.")

st.subheader("로지스틱 회귀가 본 것 — 가중치")
가중치 = pd.Series(logi.coef_[0], index=X.columns).round(3).reset_index()
가중치.columns = ["속성", "가중치"]
가중치["방향"] = ["성공 확률을 올림" if v > 0 else "성공 확률을 내림" for v in 가중치["가중치"]]
가중치그림 = px.bar(가중치.sort_values("가중치"), x="가중치", y="속성", orientation="h",
                color="방향", color_discrete_map={"성공 확률을 올림": "#d64545", "성공 확률을 내림": "#2b7fd6"})
가중치그림.add_vline(x=0, line_width=1.5, line_color="#1c2230")
가중치그림.update_layout(height=300, yaxis_title=None, legend_title_text="")
st.plotly_chart(가중치그림, width="stretch")
st.caption(f"절편 {logi.intercept_[0]:.3f} · 가중치는 표준화한 값에 곱하는 수입니다. "
           "막대가 0보다 오른쪽이면 그 값이 클수록 성공 확률이 올라가고, 왼쪽이면 내려갑니다. "
           "중요도와 달리 부호가 있고, 모두 더해도 1이 되지 않습니다.")

# ④ 팀장에게 보고하기 — 최근 개봉작은 어떻게 될까
import datetime as dt

st.subheader("팀장에게 보고하기 — 최근 개봉작 예측")
일수 = st.slider("최근 며칠 안에 개봉한 영화를 볼까요", 7, 90, 60)

개봉일 = pd.to_datetime(df["openDt"], format="%Y%m%d", errors="coerce")
최근 = (개봉일 >= pd.Timestamp(dt.date.today()) - pd.Timedelta(days=일수)).fillna(False)

if not 최근.any():
    st.info("이 기간에 개봉한 영화가 없습니다. 기간을 늘려 보세요.")
else:
    확률 = logi.predict_proba(scaler.transform(X[최근]))[:, 1]

    def 신호(p):
        if p >= 0.8:
            return "🔥 대박 조짐"
        if p >= 0.5:
            return "✨ 기대작"
        if p >= 0.2:
            return "🌱 지켜보기"
        return "💤 조용"

    보고서 = pd.DataFrame({
        "신호": [신호(p) for p in 확률],
        "영화": df.loc[최근, "movieNm"].values,
        "성공 확률": 확률.round(3),
        "개봉일": 개봉일[최근].dt.strftime("%m월 %d일").values,
        "스크린 수": df.loc[최근, "스크린 수"].values,
        "지금까지 관객": df.loc[최근, "total_audi"].values,
        "로지스틱 회귀": ["성공" if p >= 0.5 else "기준 미달" for p in 확률],
        "의사결정트리": ["성공" if v else "기준 미달" for v in tree.predict(X[최근])],
        "실제": ["✅ 이미 넘음" if a >= 기준 else "⏳ 아직" for a in df.loc[최근, "total_audi"]],
    }).sort_values("성공 확률", ascending=False)

    대박 = 보고서[보고서["신호"] == "🔥 대박 조짐"]["영화"].tolist()
    기대 = 보고서[보고서["신호"] == "✨ 기대작"]["영화"].tolist()
    if 대박:
        st.success("🔥 대박 조짐: " + " · ".join(대박))
    if 기대:
        st.info("✨ 기대작: " + " · ".join(기대))
    if not 대박 and not 기대:
        st.warning("이번 기간에는 모델이 기준을 넘을 것으로 본 영화가 없습니다. 기간이나 기준을 바꿔 보세요.")

    st.dataframe(보고서, hide_index=True, width="stretch", column_config={
        "성공 확률": st.column_config.ProgressColumn("성공 확률", min_value=0.0, max_value=1.0, format="%.2f"),
        "지금까지 관객": st.column_config.NumberColumn("지금까지 관객", format="%,d명"),
        "스크린 수": st.column_config.NumberColumn("스크린 수", format="%,d개"),
    })
    갈림 = int((보고서["로지스틱 회귀"] != 보고서["의사결정트리"]).sum())
    st.caption(f"최근 {일수}일 안에 개봉한 {len(보고서)}편입니다. "
               f"기준은 누적 관객 {기준 // 10_000}만 명({기준:,}명) 이상입니다. "
               f"두 모델의 판정이 나뉜 영화는 {갈림}편입니다. "
               "⏳인 영화는 정답이 아직 나오지 않았습니다.")

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
