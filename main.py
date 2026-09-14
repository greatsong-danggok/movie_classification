# main.py — 기록된 관객 수의 기준 충족 여부를 분류한다
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

MOVIES = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(MOVIES, dtype={"movieCd": str}).sort_values("movieCd").reset_index(drop=True)
    df["장르"] = df["genre"].str.split("|").str[0]
    df["국가"] = df["nation"].str.split("|").str[0]
    return df


df = load_data()
기준 = st.selectbox("성공 기준 (총 관객)", [500_000, 1_000_000, 3_000_000], index=1)
df["성공"] = (df["total_audi"] >= 기준).astype(int)
st.caption(f"성공 {df['성공'].sum()}편 / {len(df)}편")
st.info("기록된 데이터의 분류 연습입니다. 개봉 전 예측 성능을 뜻하지 않습니다.")

is_test = df.index % 10 < 3
features = df[["first_scrn", "first_show", "peak", "장르", "국가"]]
Xtr = pd.get_dummies(features[~is_test])
Xte = pd.get_dummies(features[is_test]).reindex(columns=Xtr.columns, fill_value=0)
X = pd.concat([Xtr, Xte]).sort_index()
is_test = df.index % 10 < 3                      # 열 편 중 앞 세 편을 테스트용으로
st.caption(f"훈련용 {(~is_test).sum()}편 · 테스트용 {is_test.sum()}편 · 전체 {len(df)}편")
Xtr, Xte = X[~is_test], X[is_test]
ytr, yte = df["성공"][~is_test], df["성공"][is_test]

scaler = StandardScaler().fit(Xtr)
logi = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)

prob = logi.predict_proba(scaler.transform(Xte))[:, 1]
st.metric("로지스틱 회귀 정확도", f"{accuracy_score(yte, (prob >= 0.5).astype(int)):.3f}")
확률표 = pd.DataFrame({"영화": df.loc[is_test, "movieNm"].values,
                     "추정 확률": prob,
                     "실제": ["성공" if v else "기준 미달" for v in yte],
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
st.caption("표는 테스트용 영화의 실제 레이블과 추정 확률, 스크린 수, 개봉일입니다.")

choices = df.loc[is_test].copy()
selected = st.selectbox("테스트용 영화", list(range(len(choices))),
                        format_func=lambda i: choices.iloc[i]["movieNm"])
p = float(prob[selected])
st.write(f"추정 성공 확률: {p:.1%}")
st.write("예측:", "성공" if p >= 0.5 else "기준 미달")
st.write("실제 레이블:", "성공" if int(yte.iloc[selected]) else "기준 미달")
st.caption("이 확률은 모델의 추정값이며 실제 성공 가능성을 보장하지 않습니다.")
