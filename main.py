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
st.caption(f"훈련용 {(~is_test).sum()}편 · 테스트용 {is_test.sum()}편 · 전체 {len(df)}편")

scaler = StandardScaler().fit(X[~is_test])
logi = LogisticRegression(max_iter=2000).fit(scaler.transform(X[~is_test]), y[~is_test])

prob = logi.predict_proba(scaler.transform(X[is_test]))[:, 1]
st.metric("로지스틱 회귀 정확도", f"{accuracy_score(y[is_test], (prob >= 0.5).astype(int)):.3f}")
확률표 = pd.DataFrame({"영화": df.loc[is_test, "movieNm"].values,
                     "추정 확률": prob.round(3),
                     "실제": ["성공" if v else "기준 미달" for v in y[is_test]],
                     "스크린 수": df.loc[is_test, "first_scrn"].values,
                     "누적 관객": df.loc[is_test, "total_audi"].values,
                     "개봉일": pd.to_datetime(df.loc[is_test, "openDt"], format="%Y%m%d",
                                           errors="coerce").dt.strftime("%Y.%m.%d").values})
확률표 = 확률표.sort_values("추정 확률", ascending=False)
fig = px.scatter(확률표, x="스크린 수", y="추정 확률", color="실제", hover_name="영화",
                 log_x=True, color_discrete_map={"성공": "#b07a00", "기준 미달": "#2b7fd6"})
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
