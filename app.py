import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="BuzzGuard", layout="wide")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed"
RESULT_PATH = ROOT / "results"

COORD_COLS = ["sync_hashtag_score", "duplicate_content_score"]


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH / "xgb_feature_AB.pkl")
    encoder = joblib.load(MODEL_PATH / "label_encoder.pkl")
    explainer = shap.TreeExplainer(model)
    return model, encoder, explainer


@st.cache_data
def load_sample():
    df = pd.read_csv(DATA_PATH / "feature_AB_sample.csv").fillna(0)
    return df


model, encoder, explainer = load_model()
FEATURES = list(model.feature_names_in_)

st.title("BuzzGuard")
st.caption("Deteksi bot (Tahap 1, supervised) dan indikasi buzzer lewat pola koordinasi (Tahap 2, unsupervised)")

tab_demo, tab_model, tab_about = st.tabs(["Demo", "Perbandingan Model", "Tentang"])

# ---------------- DEMO ----------------
with tab_demo:
    src = st.radio("Sumber data", ["Sample dataset", "Upload CSV"], horizontal=True)

    if src == "Sample dataset":
        df = load_sample()
        n = st.slider("Jumlah akun sampel", 200, min(2000, len(df)), 500, step=100)
        df = df.sample(n, random_state=42).reset_index(drop=True)
    else:
        up = st.file_uploader("CSV dengan kolom fitur seperti feature_AB.csv", type="csv")
        if up is None:
            st.stop()
        df = pd.read_csv(up).fillna(0)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Kolom hilang: {missing}")
            st.stop()
        if "user_id" not in df.columns:
            df["user_id"] = range(len(df))

    X = df[FEATURES]
    proba = model.predict_proba(X)
    human_idx = list(encoder.classes_).index("human")
    df["pred_label"] = encoder.inverse_transform(model.predict(X))
    df["p_human"] = proba[:, human_idx]

    # Tahap 2: skor koordinasi hanya untuk akun yang diprediksi human
    pct = st.slider("Persentil ambang koordinasi (Tahap 2)", 80, 99, 90)
    humans = df["pred_label"] == "human"
    df["final_label"] = df["pred_label"]
    if humans.sum() > 0:
        thr = df.loc[humans, COORD_COLS].quantile(pct / 100)
        flag = humans & (
            (df["sync_hashtag_score"] >= thr["sync_hashtag_score"])
            | (df["duplicate_content_score"] >= thr["duplicate_content_score"])
        )
        df.loc[flag, "final_label"] = "indikasi_buzzer"

    c1, c2, c3 = st.columns(3)
    c1.metric("Diprediksi bot", int((df["final_label"] == "bot").sum()))
    c2.metric("Human (koordinasi rendah)", int((df["final_label"] == "human").sum()))
    c3.metric("Indikasi buzzer", int((df["final_label"] == "indikasi_buzzer").sum()))

    st.caption("Indikasi buzzer adalah flag pola koordinasi, bukan vonis pasti.")
    st.dataframe(
        df[["user_id", "pred_label", "p_human", "final_label"] + COORD_COLS],
        use_container_width=True,
    )

    st.subheader("Penjelasan per akun (SHAP)")
    uid = st.selectbox("Pilih user_id", df["user_id"].tolist())
    i = df.index[df["user_id"] == uid][0]
    sv = explainer.shap_values(X.iloc[[i]])
    base = explainer.expected_value
    exp = shap.Explanation(
        values=sv[0], base_values=base, data=X.iloc[i], feature_names=FEATURES
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

# ---------------- MODEL ----------------
with tab_model:
    f = RESULT_PATH / "model_comparison.csv"
    if f.exists():
        st.dataframe(pd.read_csv(f), use_container_width=True)
    f2 = RESULT_PATH / "shap_importance.csv"
    if f2.exists():
        imp = pd.read_csv(f2).head(15).set_index("Feature")
        st.subheader("Top 15 fitur (SHAP)")
        st.bar_chart(imp)

# ---------------- ABOUT ----------------
with tab_about:
    st.markdown(
        """
**Alur dua tahap**
1. Tahap 1: classifier XGBoost (label asli Cresci-2017: bot vs human).
2. Tahap 2: akun yang lolos sebagai human dinilai skor koordinasinya
   (hashtag sinkron, konten duplikat). Skor tinggi diberi flag *indikasi buzzer*.

Dataset: Cresci-2017 (tidak disertakan di repo).
"""
    )
