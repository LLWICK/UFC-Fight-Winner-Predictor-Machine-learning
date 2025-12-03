import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Data & Model
# -----------------------------
@st.cache_data
def load_fighters():
    return pd.read_csv("./dataset/ufc_fighters_cleaned.csv")

@st.cache_resource
def load_model():
    return joblib.load("./models/randomForest1.pkl"), joblib.load("./models/scaler/scaler.pkl")

df_fighters = load_fighters()
model, scaler = load_model()

# Helper
def get_fighter(name): return df_fighters[df_fighters['name'] == name].iloc[0]

def compute_delta(f1, f2):
    return pd.DataFrame([{
        "delta_height": f1.height_cm - f2.height_cm,
        "delta_reach": f1.reach - f2.reach,
        "delta_age": f1.age - f2.age,
        "delta_win_rate": f1.win_rate - f2.win_rate,
        "delta_SLpM": f1.SLpM - f2.SLpM,
        "delta_StrAcc": f1.StrAcc - f2.StrAcc,
        "delta_SApM": f1.SApM - f2.SApM,
        "delta_StrDef": f1.StrDef - f2.StrDef,
        "delta_TDAvg": f1.TDAvg - f2.TDAvg,
        "delta_TDAcc": f1.TDAcc - f2.TDAcc,
        "delta_TDDef": f1.TDDef - f2.TDDef,
        "delta_SubAvg": f1.SubAvg - f2.SubAvg,
    }])


# -----------------------------------------------------
# 🔥 UI SETUP (UFC THEMED)
# -----------------------------------------------------
st.set_page_config(page_title="UFC Fight Predictor", layout="wide")

st.markdown("""
<style>
/* Center title */
h1 { text-align: center; }

/* Fighter Cards */
.fighter-card {
    padding: 25px;
    border-radius: 15px;
    color: white;
    font-size: 20px;
    font-weight: bold;
}
.red-corner {
    background: #b30000;
    box-shadow: 0 0 15px #ff3333;
}
.blue-corner {
    background: #0033cc;
    box-shadow: 0 0 15px #3366ff;
}

/* Stat bars */
.stat-bar {
    height: 10px;
    border-radius: 5px;
    margin-top: 5px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------
# 🥊 TITLE
# -----------------------------------------------------
st.markdown("<h1>🥋 UFC Fight Winner Prediction</h1>", unsafe_allow_html=True)
st.write("Select two fighters to compare and generate a fight prediction.")

st.divider()


# -----------------------------------------------------
# 🧍 Fighter Selection
# -----------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    f1_name = st.selectbox("👉 Select Fighter 1 (Red Corner)", sorted(df_fighters["name"].unique()))

with col2:
    f2_name = st.selectbox("👉 Select Fighter 2 (Blue Corner)", sorted(df_fighters["name"].unique()))

if f1_name == f2_name:
    st.warning("⚠️ Please select two different fighters.")
    st.stop()

f1 = get_fighter(f1_name)
f2 = get_fighter(f2_name)


# -----------------------------------------------------
# 🔥 FIGHTER PROFILE CARDS
# -----------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.markdown(f"""
    <div class="fighter-card red-corner">
        <h2>{f1.name}</h2>
        <p>🏋️ Weight: <b>{f1.weight_kg}</b></p>
        <p>📏 Height: <b>{f1.height_cm} cm</b></p>
        <p>🦍 Reach: <b>{f1.reach} cm</b></p>
        <p>🎯 Win Rate: <b>{round(f1.win_rate*100,1)}%</b></p>
        <p>🥊 SLpM: <b>{f1.SLpM}</b></p>
        <p>🎯 StrAcc: <b>{f1.StrAcc}</b></p>
        <p>🛡️ StrDef: <b>{f1.StrDef}</b></p>
        <p>🤼 TDAvg: <b>{f1.TDAvg}</b></p>
    </div>
    """, unsafe_allow_html=True)

with colB:
    st.markdown(f"""
    <div class="fighter-card blue-corner">
        <h2>{f2.name}</h2>
        <p>🏋️ Weight: <b>{f2.weight_kg}</b></p>
        <p>📏 Height: <b>{f2.height_cm} cm</b></p>
        <p>🦍 Reach: <b>{f2.reach} cm</b></p>
        <p>🎯 Win Rate: <b>{round(f2.win_rate*100,1)}%</b></p>
        <p>🥊 SLpM: <b>{f2.SLpM}</b></p>
        <p>🎯 StrAcc: <b>{f2.StrAcc}</b></p>
        <p>🛡️ StrDef: <b>{f2.StrDef}</b></p>
        <p>🤼 TDAvg: <b>{f2.TDAvg}</b></p>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# -----------------------------------------------------
# ⚖️ DELTA TABLE
# -----------------------------------------------------
st.subheader("📊 Tale of the Tape — Stat Advantage")

delta_df = compute_delta(f1, f2)
st.dataframe(delta_df.style.highlight_max(axis=1, color="lightgreen").highlight_min(axis=1, color="#ffcccc"))

st.divider()



# -----------------------------------------------------
# 🎯 PREDICTION SECTION
# -----------------------------------------------------
st.subheader("🥇 Fight Prediction")

if st.button("🔥 Predict Winner"):
    scaled = scaler.transform(delta_df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    if pred == 1:
        winner = f1_name
        confidence = prob[1] * 100
        color = "#b30000"
    else:
        winner = f2_name
        confidence = prob[0] * 100
        color = "#0033cc"

    st.markdown(f"""
    <div style="
        padding:25px;
        text-align:center;
        color:white;
        font-size:28px;
        font-weight:bold;
        background:{color};
        border-radius:15px;
        box-shadow:0 0 20px {color}cc;">
        🏆 Predicted Winner: {winner}<br><br>
        🔥 Confidence: {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    if confidence < 55:
        st.info("Very close matchup — low confidence prediction.")

