import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("attrition_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.set_page_config(
    page_title="HR Attrition Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Risk Intelligence System")
st.markdown("AI-powered cost-sensitive attrition risk prediction tool for HR decision support.")

# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.header("⚙️ Decision Settings")

threshold = st.sidebar.slider(
    "Attrition Risk Threshold",
    0.10, 0.90, 0.40, 0.05
)

cost_fn = st.sidebar.number_input(
    "Cost of Losing Employee (₹)",
    value=300000
)

cost_fp = st.sidebar.number_input(
    "Cost of Unnecessary Intervention (₹)",
    value=5000
)

# -----------------------------
# Input Section
# -----------------------------
st.markdown("## 🧾 Employee Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 60, 30)
    years_at_company = st.number_input("Years at Company", 0, 40, 5)

with col2:
    monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000)
    overtime = st.selectbox("Overtime", ["Yes", "No"])

overtime_val = 1 if overtime == "Yes" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Attrition Risk"):

    input_dict = {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "OverTime_Yes": overtime_val
    }

    input_df = pd.DataFrame([input_dict])

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    prob = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.markdown("## 📈 Risk Assessment")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Attrition Probability", f"{prob*100:.2f}%")
        st.progress(float(prob))

    with colB:
        if prob >= threshold:
            st.error("⚠️ HIGH RISK — Immediate HR Action Recommended")
            risk_level = "High"
        elif prob >= threshold - 0.15:
            st.warning("⚠️ MODERATE RISK — Monitor & Engage")
            risk_level = "Moderate"
        else:
            st.success("✅ LOW RISK — Stable Employee")
            risk_level = "Low"

    # -----------------------------
    # Financial Impact
    # -----------------------------
    st.markdown("## 💰 Financial Risk Analysis")

    expected_attrition_loss = prob * cost_fn
    intervention_cost = cost_fp if prob >= threshold else 0
    net_risk_exposure = expected_attrition_loss + intervention_cost

    colC, colD, colE = st.columns(3)

    colC.metric("Potential Attrition Loss", f"₹{expected_attrition_loss:,.0f}")
    colD.metric("Intervention Cost", f"₹{intervention_cost:,.0f}")
    colE.metric("Total Risk Exposure", f"₹{net_risk_exposure:,.0f}")

    # -----------------------------
    # Model Explanation
    # -----------------------------
    with st.expander("📘 Model & System Overview"):
        st.write("""
        - Model Type: Logistic Regression (Cost-Sensitive Optimization)
        - Evaluation Metric: ROC-AUC ≈ 0.78
        - Decision Threshold Adjustable for Business Needs
        - Designed to balance financial loss and operational capacity
        """)

    st.markdown("---")
    st.caption("Developed as an ML-based HR Decision Support System")