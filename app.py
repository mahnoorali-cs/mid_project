import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Disease Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a3c5e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #5a7a9a;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .card-high   { background: #ffe4e4; border-left: 5px solid #e74c3c; }
    .card-medium { background: #fff4e0; border-left: 5px solid #f39c12; }
    .card-low    { background: #e4f9e8; border-left: 5px solid #27ae60; }
    .card-info   { background: #e8f0fe; border-left: 5px solid #3b82f6; }
    .card-severe { background: #fdecea; border-left: 5px solid #c0392b; }
    .card-mild   { background: #eafaf1; border-left: 5px solid #1e8449; }
    .card-moderate { background: #fef9e7; border-left: 5px solid #d4ac0d; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a3c5e;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stTabs"] button { font-size: 1rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load & train models (cached so they run once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("Assign1_ml.csv")
    df = df.dropna()

    # ── preprocessing ──────────────────────────
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df[["Systolic_bp", "Diastolic_bp"]] = (
        df["Blood_Pressure"].str.split("/", expand=True).astype(int)
    )

    # Condition label for NLP
    def label_condition(text):
        if any(w in text for w in [
            "severe hypertension", "stage 2 hypertension", "uncontrolled hypertension",
            "chest pain", "chest tightness", "angina",
            "very high cholesterol/triglycerides", "high cardiovascular risk",
            "high risk category", "requires pacing evaluation", "elevated heart rate"
        ]):
            return "Severe Condition"
        elif any(w in text for w in [
            "Controlled BP", "borderline BP", "pre-hypertension", "asymptomatic",
            "moderate risk", "low current risk", "Active lifestyle",
            "Normal BMI", "normal lipids", "moderate activity", "Good Activity",
            "Stage 1 hypertension", "bp control", "Good capacity",
            "Normal cholesterol", "non smoker", "normal weight",
            "normal", "active", "management"
        ]):
            return "Mild Condition"
        else:
            return "Moderate Condition"

    df1 = df.copy()
    df1["Condition_Category"] = df1["Medical_Report_Text"].apply(label_condition)

    # Risk category
    def risk_category(row):
        if (row["Heart_Attack_Risk"] == 1 and row["Age"] > 65 and
                (row["Systolic_bp"] > 150 or row["Cholesterol"] > 350 or row["BMI"] > 30)) or \
                (row["Smoking"] == 1 and row["Treatment_Success_Rate"] < 60):
            return "High"
        elif (row["Systolic_bp"] >= 135 or row["Cholesterol"] > 300 or row["BMI"] < 20) and \
                row["Treatment_Success_Rate"] < 70:
            return "Medium"
        else:
            return "Low"

    df["Risk_category"] = df.apply(risk_category, axis=1)

    # ── Linear Regression — Treatment Success Rate ──
    li_x = df[["Systolic_bp", "Diastolic_bp", "Cholesterol", "BMI",
               "Triglycerides", "Exercise_Hours_Per_Week"]]
    li_y = df["Treatment_Success_Rate"]
    li_x_train, li_x_test, li_y_train, _ = train_test_split(
        li_x, li_y, test_size=0.2, random_state=42)
    li_scaler = StandardScaler()
    li_x_train = li_scaler.fit_transform(li_x_train)
    li_model = LinearRegression()
    li_model.fit(li_x_train, li_y_train)

    # ── Logistic Regression — Heart Attack Risk ──
    lo_x = df[["Systolic_bp", "Diastolic_bp", "Cholesterol", "BMI",
               "Triglycerides", "Smoking", "Age", "Heart Rate"]]
    lo_y = df["Heart_Attack_Risk"]
    lo_x_train, _, lo_y_train, _ = train_test_split(
        lo_x, lo_y, test_size=0.2, random_state=42)
    lo_scaler = StandardScaler()
    lo_x_train = lo_scaler.fit_transform(lo_x_train)
    lo_model = LogisticRegression(max_iter=1000)
    lo_model.fit(lo_x_train, lo_y_train)

    # ── Naive Bayes — Condition Category (NLP) ──
    nb_x = df1["Medical_Report_Text"]
    nb_y = df1["Condition_Category"]
    nb_x_train, _, nb_y_train, _ = train_test_split(
        nb_x, nb_y, test_size=0.2, random_state=42, stratify=nb_y)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    nb_x_train_tfidf = vectorizer.fit_transform(nb_x_train)
    nb_model = MultinomialNB()
    nb_model.fit(nb_x_train_tfidf, nb_y_train)

    # ── SVM — Risk Category ──
    svm_x = df[["Systolic_bp", "Diastolic_bp", "Heart Rate", "Cholesterol",
                "BMI", "Smoking", "Exercise_Hours_Per_Week", "Treatment_Success_Rate"]]
    svm_y = df["Risk_category"]
    svm_x_train, _, svm_y_train, _ = train_test_split(
        svm_x, svm_y, test_size=0.2, random_state=42)
    svm_scaler = StandardScaler()
    svm_x_train = svm_scaler.fit_transform(svm_x_train)
    svm_model = SVC(kernel="rbf", probability=True)
    svm_model.fit(svm_x_train, svm_y_train)

    return {
        "li_model": li_model, "li_scaler": li_scaler,
        "lo_model": lo_model, "lo_scaler": lo_scaler,
        "nb_model": nb_model, "vectorizer": vectorizer,
        "svm_model": svm_model, "svm_scaler": svm_scaler,
    }


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🏥 Patient Disease Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter patient data below to get predictions from four ML models.</p>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load models with spinner
# ─────────────────────────────────────────────
with st.spinner("Training models on dataset... (first load only)"):
    models = load_and_train()

# ─────────────────────────────────────────────
# Sidebar — Patient Input
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Patient Information")
    st.markdown("---")

    st.markdown("**Demographics**")
    age = st.slider("Age", 18, 90, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])

    st.markdown("**Vital Signs**")
    systolic = st.slider("Systolic BP (mmHg)", 80, 220, 130)
    diastolic = st.slider("Diastolic BP (mmHg)", 50, 130, 85)
    heart_rate = st.slider("Heart Rate (bpm)", 40, 110, 75)

    st.markdown("**Lab Values**")
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 220)
    bmi = st.slider("BMI", 15.0, 45.0, 26.0, step=0.1)
    triglycerides = st.slider("Triglycerides (mg/dL)", 30, 800, 300)

    st.markdown("**Lifestyle**")
    exercise = st.slider("Exercise Hours/Week", 0.0, 20.0, 5.0, step=0.5)

    st.markdown("**Doctor's Notes (for NLP)**")
    report_text = st.text_area(
        "Medical Report Text",
        value="Middle-aged patient with Stage 1 hypertension, normal BMI, moderate activity level, non smoker.",
        height=120
    )

    predict_btn = st.button("🔍 Run Predictions", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────
def run_predictions():
    smoking_val = 1 if smoking == "Yes" else 0
    li_input = np.array([[systolic, diastolic, cholesterol, bmi, triglycerides, exercise]])
    li_input_scaled = models["li_scaler"].transform(li_input)
    treatment_rate = models["li_model"].predict(li_input_scaled)[0]
    treatment_rate = np.clip(treatment_rate, 0, 100)

    lo_input = np.array([[systolic, diastolic, cholesterol, bmi,
                          triglycerides, smoking_val, age, heart_rate]])
    lo_input_scaled = models["lo_scaler"].transform(lo_input)
    heart_risk_pred = models["lo_model"].predict(lo_input_scaled)[0]
    heart_risk_prob = models["lo_model"].predict_proba(lo_input_scaled)[0][1]

    nb_tfidf = models["vectorizer"].transform([report_text])
    condition = models["nb_model"].predict(nb_tfidf)[0]

    svm_input = np.array([[systolic, diastolic, heart_rate, cholesterol,
                           bmi, smoking_val, exercise, treatment_rate]])
    svm_input_scaled = models["svm_scaler"].transform(svm_input)
    risk_cat = models["svm_model"].predict(svm_input_scaled)[0]

    return treatment_rate, heart_risk_pred, heart_risk_prob, condition, risk_cat


# ─────────────────────────────────────────────
# Results Display
# ─────────────────────────────────────────────
if predict_btn:
    treatment_rate, heart_risk_pred, heart_risk_prob, condition, risk_cat = run_predictions()

    st.markdown("## 📊 Prediction Results")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        color = "#27ae60" if treatment_rate >= 65 else ("#f39c12" if treatment_rate >= 50 else "#e74c3c")
        st.metric("Treatment Success Rate", f"{treatment_rate:.1f}%")
        st.markdown(f'<div style="height:6px;background:{color};border-radius:4px;width:{min(treatment_rate,100):.0f}%"></div>',
                    unsafe_allow_html=True)
        st.caption("Linear Regression")

    with col2:
        risk_label = "⚠️ At Risk" if heart_risk_pred == 1 else "✅ Low Risk"
        st.metric("Heart Attack Risk", risk_label)
        st.caption(f"Probability: {heart_risk_prob*100:.1f}% | Logistic Regression")

    with col3:
        st.metric("Condition Category", condition)
        st.caption("Naive Bayes (NLP on report text)")

    with col4:
        risk_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        st.metric("Patient Risk Category", f"{risk_colors.get(risk_cat,'')} {risk_cat}")
        st.caption("SVM Classifier")

    # ── Detailed cards ──
    st.markdown("---")
    st.markdown("### 🔎 Detailed Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💊 Treatment Success", "❤️ Heart Attack Risk",
        "📝 Medical Condition", "🎯 Overall Risk"
    ])

    with tab1:
        level = "low" if treatment_rate < 50 else ("medium" if treatment_rate < 65 else "high")
        css = "card-high" if level == "low" else ("card-medium" if level == "medium" else "card-low")
        st.markdown(f"""
        <div class="result-card {css}">
            <h4>Predicted Treatment Success Rate: <strong>{treatment_rate:.1f}%</strong></h4>
            <p>Based on: Systolic/Diastolic BP, Cholesterol, BMI, Triglycerides, Exercise Hours.</p>
            <p><b>Model:</b> Linear Regression &nbsp;|&nbsp;
               <b>Interpretation:</b>
               {"Below 50% — Immediate medical attention needed." if level=="low" else
                "50–65% — Moderate outlook, monitor closely." if level=="medium" else
                "Above 65% — Good prognosis with current lifestyle."}</p>
        </div>""", unsafe_allow_html=True)

    with tab2:
        css2 = "card-high" if heart_risk_pred == 1 else "card-low"
        msg = ("High Heart Attack Risk detected. Immediate cardiology review recommended."
               if heart_risk_pred == 1 else
               "Low Heart Attack Risk. Maintain healthy habits.")
        st.markdown(f"""
        <div class="result-card {css2}">
            <h4>Heart Attack Risk: <strong>{"YES ⚠️" if heart_risk_pred==1 else "NO ✅"}</strong></h4>
            <p>Confidence: <strong>{heart_risk_prob*100:.1f}%</strong></p>
            <p><b>Model:</b> Logistic Regression &nbsp;|&nbsp; <b>Interpretation:</b> {msg}</p>
            <p><b>Key factors used:</b> Age, BP, Cholesterol, BMI, Triglycerides, Smoking, Heart Rate</p>
        </div>""", unsafe_allow_html=True)

    with tab3:
        cond_css = {
            "Severe Condition": "card-severe",
            "Mild Condition": "card-mild",
            "Moderate Condition": "card-moderate",
        }.get(condition, "card-info")
        cond_advice = {
            "Severe Condition": "Urgent intervention required. Refer to specialist.",
            "Mild Condition": "Low risk. Routine monitoring and lifestyle guidance.",
            "Moderate Condition": "Monitor regularly. Medication or lifestyle adjustments may help.",
        }.get(condition, "")
        st.markdown(f"""
        <div class="result-card {cond_css}">
            <h4>Medical Condition: <strong>{condition}</strong></h4>
            <p><b>Model:</b> Naive Bayes (TF-IDF on doctor's notes)</p>
            <p><b>Report analysed:</b> "{report_text[:120]}{'...' if len(report_text)>120 else ''}"</p>
            <p><b>Recommendation:</b> {cond_advice}</p>
        </div>""", unsafe_allow_html=True)

    with tab4:
        risk_css = {"High": "card-high", "Medium": "card-medium", "Low": "card-low"}.get(risk_cat, "card-info")
        risk_advice = {
            "High": "Immediate attention needed. Schedule specialist consult and close monitoring.",
            "Medium": "Elevated risk. Regular check-ups and lifestyle modification advised.",
            "Low": "Low risk. Routine annual check-up recommended.",
        }.get(risk_cat, "")
        st.markdown(f"""
        <div class="result-card {risk_css}">
            <h4>Overall Patient Risk: <strong>{risk_colors.get(risk_cat,'')} {risk_cat}</strong></h4>
            <p><b>Model:</b> SVM Classifier</p>
            <p><b>Factors used:</b> BP, Heart Rate, Cholesterol, BMI, Smoking, Exercise, Treatment Rate</p>
            <p><b>Advice:</b> {risk_advice}</p>
        </div>""", unsafe_allow_html=True)

    # ── Summary bar ──
    st.markdown("---")
    st.markdown("### 📋 Patient Summary")
    summary_cols = st.columns(5)
    summary_cols[0].metric("Age", age)
    summary_cols[1].metric("BMI", f"{bmi:.1f}")
    summary_cols[2].metric("BP", f"{systolic}/{diastolic}")
    summary_cols[3].metric("Cholesterol", cholesterol)
    summary_cols[4].metric("Smoker", smoking)

else:
    # Placeholder before predict is clicked
    st.info("👈 Fill in patient details in the sidebar and click **Run Predictions** to see results.")
    st.markdown("""
    ### How this works
    This app runs **4 ML models** trained on patient medical data:

    | Model | Predicts |
    |---|---|
    | **Linear Regression** | Treatment Success Rate (0–100%) |
    | **Logistic Regression** | Heart Attack Risk (Yes/No) |
    | **Naive Bayes** | Medical Condition severity from doctor notes |
    | **SVM** | Overall Patient Risk Category (High/Medium/Low) |

    The models are trained live on `Assign1_ml.csv` when the app first loads.
    """)
