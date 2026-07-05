import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling & CSS Animations ---
st.markdown("""
<style>
    /* Theme color variables */
    :root {
        --primary: #FF4B4B;
        --accent: #1E90FF;
        --bg-glow: rgba(30, 144, 255, 0.1);
    }
    
    /* Animation definition */
    @keyframes fadeInSlide {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 5px rgba(255,75,75,0.4); }
        50% { box-shadow: 0 0 20px rgba(255,75,75,0.8); }
        100% { box-shadow: 0 0 5px rgba(255,75,75,0.4); }
    }

    /* Cards styling */
    .metric-card {
        background: #1a1c24;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid var(--accent);
        animation: fadeInSlide 0.6s ease-out forwards;
    }
    
    .result-card-danger {
        background: #2d1f24;
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        animation: pulseGlow 2s infinite, fadeInSlide 0.5s ease-out;
    }
    
    .result-card-success {
        background: #1f2d24;
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        animation: fadeInSlide 0.5s ease-out;
    }
</style>
""", unsafe_index=True)

# --- Load the Model Safely ---
@st.cache_resource
def load_model():
    try:
        with open("model (4).pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Main Page UI ---
st.title("🔮 Customer Churn Analytics Predictor")
st.markdown("Assess the risk of customer churn in real-time with machine learning insights.")
st.write("---")

if model is not None:
    # --- Sidebar Inputs arranged into logical groups ---
    st.sidebar.header("📋 Customer Identification")
    row_number = st.sidebar.number_input("Row Number", min_value=1, value=1, step=1, help="Unique identifier sequence number.")
    
    st.sidebar.header("📊 Demographics")
    geography = st.sidebar.selectbox("Geography / Region", ["France", "Spain", "Germany"])
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    age = st.sidebar.slider("Age (Years)", 18, 100, 35)
    
    st.sidebar.header("💰 Financial Metrics")
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=500.0)
    estimated_salary = st.sidebar.number_input("Estimated Salary ($)", min_value=0.0, value=85000.0, step=500.0)

    st.sidebar.header("📈 Engagement Profiles")
    tenure = st.sidebar.slider("Tenure (Years with Bank)", 0, 10, 5)
    num_products = st.sidebar.slider("Number of Bank Products Used", 1, 4, 2)
    has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])

    # --- Feature Engineering / Categorical Mapping ---
    # The model expects strings as trained or native features ('Geography', 'Gender') inside a structured payload.
    # We assemble the row using exact labels mapping back to the expected array structures.
    
    has_cr_card_val = 1 if has_cr_card == "Yes" else 0
    is_active_val = 1 if is_active_member == "Yes" else 0
    
    # Constructing ordered DataFrame matching feature_names_in_
    input_data = pd.DataFrame([{
        'RowNumber': row_number,
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card_val,
        'IsActiveMember': is_active_val,
        'EstimatedSalary': estimated_salary
    }])

    # --- Metrics Layout Layout Display ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <span style='color:gray; font-size:14px;'>Customer Demographics</span><br>
            <b style='font-size:20px;'>{gender}, {age} Years Old</b><br>
            <span style='font-size:13px; color:#1E90FF;'>Origin Country: {geography}</span>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card' style='border-left-color: #ff9f43;'>
            <span style='color:gray; font-size:14px;'>Financial Standing</span><br>
            <b style='font-size:20px;'>Balance: ${balance:,.2f}</b><br>
            <span style='font-size:13px; color:#ff9f43;'>Credit Rating: {credit_score}</span>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card' style='border-left-color: #2ecc71;'>
            <span style='color:gray; font-size:14px;'>Product Engagement</span><br>
            <b style='font-size:20px;'>{num_products} Active Products</b><br>
            <span style='font-size:13px; color:#2ecc71;'>Tenure period: {tenure} yrs</span>
        </div>""", unsafe_allow_html=True)

    st.write("##")

    # --- Inference Action Engine ---
    if st.button("🔥 Run Risk Analysis Assessment", use_container_width=True):
        with st.spinner("Analyzing profile indicators across estimator branches..."):
            time.sleep(0.8) # Smooth transitional dynamic pause
            
            # Predict Risk probabilities and Churn category
            prediction = model.predict(input_data)[0]
            
            # Handle both models predicting probability lists or raw values
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0][1] * 100
            else:
                prob = 100.0 if prediction == 1 else 0.0

        st.write("### Diagnostics Evaluation Result:")
        
        # Display customized color-themed container configurations depending on binary prediction flags
        if prediction == 1:
            st.markdown(f"""
            <div class='result-card-danger'>
                <h2 style='color: #ff4b4b; margin: 0;'>⚠️ High Churn Attrition Warning</h2>
                <p style='font-size: 18px;'>This customer reveals high risk alignment tendencies matching historical retention exits.</p>
                <h1 style='color: #ff4b4b; font-size: 48px; margin: 10px 0;'>{prob:.1f}% Risk Score</h1>
                <p style='color: gray; font-size:13px;'>Immediate strategic intervention management advised.</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons() if prob < 10 else None
        else:
            st.markdown(f"""
            <div class='result-card-success'>
                <h2 style='color: #2ecc71; margin: 0;'>✅ Stable Customer Account</h2>
                <p style='font-size: 18px;'>The customer behaves safely with steady indicator trends indicating retention probability safety status.</p>
                <h1 style='color: #2ecc71; font-size: 48px; margin: 10px 0;'>{prob:.1f}% Risk Score</h1>
                <p style='color: gray; font-size:13px;'>Loyalty score signals consistent baseline conditions.</p>
            </div>
            """, unsafe_allow_html=True)
            st.snow()

else:
    st.info("Please verify that your file is named precisely `model (4).pkl` and is sitting within this current execution path directory.")
