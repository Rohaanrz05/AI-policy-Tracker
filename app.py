import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

st.set_page_config(page_title="AI Policy Classifier", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_pipeline_artifacts():
    try:
        model = joblib.load("knn_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        svd = joblib.load("svd_transformer.pkl")
        encoder = joblib.load("target_encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, tfidf, svd, encoder, scaler
    except Exception as e:
        return None, None, None, None, None

model, tfidf, svd, encoder, scaler = load_pipeline_artifacts()

if model is None:
    st.error("❌ App waiting for matching .pkl files from your Colab download.")
else:
    st.title("🛡️ AI Policy & Governance Classification Hub")
    st.caption("DecodeLabs Advanced Analytics Stack — Batch 2026")
    st.markdown("---")

    col_input, col_output = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.markdown("### 📥 Stream Ingestion Window")
        policy_text = st.text_area("Raw Policy Document Context (Title / Provision Statement)", height=150, placeholder="Paste policy language here...")
        policy_impact_score = st.slider("Quantitative Policy Impact Metric Value ($W_{impact}$)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
        compute_trigger = st.button("🚀 Analyze Framework Telemetry")

    with col_output:
        st.markdown("### 📊 Predictive Engine Analytics")
        if compute_trigger and policy_text.strip():
            with st.status("Processing Vectors...", expanded=True) as status:
                # 1. Map Text to TF-IDF
                vectorized = tfidf.transform([policy_text])
                # 2. Compress via SVD (Outputs 5 features)
                reduced = svd.transform(vectorized)
                # 3. Add 1 numeric features (Makes exactly 6 features total)
                numeric_feat = np.array([[policy_impact_score]])
                combined_matrix = np.hstack((reduced, numeric_feat))
                # 4. Scale and Predict using matching 6-feature matrix
                scaled_features = scaler.transform(combined_matrix)
                prediction = model.predict(scaled_features)
                predicted_class = encoder.inverse_transform(prediction)[0]
                status.update(label="Feature Computation Complete!", state="complete")
            
            st.markdown(f"""
                <div style="background: rgba(79,70,229,0.15); border-left: 5px solid #4F46E5; padding: 1.5rem; border-radius: 8px;">
                    <span style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #818CF8;">Assigned Taxonomy Class</span>
                    <h2 style="margin: 0.3rem 0 0 0; color: #FFFFFF; font-size: 26px;">{predicted_class}</h2>
                </div>
            """, unsafe_style_html=True)
        else:
            st.info("💡 Awaiting vector transmission from the configuration window.")
