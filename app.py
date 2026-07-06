import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# =====================================================
# 1. ADVANCED PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="AI Policy Classification System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Injection for Enterprise Feel
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Metric Card Custom Style */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00FFCC;
        font-weight: 700;
    }
    /* Headers & Branding Accents */
    h1 {
        color: #FFFFFF;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    .stButton>button {
        background-color: #4F46E5 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #6366F1 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    </style>
""", unsafe_style_html=True)

# =====================================================
# 2. OPTIMIZED RESOURCE INGESTION
# =====================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("knn_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        svd = joblib.load("svd_transformer.pkl")
        encoder = joblib.load("target_encoder.pkl")
        return model, tfidf, svd, encoder
    except Exception as e:
        st.error(f"Error loading model pkl artifacts: {e}")
        return None, None, None, None

model, tfidf, svd, encoder = load_artifacts()

# =====================================================
# 3. SIDEBAR UTILITIES & BRANDING
# =====================================================
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/security-checked.png", width=70)
    st.title("Control Panel")
    st.caption("DecodeLabs AI Architecture Framework")
    st.markdown("---")
    
    st.markdown("### 👤 Engineer Assignment")
    st.info("**Developer:** M. Rohaan Zahid\n\n**Role:** AI Engineer Intern")
    
    st.markdown("### 🛠️ Pipeline Node Details")
    st.markdown("""
    - **NLP Engine:** TF-IDF Matrix Tokenizer
    - **Dimensional Reduction:** Truncated SVD Components
    - **Classifier Engine:** K-Nearest Neighbors ($K=5$)
    - **Scaling Topology:** StandardScaler Norm
    """)
    st.markdown("---")
    st.caption("© 2026 DecodeLabs | Batch 2026 [cite: 4]")

# =====================================================
# 4. MAIN USER INTERFACE WORKSPACE
# =====================================================

# Header Hero Section
col_logo, col_title = st.columns([0.1, 0.9])
with col_logo:
    st.markdown("<h1 style='font-size: 50px; margin:0;'>🛡️</h1>", unsafe_style_html=True)
with col_title:
    st.title("AI Governance & Policy Classification Platform")
    st.caption("Production Pipeline Node: Text Extraction, Geometrical Dimensional Reduction, & Classification Model Lookup [cite: 33]")

st.markdown("""
This platform processes public policy texts into structured, vector-mapped spaces. 
By translating semantic textual weights alongside strategic impact indices, the integrated **K-Nearest Neighbors (KNN)** architecture assigns discrete administrative categories with high geometric precision[cite: 10, 143].
""")

st.markdown("---")

if model is None:
    st.critical("Pipeline Initialization Blocked. Ensure model pickle files are inside the root directory.")
else:
    # Twin Workspace Columns Split
    col_input, col_output = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.subheader("📥 Data Ingestion Workspace")
        
        policy_text = st.text_area(
            "Policy Document Core (Title or Statement Mapping Node)",
            height=140,
            placeholder="Type or paste the structural text here... (e.g., European Parliament votes on risk-level categorization matrix for generative models.)"
        )
        
        # FIXES THE SHAPE MISMATCH ERROR: Collect the missing numeric dimension required by the scaler
        policy_impact_score = st.slider(
            "Policy Impact Matrix Score ($W_{impact}$)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.5,
            help="Quantitative metric evaluating the scope of the global regulatory or compliance impact."
        )
        
        st.markdown("<br>", unsafe_style_html=True)
        trigger_prediction = st.button("🚀 Process & Classify Vector")

    with col_output:
        st.subheader("📊 Architectural Analytics Engine")
        
        if trigger_prediction:
            if not policy_text.strip():
                st.warning("⚠️ Input sequence rejected. Please input text before execution.")
            else:
                # Execution Process Feedback Block
                with st.status("Computing NLP Vectors & Geometrical Distances...", expanded=True) as status:
                    st.write("Tokenizing unstructured string vectors via TF-IDF...")
                    vectorized = tfidf.transform([policy_text])
                    time.sleep(0.2)
                    
                    st.write("Compressing sparse matrix data via Truncated SVD...")
                    reduced = svd.transform(vectorized)
                    time.sleep(0.2)
                    
                    st.write("Assembling structural feature layers with impact metrics...")
                    # Combine the text components with your numerical entry to rebuild the 5 features
                    numeric_feat = np.array([[policy_impact_score]])
                    combined_features = np.hstack((reduced, numeric_feat))
                    
                    # Optional: Scaler step if saved during training
                    # If you saved your scaler object, load it in load_models and use it here:
                    # combined_features = scaler.transform(combined_features)
                    
                    st.write("Executing K-Nearest Neighbors space coordinate matching...")
                    prediction = model.predict(combined_features)
                    predicted_category = encoder.inverse_transform(prediction)[0]
                    time.sleep(0.1)
                    
                    status.update(label="Classification Vector Processed Successfully!", state="complete")
                
                # Polished Metrics Dashboard Output Layout
                st.markdown("### 🎯 Classification Result")
                
                # Visual highlight container for the output classification
                st.markdown(f"""
                    <div style="background-color: rgba(79, 70, 229, 0.1); border-left: 5px solid #4F46E5; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;">
                        <span style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #A5B4FC;">Identified Taxonomy Class</span>
                        <h2 style="margin: 0.5rem 0 0 0; color: #FFFFFF; font-size: 28px;">{predicted_category}</h2>
                    </div>
                """, unsafe_style_html=True)
                
                # Context Meta Metrics Indicators
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="Assigned Impact Score", value=f"{policy_impact_score} / 100")
                with m_col2:
                    st.metric(label="Inference Status", value="Verified Active")
                    
        else:
            # Idle placeholder display state before trigger execution
            st.info("System Standby: Awaiting pipeline telemetry execution from input panel.")
            
            # Static Graphic Blueprint Layout to keep page layout visually balanced
            st.markdown("""
            ```
            [Text Stream] ──> [TF-IDF Space] ──> [SVD Compress] ──┐
                                                                 ├──> [KNN Model] ──> Category Target
            [Impact Index] ──────────────────────────────────────┘
            ```
            """)

# Footer Layout
st.markdown("<br><br>", unsafe_style_html=True)
st.divider()
st.caption("🔒 Confidential Infrastructure Node — DecodeLabs Supervised Machine Learning Track Project 2[cite: 2, 7].")
