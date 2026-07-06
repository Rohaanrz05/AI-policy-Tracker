import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# =====================================================
# 1. ENTERPRISE PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="AI Policy Classification System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling Injections
st.markdown("""
    <style>
    /* Main Canvas Optimization */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* Premium UI Metric Typography */
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        color: #6366F1 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Button UX Overhaul */
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.25s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.4) !important;
    }
    
    /* Input Form Accent Border */
    div[data-baseweb="textarea"] {
        border-radius: 8px !important;
    }
    </style>
""", unsafe_style_html=True)

# =====================================================
# 2. CACHED MODEL ARTIFACT INGESTION
# =====================================================
@st.cache_resource
def load_pipeline_artifacts():
    try:
        model = joblib.load("knn_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        svd = joblib.load("svd_transformer.pkl")
        encoder = joblib.load("target_encoder.pkl")
        scaler = joblib.load("scaler.pkl")  # Ingesting the scaler artifact
        return model, tfidf, svd, encoder, scaler
    except Exception as e:
        return None, None, None, None, None

model, tfidf, svd, encoder, scaler = load_pipeline_artifacts()

# =====================================================
# 3. SIDEBAR CONTROLS & META PANEL
# =====================================================
with st.sidebar:
    st.markdown("<div style='text-align: center; padding-bottom: 1rem;'>", unsafe_style_html=True)
    st.image("https://img.icons8.com/nolan/128/security-checked.png", width=65)
    st.markdown("</div>", unsafe_style_html=True)
    
    st.title("System Framework")
    st.caption("DecodeLabs Advanced Analytics Stack")
    st.markdown("---")
    
    # Metadata Groupings
    st.markdown("### 👤 Engineer Assignment")
    st.info("**Developer:** M. Rohaan Zahid\n\n**Assignment Track:** AI Engineering Intern")
    
    st.markdown("### ⚙️ Engine Nodes")
    st.markdown("""
    - **NLP Vectorizer:** TF-IDF Tokenizer
    - **Dimensional Space:** Truncated SVD 
    - **Classifier Engine:** KNN ($K=5$)
    - **Feature Scaling:** StandardScaler
    """)
    st.markdown("---")
    st.caption("🔒 Architecture Security Node — Batch 2026")

# =====================================================
# 4. MAIN HUB WORKSPACE Layout
# =====================================================
# Header Hero Section Matrix
hero_logo, hero_headline = st.columns([0.08, 0.92])
with hero_logo:
    st.markdown("<h1 style='font-size: 52px; margin:0; padding-top:5px;'>🛡️</h1>", unsafe_style_html=True)
with hero_headline:
    st.title("AI Policy & Governance Classification Hub")
    st.caption("Supervised Track Milestone: Vector-Space Text Extraction & Geometrical Neighborhood Mapping")

st.markdown("""
This production interface utilizes continuous **TF-IDF mapping** alongside a compressed **Truncated SVD pipeline** to classify public policy items with strict geometric accuracy.
""")
st.markdown("---")

if model is None:
    st.error("❌ Critical System Alert: Pipeline binaries (`.pkl` objects) are missing from the current working directory.")
else:
    # Dashboard Split Columns Grid Setup
    col_input, col_output = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.markdown("### 📥 Stream Ingestion Window")
        
        policy_text = st.text_area(
            "Raw Policy Document Context (Title / Provision Statement)",
            height=150,
            placeholder="Insert standard policy language here... (e.g., Global trade delegates draft strict restrictions regarding automated model verification protocols...)"
        )
        
        # Fixing Feature Dimensions: Capturing the 5th continuous component
        policy_impact_score = st.slider(
            "Quantitative Policy Impact Metric Value ($W_{impact}$)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.5,
            help="Assigns structural parameter weight regarding the scope of the global regulatory item."
        )
        
        st.markdown("<br>", unsafe_style_html=True)
        compute_trigger = st.button("🚀 Analyze Framework Telemetry")

    with col_output:
        st.markdown("### 📊 Predictive Engine Analytics")
        
        if compute_trigger:
            if not policy_text.strip():
                st.warning("⚠️ Telemetry execution rejected. Source text buffer cannot be empty.")
            else:
                # =====================================================================
                # UPDATED INFERENCE STEP: FORCING 6 FEATURES FOR THE SCALER
                # =====================================================================
                
                with st.status("Initializing NLP Core Pipelines...", expanded=True) as status:
                    st.write("Extracting token matrix elements via TF-IDF...")
                    vectorized = tfidf.transform([policy_text])
                    time.sleep(0.15)
                    
                    st.write("Executing dimensionality compression via Truncated SVD...")
                    reduced = svd.transform(vectorized) 
                    time.sleep(0.15)
                    
                    # Check if your saved SVD model outputs 5 components automatically.
                    # If it only outputs 4 components, we manually pad a placeholder 0 
                    # to satisfy the scaler's demand for 6 features total.
                    if reduced.shape[1] == 4:
                        st.write("Adapting vector matrix to match 6-feature layout...")
                        placeholder_padding = np.zeros((1, 1))
                        reduced = np.hstack((reduced, placeholder_padding))
                    
                    st.write("Concatenating numerical metric arrays to feature coordinates...")
                    numeric_feat = np.array([[policy_impact_score]])
                    combined_matrix = np.hstack((reduced, numeric_feat)) # This makes exactly 6 columns!
                    
                    st.write("Applying standardization scaling parameters...")
                    scaled_features = scaler.transform(combined_matrix)
                    time.sleep(0.1)
                    
                    st.write("Running geometric distance search across KNN tree nodes...")
                    prediction = model.predict(scaled_features)
                    predicted_class = encoder.inverse_transform(prediction)[0]
                    
                    status.update(label="Feature Computation Complete!", state="complete")
                
                # Presentation Interface Metrics Block
                st.markdown("### 🎯 Classification Target Mapping")
                
                # Highlight card for classification taxonomy
                st.markdown(f"""
                    <div style="background: linear-gradient(90deg, rgba(79,70,229,0.15) 0%, rgba(99,102,241,0.03) 100%); 
                                border-left: 5px solid #4F46E5; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;">
                        <span style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #818CF8;">Assigned Structural Taxonomy</span>
                        <h2 style="margin: 0.3rem 0 0 0; color: #FFFFFF; font-size: 26px; font-weight: 700;">{predicted_class}</h2>
                    </div>
                """, unsafe_style_html=True)
                
                # Meta Indicators Columns Row
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(label="Ingested Impact Rating", value=f"{policy_impact_score} / 100")
                with metric_col2:
                    st.metric(label="Inference State", value="Active / Normal")
        else:
            # Standby Placeholder Block State
            st.info("💡 Main Interface Standby: Awaiting vector transmission from the left-hand configuration window.")
            
            # Blueprint Code Frame Block Layout Representation
            st.markdown("""
            ```text
            [ Text Entry Stream ] ──> [ TF-IDF Processing ] ──> [ SVD Dimensional Compression ] ──┐
                                                                                                 ├──> [ Scaler Engine ] ──> [ KNN Engine ]
            [ Slider Metric Value ] ─────────────────────────────────────────────────────────────┘
            ```
            """)

# =====================================================
# 5. FOOTER ARCHITECTURE
# =====================================================
st.markdown("<br><br>", unsafe_style_html=True)
st.divider()
st.caption("🔒 DecodeLabs AI Internship — Phase 2 Predictive Classification Pipeline Node.")
