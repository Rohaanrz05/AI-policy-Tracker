import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# =====================================================
# 1. ENTERPRISE PAGE CONFIGURATION & THEME
# =====================================================
st.set_page_config(
    page_title="AI Policy Classification System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Style Overrides
st.markdown("""
    <style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 95%; }
    div[data-testid="stMetricValue"] { font-size: 26px !important; color: #6366F1 !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { font-size: 13px !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%) !important;
        color: white !important; border-radius: 8px !important; border: none !important;
        padding: 0.7rem 2rem !important; font-weight: 600 !important; letter-spacing: 0.5px;
        width: 100%; transition: all 0.25s ease-in-out;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(79, 70, 229, 0.4) !important; }
    div[data-baseweb="textarea"] { border-radius: 8px !important; }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# 2. CACHED LIVE PIPELINE TRAINING (FIXED FILENAMES)
# =====================================================
@st.cache_resource
def initialize_and_train_pipeline():
    try:
        # Fixed: Ensuring standard filename is consistently referenced
        DATA_FILENAME = 'ai_policy_tracker_2026 (1).csv'
        df = pd.read_csv(DATA_FILENAME)
        
        df['title'] = df['title'].fillna("").astype(str)
        df['policy_impact_score'] = pd.to_numeric(df['policy_impact_score'], errors='coerce').fillna(0)

        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(df['category'].astype(str))

        tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        X_tfidf = tfidf.fit_transform(df['title'])

        svd = TruncatedSVD(n_components=5, random_state=42)
        X_text_features = svd.fit_transform(X_tfidf)

        X_numeric = df['policy_impact_score'].values.reshape(-1, 1)
        X_combined = np.hstack((X_text_features, X_numeric))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Adjust neighbors dynamically if dataset is extremely small
        n_neighbors = min(5, len(X_train))
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_model.fit(X_train, y_train)
        
        accuracy = knn_model.score(X_test, y_test) * 100
        return knn_model, tfidf, svd, target_encoder, scaler, accuracy, None
    except Exception as e:
        return None, None, None, None, None, 0.0, str(e)

knn_model, tfidf, svd, target_encoder, scaler, model_accuracy, error_msg = initialize_and_train_pipeline()

# =====================================================
# 3. SIDEBAR CONTROLS & META PANEL
# =====================================================
with st.sidebar:
    st.markdown("<div style='text-align: center; padding-bottom: 1rem;'>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/nolan/128/security-checked.png", width=65)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.title("System Framework")
    st.caption("DecodeLabs Advanced Analytics Stack")
    st.markdown("---")
    
    st.markdown("### 👤 Engineer Assignment")
    st.info("**Developer:** M. Rohaan Zahid\n\n**Assignment Track:** AI Engineering Intern")
    
    if error_msg is None:
        st.markdown("### 📈 Real-Time Engine Health")
        st.metric(label="Live Pipeline Accuracy", value=f"{model_accuracy:.2f}%")
    st.markdown("---")
    st.caption("🔒 Architecture Security Node — Batch 2026")

# =====================================================
# 4. MAIN HUB WORKSPACE DESIGN
# =====================================================
hero_logo, hero_headline = st.columns([0.08, 0.92])
with hero_logo:
    st.markdown("<h1 style='font-size: 52px; margin:0; padding-top:5px;'>🛡️</h1>", unsafe_allow_html=True)
with hero_headline:
    st.title("AI Policy & Governance Classification Hub")
    st.caption("Supervised Track Milestone: Vector-Space Text Extraction & Geometrical Neighborhood Mapping")

st.markdown("""
This production dashboard processes unstructured legal texts into structured, vector-mapped spaces. 
By compiling mathematical text features alongside a physical policy impact rating, it uses **K-Nearest Neighbors (KNN)** to catalog items with high geometric accuracy.
""")
st.markdown("---")

if error_msg:
    # Fixed: Streamlined error code output to look for uniform dataset name
    st.error(f"❌ Initialization Error: Could not read 'ai_policy_tracker_2026.csv'. Details: {error_msg}")
else:
    col_input, col_output = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.markdown("### 📥 Stream Ingestion Window")
        
        policy_text = st.text_area(
            "Raw Policy Document Context (Title / Provision Statement)",
            height=150,
            placeholder="Insert standard policy language here... (e.g., Global trade delegates draft strict restrictions regarding automated model verification protocols...)"
        )
        
        policy_impact_score = st.slider(
            "Quantitative Policy Impact Metric Value ($W_{impact}$)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.5,
            help="Assigns structural parameter weight regarding the scope of the global regulatory item."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        compute_trigger = st.button("🚀 Analyze Framework Telemetry")

    with col_output:
        st.markdown("### 📊 Predictive Engine Analytics")
        
        if compute_trigger:
            if not policy_text.strip():
                st.warning("⚠️ Telemetry execution rejected. Source text buffer cannot be empty.")
            else:
                with st.status("Computing NLP Vectors & Geometrical Distances...", expanded=True) as status:
                    st.write("Extracting token matrix elements via TF-IDF...")
                    vectorized = tfidf.transform([policy_text])
                    time.sleep(0.1)
                    
                    st.write("Executing dimensionality compression via Truncated SVD...")
                    reduced = svd.transform(vectorized)
                    time.sleep(0.1)
                    
                    st.write("Concatenating numerical metric arrays to feature coordinates...")
                    numeric_feat = np.array([[policy_impact_score]])
                    combined_matrix = np.hstack((reduced, numeric_feat))
                    
                    st.write("Applying standardization scaling parameters...")
                    scaled_features = scaler.transform(combined_matrix)
                    
                    st.write("Running geometric distance search across KNN tree nodes...")
                    prediction = knn_model.predict(scaled_features)
                    predicted_class = target_encoder.inverse_transform(prediction)[0]
                    
                    status.update(label="Feature Computation Complete!", state="complete")
                
                st.markdown("### 🎯 Classification Target Mapping")
                
                st.markdown(f"""
                    <div style="background: linear-gradient(90deg, rgba(79,70,229,0.15) 0%, rgba(99,102,241,0.03) 100%); 
                                border-left: 5px solid #4F46E5; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;">
                        <span style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #818CF8;">Assigned Structural Taxonomy</span>
                        <h2 style="margin: 0.3rem 0 0 0; color: #FFFFFF; font-size: 26px; font-weight: 700;">{predicted_class}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(label="Ingested Impact Rating", value=f"{policy_impact_score} / 100")
                with metric_col2:
                    st.metric(label="Inference State", value="Active / Normal")
        else:
            st.info("💡 Main Interface Standby: Awaiting vector transmission from the left-hand configuration window.")
            st.markdown("""
            ```text
            [ Text Entry Stream ] ──> [ TF-IDF Processing ] ──> [ SVD Dimensional Compression ] ──┐
                                                                                                  ├──> [ Scaler Engine ] ──> [ KNN Engine ]
            [ Slider Metric Value ] ─────────────────────────────────────────────────────────────┘
            ```
            """)

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.caption("🔒 DecodeLabs AI Internship — Phase 2 Predictive Classification Pipeline Node.")
