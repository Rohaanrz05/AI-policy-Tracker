import streamlit as st
import joblib
import numpy as np

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Policy Classification System",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
.main .block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.stButton > button{
    width:100%;
    border-radius:10px;
    font-weight:bold;
}

.prediction-box{
    padding:20px;
    border-radius:10px;
    background-color:#1e293b;
    border-left:5px solid #6366F1;
    margin-top:15px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL FILES
# ==========================================================

@st.cache_resource
def load_models():

    model = joblib.load("knn_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    svd = joblib.load("svd_transformer.pkl")
    encoder = joblib.load("target_encoder.pkl")

    return model, tfidf, svd, encoder

try:
    model, tfidf, svd, encoder = load_models()
    model_loaded = True

except Exception as e:
    model_loaded = False
    error_message = str(e)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🛡️ AI Policy Classifier")

    st.markdown("---")

    st.markdown("""
### Developer
**M. Rohaan Zahid**

### Internship
DecodeLabs AI Internship

### Machine Learning Pipeline
- TF-IDF Vectorizer
- Truncated SVD
- K-Nearest Neighbors
- NLP Classification

### Project Goal
Automatically classify AI policy and governance documents into predefined categories.
""")

# ==========================================================
# HEADER
# ==========================================================

st.title("🤖 AI Policy Classification System")

st.markdown("""
This application uses **Natural Language Processing (NLP)** and **Machine Learning**
to classify AI policy documents into relevant categories.

Enter a policy statement, regulation, governance update, or AI-related document title below.
""")

st.divider()

# ==========================================================
# MODEL CHECK
# ==========================================================

if not model_loaded:

    st.error("Unable to load model files.")

    st.code(error_message)

    st.stop()

# ==========================================================
# USER INPUT
# ==========================================================

policy_text = st.text_area(
    "Enter Policy Text",
    height=200,
    placeholder="Example: Government introduces new AI safety regulations requiring transparency in foundation models..."
)

# ==========================================================
# PREDICTION BUTTON
# ==========================================================

predict_btn = st.button("🚀 Classify Policy")

# ==========================================================
# PREDICTION LOGIC
# ==========================================================

if predict_btn:

    if policy_text.strip() == "":

        st.warning("Please enter policy text.")

    else:

        with st.spinner("Analyzing policy document..."):

            # TF-IDF
            vectorized_text = tfidf.transform([policy_text])

            # SVD
            reduced_features = svd.transform(vectorized_text)

            # Prediction
            prediction = model.predict(reduced_features)

            predicted_category = encoder.inverse_transform(prediction)[0]

        st.success("Classification Completed Successfully")

        st.markdown(
            f"""
            <div class="prediction-box">
                <h3>Predicted Category</h3>
                <h2>{predicted_category}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Optional KNN Confidence Approximation
        try:

            distances, indices = model.kneighbors(
                reduced_features,
                n_neighbors=5
            )

            confidence = max(
                0,
                100 - (np.mean(distances) * 100)
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.1f}%"
            )

        except:
            pass

# ==========================================================
# EXAMPLES
# ==========================================================

st.divider()

st.subheader("Example Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "Government introduces strict AI safety regulations for large language models."
    )

with col2:
    st.info(
        "International trade organizations propose new AI export restrictions."
    )

with col3:
    st.info(
        "Technology companies announce ethical AI governance standards."
    )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "DecodeLabs AI Internship Project 2 | AI Policy Classification System"
)
