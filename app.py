import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Policy Classification System",
    page_icon="🤖",
    layout="centered"
)

# =====================================================
# LOAD TRAINED FILES
# =====================================================

@st.cache_resource
def load_models():
    model = joblib.load("knn_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    svd = joblib.load("svd_transformer.pkl")
    encoder = joblib.load("target_encoder.pkl")
    return model, tfidf, svd, encoder

model, tfidf, svd, encoder = load_models()

# =====================================================
# HEADER
# =====================================================

st.title("🤖 AI Policy Classification System")

st.markdown("""
This application uses **Machine Learning**, **Natural Language Processing (NLP)**,
**TF-IDF**, and **K-Nearest Neighbors (KNN)** to classify AI policy documents
into predefined categories.
""")

# =====================================================
# USER INPUT
# =====================================================

policy_text = st.text_area(
    "Enter Policy Title or Policy Statement",
    height=150,
    placeholder="Example: Government introduces new AI safety regulations..."
)

# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict Category"):

    if policy_text.strip() == "":
        st.warning("Please enter policy text.")
    else:

        # TF-IDF Transformation
        vectorized = tfidf.transform([policy_text])

        # SVD Reduction
        reduced = svd.transform(vectorized)

        # Prediction
        prediction = model.predict(reduced)

        predicted_category = encoder.inverse_transform(prediction)[0]

        st.success("Prediction Completed")

        st.subheader("Predicted Category")

        st.info(predicted_category)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Project Information")

st.sidebar.markdown("""
### AI Policy Classification System

**Developed By:** M. Rohaan Zahid

### Technologies Used
- Python
- Streamlit
- Scikit-Learn
- TF-IDF
- Truncated SVD
- KNN Classifier

### Features
- NLP-based Classification
- Real-Time Predictions
- User-Friendly Interface
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption("DecodeLabs AI Internship Project 2")
