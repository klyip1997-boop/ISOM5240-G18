import streamlit as st
from transformers import pipeline

# 1. Page Configuration & UI Setup
st.set_page_config(page_title="Sephora AI Analyzer", page_icon="💄", layout="centered")
st.title("💄 Sephora Customer Feedback AI")
st.write("Welcome to the automated feedback triage system. Paste a customer email below to summarize the core issue and detect the exact emotion for priority routing.")

# 2. Load Models (Cached for high efficiency)
@st.cache_resource
def load_pipelines():
    # Pipeline 1 (Pre-trained): Summarization model to shorten emails
    # Blueprint specified: sshleifer/distilbart-cnn-12-6
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    # Pipeline 2 (Fine-tuned): Emotion Classifier
    # IMPORTANT: Replace YOUR_HF_USERNAME below with your actual Hugging Face username!
    hf_model_url = "klyip/emotion-classifier"
    emotion_classifier = pipeline("text-classification", model=hf_model_url)
    
    return summarizer, emotion_classifier

# Load models with a visual spinner for the user
with st.spinner("Initializing Sephora AI Models from Hugging Face..."):
    try:
        summarizer, emotion_classifier = load_pipelines()
        st.success("System Ready!")
    except Exception as e:
        st.error(f"System Error: Could not load models. Did you replace 'YOUR_HF_USERNAME'? Details: {e}")

# 3. User Input Section
st.markdown("### Step 1: Input Customer Email")
user_input = st.text_area(
    "Paste the email text here:", 
    height=150, 
    placeholder="Example: I am absolutely furious! My package arrived completely destroyed and customer service has been ignoring my calls for a week. I demand a refund immediately."
)

# 4. Processing & Output Section
if st.button("Analyze Feedback"):
    # Ensure the user actually typed something long enough to summarize
    if len(user_input.strip()) < 30:
        st.warning("Please paste a longer email (at least 30 characters) for the AI to analyze.")
    else:
        st.markdown("---")
        st.markdown("### Step 2: AI Analysis Results")
        
        with st.spinner("Processing text through AI pipelines..."):
            try:
                # --- Pipeline 1: Summarization ---
                # We limit max_length to keep the summary concise for the customer service rep
                summary_result = summarizer(user_input, max_length=50, min_length=10, do_sample=False)
                st.subheader
