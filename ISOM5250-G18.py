import streamlit as st
from transformers import pipeline

# 1. Page Configuration & UI Setup

st.set_page_config(page_title="Sephora AI Analyzer", page_icon="💄", layout="centered")
st.title("💄 Sephora Customer Service AI")
st.write("Welcome to the automated feedback triage system. Paste a customer email below to detect their emotion and generate a draft response.")

# 2. Load Models (Cached for high efficiency and fast runtime)
@st.cache_resource
def load_pipelines():
    # Pipeline 1: Text Generation (Drafting Auto-Replies)
    # Using distilgpt2 because it is highly efficient for cloud deployment
    auto_replier = pipeline("text-generation", model="distilgpt2")
    
    # Pipeline 2: Fine-Tuned Emotion Classifier
    # IMPORTANT: Replace YOUR_HF_USERNAME below!
    hf_model_url = "klyipaf/emotion-classifier"
    emotion_classifier = pipeline("text-classification", model=hf_model_url)
    
    return auto_replier, emotion_classifier

# Load models with a visual spinner for the user
with st.spinner("Initializing AI Models from Hugging Face..."):
    try:
        auto_replier, emotion_classifier = load_pipelines()
        st.success("System Ready!")
    except Exception as e:
        st.error(f"System Error: Could not load models. Did you replace 'YOUR_HF_USERNAME'? Details: {e}")

# 3. User Input Section
st.markdown("### Step 1: Input Customer Email")
user_input = st.text_area("Paste the email text here:", height=150, placeholder="Example: I am absolutely furious! My package arrived completely destroyed...")

# 4. Processing & Output Section
if st.button("Analyze Feedback & Generate Reply"):
    if user_input.strip() == "":
        st.warning("Please paste some text before analyzing.")
    else:
        st.markdown("---")
        st.markdown("### Step 2: AI Analysis Results")
        
        with st.spinner("Processing text..."):
            try:
                # --- A. Emotion Classification ---
                emotion_result = emotion_classifier(user_input)
                label = emotion_result[0]['label']
                score = emotion_result[0]['score']
                
                # Map model labels to business-friendly terms
                emotion_map = {
                    "LABEL_0": "Sadness 😢", "LABEL_1": "Joy 😄", 
                    "LABEL_2": "Love 🥰", "LABEL_3": "Anger 😡", 
                    "LABEL_4": "Fear 😨", "LABEL_5": "Surprise 😲"
                }
                final_emotion = emotion_map.get(label, "Unknown Emotion")
                
                st.subheader("🧠 Detected Customer Emotion")
                st.info(f"**{final_emotion}** (AI Confidence Score: {score*100:.1f}%)")

                # --- B. Auto-Reply Generation ---
                st.subheader("🤖 Suggested AI Auto-Reply Draft")
                # Provide a starting prompt to guide the AI's response
                prompt = "Dear Customer, thank you for contacting Sephora. We understand your feedback and "
                reply_result = auto_replier(prompt, max_new_tokens=40, num_return_sequences=1, pad_token_id=50256)
                
                # Clean up and display the generated text
                generated_text = reply_result[0]['generated_text']
                st.success(generated_text)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")




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
    hf_model_url = "YOUR_HF_USERNAME/sephora-emotion-classifier"
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
