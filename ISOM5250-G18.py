import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(page_title="Sephora Customer Feedback AI Triage System", page_icon="💄", layout="centered")
st.title("💄 Sephora Customer Feedback AI Triage System")
st.write("Welcome to the automated feedback triage system. Paste a customer feedback below to summarize the core issue and detect the exact emotion for priority routing.")

# load models
@st.cache_resource
def load_pipelines():
    # Pipeline 1 (Pre-trained): Summarization model to shorten emails
    # Blueprint specified: sshleifer/distilbart-cnn-12-6
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    
    # Pipeline 2 (Fine-tuned): Emotion Classifier
    # IMPORTANT: Replace YOUR_HF_USERNAME below with your actual Hugging Face username!
    hf_model_url = "klyipaf/emotion-classifier"
    emotion_classifier = pipeline("text-classification", model=hf_model_url)
    
    return summarizer, emotion_classifier

# Load models with a visual spinner for the user
with st.spinner("Initializing AI Models ..."):
    try:
        summarizer, emotion_classifier = load_pipelines()
        st.success("System Ready!")
    except Exception as e:
        st.error(f"System Error: Could not load models. Details: {e}")

# User input feedback
st.markdown("### Step 1: Input Customer Feedback")
user_input = st.text_area(
    "Paste the email text here:", 
    height=150, 
    placeholder="Example: I’m contacting you to report an issue with my recent order, #SF-10493827, placed on March 8, 2026. The package arrived on March 14, but several items were not in acceptable condition. The Glow Serum bottle was leaking inside the box, and the outer packaging for the Mini Eyeshadow Palette was crushed and partially opened. The protective padding also seemed insufficient for glass items. I reached out through live chat on March 15 and provided photos of the damage as requested. The agent confirmed the claim and said I would receive a replacement confirmation email within 24–48 hours. It has now been five days and I still haven’t received an update, tracking number, or refund notice. I’ve checked my spam folder and my Sephora account order history, but there is no status change.Please either ship replacements for the damaged products immediately or issue a full refund for the affected items. I would appreciate written confirmation of the resolution and an estimated timeline."
)

# Process and output result
if st.button("Analyze Feedback"):
    # Ensure the user actually typed something long enough to summarize
    if len(user_input.strip()) < 30:
        st.warning("Please paste a longer feedback (at least 30 characters) for the AI to analyze.")
    else:
        st.markdown("---")
        st.markdown("### Step 2: AI Analysis Results")
        
        with st.spinner("Processing text through AI pipelines..."):
            try:
                # --- Pipeline 1: Summarization ---
                summary_result = summarizer(user_input, max_length=50, min_length=10, do_sample=False)
                st.subheader("📝 Core Issue Summary")
                st.info(summary_result[0]['summary_text'])

                # --- Pipeline 2: Emotion Classification ---
                emotion_result = emotion_classifier(user_input)
                label = emotion_result[0]['label']
                score = emotion_result[0]['score']
                
                # Map dair-ai/emotion model labels to business-friendly terms
                emotion_map = {
                    "LABEL_0": "Sadness 😢", "LABEL_1": "Joy 😄", 
                    "LABEL_2": "Love 🥰", "LABEL_3": "Anger 😡", 
                    "LABEL_4": "Fear 😨", "LABEL_5": "Surprise 😲"
                }
                final_emotion = emotion_map.get(label, "Unknown Emotion")
                
                st.subheader("🧠 Detected Customer Emotion")
                st.success(f"**{final_emotion}** (AI Confidence Score: {score*100:.1f}%)")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
