import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sephora AI Analyzer", page_icon="💄")
st.title("💄 Sephora Customer Feedback AI")
st.write("Paste a customer service email below. This app uses deep learning to summarize the issue and detect the customer's core emotion.")

@st.cache_resource
def load_pipelines():
    # Pipeline 1: TINY Pre-trained Summarizer (Safe for Streamlit memory limits)
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    
    # Pipeline 2: Your Custom Fine-Tuned Emotion Classifier 
    # REPLACE THIS STRING with your actual Hugging Face username and repo name!
    # Example: "johnsmith/sephora-emotion-classifier"
    emotion_classifier = pipeline("text-classification", model="klyipaf/emotion-classifier")
    
    return summarizer, emotion_classifier

with st.spinner("Downloading AI Models from Hugging Face... This takes about 60 seconds on the first run."):
    try:
        summarizer, emotion_classifier = load_pipelines()
        st.success("Pipelines loaded successfully from Hugging Face!")
    except Exception as e:
        st.error(f"Error loading models. Did you replace 'YOUR_HF_USERNAME' with your actual username? Error details: {e}")

st.markdown("### Step 1: Input Customer Email")
user_input = st.text_area("Paste the email text here:", height=150)

if st.button("Analyze Feedback"):
    if user_input.strip() == "":
        st.warning("Please paste some text before analyzing.")
    else:
        st.markdown("---")
        st.markdown("### Step 2: AI Analysis Results")
        
        with st.spinner("Processing text..."):
            try:
                # Run Summarizer
                summary_result = summarizer(user_input, max_length=45, min_length=10, do_sample=False)
                st.subheader("📝 Quick Summary")
                st.info(summary_result[0]['summary_text'])
                
                # Run Emotion Classifier
                emotion_result = emotion_classifier(user_input)
                label = emotion_result[0]['label']
                score = emotion_result[0]['score']
                
                emotion_map = {
                    "LABEL_0": "Sadness 😢", "LABEL_1": "Joy 😄", 
                    "LABEL_2": "Love 🥰", "LABEL_3": "Anger 😡", 
                    "LABEL_4": "Fear 😨", "LABEL_5": "Surprise 😲"
                }
                final_emotion = emotion_map.get(label, "Unknown Emotion")
                
                st.subheader("🧠 Detected Customer Emotion")
                st.success(f"**{final_emotion}** (AI Confidence Score: {score*100:.1f}%)")
            except Exception as e:
                st.error(f"Something went wrong during analysis: {e}")

