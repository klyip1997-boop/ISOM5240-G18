import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sephora AI Analyzer", page_icon="💄")
st.title("💄 Sephora Customer Feedback AI")
st.write("Paste a customer service email below. This app uses deep learning to detect the customer's core emotion and generate a suggested auto-reply.")

@st.cache_resource
def load_pipelines():
    # Pipeline 1: Text Generation (Explicitly supported by Streamlit!)
    # We use distilgpt2 because it is tiny and lightning-fast.
    auto_replier = pipeline("text-generation", model="distilgpt2")
    
    # Pipeline 2: Your Custom Fine-Tuned Emotion Classifier 
    # REPLACE THIS STRING with your actual Hugging Face username and repo name!
    # Example: "johnsmith/sephora-emotion-classifier"
    emotion_classifier = pipeline("text-classification", model="klyipaf/emotion-classifier")
    
    return auto_replier, emotion_classifier

with st.spinner("Downloading AI Models from Hugging Face..."):
    try:
        auto_replier, emotion_classifier = load_pipelines()
        st.success("Pipelines loaded successfully from Hugging Face!")
    except Exception as e:
        st.error(f"Error loading models. Did you replace 'YOUR_HF_USERNAME'? Error details: {e}")

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
                # 1. Run Emotion Classifier
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

                # 2. Run Text Generation (Suggested Auto-Reply)
                st.subheader("🤖 Suggested AI Auto-Reply Draft")
                # We give the AI a prompt to start the reply based on the emotion
                prompt = "Dear Customer, thank you for contacting Sephora. We understand your feedback and "
                reply_result = auto_replier(prompt, max_new_tokens=30, num_return_sequences=1, pad_token_id=50256)
                
                st.info(reply_result[0]['generated_text'])
                
            except Exception as e:
                st.error(f"Something went wrong during analysis: {e}")
