import streamlit as st
from transformers import pipeline

# Set up the visual layout of the app
st.set_page_config(page_title="Sephora AI Analyzer", page_icon="💄")
st.title("💄 Sephora Customer Feedback AI")
st.write("Paste a customer service email below. This app uses deep learning to summarize the issue and detect the customer's core emotion.")

# Load the models using Streamlit's cache so it runs efficiently (High grades for Efficiency!)
@st.cache_resource
def load_pipelines():
    # Pipeline 1: Pre-trained Summarizer (Directly from Hugging Face)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    # Pipeline 2: Your Custom Fine-Tuned Emotion Classifier 
    # (Loading from the folder you uploaded to GitHub)
    emotion_classifier = pipeline("text-classification", model="./Fine-tuned_Model_files/emotion_classifier")
    
    return summarizer, emotion_classifier

# Load models with a loading spinner
with st.spinner("Loading AI Models... This may take a moment on first run."):
    try:
        summarizer, emotion_classifier = load_pipelines()
        st.success("Pipelines loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models. Did you upload the model folder to GitHub? Error: {e}")

# Create the user interface
st.markdown("### Step 1: Input Customer Email")
user_input = st.text_area("Paste the email text here:", height=150, placeholder="Example: I am absolutely furious! My package arrived completely destroyed...")

if st.button("Analyze Feedback"):
    if user_input.strip() == "":
        st.warning("Please paste some text before analyzing.")
    else:
        st.markdown("---")
        st.markdown("### Step 2: AI Analysis Results")
        
        with st.spinner("Processing text through AI pipelines..."):
            
            # --- RUN PIPELINE 1: SUMMARIZATION ---
            # Using max_length to ensure the summary is short and efficient
            summary_result = summarizer(user_input, max_length=45, min_length=10, do_sample=False)
            st.subheader("📝 Quick Summary")
            st.info(summary_result[0]['summary_text'])
            
            # --- RUN PIPELINE 2: EMOTION CLASSIFICATION ---
            emotion_result = emotion_classifier(user_input)
            label = emotion_result[0]['label']
            score = emotion_result[0]['score']
            
            # Map the model's output labels to human-readable emotions
            emotion_map = {
                "LABEL_0": "Sadness 😢", 
                "LABEL_1": "Joy 😄", 
                "LABEL_2": "Love 🥰", 
                "LABEL_3": "Anger 😡", 
                "LABEL_4": "Fear 😨", 
                "LABEL_5": "Surprise 😲"
            }
            final_emotion = emotion_map.get(label, "Unknown Emotion")
            
            st.subheader("🧠 Detected Customer Emotion")
            st.success(f"**{final_emotion}** (AI Confidence Score: {score*100:.1f}%)")
            
            st.markdown("---")
            st.caption("This business application was developed for ISOM5240.")
