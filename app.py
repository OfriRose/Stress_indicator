import streamlit as st
from google import genai
import json
import pandas as pd

# App Configuration
st.set_page_config(page_title="AutiStress AI Analyzer", layout="wide")
st.title("🧩 AutiStress AI Analyzer (v2.0)")
st.subheader("Detecting Physiological Stress Points via Gemini 2.0 Flash")

# Sidebar for Configuration
st.sidebar.header("Settings")

# Use Streamlit's secrets management for the API key
api_key = st.secrets.get("GEMINI_API_KEY")

if api_key:
    # New Client-based architecture
    client = genai.Client(api_key=api_key)

    # Model selection
    model_choice = st.sidebar.selectbox(
        "Choose a Gemini model:",
        ("gemini-2.5-flash")
    )

    # File Uploader
    uploaded_file = st.file_uploader("Upload Physiological Data (JSON)", type=['json'])

    if uploaded_file is not None:
        # Load and Parse Data
        data = json.load(uploaded_file)
        df = pd.DataFrame(data['daily_data'])

        # Visualization
        st.write("### Vital Signs Timeline")
        st.line_chart(df.set_index('timestamp')[['heart_rate', 'hrv_ms']])

        # Prompt editor
        default_prompt = f"""
        You are an expert in physiological data analysis and sensory regulation for autistic individuals.
        Analyze the following user data: {json.dumps(data)}
        
        Identify timestamps where stress is likely (high Heart Rate without physical activity, and a drop in HRV).
        Explain the physiological reasoning and suggest a brief regulation activity.
        Format the response as a clear list in English.
        """
        prompt_text = st.text_area("Analysis Prompt:", default_prompt, height=250)


        if st.button(f"Analyze Stress Points with {model_choice}"):
            with st.spinner(f"{model_choice} is analyzing the data..."):
                try:
                    # New syntax for generating content
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=prompt_text
                    )

                    st.success("Analysis Complete!")
                    st.markdown("### 🔍 AI Insights:")
                    st.markdown(response.text)

                except Exception as e:
                    st.error(f"API Error: {e}")
                    st.info(f"Ensure your API Key has access to {model_choice} models in AI Studio.")
else:
    st.info("Please add your Gemini API Key to the Streamlit secrets to begin.")
    st.code("GEMINI_API_KEY = 'YOUR_API_KEY'", language="toml")

with st.expander("System Info"):
    st.write("Using the new Google GenAI SDK architecture.")
