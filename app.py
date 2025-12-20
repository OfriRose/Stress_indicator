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
api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

if api_key:
    # New Client-based architecture
    client = genai.Client(api_key=api_key)
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Physiological Data (JSON)", type=['json'])

    if uploaded_file is not None:
        # Load and Parse Data
        data = json.load(uploaded_file)
        df = pd.DataFrame(data['daily_data'])
        
        # Visualization
        st.write("### Vital Signs Timeline")
        st.line_chart(df.set_index('timestamp')[['heart_rate', 'hrv_ms']])

        if st.button("Analyze Stress Points with Gemini 2.0"):
            with st.spinner("Gemini 2.0 Flash is analyzing the data..."):
                # New prompt logic
                prompt_text = f"""
                You are an expert in physiological data analysis and sensory regulation for autistic individuals.
                Analyze the following user data: {json.dumps(data)}
                
                Identify timestamps where stress is likely (high Heart Rate without physical activity, and a drop in HRV).
                Explain the physiological reasoning and suggest a brief regulation activity.
                Format the response as a clear list in English.
                """
                
                try:
                    # New syntax for generating content
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_text
                    )
                    
                    st.success("Analysis Complete!")
                    st.markdown("### 🔍 AI Insights:")
                    st.write(response.text)
                    
                except Exception as e:
                    st.error(f"API Error: {e}")
                    st.info("Ensure your API Key has access to Gemini 2.0 models in AI Studio.")
else:
    st.info("Please enter your API Key in the sidebar to begin.")

with st.expander("System Info"):
    st.write("Using the new Google GenAI SDK architecture.")