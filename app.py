import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Course Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load Trained Model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Header Section
st.title("🎓 Student Performance Predictor")
st.write("Estimate your predicted marks based on your study habits and course load.")
st.markdown("---")

# User Inputs
st.subheader("📥 Enter Student Data")

col1, col2 = st.columns(2)

with col1:
    number_courses = st.number_input(
        "Number of Courses", 
        min_value=1, 
        max_value=15, 
        value=5, 
        step=1,
        help="Total registered courses"
    )

with col2:
    time_study = st.number_input(
        "Daily Study Time (Hours)", 
        min_value=0.0, 
        max_value=24.0, 
        value=4.5, 
        step=0.5,
        help="Average hours spent studying per day"
    )

# Prediction Logic
st.markdown("---")
if st.button("🚀 Calculate Predicted Marks", use_container_width=True):
    # Align features with the trained model input
    input_data = pd.DataFrame({
        "number_courses": [number_courses],
        "time_study": [time_study]
    })
    
    # Generate Prediction
    prediction = model.predict(input_data)[0]
    
    # Result Display
    st.success("Analysis Complete!")
    st.metric(
        label="Predicted Marks / Score", 
        value=f"{prediction:.2f}"
    )
