import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

def create_download_link(df):
    """Generates an HTML link to download a pandas DataFrame as a CSV file."""
    csv_data = df.to_csv(index=False)
    b64_data = base64.b64encode(csv_data.encode()).decode()
    download_href = f'<a href="data:file/csv;base64,{b64_data}" download="prediction_results.csv">Download Results as CSV</a>'
    return download_href

st.set_page_config(page_title="Cardio Health Analyzer", layout="wide")
st.title("Cardiovascular Health Analyzer")

tab_single, tab_batch, tab_info = st.tabs(['Single Prediction', 'Batch Analysis', 'Model Performance'])

# --- Tab 1: Single Prediction ---
with tab_single:
    st.header("Enter Patient Details for Analysis")
    
    patient_age = st.number_input("Patient's Age (in years)", min_value=0, max_value=150)
    gender = st.selectbox("Gender", ["Male", "Female"])
    pain_type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    bp_resting = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    serum_chol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_sugar = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    ecg_results = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_heart_rate = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_induced_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    st_depression = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_segment_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    if st.button("Analyze Patient Data"):
        gender_numeric = 0 if gender == "Male" else 1
        pain_type_numeric = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(pain_type)
        fasting_sugar_numeric = 1 if fasting_sugar == "> 120 mg/dl" else 0
        ecg_results_numeric = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(ecg_results)
        exercise_angina_numeric = 1 if exercise_induced_angina == "Yes" else 0
        st_slope_numeric = ["Upsloping", "Flat", "Downsloping"].index(st_segment_slope)

        user_input_df = pd.DataFrame({
            'Age': [patient_age], 'Sex': [gender_numeric], 'ChestPainType': [pain_type_numeric],
            'RestingBP': [bp_resting], 'Cholesterol': [serum_chol], 'FastingBS': [fasting_sugar_numeric],
            'RestingECG': [ecg_results_numeric], 'MaxHR': [max_heart_rate], 'ExerciseAngina': [exercise_angina_numeric],
            'Oldpeak': [st_depression], 'ST_Slope': [st_slope_numeric]
        })

        model_display_names = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
        model_filenames = ['tree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl']
        
        all_predictions = []
        for model_file in model_filenames:
            try:
                with open(model_file, 'rb') as file:
                    model = pickle.load(file)
                    prediction = model.predict(user_input_df)
                    all_predictions.append(prediction)
            except FileNotFoundError:
                st.error(f"Error: Model file '{model_file}' not found.")
                all_predictions.append([None])
            except Exception as e:
                st.error(f"An error occurred with {model_file}: {e}")
                all_predictions.append([None])
        
        st.subheader('Analysis Report')
        st.markdown('---')

        for i, result in enumerate(all_predictions):
            st.subheader(model_display_names[i])
            if result is not None and result[0] is not None:
                if result[0] == 0:
                    st.success("Result: Low Risk of Heart Disease.")
                else:
                    st.warning("Result: High Risk of Heart Disease.")
            else:
                st.error("Prediction could not be generated.")
            st.markdown('---')

# --- Tab 2: Batch Analysis ---
with tab_batch:
    st.header("Analyze Health Data from a CSV File")
    st.subheader('CSV File Instructions:')
    st.info("""
    1. The file must not contain any missing (NaN) values.
    2. It must have 11 columns in this specific order: 'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'.
    3. Please use numeric values for categorical data as specified in the single prediction tab.
    """)

    uploaded_file = st.file_uploader("Select a CSV file for analysis", type=["csv"], key="batch_uploader")

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        
        expected_cols = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
        ]

        if set(expected_cols).issubset(uploaded_df.columns):
            try:
                lr_model = pickle.load(open('LogisticRegression.pkl', 'rb'))
                
                batch_predictions = []
                for i in range(len(uploaded_df)):
                    row_data = uploaded_df[expected_cols].iloc[i].values
                    prediction = lr_model.predict([row_data])[0]
                    batch_predictions.append(prediction)
                
                uploaded_df['Logistic_Regression_Prediction'] = batch_predictions

                st.subheader("Batch Prediction Results:")
                st.dataframe(uploaded_df)

                st.markdown(create_download_link(uploaded_df), unsafe_allow_html=True)

            except FileNotFoundError:
                st.error("Error: 'LogisticRegression.pkl' model file not found.")
            except Exception as e:
                st.error(f"An unexpected error occurred during batch processing: {e}")
        else:
            st.warning("The uploaded CSV does not have the required columns. Please check instructions.")
    else:
        st.info("Please upload a CSV file to begin batch analysis.")

# --- Tab 3: Model Performance ---
with tab_info:
    st.header("Machine Learning Model Performance")
    
    accuracy_data = {
        'Decision Trees': 82.73, 
        'Logistic Regression': 87.12, 
        'Random Forest': 86.05, 
        'Support Vector Machine': 86.91
    }

    perf_df = pd.DataFrame(list(accuracy_data.items()), columns=['Model', 'Accuracy (%)'])
    
    try:
        import plotly.express as px
        fig = px.bar(perf_df, y='Accuracy (%)', x='Model', title='Model Accuracy Comparison', text='Accuracy (%)')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.error("Plotly is not installed. Please run: pip install plotly")

