import streamlit as st
import pandas as pd
import numpy as np
import joblib
import dill
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .risk-very-low {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .risk-low {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .risk-moderate {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        border-left-color: #fd7e14;
    }
    .risk-very-high {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:     
        # Load preprocessor with dill
        with open('diabetes_preprocessor_advanced.dill', 'rb') as f:
            preprocessor = dill.load(f)

        # Load model
        model = joblib.load('best_diabetes_model_advanced.pkl')
        
        # Load threshold
        threshold = joblib.load('optimal_threshold_advanced.pkl')
        
        # Return success info instead of writing to streamlit
        return preprocessor, model, threshold, {
            'success': True,
            'preprocessor_type': str(type(preprocessor)),
            'model_type': str(type(model)),
            'threshold': threshold,
            'has_preprocess_method': hasattr(preprocessor, 'preprocess_new_data'),
            'available_methods': [method for method in dir(preprocessor) if not method.startswith('_')]
        }
    except Exception as e:
        return None, None, None, {'success': False, 'error': str(e)}

def predict_diabetes_risk(patient_data, preprocessor, model, threshold):
    """Predict diabetes risk for a patient"""
    try:
        # Preprocess data with uncertainty information
        patient_processed, uncertainty_info = preprocessor.preprocess_new_data(
            patient_data, 
            return_uncertainty=True
        )
        
        # Get probability
        risk_prob = model.predict_proba(patient_processed)[0, 1]
        
        # Apply optimal threshold
        prediction = int(risk_prob >= threshold)
        
        # Risk categorization
        if risk_prob < 0.2:
            category = "Very Low Risk"
            color = "🟢"
            css_class = "risk-very-low"
            recommendation = "Continue regular preventive care"
        elif risk_prob < 0.4:
            category = "Low Risk"
            color = "🟢"
            css_class = "risk-low"
            recommendation = "Maintain healthy lifestyle"
        elif risk_prob < 0.6:
            category = "Moderate Risk"
            color = "🟡"
            css_class = "risk-moderate"
            recommendation = "Enhanced monitoring recommended"
        elif risk_prob < 0.8:
            category = "High Risk"
            color = "🟠"
            css_class = "risk-high"
            recommendation = "Medical consultation advised"
        else:
            category = "Very High Risk"
            color = "🔴"
            css_class = "risk-very-high"
            recommendation = "Immediate medical attention recommended"
        
        # Calculate overall confidence based on imputation uncertainty
        overall_confidence = uncertainty_info['overall_confidence'][0]
        
        # Add confidence indicator to recommendation if uncertainty is high
        if overall_confidence < 0.7:
            recommendation += " (Note: Some values were estimated - consider retesting)"
        
        return {
            'risk_probability': risk_prob,
            'prediction': prediction,
            'risk_category': category,
            'color': color,
            'css_class': css_class,
            'recommendation': recommendation,
            'threshold_used': threshold,
            'confidence': overall_confidence,
            'imputed_features': uncertainty_info['imputed_features']
        }
    
    except Exception as e:
        return {'error': str(e)}

def create_risk_gauge(risk_prob):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 40], 'color': "yellow"},
                {'range': [40, 60], 'color': "orange"},
                {'range': [60, 80], 'color': "red"},
                {'range': [80, 100], 'color': "darkred"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(patient_data):
    """Create a feature importance visualization"""
    # Sample feature importance values - in real app, get from actual model
    features = ['Glucose', 'BMI', 'Age', 'Pregnancies', 'Insulin', 'Blood Pressure', 'Skin Thickness', 'Diabetes Pedigree']
    importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.05]
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance in Diabetes Risk Assessment",
        labels={'x': 'Importance Score', 'y': 'Features'}
    )
    fig.update_layout(height=400)
    return fig

def generate_report(patient_data, results):
    """Generate a detailed medical report"""
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""
DIABETES RISK ASSESSMENT REPORT
Generated: {current_time}
{'='*50}

PATIENT INFORMATION:
- Age: {patient_data.get('Age', 'Unknown')} years
- Pregnancies: {patient_data.get('Pregnancies', 'Unknown')}
- BMI: {patient_data.get('BMI', 'Unknown')}
- Glucose: {patient_data.get('Glucose', 'Unknown')} mg/dL
- Blood Pressure: {patient_data.get('BloodPressure', 'Unknown')} mm Hg
- Insulin: {patient_data.get('Insulin', 'Unknown')} mu U/ml
- Skin Thickness: {patient_data.get('SkinThickness', 'Unknown')} mm
- Diabetes Pedigree Function: {patient_data.get('DiabetesPedigreeFunction', 'Unknown')}

ASSESSMENT RESULTS:
- Risk Probability: {results['risk_probability']:.1%}
- Risk Classification: {results['risk_category']}
- Clinical Recommendation: {results['recommendation']}
- Decision Threshold Used: {results['threshold_used']:.3f}

INTERPRETATION:
{
'This assessment indicates a high likelihood of diabetes. Immediate medical consultation is recommended for proper diagnosis and treatment planning.'
if results['prediction'] == 1 and results['risk_probability'] > 0.7
else 'This assessment suggests low to moderate diabetes risk. Continue regular health monitoring and maintain healthy lifestyle habits.'
}

DISCLAIMER:
This automated assessment is for informational purposes only and should not replace professional medical diagnosis. Please consult with healthcare providers for proper medical evaluation.
"""
    return report

def main():
    # Load model
    preprocessor, model, threshold, load_info = load_model()
    
    if not load_info['success']:
        st.error(f"❌ Failed to load required model files: {load_info['error']}")
        st.stop()
    
    # Display debug info if needed (optional)
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write(f"✅ Loaded preprocessor: {load_info['preprocessor_type']}")
        st.sidebar.write(f"✅ Loaded model: {load_info['model_type']}")
        st.sidebar.write(f"✅ Loaded threshold: {load_info['threshold']}")
        if load_info['has_preprocess_method']:
            st.sidebar.write("✅ Preprocessor has preprocess_new_data method")
        else:
            st.sidebar.write("⚠️ Preprocessor doesn't have preprocess_new_data method")
            st.sidebar.write(f"Available methods: {load_info['available_methods']}")
    
    # Header
    st.markdown('<div class="main-header">🏥 Diabetes Risk Assessment System</div>', unsafe_allow_html=True)
    
    # Sidebar for input
    st.sidebar.header("📝 Patient Information")
    st.sidebar.markdown("Enter patient details below:")
    
    # Patient input form
    with st.sidebar:
        pregnancies = st.number_input("👶 Number of pregnancies", min_value=0, max_value=20, value=0, help="Enter 0 if male or never pregnant")
        glucose = st.number_input("🩸 Plasma glucose concentration (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        blood_pressure = st.number_input("💓 Diastolic blood pressure (mm Hg)", min_value=0.0, max_value=150.0, value=70.0, step=1.0)
        skin_thickness = st.number_input("📏 Triceps skin fold thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        insulin = st.number_input("💉 2-Hour serum insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
        bmi = st.number_input("⚖️ Body mass index", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input("🧬 Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5, step=0.001, format="%.3f")
        age = st.number_input("👤 Age (years)", min_value=1, max_value=120, value=30)
        
        predict_button = st.button("🔍 Assess Risk", type="primary")
    
# Main content area
    if predict_button:
        with st.spinner("Analyzing patient data..."):
            try:
                # Prepare patient data
                patient_data = {
                    'Pregnancies': pregnancies,
                    'Glucose': glucose,
                    'BloodPressure': blood_pressure,
                    'SkinThickness': skin_thickness,
                    'Insulin': insulin,
                    'BMI': bmi,
                    'DiabetesPedigreeFunction': diabetes_pedigree,
                    'Age': age
                }
                
                st.write("✅ Patient data prepared")
                
                # Make prediction
                results = predict_diabetes_risk(patient_data, preprocessor, model, threshold)
                
                st.write("✅ Prediction completed")
                
                if 'error' in results:
                    st.error(f"❌ Error in prediction: {results['error']}")
                    st.write("Patient data:", patient_data)
                else:
                    st.success("✅ Risk assessment completed successfully!")
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📊 Risk Assessment Results")
                        
                        # Risk card
                        risk_card_html = f"""
                        <div class="risk-card {results['css_class']}">
                            <h3>{results['color']} {results['risk_category']}</h3>
                            <p><strong>Risk Probability:</strong> {results['risk_probability']:.1%}</p>
                            <p><strong>Recommendation:</strong> {results['recommendation']}</p>
                        </div>
                        """
                        st.markdown(risk_card_html, unsafe_allow_html=True)
                        
                        # Detailed interpretation
                        st.subheader("📋 Detailed Interpretation")
                        
                        if results['prediction'] == 1:
                            st.warning(f"""
                            **⚠️ Positive Diabetes Risk Indicator**
                            
                            - This patient has a **{results['risk_probability']:.1%}** probability of having diabetes
                            - Based on the optimal threshold ({results['threshold_used']:.3f}), this patient is **classified as likely to have diabetes**
                            - **Further medical evaluation is recommended**
                            """)
                        else:
                            st.success(f"""
                            **✅ Negative Diabetes Risk Indicator**
                            
                            - This patient has a **{results['risk_probability']:.1%}** probability of having diabetes
                            - Based on the optimal threshold ({results['threshold_used']:.3f}), this patient is **classified as unlikely to have diabetes**
                            - **Continue with regular health monitoring**
                            """)
                    
                    with col2:
                        st.subheader("🎯 Risk Gauge")
                        gauge_fig = create_risk_gauge(results['risk_probability'])
                        st.plotly_chart(gauge_fig, use_container_width=True)

                    # Feature importance chart
                    st.subheader("📈 Feature Importance Analysis")
                    importance_fig = create_feature_importance_chart(patient_data)
                    st.plotly_chart(importance_fig, use_container_width=True)
                    
                    # Patient data summary
                    st.subheader("📄 Patient Data Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Age", f"{age} years")
                        st.metric("Pregnancies", pregnancies)
                    
                    with col2:
                        st.metric("Glucose", f"{glucose} mg/dL")
                        st.metric("Blood Pressure", f"{blood_pressure} mm Hg")
                    
                    with col3:
                        st.metric("BMI", f"{bmi:.1f}")
                        st.metric("Insulin", f"{insulin} mu U/ml")
                    
                    with col4:
                        st.metric("Skin Thickness", f"{skin_thickness} mm")
                        st.metric("Diabetes Pedigree", f"{diabetes_pedigree:.3f}")
                    
                    # Generate and download report
                    st.subheader("📄 Generate Report")
                    
                    if st.button("📋 Generate Detailed Report"):
                        report = generate_report(patient_data, results)
                        st.text_area("Medical Report", report, height=400)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Report",
                            data=report,
                            file_name=f"diabetes_risk_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
    
            except Exception as e:
                st.error(f"❌ Unexpected error during prediction: {str(e)}")
                st.write("Stack trace:", e)
                import traceback
                st.code(traceback.format_exc())


    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Diabetes Risk Assessment System
        
        This advanced system uses machine learning to assess diabetes risk based on key health indicators.
        
        ### How to use:
        1. **Enter patient information** in the sidebar
        2. **Click "Assess Risk"** to get the prediction
        3. **Review the results** and recommendations
        4. **Generate a report** for medical records
        
        ### Features:
        - 🎯 **Accurate Risk Assessment** using trained ML models
        - 📊 **Visual Risk Indicators** with color-coded categories
        - 📈 **Feature Importance Analysis** 
        - 📄 **Detailed Medical Reports**
        - 🔍 **Interactive Visualizations**
        
        ### Important Disclaimer:
        This tool is for **educational and informational purposes only**. It should not replace professional medical diagnosis or treatment. Always consult with healthcare providers for proper medical evaluation.
        """)
        
        # Sample cases
        st.subheader("🧪 Sample Test Cases")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👤 Low Risk Patient"):
                st.session_state.update({
                    'pregnancies': 1, 'glucose': 85, 'blood_pressure': 66, 'skin_thickness': 29,
                    'insulin': 0, 'bmi': 26.6, 'diabetes_pedigree': 0.351, 'age': 31
                })
                st.rerun()
        
        with col2:
            if st.button("👤 Moderate Risk Patient"):
                st.session_state.update({
                    'pregnancies': 6, 'glucose': 148, 'blood_pressure': 72, 'skin_thickness': 35,
                    'insulin': 0, 'bmi': 33.6, 'diabetes_pedigree': 0.627, 'age': 50
                })
                st.rerun()
        
        with col3:
            if st.button("👤 High Risk Patient"):
                st.session_state.update({
                    'pregnancies': 8, 'glucose': 183, 'blood_pressure': 64, 'skin_thickness': 0,
                    'insulin': 0, 'bmi': 23.3, 'diabetes_pedigree': 0.672, 'age': 32
                })
                st.rerun()

if __name__ == "__main__":
    main()