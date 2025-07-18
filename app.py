import streamlit as st
import pandas as pd
import numpy as np
import joblib
import dill
import datetime
import pytz
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from sklearn.ensemble import VotingClassifier
import os
import traceback
    
# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark mode support
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A9EFF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
        color: inherit;
    }
    .risk-very-low, .risk-low {
        background-color: rgba(40, 167, 69, 0.15);
        border-left-color: #28a745;
    }
    .risk-moderate {
        background-color: rgba(255, 193, 7, 0.15);
        border-left-color: #ffc107;
    }
    .risk-high, .risk-very-high {
        background-color: rgba(220, 53, 69, 0.15);
        border-left-color: #dc3545;
    }
    .metric-card {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: inherit;
    }
    
    /* Dark mode specific overrides */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #64B5F6;
        }
        .risk-very-low, .risk-low {
            background-color: rgba(76, 175, 80, 0.2);
        }
        .risk-moderate {
            background-color: rgba(255, 193, 7, 0.2);
        }
        .risk-high, .risk-very-high {
            background-color: rgba(244, 67, 54, 0.2);
        }
        .metric-card {
            background-color: rgba(255, 255, 255, 0.05);
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model files with proper error handling"""
    required_files = {
        'preprocessor': 'diabetes_preprocessor_advanced.dill',
        'model': 'best_diabetes_model_advanced.pkl',
        'threshold': 'optimal_threshold_advanced.pkl'
    }
    
    # Check if all required files exist
    missing_files = []
    for name, filename in required_files.items():
        if not os.path.exists(filename):
            missing_files.append(filename)
    
    if missing_files:
        return None, None, None, {
            'success': False,
            'error': f"Missing required files: {', '.join(missing_files)}"
        }
    
    try:
        # Load preprocessor with dill
        with open(required_files['preprocessor'], 'rb') as f:
            preprocessor = dill.load(f)

        # Load model
        model = joblib.load(required_files['model'])
        
        # Load threshold
        threshold = joblib.load(required_files['threshold'])
        
        return preprocessor, model, threshold, {
            'success': True,
            'preprocessor_type': str(type(preprocessor)),
            'model_type': str(type(model)),
            'threshold': threshold,
            'has_preprocess_method': hasattr(preprocessor, 'preprocess_new_data')
        }
        
    except Exception as e:
        return None, None, None, {
            'success': False,
            'error': f"Error loading model files: {str(e)}"
        }

# Local Timezone
local_tz = pytz.timezone('Africa/Lagos')  # Lagos timezone
        
def predict_diabetes_risk(patient_data, preprocessor, model, threshold):
    """Predict diabetes risk for a patient"""
    try:
        # Check if preprocessor has the required method
        if not hasattr(preprocessor, 'preprocess_new_data'):
            raise AttributeError("Preprocessor does not have 'preprocess_new_data' method")
        
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
        risk_categories = [
            (0.2, "Very Low Risk", "🟢", "risk-very-low", "Continue regular preventive care"),
            (0.4, "Low Risk", "🟢", "risk-low", "Maintain healthy lifestyle"),
            (0.6, "Moderate Risk", "🟡", "risk-moderate", "Enhanced monitoring recommended"),
            (0.8, "High Risk", "🟠", "risk-high", "Medical consultation advised"),
            (1.0, "Very High Risk", "🔴", "risk-very-high", "Immediate medical attention recommended")
        ]
        
        for threshold_val, category, color, css_class, recommendation in risk_categories:
            if risk_prob < threshold_val:
                break
        
        # Calculate overall confidence
        overall_confidence = uncertainty_info.get('overall_confidence', [0.8])[0]
        
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
            'imputed_features': uncertainty_info.get('imputed_features', []),
            'uncertainty_info': uncertainty_info
        }
    
    except Exception as e:
        return {'error': str(e)}

def create_risk_gauge(risk_prob):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk Probability (%)"},
        delta={'reference': 50},
        gauge={
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

def extract_importance_from_estimator(estimator, name):
    """Extract importance from a single estimator with multiple fallback methods"""
    try:
        # Method 1: Direct feature_importances_
        if hasattr(estimator, 'feature_importances_'):
            print(f"✓ {name}: Used Method 1 - feature_importances_")
            return estimator.feature_importances_
        
        # Method 2: Coefficients (for linear models)
        if hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            print(f"✓ {name}: Used Method 2 - coef_ (coefficients)")
            return np.abs(coef[0] if coef.ndim > 1 else coef)
        
        # Method 3: Pipeline with named_steps
        if hasattr(estimator, 'named_steps'):
            for step_name in ['classifier', 'regressor', 'model']:
                if hasattr(estimator.named_steps, step_name):
                    print(f"✓ {name}: Used Method 3 - named_steps.{step_name}")
                    final_est = getattr(estimator.named_steps, step_name)
                    return extract_importance_from_estimator(final_est, f"{name}_{step_name}")
        
        # Method 4: Pipeline with steps
        if hasattr(estimator, 'steps'):
            print(f"✓ {name}: Used Method 4 - pipeline steps")
            final_est = estimator.steps[-1][1]
            return extract_importance_from_estimator(final_est, f"{name}_final")
        
        print(f"✗ {name}: No compatible method found")
        return None
        
    except Exception as e:
        print(f"✗ {name}: Error in extraction - {str(e)}")
        return None

def get_feature_importance(model, preprocessor, patient_data=None):
    """Extract feature importance from ensemble models - ALWAYS FRESH"""
    try:
        # Clear any cached feature importance to ensure fresh calculation
        if hasattr(model, '_feature_importance_cache'):
            delattr(model, '_feature_importance_cache')
        
        # Get feature names from preprocessor
        feature_names = getattr(preprocessor, 'feature_names', [])
        
        if not feature_names:
            if hasattr(preprocessor, 'get_feature_names_out'):
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    pass
        
        if not feature_names:
            return None, None
        
        # Check if model is VotingClassifier
        if isinstance(model, VotingClassifier):
            # Get all estimators from the ensemble - FRESH calculation
            estimators = model.estimators_
            estimator_names = [name for name, _ in model.estimators]
            
            # Collect importance from each estimator
            all_importances = []
            valid_estimators = []
            
            for name, estimator in zip(estimator_names, estimators):
                try:
                    # Force fresh calculation by accessing the estimator directly
                    importance = extract_importance_from_estimator(estimator, name)
                    
                    if importance is not None and len(importance) == len(feature_names):
                        all_importances.append(importance)
                        valid_estimators.append(name)
                        
                except Exception as e:
                    continue
            
            if all_importances:
                # Average importance across all valid estimators - FRESH calculation
                avg_importance = np.mean(all_importances, axis=0)
                return avg_importance, feature_names
            else:
                return None, None
        
        else:
            # Single model - FRESH calculation
            importance = extract_importance_from_estimator(model, "Single Model")
            return importance, feature_names
    
    except Exception as e:
        st.error(f"Error extracting feature importance: {str(e)}")
        return None, None

def create_feature_importance_chart(preprocessor, model, patient_data):
    """Create a feature importance visualization - ALWAYS FRESH"""
    try:
        # Get feature importance and names - FRESH calculation
        importance_scores, feature_names = get_feature_importance(model, preprocessor, patient_data)
        
        if importance_scores is not None and feature_names is not None:
            # Create a dataframe for plotting - FRESH data
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=True)
            
            # Take top 15 features if there are too many
            if len(importance_df) > 15:
                importance_df = importance_df.tail(15)
            
            # Create bar chart with timestamp to ensure fresh rendering
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title=f"Feature Importance in Diabetes Risk Assessment (Updated: {datetime.datetime.now(local_tz).strftime('%H:%M:%S')})",
                labels={'importance': 'Importance Score', 'feature': 'Features'},
                color='importance',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                height=max(400, len(importance_df) * 30),
                showlegend=False,
                title_font_size=16,
                font=dict(size=12),
                margin=dict(l=150, r=50, t=50, b=50)
            )
            
            return fig
        else:
            return create_fallback_chart(patient_data)
        
    except Exception as e:
        st.error(f"Error creating feature importance chart: {str(e)}")
        return create_fallback_chart(patient_data)

def create_fallback_chart(patient_data):
    """Create a fallback feature importance chart based on medical knowledge"""
    medical_importance = {
        'Glucose': 0.25,
        'BMI': 0.20,
        'Age': 0.15,
        'DiabetesPedigreeFunction': 0.12,
        'Insulin': 0.10,
        'Pregnancies': 0.08,
        'BloodPressure': 0.06,
        'SkinThickness': 0.04
    }
    
    features = list(medical_importance.keys())
    importance = list(medical_importance.values())
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance (Medical Knowledge-Based)",
        labels={'x': 'Relative Importance', 'y': 'Features'},
        color=importance,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16,
        font=dict(size=12),
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    return fig

def generate_comprehensive_report(patient_data, results, preprocessor):
    """Generate a comprehensive medical report - UPDATED VERSION"""
    try:
        # Use local timezone
        current_time = datetime.datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
        
        # Debug: Check what we have
        st.write("DEBUG - Generating report with:")
        st.write(f"Patient data keys: {list(patient_data.keys()) if patient_data else 'None'}")
        st.write(f"Results keys: {list(results.keys()) if results else 'None'}")
        
        # Safe extraction of results with defaults
        risk_prob = results.get('risk_probability', 0.0)
        confidence = results.get('confidence', 0.0)
        imputed_features = results.get('imputed_features', [])
        risk_category = results.get('risk_category', 'Unknown')
        recommendation = results.get('recommendation', 'Consult healthcare provider')
        threshold_used = results.get('threshold_used', 0.5)
        
        # Build imputation details with actual values
        imputation_details = ""
        if imputed_features:
            imputation_details = "\nIMPUTATION DETAILS:\n"
            for feature in imputed_features:
                actual_value = patient_data.get(feature, 'Unknown')
                imputation_details += f"- {feature}: {actual_value} (Estimated - uncertainty quantified)\n"
        
        # Clinical interpretation
        if risk_prob > 0.8:
            clinical_notes = "Very high diabetes risk detected - Immediate medical evaluation recommended"
        elif risk_prob > 0.6:
            clinical_notes = "High diabetes risk detected - Medical consultation recommended within 2-4 weeks"
        elif risk_prob > 0.4:
            clinical_notes = "Moderate diabetes risk detected - Regular monitoring recommended"
        else:
            clinical_notes = "Low diabetes risk detected - Continue standard preventive care"
        
        # Safe data extraction with better error handling
        def safe_get(key, default='Not provided'):
            try:
                if not patient_data:
                    return default
                value = patient_data.get(key, default)
                if value is None or value == '':
                    return default
                return str(value)
            except Exception as e:
                st.warning(f"Error accessing {key}: {str(e)}")
                return default
        
        # BMI categorization
        bmi_value = float(safe_get('BMI', 0))
        if bmi_value < 18.5:
            bmi_status = 'Underweight (<18.5)'
        elif bmi_value < 25:
            bmi_status = 'Normal (18.5-25)'
        elif bmi_value < 30:
            bmi_status = 'Overweight (25-30)'
        elif bmi_value < 35:
            bmi_status = 'Obese_I (30-35)'
        else:
            bmi_status = 'Obese_II+ (>35)'
        
        # Glucose categorization
        glucose_value = float(safe_get('Glucose', 0))
        if glucose_value < 140:
            glucose_status = 'Normal_Glucose (<140)'
        elif glucose_value < 200:
            glucose_status = 'Elevated_Glucose (140-199)'
        else:
            glucose_status = 'High_Glucose (≥200)'
        
        # Build the report
        report = f"""DIABETES RISK ASSESSMENT REPORT
Generated: {current_time}
{'='*60}

PATIENT INFORMATION:
- Age: {safe_get('Age')} years
- Number of Pregnancies: {safe_get('Pregnancies')}
- Body Mass Index (BMI): {safe_get('BMI')}
- Plasma Glucose: {safe_get('Glucose')} mg/dL
- Diastolic Blood Pressure: {safe_get('BloodPressure')} mm Hg
- 2-Hour Serum Insulin: {safe_get('Insulin')} mu U/ml
- Triceps Skin Fold Thickness: {safe_get('SkinThickness')} mm
- Diabetes Pedigree Function: {safe_get('DiabetesPedigreeFunction')}

ASSESSMENT RESULTS:
- Risk Probability: {risk_prob:.1%}
- Risk Classification: {risk_category}
- Prediction Confidence: {confidence:.1%}
- Decision Threshold Used: {threshold_used:.3f}
- Clinical Recommendation: {recommendation}

{imputation_details}

CLINICAL NOTES:
{clinical_notes}

RISK FACTORS ASSESSMENT:
- Glucose Level: {glucose_status}
- BMI Status: {bmi_status}
- Age Factor: {'High Risk' if float(safe_get('Age', 0)) > 40 else 'Low Risk'}
- Blood Pressure: {'Elevated' if float(safe_get('BloodPressure', 0)) > 80 else 'Normal'}

DISCLAIMER:
This automated assessment is for informational/research purposes and clinical decision support. 
It is not intended to replace a physician's diagnosis. Follow-up diagnostic testing is required.

System Information:
- Model Type: Advanced Ensemble
- Preprocessor: {type(preprocessor).__name__ if preprocessor else 'Unknown'}
- Report Generated: {current_time}
"""
        return report
        
    except Exception as e:
        error_msg = f"Error in report generation: {str(e)}"
        st.error(error_msg)
        st.write("Full error traceback:")
        st.code(traceback.format_exc())
        
        # Return a basic report even if detailed generation fails
        return f"""DIABETES RISK ASSESSMENT REPORT
Generated: {current_time}
{'='*60}

ERROR GENERATING DETAILED REPORT: {str(e)}

BASIC RESULTS:
- Risk Probability: {results.get('risk_probability', 'Unknown')}
- Risk Classification: {results.get('risk_category', 'Unknown')}
- Recommendation: {results.get('recommendation', 'Unknown')}

Please contact system administrator for detailed report.
"""

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'show_report': False,
        'current_report': "",
        'current_patient_data': {},
        'current_results': {},
        # Remove prediction_made from defaults - let it be set only by prediction logic
        'Pregnancies': 0,
        'Glucose': 120.0,
        'BloodPressure': 70.0,
        'SkinThickness': 20.0,
        'Insulin': 80.0,
        'BMI': 25.0,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 30
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Only set prediction_made to False if it doesn't exist at all
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

def load_sample_case_data(case_data):
    """Load sample case data into session state"""
    for key, value in case_data.items():
        st.session_state[key] = value

def main():
    # Initialize session state
    initialize_session_state()

    # Load model
    preprocessor, model, threshold, load_info = load_model()
    
    if not load_info['success']:
        st.error(f"❌ Failed to load required model files: {load_info['error']}")
        st.info("Please ensure the following files are in the current directory:")
        st.code("""
- diabetes_preprocessor_advanced.dill
- best_diabetes_model_advanced.pkl
- optimal_threshold_advanced.pkl
        """)
        st.stop()
    
    # Debug info in sidebar
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.success("✅ All model files loaded successfully")
        st.sidebar.write(f"Model type: {load_info['model_type']}")
        st.sidebar.write(f"Threshold: {load_info['threshold']}")
        st.sidebar.write(f"Has preprocess method: {load_info['has_preprocess_method']}")
        st.sidebar.write(f"Prediction made: {st.session_state.prediction_made}")
        st.sidebar.write(f"Current patient data: {bool(st.session_state.current_patient_data)}")
        st.sidebar.write(f"Current results: {bool(st.session_state.current_results)}")
    
    # Header
    st.markdown('<div class="main-header">🏥 Diabetes Risk Assessment System</div>', unsafe_allow_html=True)

    # Home button - show only when prediction_made is True
    with st.sidebar:
        if st.session_state.get('prediction_made', False):
            if st.button("🏠 Return to Home", key="home_button", use_container_width=True):
                st.session_state.prediction_made = False
                # Clear all prediction-related session state
                for key in ['current_patient_data', 'current_results', 'show_report']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            st.markdown("---")
    
    # Sidebar for input
    st.sidebar.header("📝 Patient Information")
    st.sidebar.markdown("Enter patient details below:")
    
    # Patient input form
    with st.sidebar:
        pregnancies = st.number_input("👶 Number of pregnancies", 
                                    min_value=0, max_value=20, 
                                    value=st.session_state.Pregnancies, 
                                    help="Enter 0 if male or never pregnant")
        
        glucose = st.number_input("🩸 Plasma glucose concentration (mg/dL)", 
                                min_value=0.0, max_value=300.0, 
                                value=st.session_state.Glucose, 
                                step=1.0)
        
        blood_pressure = st.number_input("💓 Diastolic blood pressure (mm Hg)", 
                                       min_value=0.0, max_value=150.0, 
                                       value=st.session_state.BloodPressure, 
                                       step=1.0)
        
        skin_thickness = st.number_input("📏 Triceps skin fold thickness (mm)", 
                                       min_value=0.0, max_value=100.0, 
                                       value=st.session_state.SkinThickness, 
                                       step=1.0)
        
        insulin = st.number_input("💉 2-Hour serum insulin (mu U/ml)", 
                                min_value=0.0, max_value=900.0, 
                                value=st.session_state.Insulin, 
                                step=1.0)
        
        bmi = st.number_input("⚖️ Body mass index", 
                            min_value=0.0, max_value=70.0, 
                            value=st.session_state.BMI, 
                            step=0.1)
        
        diabetes_pedigree = st.number_input("🧬 Diabetes pedigree function", 
                                          min_value=0.0, max_value=3.0, 
                                          value=st.session_state.DiabetesPedigreeFunction, 
                                          step=0.001, format="%.3f")
        
        age = st.number_input("👤 Age (years)", 
                            min_value=1, max_value=120, 
                            value=st.session_state.Age)
        
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
                
                # Make prediction
                results = predict_diabetes_risk(patient_data, preprocessor, model, threshold)
                
                if 'error' in results:
                    st.error(f"❌ Error in prediction: {results['error']}")
                    with st.expander("Debug Information"):
                        st.write("Patient data:", patient_data)
                        st.write("Error details:", results['error'])
                else:
                    # Store data for report generation
                    st.session_state.current_patient_data = patient_data
                    st.session_state.current_results = results
                    st.session_state.prediction_made = True
                    st.session_state.show_report = False  # Reset report flag
                    
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
                            <p><strong>Prediction Confidence:</strong> {results['confidence']:.1%}</p>
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
                            - **Prediction confidence: {results['confidence']:.1%}**
                            - **Further medical evaluation is recommended**
                            """)
                        else:
                            st.success(f"""
                            **✅ Negative Diabetes Risk Indicator**
                            
                            - This patient has a **{results['risk_probability']:.1%}** probability of having diabetes
                            - Based on the optimal threshold ({results['threshold_used']:.3f}), this patient is **classified as unlikely to have diabetes**
                            - **Prediction confidence: {results['confidence']:.1%}**
                            - **Continue with regular health monitoring**
                            """)
                        
                        # Show imputed features if any
                        if results['imputed_features']:
                            st.info(f"📝 **Note:** The following features were estimated due to missing/zero values: {', '.join(results['imputed_features'])}")
                    
                    with col2:
                        st.subheader("🎯 Risk Gauge")
                        gauge_fig = create_risk_gauge(results['risk_probability'])
                        st.plotly_chart(gauge_fig, use_container_width=True)

                    # Feature importance chart
                    st.subheader("📈 Feature Importance Analysis")
                    importance_fig = create_feature_importance_chart(preprocessor, model, patient_data)
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
                    
            except Exception as e:
                st.error(f"❌ Unexpected error during prediction: {str(e)}")
                with st.expander("Debug Information"):
                    st.code(traceback.format_exc())

    # Report generation section - MOVED OUTSIDE prediction block
    if st.session_state.prediction_made and st.session_state.current_patient_data and st.session_state.current_results:
        st.subheader("📄 Generate Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Generate Detailed Report", key="generate_report"):
                with st.spinner("Generating report..."):
                    try:
                        st.session_state.current_report = generate_comprehensive_report(
                            st.session_state.current_patient_data, 
                            st.session_state.current_results, 
                            preprocessor
                        )
                        st.session_state.show_report = True
                        st.success("✅ Report generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        st.session_state.show_report = False
                        st.code(traceback.format_exc())
        
        with col2:
            if st.session_state.show_report and st.session_state.current_report:
                st.download_button(
                    label="📥 Download Report",
                    data=st.session_state.current_report,
                    file_name=f"diabetes_risk_report_{datetime.datetime.now(local_tz).strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_report"
                )
        
        # Show report if generated
        if st.session_state.show_report and st.session_state.current_report:
            st.subheader("📄 Detailed Report")
            st.text_area("Medical Report", st.session_state.current_report, height=400, key="report_display")

    # Welcome message when no prediction has been made
    if not st.session_state.prediction_made:
        st.markdown("""
        ## Welcome to the Diabetes Risk Assessment System
        
        This advanced system uses machine learning to assess diabetes risk based on key health and lifestyle indicators.
        
        ### How to use:
        1. **Enter patient information** in the sidebar
        2. **Click "Assess Risk"** to get the prediction
        3. **Review the results** and recommendations
        4. **Generate a report** for medical records
        
        ### Features:
        - 🎯 **Accurate Risk Assessment** using trained ML models
        - 📊 **Visual Risk Indicators** with color-coded categories
        - 📈 **Feature Importance Analysis** with engineered features
        - 📄 **Comprehensive Medical Reports** with uncertainty quantification
        - 🔍 **Interactive Visualizations**
        - 🧮 **Multiple Imputation** for handling missing values
        
        ### Clinical Disclaimer:
        This automated assessment is for informational/research purposes and clinical decision support. It is not intended to replace a physician's diagnosis. Follow-up diagnostic testing is required.
        """)

        # Sample cases
        st.subheader("🧪 Sample Test Cases")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👤 Low Risk Patient"):
                load_sample_case_data({
                    'Pregnancies': 1, 'Glucose': 85.0, 'BloodPressure': 66.0, 'SkinThickness': 29.0,
                    'Insulin': 0.0, 'BMI': 26.6, 'DiabetesPedigreeFunction': 0.351, 'Age': 31
                })
                st.rerun()
        
        with col2:
            if st.button("👤 Moderate Risk Patient"):
                load_sample_case_data({
                    'Pregnancies': 6, 'Glucose': 148.0, 'BloodPressure': 72.0, 'SkinThickness': 35.0,
                    'Insulin': 0.0, 'BMI': 33.6, 'DiabetesPedigreeFunction': 0.627, 'Age': 50
                })
                st.rerun()
        
        with col3:
            if st.button("👤 High Risk Patient"):
                load_sample_case_data({
                    'Pregnancies': 8, 'Glucose': 183.0, 'BloodPressure': 64.0, 'SkinThickness': 0.0,
                    'Insulin': 0.0, 'BMI': 23.3, 'DiabetesPedigreeFunction': 0.672, 'Age': 32
                })
                st.rerun()

if __name__ == "__main__":
    main()
