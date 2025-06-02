import streamlit as st
import pandas as pd
import joblib

# Load the trained model and pre-fitted scaler
model = joblib.load(open('logreg_model.pkl', 'rb'))  # Logistic regression model
scaler = joblib.load(open('scaler.pkl', 'rb'))  # Pre-fitted scaler for scaled features

def preprocess_input_data(age, inverse_oxygen_saturation, inverse_energy_level, smoking, breathing_issue,
                          throat_discomfort, stress_immune, pollution_exposure,
                          smoking_family_history, gender):
    """
    Reverse the inverses and dynamically adjust the scaled values for correct risk interpretation.
    """
    # Prepare DataFrame for scaling
    scaled_features = pd.DataFrame({
        'AGE': [age],
        'INVERSE_OXYGEN_SATURATION': [inverse_oxygen_saturation],
        'INVERSE_ENERGY_LEVEL': [inverse_energy_level]
    })

    # Scale the features using the scaler
    scaled_features = scaler.transform(scaled_features)

    # **Reverse scaled values for risk alignment**
    reversed_energy_level = 1 - scaled_features[0][2]  # Flip the energy level scaling
    reversed_oxygen_saturation = 1 - scaled_features[0][1]  # Flip the oxygen saturation scaling

    # Create input data with reversed scaled values and other unscaled features in the correct order
    input_data = pd.DataFrame({
        'SMOKING': [smoking],
        'INVERSE_ENERGY_LEVEL': [reversed_energy_level],
        'BREATHING_ISSUE': [breathing_issue],
        'THROAT_DISCOMFORT': [throat_discomfort],
        'STRESS_IMMUNE': [stress_immune],
        'INVERSE_OXYGEN_SATURATION': [reversed_oxygen_saturation],
        'AGE': [scaled_features[0][0]],  # Scaled
        'EXPOSURE_TO_POLLUTION': [pollution_exposure],
        'SMOKING_FAMILY_HISTORY': [smoking_family_history],
        'GENDER': [gender]
    })

    return input_data

def main():
    st.title("Lung Cancer Risk Prediction App")
    st.write("Enter the patient's information below:")

    # Collect user inputs
    smoking = 1 if st.selectbox("Do you smoke?", options=["YES", "NO"]) == "YES" else 0
    inverse_energy_level = 1 / st.slider("Energy Level (33-76)", min_value=33, max_value=76)
    breathing_issue = 1 if st.selectbox("Do you have breathing issues?", options=["YES", "NO"]) == "YES" else 0
    throat_discomfort = 1 if st.selectbox("Do you have throat discomfort?", options=["YES", "NO"]) == "YES" else 0
    stress_immune = st.selectbox("Stress Immune (1 = High, 0 = Low)", options=[1, 0])
    inverse_oxygen_saturation = 1 / st.slider("Oxygen Saturation (89.9-99.9)", min_value=89.9, max_value=99.9, step=0.1)
    age = st.number_input("Enter Age (32-84)", min_value=32, max_value=84)
    pollution_exposure = 1 if st.selectbox("Are you exposed to pollution?", options=["YES", "NO"]) == "YES" else 0
    smoking_family_history = 1 if st.selectbox("Is there a family history of smoking?", options=["YES", "NO"]) == "YES" else 0
    gender = st.selectbox("Gender (1 = Female, 0 = Male)", options=[1, 0])

    # Preprocess inputs
    input_data = preprocess_input_data(
        age, inverse_oxygen_saturation, inverse_energy_level, smoking, breathing_issue,
        throat_discomfort, stress_immune, pollution_exposure, smoking_family_history, gender
    )

    # Predict button functionality
    if st.button('Predict'):
        # Make predictions using the trained model
        make_prediction = model.predict_proba(input_data)
        prob_not_at_risk = make_prediction[0][0]
        prob_at_risk = make_prediction[0][1]

        # Display predictions
        st.success(f"The probability of being at risk of lung cancer: {prob_at_risk * 100:.2f}%")
        st.success(f"The probability of not being at risk of lung cancer: {prob_not_at_risk * 100:.2f}%")

if __name__ == '__main__':
    main()
