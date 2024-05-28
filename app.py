import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained linear regression model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))  # Replace with your actual model file

# Load the dataset (assuming it's in CSV format)
@st.cache_data
def load_data():
    data = pd.read_csv('diabetes.csv')  # Replace with your actual dataset file
    return data

data = load_data()

# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        precautions = (
            'The person is diabetic\n\n'
            '**FOLLOW THE FOLLOWING PRECAUTIONS:**\n'
            '1. Make healthy eating and physical activity part of your daily routine\n'
            '2. Maintain a healthy weight\n'
            '3. Monitor your blood sugar and follow your healthcare provider\'s instructions\n'
            '4. Take medications as directed\n'
            '5. Be active most days\n'
            '6. Test your blood sugar often\n'
            '7. Learn ways to manage stress\n'
            '8. When traveling, always have glucose tablets or orange juice at hand'
        )
        return precautions

# Main function to run the app
def main():
    st.image('https://th.bing.com/th/id/OIP.kVgRxzxr8vUQ8y2m0NDtUwHaHa?pid=ImgDet&w=207&h=207&c=7&dpr=1.3', width=200)
    st.title('DIABETES PREDICTION WEB APP')
    st.subheader('Enter all the parameters to get result')
    st.toast('Consult from your doctor')

    # Sidebar for user input
    with st.sidebar:
        st.header('User Input Parameters')
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Glucose level')
        BloodPressure = st.text_input('Blood Pressure value')
        SkinThickness = st.text_input('Skin Thickness value')
        Insulin = st.text_input('Insulin value')
        BMI = st.text_input('BMI value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        Age = st.text_input('Age of the person')

    # When 'Diabetes Test Result' is clicked, make the prediction and display it
    if st.button('Diabetes Test Result'):
        result = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        if 'not diabetic' in result:
            st.success(result)
        else:
            st.error(result)

if __name__ == '__main__':
    main()
