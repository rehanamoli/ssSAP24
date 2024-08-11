import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load('temperature_model.pkl')

# Step 2: Load and preprocess the data
data = pd.read_csv('IOT-temp.csv', parse_dates=["noted_date"], dayfirst=True)
data = data.drop(columns=["id", "room_id/id"], errors="ignore")
data = data.sort_values(by=["noted_date"]).reset_index(drop=True)
data_cleaned = data.drop_duplicates()
data_cleaned = data_cleaned.groupby(["noted_date", "out/in"])["temp"].agg("mean").reset_index()
data_cleaned_pivoted = data_cleaned.pivot(index="noted_date", columns="out/in", values="temp")

data_cleaned_pivoted.dropna(inplace=True)

if np.any(np.isnan(data_cleaned_pivoted['Out'])) or np.any(np.isnan(data_cleaned_pivoted['In'])):
    st.error("NaN values detected in the dataset after dropping. Please check the data.")
    raise ValueError("NaN values detected in the dataset.")

if np.any(np.isinf(data_cleaned_pivoted['Out'])) or np.any(np.isinf(data_cleaned_pivoted['In'])):
    st.error("Infinite values detected in the dataset. Please check the data.")
    raise ValueError("Infinite values detected in the dataset.")

X = data_cleaned_pivoted['Out'].values.reshape(-1, 1)
data_cleaned_pivoted['predicted_in'] = model.predict(X)

st.title('Temperature Prediction and Time Series Analysis')

outside_temp_input = st.number_input('Enter the outside temperature:', min_value=-30.0, max_value=50.0)

predicted_inside_temp = model.predict([[outside_temp_input]])[0]
st.write(f'Predicted inside temperature: {predicted_inside_temp:.2f}°C')

st.subheader('Time Series Analysis')
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(data_cleaned_pivoted.index, data_cleaned_pivoted['Out'], label='Outside Temperature', color='blue')
ax.plot(data_cleaned_pivoted.index, data_cleaned_pivoted['In'], label='Inside Temperature', color='red')
ax.plot(data_cleaned_pivoted.index, data_cleaned_pivoted['predicted_in'], label='Predicted Inside Temperature', color='green', linestyle='--')
ax.set_xlabel('Date and Time')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Time Series of Inside and Outside Temperatures')
ax.legend()

st.pyplot(fig)


st.subheader('Model Details')
st.write(f"Model Coefficient (Slope): {model.coef_[0]:.2f}")
st.write(f"Model Intercept: {model.intercept_:.2f}")
