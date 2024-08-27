import streamlit as st
import numpy as np
import pickle

# model = pickle.load(open('model.pkl', 'rb'))
# sc = pickle.load(open('standscaler.pkl', 'rb'))
# ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
model = pickle.load(open('model/model.pkl', 'rb'))
sc = pickle.load(open('model/standscaler.pkl', 'rb'))
ms = pickle.load(open('model/minmaxscaler.pkl', 'rb'))

st.title('CropGenie 🌱')
    

with st.form(key='input_form'):
    st.header('Enter Crop Features')

    nitrogen = st.number_input('Nitrogen', min_value=0.0, step=0.01)
    phosphorus = st.number_input('Phosphorus', min_value=0.0, step=0.01)
    potassium = st.number_input('Potassium', min_value=0.0, step=0.01)
    temperature = st.number_input('Temperature (°C)', min_value=0.0, step=0.01)
    humidity = st.number_input('Humidity (%)', min_value=0.0, step=0.01)
    ph = st.number_input('pH', min_value=0.0, step=0.01)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, step=0.01)

    submit_button = st.form_submit_button('Get Recommendation')

    if submit_button:
        features = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1)

        scaled_features = ms.transform(features)
        final_features = sc.transform(scaled_features)

        prediction = model.predict(final_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        crop = crop_dict.get(prediction[0], "Unknown")
        result = f"**{crop}** is the best crop to be cultivated with the provided data."

        st.subheader('Recommendation')
        st.write(result)

    st.write("© 2024 CropGenie - Om Laulkar. All rights reserved.")
