import streamlit as st
from tensorflow import keras
import numpy as np
import cv2

model = keras.models.load_model("money_model.h5")

uploaded_file = st.file_uploader("Select an Image: ", type=["jpg", "jpeg", "png"])

result = ""
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, 1)

    st.image(input_image, channels="BGR")
    input_image_resized = cv2.resize(input_image, (224, 224))
    input_image_scaled = input_image_resized/255
    input_image_reshaped = np.reshape(input_image_scaled, [1,224, 224,3])

    input_prediction = model.predict(input_image_reshaped)
    input_pred_label = np.argmax(input_prediction)

    if input_pred_label == 0:
        result ='This is 5 Saudi Riyals'
    elif input_pred_label == 1:
        result ='This is 10 Saudi Riyals'
    else:
        result ='This is 50 Saudi Riyals'
st.success(result)

