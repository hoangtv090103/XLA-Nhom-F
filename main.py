# -*- coding: utf-8 -*-
import os
import streamlit as st
import warnings
import tensorflow as tf

import logging

_logger = logging.getLogger(__name__)

def load_model():
    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'model/final_best_model.keras')
        print(model_dir)
        model = tf.keras.models.load_model(model_dir)
        st.write("Mô hình đã được tải.")
    except Exception as e:
        st.write("Lỗi khi tải mô hình:", e)
        model = None
    return model

if __name__ == '__main__':
    st.title("Phân loại MRI")
    warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
    try:
        model = load_model()
        _logger.info("Model loaded")
    except Exception as e:
        st.write("Lỗi khi tải mô hình:", e)
        model = None
    if model is None:
        st.write("Không thể tải mô hình.")
        st.stop()
    uploaded_file = st.file_uploader("Chọn file MRI", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.write("Vui lòng tải lên một file MRI.")
    elif uploaded_file is not None:
        st.image(uploaded_file, caption='MRI', use_column_width=True)
        try:
            # Preprocess image
            image = tf.io.decode_image(uploaded_file.getvalue(), channels=3)
            image = tf.image.resize(image, [224, 224])
            image = tf.cast(image, tf.float32)
            image = tf.divide(tf.cast(image, tf.float32), 255.0)
            image = tf.expand_dims(image, axis=0)

            # Predict
            labels = {0: f'U thần kinh đệm', 1: f'U màng não', 2: f'Không u', 3: f'U tuyến yên'}

            # model.predict: dự đoán nhãn của ảnh
            prediction = model.predict(image)
            
            # argmax: trả về chỉ số của phần tử lớn nhất trong mảng
            st.write("Dự đoán:", labels[prediction.argmax()]) 
            st.write("Xác suất:", prediction)
        except Exception as e:
            st.write("Lỗi khi dự đoán:", e)
            st.write("Vui lòng cài đặt TensorFlow bằng lệnh: pip install tensorflow")