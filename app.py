# vgg16_predict.py
# pip install streamlit tensorflow pillow huggingface_hub
# streamlit run vgg16_predict.py
# https://vgg16app-bgnstkinr6jtirpwqqmfdp.streamlit.app/

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download
import json
from PIL import Image
import io

st.title("🐠🐡🦈 물고기 이미지 분류기")
st.write("🐠🐡🦈 이 이미지 분류기는 angel_fish, betta, electric_blue_acara, guppy 분류기입니다.")
st.image(["angel_fish.jpg","betta.jpg","electric_blue_acara.jpg","guppy.jpg"],caption=["angel_fish.jpg","betta.jpg","electric_blue_acara.jpg","guppy.jpg"],width=300)
#use_column_width=True
# 모델 및 클래스 불러오기
@st.cache_resource
def load_model_and_labels():
    model_path = hf_hub_download(repo_id="Mrcho5710/fish_text", filename="EffiB0_test.h5")
    label_path = hf_hub_download(repo_id="Mrcho5710/fish_text", filename="EffiB0_test.json")

    model = load_model(model_path)
    with open(label_path, 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model_and_labels()

# 사용자 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요 (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 로딩
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='업로드된 이미지', use_column_width=True)

    # 전처리
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # 예측
    predictions = model.predict(img_array)
    pred_probs = predictions[0]
    max_prob = np.max(pred_probs)
    predicted_class = class_names[np.argmax(pred_probs)]

    # 출력
    if max_prob <= 0.6:
        st.markdown("### ⚠️ 학습한 클래스가 아니거나, 분류를 실패했습니다.")
    else:
        st.markdown(f"### ✅ 예측 결과: **{predicted_class}**")

    st.markdown("### 🔢 클래스별 확률")
    for i, prob in enumerate(pred_probs):
        st.write(f"{class_names[i]}: {prob:.4f}")
