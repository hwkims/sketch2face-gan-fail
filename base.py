import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 1. 모델 로드 (예외 처리 포함)
model_path = 'generator_final.h5'  # h5 모델 파일 경로

try:
    model = load_model(model_path, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. 스케치 이미지 전처리 함수
def preprocess_image(image_path):
    # 이미지 로드 (컬러를 그레이스케일로 변환)
    img = Image.open(image_path).convert('L')

    # 이미지를 모델에 맞게 크기 조정 (예: 256x256)
    img = img.resize((256, 256))

    # 이미지를 numpy 배열로 변환하고 정규화
    img_array = np.array(img) / 255.0

    # 그레이스케일 이미지를 RGB로 변환 (채널 3개 추가)
    img_array = np.stack([img_array] * 3, axis=-1)  # (256, 256, 3) 형태로 변환

    # 차원을 맞추기 위해 배치 차원 추가 (1, 256, 256, 3)
    return np.expand_dims(img_array, axis=0)

# 3. 모델 예측 함수
def generate_face_from_sketch(sketch_path):
    # 이미지 전처리
    preprocessed_img = preprocess_image(sketch_path)

    # 예측
    try:
        generated_face = model.predict(preprocessed_img)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # 예측 결과 (생성된 얼굴) - 예를 들어 256x256 이미지
    generated_face = generated_face[0]  # 배치 차원 제거

    # 정규화된 값 범위 (0, 1)로 클리핑
    generated_face = np.clip(generated_face, 0, 1)

    # 4. 결과 시각화
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title("Input Sketch")
    sketch_img = Image.open(sketch_path)
    plt.imshow(sketch_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Generated Face")
    plt.imshow(generated_face)
    plt.axis('off')

    plt.show()

# 5. 스케치 파일 경로로 테스트
sketch_image_path = 'image.jpg'  # 스케치 이미지 파일 경로
generate_face_from_sketch(sketch_image_path)