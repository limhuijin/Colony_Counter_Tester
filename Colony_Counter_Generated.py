import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# 모델 로드
model_path = 'C:/Users/gabri/Desktop/coding/Colony_Counter_Tester/model/colony_detector_model_02.keras'
model = tf.keras.models.load_model(model_path)

# 콜로니 검출 및 시각화 함수
def detect_and_visualize(image_path, model):
    original_image = Image.open(image_path).convert("RGB")  # RGBA를 RGB로 변환
    image = original_image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    bbox = predictions[0]
    
    # bounding box 좌표를 원래 이미지 크기에 맞게 변환
    img_width, img_height = original_image.size
    bbox = [
        bbox[0] * img_width,
        bbox[1] * img_height,
        bbox[2] * img_width,
        bbox[3] * img_height
    ]

    # 원래 이미지에 bounding box 시각화
    draw = ImageDraw.Draw(original_image)
    draw.rectangle(
        [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
        outline="red",
        width=3
    )

    plt.imshow(original_image)
    plt.axis('off')  # 축 숨기기
    plt.show()

# 예시 이미지 경로
image_base_path = 'C:/Users/gabri/Desktop/coding/Colony_Counter_Tester/images'
image_filenames = [
    'colony_01.png',
]

# 이미지 파일 이름 리스트에서 이미지가 존재하는 첫 번째 경로 사용
example_image_path = None
for filename in image_filenames:
    path = os.path.join(image_base_path, filename)
    if os.path.exists(path):
        example_image_path = path
        break

if example_image_path:
    detect_and_visualize(example_image_path, model)
else:
    print("No valid image found in the specified paths.")