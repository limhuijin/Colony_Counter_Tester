import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import tensorflow as tf

# 모델 로드
model_path = 'C:/Users/gabri/Desktop/coding/Colony_Counter_Tester/model/colony_detector_model_04.keras'
model = tf.keras.models.load_model(model_path)

# 콜로니 검출 및 시각화 함수
def detect_and_visualize(image_path, model):
    original_image = Image.open(image_path).convert("RGB")  # RGBA를 RGB로 변환
    image = original_image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    
    # 다수의 bounding box를 예측한다고 가정
    bboxes = predictions[0].reshape(-1, 4)

    draw = ImageDraw.Draw(original_image)
    font = ImageFont.load_default()

    # bounding box 좌표를 원래 이미지 크기에 맞게 변환 및 시각화
    img_width, img_height = original_image.size
    for bbox in bboxes:
        x, y, width, height = bbox
        x = x * img_width
        y = y * img_height
        width = width * img_width
        height = height * img_height

        draw.rectangle(
            [x, y, x + width, y + height],
            outline="red",
            width=3
        )

        # 좌표 표시
        draw.text((x, y), f"({x:.0f}, {y:.0f})", fill="red", font=font)

    # 검출된 콜로니 좌표 출력
    for bbox in bboxes:
        x, y, width, height = bbox
        x = x * img_width
        y = y * img_height
        width = width * img_width
        height = height * img_height
        print(f"Detected colony bounding box coordinates: [{x:.2f}, {y:.2f}, {width:.2f}, {height:.2f}]")
        
    plt.imshow(original_image)
    plt.axis('off')  # 축 숨기기
    plt.show()


# 예시 이미지 경로
example_image_path = 'C:/Users/gabri/Desktop/coding/Colony_Counter_Tester/images/colony_01.png'
detect_and_visualize(example_image_path, model)
