import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import requests
from io import BytesIO

# ndjson 파일 경로
file_path = 'C:/Users/gabri/Desktop/coding/Colony_Counter_Tester/dataset/Colony_Counterbox.ndjson'

# 파일 내용 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# 데이터 프레임으로 변환
df = pd.json_normalize(data)

# 데이터 프레임의 열 이름 확인
print(df.columns)

# 데이터 프레임 확인
print(df.head())

# 10번째 이미지 데이터 추출
row = df.iloc[9]

# 이미지 URL 가져오기
image_url = row['data_row.row_data']

# 이미지 URL에서 이미지 불러오기
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
draw = ImageDraw.Draw(image)

# 이미지 크기 확인
image_width, image_height = image.size

# 콜로니 위치 정보가 포함된 열을 파싱하여 이미지에 표시
project_key = 'projects.clynqj52408k707zxhge54g8q.labels'
if project_key in row and row[project_key]:
    labels = row[project_key]
    if labels:
        annotations = labels[0]['annotations']
        objects = annotations['objects']
        for colony in objects:
            bounding_box = colony['bounding_box']
            top = bounding_box['top'] / 100 * image_height
            left = bounding_box['left'] / 100 * image_width
            height = bounding_box['height'] / 100 * image_height
            width = bounding_box['width'] / 100 * image_width

            # 콜로니 사각형 그리기
            draw.rectangle([left, top, left + width, top + height], outline="red", width=2)

# 이미지 표시
plt.imshow(image)
plt.axis('off')
plt.show()

# 콜로니 개수 시각화
colony_count = len(objects)
plt.bar(['Colonies'], [colony_count])
plt.ylabel('Count')
plt.title('Colony Count')
plt.show()
