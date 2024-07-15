import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import natsort
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.layers import Reshape

df = pd.read_csv("/content/drive/MyDrive/RFMiD_Testing_Labels.csv")
df = df.iloc[:92,1:]



folder_dir = '/content/drive/MyDrive/Training/'
folder_list = os.listdir(folder_dir)
sorted_file_list = natsort.natsorted(folder_list)
total_img = []

for file in sorted_file_list:
    if file.endswith('.png'):  # Ensure only PNG files are processed
        img_path = os.path.join(folder_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            resize_img = cv2.resize(img, (256, 256))
            total_img.append(resize_img)

total_img = np.array(total_img)


x_train, x_test,y_train, y_test= train_test_split(total_img, values,  test_size=0.2, shuffle=True)


#정규화
x_train = x_train.reshape(x_train.shape[0], 256, 256, 3)  # x_train.shape로 알 수 있음
x_test = x_test.reshape(x_test.shape[0], 256, 256, 3) # x_test.shape로 알 수 있음
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


"""
원 핫 인코딩
y_train = to_categorical(y_train, 46)
y_test =to_categorical(y_test, 46) # 46 원핫 인코딩할 피쳐 수
"""

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(256, 256, 3),
                 activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) 
model.add(Flatten()) 
model.add(Dense(128,  activation='relu'))
model.add(Dropout(0.5))
#다중 레이블 분류
model.add(Dense(46, activation='sigmoid'))


# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#softmax이면 mean_squared_error or categorial_crossentropy

# Model checkpoint and early stopping
modelpath = "./MNIST_CNN.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(x_train, y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0,
                    callbacks=[early_stopping_callback, checkpointer])

# Evaluate the model
print("\nTest Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))


"""
이미지 불러오기 코드

folder_dir = './dataset/Training/'
folder_list = os.listdir(folder_dir)
sorted_file_list = natsort.natsorted(folder_list)

total_img = []
for dir in sorted_file_list:
    img_dir = folder_dir+dir
    img = cv2.imread(img_dir)
    resize_img = cv2.resize(img, (256, 256))
    total_img.append(resize_img)

total_img = np.array(total_img)
print(total_img.shape)


상관관계 추출 코드
# 데이터 사이의 상관 관계를 확인합니다.
corr=data_set.corr()
corr_sort=corr.sort_values('Risk1Y', ascending=False)

# 사망 결과와 가장 관련도 높은 데이터 5개를 확인합니다.
corr_sort['Risk1Y'].head(8)
x = data_set[corr_sort['Risk1Y'].head(8).index[1:]]
y = data_set['Risk1Y']
x

## 모델에 시험 데이터를 넣어보자
y_pred = model.predict(x_test)

## Sigmoid 적용
y_pred = y_pred.flatten() # 차원 펴주기
y_pred = np.where(y_pred > 0.5, 1.0, 0.0) #0.5보다크면 1.0, 작으면 0.0

from sklearn.metrics import accuracy_score
print('정확도:',accuracy_score(y_test, y_pred))

#예측 실제값 비교
print('예측 : 실제')
for i in range(len(y_pred)):
  print(y_pred[i],':',y_test.iloc[i]) #예측값과 실제값 비교해라
## 내가 직접 입력하는 손님의 정보

## total_bill,	sex,	smoker,	day,	time,	size
total_bill = input('총 식사 금액은 얼마인가요? (소숫점까지 적어주세요. 예)15.11)')
sex = input('손님의 성별은 무엇인가요? 0: 여성/ 1: 남성')
smoker = input('손님의 흡연 유무는 어떤가요? 0: 흡연/ 1: 비흡연')
day = input('손님이 방문한 요일이 무엇인가요? 0: 목/ 1: 금/ 2: 토/ 3: 일')
time = input('손님이 방문한 타임은 언제인가요? 0: 점심/ 1: 저녁')
size = input('손님의 그룹은 몇병인가요? 1~n')

data = [total_bill,	sex,	smoker,	day,	time,	size]
print('\n입력한 손님의 정보는 아래와 같습니다.')
print('총비용:',data[0],' /성별:',data[1],' /흡연유무:',data[2],' /요일:',data[3],' /시간대:',data[4],' /규모:',data[5])

## 우리가 만든 회귀 모델에 입력하기 위한 데이터 형태로 변경합니다.
input_data = np.reshape(data*2, (2,6))
input_data = input_data.astype(float)
y_pred = model.predict(input_data)
print('총비용:',data[0],' /성별:',data[1],' /흡연유무:',data[2],' /요일:',data[3],' /시간대:',data[4],' /규모:',data[5])
print('이 손님이 나에게 줄 Tip:',y_pred[0])

label이 1차원 데이터이면 => 다중 클래스분류 (softmax) ,원핫인코딩 -> categorical_cross entropy
label이 2차원 데이터이면 => 다중 레이블분류 (sigmoid) -> binary_cross entropy
"""
