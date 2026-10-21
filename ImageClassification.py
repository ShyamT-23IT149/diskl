from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


import numpy as np
import cv2
import os

data = []
labels = []

for label,folders in enumerate(["muffin","chihuahua"]):
    for file in os.listdir(folders):
        img = cv2.imread(os.path.join(folders,file))
        img = cv2.resize(img,(64,64))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        data.append(img.flatten())
        labels.append(label)

X = np.array(data)
y = np.array(labels)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = SVC()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_pred,y_test))