import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize
import glob

# Đọc dữ liệu IRIS
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

# Đọc dữ liệu ảnh nha khoa
dental_data_path = 'C:/Users/Admin/PycharmProjects/bth5/nhakhoa'
image_size = (64, 64)  # kích thước ảnh chuẩn để huấn luyện

X_dental = []
y_dental = []

for class_folder in os.listdir(dental_data_path):
    class_path = os.path.join(dental_data_path, class_folder)
    if os.path.isdir(class_path):
        for image_path in glob.glob(os.path.join(class_path, '*.jpg')):  # Giả sử ảnh là file .jpg
            image = imread(image_path, as_gray=True)  # Đọc ảnh ở dạng grayscale
            image = resize(image, image_size)  # Resize ảnh
            X_dental.append(image.flatten())  # Chuyển ảnh thành vector
            y_dental.append(class_folder)  # Tên thư mục là nhãn của lớp

# Chuyển đổi y_dental thành các mã số bằng LabelEncoder
le = LabelEncoder()
y_dental = le.fit_transform(y_dental)

# Chia dữ liệu nha khoa thành tập huấn luyện và kiểm tra
X_train_dental, X_test_dental, y_train_dental, y_test_dental = train_test_split(X_dental, y_dental, test_size=0.3, random_state=42)

# Naive Bayes cho IRIS
nb_model_iris = GaussianNB()
nb_model_iris.fit(X_train_iris, y_train_iris)
y_pred_iris_nb = nb_model_iris.predict(X_test_iris)
print("Độ chính xác của Naive Bayes (IRIS):", accuracy_score(y_test_iris, y_pred_iris_nb))
print("Ma trận nhầm lẫn (Naive Bayes - IRIS):\n", confusion_matrix(y_test_iris, y_pred_iris_nb))

# Naive Bayes cho nha khoa
nb_model_dental = GaussianNB()
nb_model_dental.fit(X_train_dental, y_train_dental)
y_pred_dental_nb = nb_model_dental.predict(X_test_dental)
print("Độ chính xác của Naive Bayes (Nha khoa):", accuracy_score(y_test_dental, y_pred_dental_nb))
print("Ma trận nhầm lẫn (Naive Bayes - Nha khoa):\n", confusion_matrix(y_test_dental, y_pred_dental_nb))

# CART cho IRIS
cart_model_iris = DecisionTreeClassifier(criterion='gini')
cart_model_iris.fit(X_train_iris, y_train_iris)
y_pred_iris_cart = cart_model_iris.predict(X_test_iris)
print("Độ chính xác của CART (IRIS):", accuracy_score(y_test_iris, y_pred_iris_cart))
print("Ma trận nhầm lẫn (CART - IRIS):\n", confusion_matrix(y_test_iris, y_pred_iris_cart))

# CART cho nha khoa
cart_model_dental = DecisionTreeClassifier(criterion='gini')
cart_model_dental.fit(X_train_dental, y_train_dental)
y_pred_dental_cart = cart_model_dental.predict(X_test_dental)
print("Độ chính xác của CART (Nha khoa):", accuracy_score(y_test_dental, y_pred_dental_cart))
print("Ma trận nhầm lẫn (CART - Nha khoa):\n", confusion_matrix(y_test_dental, y_pred_dental_cart))

# ID3 cho IRIS
id3_model_iris = DecisionTreeClassifier(criterion='entropy')
id3_model_iris.fit(X_train_iris, y_train_iris)
y_pred_iris_id3 = id3_model_iris.predict(X_test_iris)
print("Độ chính xác của ID3 (IRIS):", accuracy_score(y_test_iris, y_pred_iris_id3))
print("Ma trận nhầm lẫn (ID3 - IRIS):\n", confusion_matrix(y_test_iris, y_pred_iris_id3))

# ID3 cho nha khoa
id3_model_dental = DecisionTreeClassifier(criterion='entropy')
id3_model_dental.fit(X_train_dental, y_train_dental)
y_pred_dental_id3 = id3_model_dental.predict(X_test_dental)
print("Độ chính xác của ID3 (Nha khoa):", accuracy_score(y_test_dental, y_pred_dental_id3))
print("Ma trận nhầm lẫn (ID3 - Nha khoa):\n", confusion_matrix(y_test_dental, y_pred_dental_id3))

# Mạng nơ-ron nhân tạo cho IRIS với max_iter cao hơn và learning_rate_init nhỏ hơn
nn_model_iris = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, learning_rate_init=0.001, random_state=42)
nn_model_iris.fit(X_train_iris, y_train_iris)
y_pred_iris_nn = nn_model_iris.predict(X_test_iris)
print("Độ chính xác của Mạng nơ-ron (IRIS):", accuracy_score(y_test_iris, y_pred_iris_nn))
print("Ma trận nhầm lẫn (Mạng nơ-ron - IRIS):\n", confusion_matrix(y_test_iris, y_pred_iris_nn))

# Mạng nơ-ron nhân tạo cho nha khoa với các thay đổi tương tự
nn_model_dental = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, learning_rate_init=0.001, random_state=42)
nn_model_dental.fit(X_train_dental, y_train_dental)
y_pred_dental_nn = nn_model_dental.predict(X_test_dental)
print("Độ chính xác của Mạng nơ-ron (Nha khoa):", accuracy_score(y_test_dental, y_pred_dental_nn))
print("Ma trận nhầm lẫn (Mạng nơ-ron - Nha khoa):\n", confusion_matrix(y_test_dental, y_pred_dental_nn))
