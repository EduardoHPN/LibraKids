import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib # Biblioteca para salvar o modelo

# Caminho para a pasta com as imagens
DATASET_PATH = 'data'
MODEL_PATH = 'ml_model/hand_gestures_model.pkl'

# Prepara os dados de treinamento
def load_data():
    X = [] # Lista para armazenar as imagens processadas
    y = [] # Lista para armazenar as labels (letras)
    
    # Itera sobre cada pasta (A, B, C, ...)
    for letter_folder in sorted(os.listdir(DATASET_PATH)):
        folder_path = os.path.join(DATASET_PATH, letter_folder)
        if os.path.isdir(folder_path):
            print(f"Lendo imagens da pasta: {letter_folder}")
            # Itera sobre cada imagem na pasta
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                # Lê a imagem e converte para escala de cinza
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Redimensiona a imagem para um tamanho fixo (ex: 50x50)
                    img = cv2.resize(img, (50, 50))
                    # Transforma a imagem em um array 1D
                    X.append(img.flatten())
                    y.append(letter_folder)
    return np.array(X), np.array(y)

# Treina o modelo
def train_model(X, y):
    # Divide os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Cria o classificador (SVM - Support Vector Machine)
    print("Iniciando o treinamento do modelo...")
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    # Avalia a acurácia do modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy * 100:.2f}%")
    
    return model

# Salva o modelo treinado
def save_model(model):
    if not os.path.exists('ml_model'):
        os.makedirs('ml_model')
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")

if __name__ == '__main__':
    X, y = load_data()
    if len(X) > 0:
        model = train_model(X, y)
        save_model(model)
    else:
        print("Nenhuma imagem encontrada. Verifique o caminho para o dataset.")