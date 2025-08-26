import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Caminhos
DATASET_PATH = 'data'
MODEL_PATH = 'ml_model/knn_landmarks_model.pkl'

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

def extract_hand_landmarks(image_path):
    """
    Extrai os 21 pontos de referência da mão de uma imagem.
    Retorna um array de 42 coordenadas (x, y) normalizadas ou None se a mão não for detectada.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Converte a imagem para RGB (MediaPipe exige esse formato)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Processa a imagem para encontrar os pontos da mão
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        # Pega os pontos da primeira mão detectada
        landmarks = results.multi_hand_landmarks[0].landmark
        
        # Converte os 21 pontos para um array 1D de 42 coordenadas normalizadas (x, y)
        features = np.array([[landmark.x, landmark.y] for landmark in landmarks]).flatten()
        return features
    else:
        return None

def load_and_process_data():
    """Carrega as imagens, extrai os pontos de referência e prepara o dataset."""
    X = []
    y = []
    
    for letter_folder in sorted(os.listdir(DATASET_PATH)):
        folder_path = os.path.join(DATASET_PATH, letter_folder)
        if os.path.isdir(folder_path):
            print(f"Processando imagens da pasta: {letter_folder}")
            for image_file in os.listdir(folder_path):
                if image_file.endswith('.png'):
                    image_path = os.path.join(folder_path, image_file)
                    
                    # Extrai os landmarks
                    landmarks = extract_hand_landmarks(image_path)
                    
                    if landmarks is not None:
                        X.append(landmarks)
                        y.append(letter_folder)
                        
    return np.array(X), np.array(y)

def train_and_save_model(X, y):
    """
    Treina um classificador K-NN para encontrar o gesto mais próximo.
    O K-NN é ideal para esse método, pois ele se baseia em proximidade.
    """
    if not os.path.exists('ml_model'):
        os.makedirs('ml_model')
        
    # Divide os dados para avaliação
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # O K-NN (K-Nearest Neighbors) é o modelo perfeito para essa abordagem.
    # Ele simplesmente armazena os dados de treinamento e, na predição,
    # encontra os 'k' vizinhos mais próximos da nova imagem.
    model = KNeighborsClassifier(n_neighbors=5) 
    
    print("Iniciando o treinamento do modelo K-NN com pontos de referência...")
    model.fit(X_train, y_train)
    
    # Avalia a acurácia do modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo K-NN: {accuracy * 100:.2f}%")
    
    # Salva o modelo
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo K-NN salvo em: {MODEL_PATH}")
    
    return model

if __name__ == '__main__':
    print("Iniciando a extração dos pontos de referência das imagens...")
    X, y = load_and_process_data()
    
    if len(X) > 0:
        print(f"\n{len(X)} imagens processadas com sucesso. {len(np.unique(y))} gestos distintos encontrados.")
        train_and_save_model(X, y)
    else:
        print("Nenhuma imagem encontrada ou nenhuma mão foi detectada. Verifique seu dataset e o caminho.")