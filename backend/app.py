# backend/app.py

from flask import Flask, render_template, url_for, flash, redirect, request, jsonify
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
from flask_bcrypt import Bcrypt
from models import db, User
from forms import RegistrationForm, LoginForm
import os
import joblib
import cv2
import numpy as np
import base64
import mediapipe as mp
# Não é mais necessário importar HOG e resize, pois a abordagem foi alterada
# from skimage.feature import hog
# from skimage.transform import resize

# Adiciona essas duas linhas para suprimir avisos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# --- Configurações Iniciais do Flask e das Extensões ---
app.config['SECRET_KEY'] = 'sua_chave_secreta_aqui'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inicializa as extensões com a instância do app
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# --- Carregamento dos Modelos de IA e Utilitários ---
try:
    print("Carregando os modelos de IA...")
    # Carrega o modelo K-NN com os pontos de referência
    model = joblib.load('ml_model/knn_landmarks_model.pkl')
    model_classes = model.classes_

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    print("Modelos e MediaPipe carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar os modelos: {e}")
    model = None
    hands = None
    model_classes = None

# --- Funções de Autenticação do Flask-Login ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Rotas da Aplicação ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Sua conta foi criada! Agora você pode fazer login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Cadastrar', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Login bem-sucedido!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login falhou. Por favor, verifique seu nome de usuário e senha.', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

# --- Rotas Protegidas ---
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/aprender')
@login_required
def aprender_letras():
    return render_template('aprender.html')

@app.route('/em_breve')
@login_required
def em_breve():
    return render_template('em_breve.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/data_recorder')
@login_required
def data_recorder():
    return render_template('data_recorder.html')

# --- Rota de Predição (usando pontos de referência) ---
# Seu código completo do app.py, com a rota predict_letter corrigida abaixo.

@app.route('/predict_letter', methods=['POST'])
@login_required
def predict_letter():
    data = request.json
    if not data or 'image' not in data or 'target_letter' not in data:
        return jsonify({'success': False, 'message': 'Dados de imagem ou rótulo faltando.'}), 400
    
    if hands is None or model is None:
        return jsonify({'success': False, 'message': 'Modelos de IA não carregados. O servidor pode estar em manutenção.'}), 503
    
    image_data = data['image'].split(',')[1] 
    target_letter = data['target_letter']

    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({'success': False, 'message': 'Erro ao processar a imagem.'}), 400

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return jsonify({'success': True, 'correct': False, 'predicted_letter': 'Nenhuma mão detectada', 'message': 'Nenhuma mão detectada.'})
    
    # Extrai os 21 pontos de referência da mão
    landmarks = results.multi_hand_landmarks[0].landmark
    processed_landmarks = np.array([[landmark.x, landmark.y] for landmark in landmarks]).flatten()
    
    # Obtém as probabilidades para todas as classes
    probas = model.predict_proba([processed_landmarks])[0]
    
    # Encontra a classe com a maior probabilidade e o seu valor
    predicted_label_index = np.argmax(probas)
    confidence_score = probas[predicted_label_index]
    
    # *** MUDANÇA AQUI: LIMIAR MAIS ALTO ***
    CONFIDENCE_THRESHOLD = 0.8 # Aumentamos o rigor para 80% de certeza
    
    # Usa a probabilidade para tomar a decisão
    predicted_letter = model_classes[predicted_label_index]
    is_correct = False
    
    # *** MUDANÇA AQUI: NOVA LÓGICA DE VALIDAÇÃO ***
    if confidence_score < CONFIDENCE_THRESHOLD:
        # Se a confiança for baixa, o movimento é inválido
        predicted_letter = 'Movimento Inválido'
        message = 'Movimento Inválido.'
    elif predicted_letter.upper() != target_letter.upper():
        # Se a letra prevista não for a letra alvo, é um erro de gesto
        message = f"Gesto incorreto. Você fez a letra: {predicted_letter}"
    else:
        # Se a letra prevista for a letra alvo e a confiança for alta, é um acerto
        is_correct = True
        message = 'Gesto Correto!'

    # Encontra a caixa (bounding box) da mão para fins visuais
    h, w, _ = img_bgr.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for landmark in results.multi_hand_landmarks[0].landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    margin = 20
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)
    
    return jsonify({
        'success': True,
        'correct': is_correct,
        'predicted_letter': predicted_letter,
        'target_letter': target_letter,
        'box': [x_min, y_min, x_max, y_max],
        'confidence': float(confidence_score),
        'message': message
    })

# Rota para salvar a imagem de treino (agora mais alinhada à abordagem de pontos)
@app.route('/record_data', methods=['POST'])
@login_required
def record_data():
    data = request.json
    if not data or 'image' not in data or 'label' not in data:
        return jsonify({'success': False, 'message': 'Dados de imagem ou rótulo faltando.'}), 400
    
    if hands is None:
        return jsonify({'success': False, 'message': 'MediaPipe não está carregado. O servidor pode estar em manutenção.'}), 503

    image_data_base64 = data['image'].split(',')[1]
    label = data['label'].upper()

    try:
        nparr = np.frombuffer(base64.b64decode(image_data_base64), np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'success': False, 'message': 'Erro ao processar a imagem decodificada.'}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            return jsonify({'success': False, 'message': 'Nenhuma mão detectada para salvar.'})
        
        # Como o novo modelo usa landmarks, podemos salvar a imagem completa
        # em vez de apenas o recorte, o que é mais útil para re-treino ou análise.
        # Encontra a caixa (bounding box) da mão
        h, w, _ = img_bgr.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for landmark in results.multi_hand_landmarks[0].landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        
        # Recorta a imagem para pegar apenas a mão
        hand_img = img_bgr[y_min:y_max, x_min:x_max]

        # Garante que a pasta para a letra exista
        folder_path = os.path.join('data', label)
        os.makedirs(folder_path, exist_ok=True)

        existing_images = [f for f in os.listdir(folder_path) if f.startswith(label.lower()) and f.endswith('.png')]
        file_count = len(existing_images) + 1
        
        file_name = f"{label.lower()}_{file_count}.png"
        file_path = os.path.join(folder_path, file_name)

        # Salva a imagem recortada
        cv2.imwrite(file_path, hand_img)
        
        return jsonify({'success': True, 'message': f'Mão da imagem salva com sucesso como {file_name}.'})
    
    except Exception as e:
        print(f"Erro na rota /record_data: {e}")
        return jsonify({'success': False, 'message': f'Erro ao salvar imagem: {e}'}), 500


# --- Ponto de Entrada da Aplicação ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)