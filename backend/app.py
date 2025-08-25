# backend/app.py

from flask import Flask, render_template, url_for, flash, redirect, request, jsonify
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
from flask_bcrypt import Bcrypt
from models import db, User 
from forms import RegistrationForm, LoginForm 

import joblib
import cv2
import numpy as np
import base64
import mediapipe as mp

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
    print("Carregando o modelo de IA e o detector de mãos...")
    # Carrega o modelo de classificação (SVM ou RandomForest)
    model = joblib.load('ml_model/hand_gestures_model.pkl')
    # Não precisamos mais do label encoder, pois o joblib já lida com as classes
    
    # Configura o detector de mãos do MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    print("Modelos carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar os modelos: {e}")
    model = None
    hands = None

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

@app.route('/predict_letter', methods=['POST'])
@login_required
def predict_letter():
    data = request.json
    if not data or 'image' not in data or 'target_letter' not in data:
        return jsonify({'success': False, 'message': 'Dados de imagem ou letra alvo faltando.'}), 400
    
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
    
    hand_img = img_bgr[y_min:y_max, x_min:x_max]
    hand_img_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.resize(hand_img_gray, (50, 50))
    
    # O modelo joblib espera um array 1D
    prediction = model.predict([processed_img.flatten()])[0]
    
    is_correct = (prediction.upper() == target_letter.upper())
    
    return jsonify({
        'success': True,
        'correct': is_correct,
        'predicted_letter': prediction,
        'target_letter': target_letter,
        'box': [x_min, y_min, x_max, y_max]
    })

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

# --- Ponto de Entrada da Aplicação ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)