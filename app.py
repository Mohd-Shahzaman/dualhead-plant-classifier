import os
import re
import sqlite3
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from core.remedies import REMEDIES

app = Flask(__name__)
app.secret_key = 'agro-ai-secret-key-change-in-production'

# ── Flask-Login setup ──────────────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth'

class User(UserMixin):
    def __init__(self, id, email, password_hash):
        self.id = id
        self.email = email
        self.password_hash = password_hash

# ── SQLite user store ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_DIR, 'users.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def get_user_by_id(user_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute('SELECT id, email, password_hash FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return User(*row) if row else None

def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute('SELECT id, email, password_hash FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    return User(*row) if row else None

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(int(user_id))

init_db()

# ── ML model paths ─────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join(BASE_DIR, 'models/saved/plant_disease_dual_head.h5')
PLANT_LB_PATH   = os.path.join(BASE_DIR, 'models/encoders/plant_lb.pkl')
DISEASE_LB_PATH = os.path.join(BASE_DIR, 'models/encoders/disease_lb.pkl')

model = None
plant_lb = None
disease_lb = None

def load_assets():
    global model, plant_lb, disease_lb
    try:
        print("[*] Initializing AI Engine...")
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("[+] Dual-Head Model Loaded Successfully.")
        else:
            print(f"[ERROR] Model not found: {MODEL_PATH}")

        if os.path.exists(PLANT_LB_PATH):
            with open(PLANT_LB_PATH, 'rb') as f:
                plant_lb = pickle.load(f)
            print(f"[+] Plant Encoder Loaded ({len(plant_lb.classes_)} classes).")

        if os.path.exists(DISEASE_LB_PATH):
            with open(DISEASE_LB_PATH, 'rb') as f:
                disease_lb = pickle.load(f)
            print(f"[+] Disease Encoder Loaded ({len(disease_lb.classes_)} classes).")

        if model:
            print("[*] Pre-warming Neural Engine...")
            dummy = preprocess_input(np.zeros((1, 224, 224, 3)))
            model.predict(dummy)
            print("[+] Engine Ready.")

    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load assets: {e}")

load_assets()

# ── Prediction ─────────────────────────────────────────────────────────────────
def get_dual_prediction(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x   = img_to_array(img)
    x   = preprocess_input(x)
    x   = np.expand_dims(x, axis=0)

    plant_preds, disease_preds = model.predict(x)

    plant_idx   = np.argmax(plant_preds[0])
    disease_idx = np.argmax(disease_preds[0])

    plant_name   = plant_lb.classes_[plant_idx]
    disease_name = disease_lb.classes_[disease_idx]
    plant_conf   = float(np.max(plant_preds[0]))   * 100
    disease_conf = float(np.max(disease_preds[0])) * 100

    return {
        "plant":        plant_name,
        "disease":      disease_name,
        "plant_conf":   f"{plant_conf:.2f}%",
        "disease_conf": f"{disease_conf:.2f}%"
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/auth')
def auth():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    mode = request.args.get('mode', 'login')
    next_url = request.args.get('next', '')
    return render_template('auth.html', mode=mode, error=None, next=next_url)

@app.route('/login', methods=['POST'])
def login():
    email    = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    next_url = request.form.get('next', '') or url_for('index')
    user = get_user_by_email(email)
    if user and check_password_hash(user.password_hash, password):
        login_user(user, remember=True)
        return redirect(next_url)
    return render_template('auth.html', mode='login', error='Invalid email or password.', next=next_url)

@app.route('/signup', methods=['POST'])
def signup():
    email    = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    next_url = request.form.get('next', '') or url_for('index')
    if not email or len(password) < 6:
        return render_template('auth.html', mode='signup', error='Password must be at least 6 characters.', next=next_url)
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)',
                     (email, generate_password_hash(password)))
        conn.commit()
        conn.close()
        user = get_user_by_email(email)
        login_user(user, remember=True)
        return redirect(next_url)
    except sqlite3.IntegrityError:
        return render_template('auth.html', mode='signup', error='An account with this email already exists.', next=next_url)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

@app.route('/app')
@login_required
def index():
    return render_template('index.html', user_email=current_user.email)

@app.route('/predict', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "No file selected"}), 400

    upload_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, secure_filename(f.filename))
    f.save(file_path)

    try:
        if model is None:
            return jsonify({"status": "error", "message": "AI Model not loaded."})

        result = get_dual_prediction(file_path)
        plant_type   = result['plant']
        disease_type = result['disease']

        def clean(s):
            return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

        plant_match = disease_match = None

        for p in REMEDIES:
            if clean(p) == clean(plant_type):
                plant_match = p
                break

        if plant_match:
            for d in REMEDIES[plant_match]:
                if clean(d) == clean(disease_type):
                    disease_match = d
                    break

        if plant_match and disease_match:
            info = REMEDIES[plant_match][disease_match]
        else:
            p_display = plant_type.replace('_', ' ').title()
            d_display = disease_type.replace('_', ' ').title()
            info = {
                "title":          f"Diagnosis: {p_display} — {d_display}",
                "recommendation": "Maintain standard plant care. Consult a specialist if symptoms worsen.",
                "organic":        "Ensure good air circulation and use general purpose organic fertilizer.",
                "chemical":       "N/A — Monitor condition and apply preventive treatment if needed."
            }

        return jsonify({"status": "success", "prediction": result, "treatment": info})

    except Exception as e:
        print(f"[PREDICTION ERROR] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=5000)