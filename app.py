import os
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template
import joblib
import logging

# ===== Initialize Flask App =====
app = Flask(__name__)

# ===== Paths =====
MODEL_PATH = os.path.join('model', 'credit_default_model.pkl')
PRED_LOG = os.path.join('logs', 'app_logs.csv')

# ===== Load the Model =====
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
else:
    print("⚠️ Model file not found. Please run train_model.py first.")

# ===== Ensure Logs Directory Exists =====
os.makedirs('logs', exist_ok=True)

# ===== Configure Logging =====
logging.basicConfig(
    filename=os.path.join('logs', 'flask_app.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# ===== Helper: Get Client IP =====
def get_client_ip():
    # Try to get real client IP behind proxies
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr or 'unknown'


# ===== Routes =====
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return 'Model not found. Train the model first.', 500

    # Collect form data as a dictionary
    data = request.form.to_dict()

    try:
        # Convert to DataFrame
        X = pd.DataFrame([data])

        # Convert numeric columns
        for c in X.columns:
            try:
                X[c] = pd.to_numeric(X[c])
            except Exception:
                pass

        # Predict
        proba = float(model.predict_proba(X)[:, 1][0])
        label = int(model.predict(X)[0])

    except Exception as e:
        app.logger.exception('Prediction failed')
        return f'Prediction failed: {e}', 400

    # Log prediction
    log_row = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'client_ip': get_client_ip(),
        'user_agent': request.headers.get('User-Agent', 'unknown'),
        'input': str(data),
        'pred_proba': proba,
        'pred_label': label
    }
    try:
        pd.DataFrame([log_row]).to_csv(PRED_LOG, mode='a', header=not os.path.exists(PRED_LOG), index=False)
    except Exception:
        app.logger.exception('Failed to write prediction log')

    app.logger.info('Prediction made: %.4f label=%d input_keys=%s', proba, label, ','.join(data.keys()))

    # Render result page
    return render_template('result.html', probability=proba, label=label, input=data)


# ===== Main Entry Point =====
if __name__ == '__main__':
    # For development; use gunicorn in production
    app.run(host='0.0.0.0', port=5000, debug=True)
