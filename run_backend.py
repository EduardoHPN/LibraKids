"""Ponto de entrada do backend Flask (autenticação + inferência de sinais/letras).

Requer `pip install -e .[api]` (instala o pacote `libras` + Flask/SQLAlchemy/bcrypt).
"""
from libras.api import create_app

app = create_app()

if __name__ == "__main__":
    print("\nBackend Libras rodando em http://localhost:5001")
    print("   Auth:    POST /register, /login, /logout, GET /me")
    print("   Letras:  POST /predict_letter_landmarks  (+ /record_data se LIBRAS_ENABLE_RECORD_DATA=1)")
    print("   Sinais:  POST /predict_sequence, GET /classes")
    print("   Status:  GET /health")
    app.run(host="0.0.0.0", port=5001, debug=False)