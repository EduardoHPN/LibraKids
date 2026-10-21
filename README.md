#  Libras Recognition

Sistema de reconhecimento de palavras em Libras em tempo real com
**MediaPipe Holistic + LSTM Bidirecional + TensorFlow** e feedback
automático sobre a execução do sinal.

##  Estrutura do Projeto

```
libras-recognition/
├── README.md, pyproject.toml, requirements.txt, .gitignore
│
├── configs/                   ← Configurações YAML
│   └── default.yaml
│
├── src/libras/                ← Pacote Python principal
│   ├── config.py              ← Carrega YAML + define caminhos
│   ├── data/                  ← Download, preprocess, loader
│   ├── models/                ← Arquiteturas (LSTM)
│   ├── training/              ← Callbacks, trainer, evaluator
│   ├── inference/             ← Predictor, feedback, hud, webcam
│   └── utils/                 ← MediaPipe + I/O
│
├── scripts/                   ← CLI numerados (01_ ... 05_)
├── notebooks/                 ← Jupyter para exploração
├── data/                      ← raw/, interim/, processed/
├── artifacts/                 ← models/, logs/, reports/
├── tests/                     ← Testes unitários
└── docs/                      ← Documentação técnica
```

##  Como Começar

```bash
# 1. Ambiente
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .                # Instala 'libras' como pacote editável

# 2. Pipeline numerado
python scripts/01_download.py
python scripts/02_preprocess.py --test    # testa com 30 vídeos
python scripts/02_preprocess.py            # dataset completo
python scripts/03_train.py
python scripts/04_evaluate.py
python scripts/05_run_webcam.py
```

##  Configuração

Toda configuração está em `configs/default.yaml`. Para variantes:

```bash
cp configs/default.yaml configs/small.yaml
python scripts/03_train.py --config configs/small.yaml
```

##  Backend (API) e Banco de Dados

A API Flask (`src/libras/api/`) usa PostgreSQL (Supabase) em produção. A URL do
banco vem da variável `DATABASE_URL`; sem ela, a API cai num SQLite local em
`instance/site.db` (**apenas para desenvolvimento** — um aviso aparece no log).

```bash
pip install -e .[api]
cp .env.example .env            # preencha DATABASE_URL (o .env não é versionado)
```

O schema é gerenciado por migrações (Flask-Migrate/Alembic, pasta `migrations/`):

```bash
flask --app run_backend db upgrade                    # aplica as migrações no banco
flask --app run_backend db migrate -m "descricao"     # gera migração após mudar um model
python run_backend.py                                 # sobe a API em localhost:5001
```

> Já tem um `instance/site.db` criado antes das migrações (via `db.create_all()`)?
> Marque-o como atualizado com `flask --app run_backend db stamp head` em vez de `db upgrade`.

Para copiar os usuários do SQLite local para o banco de `DATABASE_URL` (sem
duplicar; hashes bcrypt copiados como estão; rode o `db upgrade` antes):

```bash
python scripts/migrate_sqlite_to_postgres.py --dry-run   # só mostra
python scripts/migrate_sqlite_to_postgres.py
```

**Rotas de predição.** O MediaPipe roda no cliente (app ou navegador), que
envia só landmarks para `POST /predict_sequence` (palavras) e
`POST /predict_letter_landmarks` (letras). As rotas são stateless. O formato
dos payloads está em [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md). A rota
de coleta `/record_data`, que grava imagens em disco e usa MediaPipe no
servidor, só é registrada com `LIBRAS_ENABLE_RECORD_DATA=1`.

**Observações sobre o Supabase**

- Use a connection string do **Session pooler**: host `*.pooler.supabase.com`,
  porta **5432**. A conexão direta (`db.<ref>.supabase.co`) é só IPv6 e costuma
  falhar em containers; o **Transaction pooler** (porta 6543) não suporta os
  prepared statements que o psycopg 3 usa.
- Exija TLS com `?sslmode=require` no fim da URL.
- `postgres://` e `postgresql://` são convertidas automaticamente para o driver
  `postgresql+psycopg://`.

##  Versões Compatíveis

```
python 3.10.x | tensorflow 2.12.0 | mediapipe 0.10.9
protobuf 3.20.3 | opencv-python 4.8.x
```

Veja `docs/ARCHITECTURE.md` para arquitetura técnica detalhada.
