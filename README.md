#  Libras Recognition

Sistema de reconhecimento de palavras em Libras em tempo real com
**MediaPipe Holistic + LSTM Bidirecional + TensorFlow** e feedback
automático sobre a execução do sinal.

##  Estrutura do Projeto

```
libras-recognition/
├── README.md, pyproject.toml, .gitignore
├── requirements.txt           ← Runtime da API (é o que entra na imagem Docker)
├── requirements-train.txt     ← Runtime + treino, notebooks e dataset
├── requirements-dev.txt       ← Testes (pytest)
├── Dockerfile, .dockerignore  ← Imagem da API
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
# 1. Ambiente (Python 3.11)
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e .[train,dev]     # Instala 'libras' como pacote editável, com treino e testes

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
pip install -e .[api,dev]       # só a API e os testes, sem MediaPipe nem OpenCV
cp .env.example .env            # preencha DATABASE_URL (o .env não é versionado)
```

Sem o extra `train`, os testes que dependem do MediaPipe são pulados.

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

##  Imagem Docker da API

A imagem usa `python:3.11-slim`, roda com usuário não-root e contém só o runtime:
`src/`, `migrations/`, `run_backend.py` e os modelos usados na inferência. O
`.dockerignore` é uma lista de liberação, então o `.env` não entra na imagem.

```bash
docker build -t librakids-api .
docker run --rm -p 8080:8080 --env-file .env librakids-api
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict_sequence \
  -H "Content-Type: application/json" --data-binary @tests/fixtures/predict_sequence.json
```

No Windows, use `curl.exe` no PowerShell. O `/health` deve mostrar
`lstm_loaded: true` e `letter_loaded: true`.

**Variáveis de ambiente**

| Variável | Padrão | Uso |
|---|---|---|
| `DATABASE_URL` | nenhum (obrigatória) | Banco PostgreSQL. Sem ela, o container não sobe: o fallback SQLite precisa criar `/app/instance`, e o usuário não-root não tem permissão. |
| `PORT` | `8080` | Porta do gunicorn e do `HEALTHCHECK`. |
| `WEB_CONCURRENCY` | `1` | Número de workers. Cada um carrega o TensorFlow e os dois modelos na memória. |

O `--env-file` não remove aspas: escreva a `DATABASE_URL` no `.env` sem elas.

**Migrações.** Não rodam a cada start. Depois de criar uma migração nova, aplique-a
num comando separado, antes de subir a nova versão da API:

```bash
docker run --rm --env-file .env librakids-api flask --app run_backend db upgrade
```

**`/record_data` não funciona nesta imagem.** A rota precisa de MediaPipe e OpenCV,
que ficam fora da imagem porque esse processamento roda no cliente. Com
`LIBRAS_ENABLE_RECORD_DATA=1`, a API falha ao subir. Use a rota só em ambiente local
com `pip install -e .[train]`.

##  Versões Compatíveis

```
python 3.11.x | tensorflow-cpu 2.12.0 | mediapipe 0.10.9
protobuf 3.20.3 | opencv-python 4.8.x | SQLAlchemy 2.1.5 (exige python 3.11+)
```

Veja `docs/ARCHITECTURE.md` para arquitetura técnica detalhada.
