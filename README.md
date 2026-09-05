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

##  Versões Compatíveis

```
python 3.10.x | tensorflow 2.12.0 | mediapipe 0.10.9
protobuf 3.20.3 | opencv-python 4.8.x
```

Veja `docs/ARCHITECTURE.md` para arquitetura técnica detalhada.
