# Retreino com landmarks do MediaPipe Tasks

**Por quê.** O modelo atual foi treinado com o Holistic legado. Com landmarks
do MediaPipe Tasks, que é o que o app e o navegador enviam, o top-1 cai de
93,8% para 85,2%, principalmente por causa do z da pose (ver
[`VALIDACAO_MEDIAPIPE_TASKS.md`](VALIDACAO_MEDIAPIPE_TASKS.md)). Treinar com
landmarks do Tasks elimina essa diferença entre treino e uso. O retreino já
era necessário de qualquer forma para o formato de 438 dimensões.

## Dois conjuntos, para comparar

| Config | Dados | Dimensões | Diferença |
|---|---|---|---|
| `configs/tasks.yaml` | `data/processed_custom_tasks/` | 438 | pose com x, y, z, visibility |
| `configs/tasks_sem_z.yaml` | `data/processed_custom_tasks_sem_z/` | 405 | pose **sem o z**, mais robusta entre runtimes |

Os dois saem dos mesmos 338 vídeos, extraídos pelo `HolisticLandmarker`
(`mediapipe==0.10.35`, modelo `float16/1`). Cada pasta tem um
`dataset_info.json` com a origem dos dados.

A API reconhece qualquer um dos dois pelo shape de entrada do modelo. Não é
preciso mudar nada no servidor: `GET /health` mostra `lstm_keypoint_dim`
igual a 438 ou 405.

## Passo a passo

### 1. Gerar os dados (local, já feito)

```bash
# landmarks brutos (uma vez; cada backend no seu venv; ver o cabeçalho do script)
.venv-tasks/Scripts/python scripts/11_extract_landmarks.py --backend tasks \
    --model artifacts/mediapipe_tasks/holistic_landmarker.task \
    --videos <raw_custom> --out data/landmarks

# conjuntos de treino
python scripts/13_build_training_set.py --cache data/landmarks --out data/processed_custom_tasks
python scripts/13_build_training_set.py --cache data/landmarks --out data/processed_custom_tasks_sem_z --no-pose-z
```

Vídeos novos: rode o passo 11 de novo (ele pula os já extraídos) e depois o 13.

### 2. Treinar no Colab

1. Envie as pastas `data/processed_custom_tasks/` (18 MB) e
   `data/processed_custom_tasks_sem_z/` (16 MB) para o Colab, no mesmo
   caminho relativo à raiz do repositório.
2. Use o código da branch `lucas`. O treino precisa de
   `libras.features.keypoints` e da versão nova do `augmentation.py`, que
   aceita 405 dimensões.
3. Treine as duas variantes:
   ```bash
   python scripts/03_train.py --config configs/tasks.yaml
   python scripts/04_evaluate.py      # avalia o último treino; anote a acurácia de teste
   # guarde artifacts/models/ desta rodada antes da próxima (o treino sobrescreve)
   python scripts/03_train.py --config configs/tasks_sem_z.yaml
   python scripts/04_evaluate.py
   ```
4. **Mantenha o TensorFlow 2.12 no Colab.** O modelo atual foi salvo pelo
   Keras 2.12.0, a mesma versão do servidor. Um modelo salvo pelo Keras 3
   (TensorFlow 2.16 ou mais novo) não carrega no servidor.

### 3. Escolher e instalar

- Se a variante sem z ficar até cerca de 2 pontos abaixo da com z na acurácia
  de teste, **prefira a sem z**. Ela depende menos da versão do MediaPipe de
  cada cliente (navegador, Android, iOS).
- Copie `best_model.keras` e `metadata.json` da variante escolhida para
  `artifacts/models/`.
- Confira localmente: `pytest` e `GET /health`, que deve mostrar
  `lstm_keypoint_dim` igual a 438 ou 405.

## Ressalvas

- **Um único sinalizador** em todos os vídeos. A acurácia com outras pessoas
  não está medida, e gravar outros sinalizadores é o próximo ganho grande.
- **Divisão por vídeo, não por sessão.** O split é feito por vídeo. Vídeos de
  uma mesma sessão de gravação podem cair em treino e teste, o que deixa a
  acurácia de teste otimista.
- **Ferramentas locais seguem no Holistic legado.** O
  `scripts/05_run_webcam.py` e o `02/07_preprocess*.py` usam o legado e o
  formato de 438. Com um modelo de 405 dimensões, a webcam local deixa de
  funcionar até ser migrada.
