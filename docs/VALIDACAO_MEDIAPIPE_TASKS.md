# Validação numérica: Holistic legado × MediaPipe Tasks (passo 2.2.2)

O LSTM foi treinado com landmarks do **Holistic legado** (`mediapipe==0.10.9`,
Python). O app e o navegador vão usar o **MediaPipe Tasks**
(`HolisticLandmarker`). Esta validação mede quanto os landmarks e a acurácia
mudam ao trocar um pelo outro, com os mesmos vídeos.

## Método

| Item | Valor |
|---|---|
| Vídeos | 338 `.mp4` de `libras-recognition/data/raw_custom`: 11 classes, 1 sinalizador, 30 fps, sem espelhar |
| Legado | `mp.solutions.holistic`, `mediapipe==0.10.9`, `model_complexity=1`, confiança 0,5/0,5 (= `configs/default.yaml`) |
| Tasks | `HolisticLandmarker` em modo VIDEO com opções padrão (as mesmas do app e do navegador), `mediapipe==0.10.35` (Python) |
| Modelo do Tasks | `holistic_landmarker/float16/1/holistic_landmarker.task`, SHA-256 `e2dab611…a50c3f8`, o mesmo arquivo do `MediapipeService` do Angular |
| LSTM | `artifacts/models/best_model.keras` (entrada 30×1662), SHA-256 `dbb052ef…980189` |
| Vetores | `libras.features.keypoints`, a mesma função que a API usa |

Para reproduzir: `scripts/11_extract_landmarks.py` (uma vez por backend, cada
um no seu venv) e depois `scripts/12_compare_landmarks.py --cache <pasta>`.

**Ressalvas**

- **Acurácia absoluta otimista.** Os vídeos são os mesmos usados no treino,
  então os números absolutos (por exemplo, 93,8% no legado) são otimistas. O
  que importa aqui é a **diferença** entre os dois backends.
- **Um único sinalizador.** A generalização para outras pessoas não foi medida.
- **Runtime Python diferente do navegador.** No Windows, o
  `HolisticLandmarker` em Python das versões 0.10.14 a 0.10.21 aborta
  (`Check failed: holder_ != nullptr The packet is empty`). Por isso a
  validação usou a 0.10.35. O Angular usa `@mediapipe/tasks-vision`
  0.10.22-rc. O modelo `.task` é o mesmo, mas os runtimes são diferentes.

## Conclusões

1. **Posições x/y praticamente iguais em face e mãos.** A mediana fica em 0,005
   larguras de ombro ou menos. Na parte superior da pose (pontos 0–22), a
   mediana é 0,036. Os pontos 23–32 (quadril e pernas) ficam fora do quadro e
   divergem mais.
2. **Convenção esquerda/direita consistente.** A mão aparece trocada em só
   0,6% das comparações. O mapeamento `leftHandLandmarks → left_hand` do
   contrato está correto, sem precisar inverter nada.
3. **`visibility` sempre presente no Tasks** (0% ausente) e próximo do legado
   (diferença mediana de 0,006). O app deve enviar o valor como vem.
4. **Detecção das mãos um pouco menor no Tasks** (cerca de 2 pontos
   percentuais), e os dois backends discordam sobre a presença da mão em cerca
   de 5% dos frames.
5. **A acurácia cai de 93,8% para 85,2% (top-1).** No top-3 a queda é pequena
   (96,7% para 95,6%). As classes mais afetadas são Sim, Bom dia, Casa e Por
   favor.
6. **A maior causa é o z da pose.** Ele responde por cerca de 65% da diferença
   nos vetores, e o Tasks tem um deslocamento sistemático (mediana −1,09
   contra −0,85). Trocar só o z da pose do Tasks pelo do legado recupera a
   acurácia para 91,1%. As mãos explicam a maior parte do resto: com o z e as
   mãos do legado, chega a 92,9%.
7. **O modelo atual depende demais do z da pose.** Sem ele, até o legado cai
   para 58,9%. O z é uma estimativa de profundidade que muda entre versões e
   implementações do MediaPipe, então é uma feature frágil para um modelo que
   vai rodar em navegador, Android e iOS.

## Recomendações

- **Retreinar com landmarks extraídos pelo MediaPipe Tasks**, com o mesmo
  `.task` e a versão fixada. Isso elimina a diferença entre treino e uso. O
  retreino já é necessário de qualquer forma para o formato de 438 dimensões.
- **Avaliar no retreino uma versão sem o z da pose** (e sem os pontos 23–32,
  que ficam fora do quadro). O objetivo é deixar o modelo robusto entre
  plataformas e versões do MediaPipe.
- **Paliativo, se o app precisar sair antes do retreino:** calibrar o z da pose
  no servidor com um ajuste linear (`0,863·z + 0,061`) recupera parte da
  acurácia (89,9%). Fica amarrado a esta versão do runtime, então não é
  recomendado como solução permanente.
- **Fixar a mesma versão do Tasks em todos os clientes**, sempre que possível.

---

## Relatório gerado (`scripts/12_compare_landmarks.py`)

Vídeos comparados: **338** (41992 frames) — modelo LSTM com entrada (30, 1662).

### 1. Detecção (frames em que a parte foi encontrada)

| Parte | Legado | Tasks | Mesma decisão |
|---|---|---|---|
| pose | 100.0% | 100.0% | 100.0% |
| face | 99.8% | 99.8% | 99.9% |
| left_hand | 49.0% | 47.1% | 95.0% |
| right_hand | 69.4% | 67.6% | 95.0% |

### 2. Diferença de posição x/y (frames em que os dois detectaram)

Em larguras de ombro: 0,05 = 5% da distância entre os ombros.

| Parte | Mediana | p90 | p99 |
|---|---|---|---|
| pose (pontos 0–22) | 0.036 | 0.154 | 0.786 |
| pose (pontos 23–32) | 0.143 | 0.296 | 0.499 |
| pose | 0.053 | 0.235 | 0.659 |
| face (lábios+sobrancelhas) | 0.005 | 0.010 | 0.017 |
| face | 0.005 | 0.012 | 0.022 |
| left_hand | 0.001 | 0.036 | 0.206 |
| right_hand | 0.001 | 0.025 | 0.142 |

`visibility` da pose: ausente (NaN) em 0.0% dos pontos do Tasks; diferença mediana para o legado 0.006 (p90 0.096).

Lateralidade: mão trocada (esquerda↔direita) em **0.6%** de 48342 comparações de mão.

### 3. Vetores finais (após `normalize_sequence` para 30 frames)

Diferença relativa ‖tasks − legado‖ / ‖legado‖ por vídeo:

| Formato | Mediana | p90 | Máx |
|---|---|---|---|
| 1662 (legado) | 13.3% | 19.8% | 38.1% |
| 438 (atual) | 27.0% | 38.3% | 67.6% |

De onde vem a diferença (fração de ‖tasks − legado‖² por componente):

| Formato | pose x/y | pose z | pose visibility | face x/y | face z | mãos x/y | mãos z |
|---|---|---|---|---|---|---|---|
| 1662 (legado) | 2.6% | 64.9% | 3.4% | 1.9% | 0.0% | 26.9% | 0.2% |
| 438 (atual) | 3.0% | 65.9% | 0.6% | 1.2% | 0.0% | 29.2% | 0.1% |

### 4. LSTM versionado

| Landmarks de | Top-1 | Top-3 |
|---|---|---|
| Holistic legado | 93.8% | 96.7% |
| MediaPipe Tasks | 85.2% | 95.6% |

Mesma palavra no top-1 nos dois: **87.6%** dos vídeos. Diferença na probabilidade da classe correta: mediana 0.001, p90 0.517.

Por classe (top-1):

| Classe | Vídeos | Legado | Tasks |
|---|---|---|---|
| Amigo | 20 | 100.0% | 100.0% |
| Bom dia | 39 | 89.7% | 74.4% |
| Casa | 26 | 96.2% | 76.9% |
| De nada | 28 | 85.7% | 89.3% |
| Desculpa | 24 | 100.0% | 95.8% |
| Família | 20 | 100.0% | 100.0% |
| Obrigado | 41 | 97.6% | 90.2% |
| Oi | 41 | 90.2% | 90.2% |
| Por favor | 31 | 100.0% | 80.6% |
| Sim | 42 | 90.5% | 69.0% |
| Trabalho | 26 | 88.5% | 88.5% |

### 5. Ablação — o que explica a diferença de acurácia

Troca partes dos landmarks do Tasks pelas do legado e mede o LSTM de novo.

| Landmarks | Top-1 | Top-3 |
|---|---|---|
| Holistic legado (referência) | 93.8% | 96.7% |
| MediaPipe Tasks | 85.2% | 95.6% |
| Tasks, z da pose do legado | 91.1% | 97.0% |
| Tasks, z da pose calibrado (0.863·z +0.061) | 89.9% | 95.9% |
| Tasks, mãos do legado | 87.9% | 96.4% |
| Tasks, z da pose e mãos do legado | 92.9% | 97.6% |
| Legado com z da pose = 0 | 58.9% | 88.5% |
