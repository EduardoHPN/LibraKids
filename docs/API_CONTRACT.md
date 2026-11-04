# Contrato da API de reconhecimento (app mobile e Angular)

O MediaPipe roda **no cliente** (celular ou navegador). O cliente envia os
**landmarks brutos**, sem normalizar, e o servidor faz todo o
pré-processamento: seleção do subconjunto da face, `normalize_spatial` e
`normalize_sequence`. Assim existe uma única fonte de verdade
(`src/libras/features/keypoints.py`), a mesma usada no treino.

As rotas são **stateless**: cada requisição traz tudo o que é preciso. Nenhum
landmark nem buffer fica no servidor entre chamadas. O banco só guarda as
sessões de login.

## Autenticação

`POST /predict_sequence`, `POST /predict_letter_landmarks`, `POST /record_data`,
`GET /me` e `POST /logout` exigem o header `Authorization: Bearer <token>`, com o
`access_token` devolvido por `POST /login`. As rotas `/health`, `/classes`,
`/register` e `/login` são públicas.

Sem um token válido, a resposta é sempre `401 {"message": "Nao autenticado."}`,
esteja ele ausente, malformado, inexistente, vencido ou encerrado. O cliente deve
esquecer o token e voltar ao login. O 401 de `/login` é outro caso: significa
senha errada e traz sua própria mensagem.

- **Sessões:** cada login cria uma sessão própria, então a mesma conta pode ficar
  ativa no app e no site ao mesmo tempo. `/logout` encerra só a sessão do token enviado.
- **Validade:** `LIBRAS_TOKEN_TTL_HOURS` horas contadas do login (padrão: 168 h),
  sem renovação por uso.
- **Banco fora do ar:** as rotas protegidas respondem 500, não 401. Nesse caso, o
  cliente deve manter o token.

---

## Convenções dos landmarks

| Campo | Pontos | Coordenadas por ponto | Observação |
|---|---|---|---|
| `pose` | 33 | `[x, y, z, visibility]` | topologia do MediaPipe Pose |
| `face` | 468 **ou** 478 | `[x, y, z]` | malha completa; com 478 (Tasks, com íris) o servidor usa os 468 primeiros |
| `left_hand` | 21 | `[x, y, z]` | mão **esquerda do sinalizador** |
| `right_hand` | 21 | `[x, y, z]` | mão **direita do sinalizador** |
| `hand` (letras) | 21 | `[x, y]` ou `[x, y, z]` | o modelo de letras usa só `x, y`; envie `handedness` junto |

- **Coordenadas normalizadas pela imagem.** `x` vai de 0 a 1 pela largura e
  `y` de 0 a 1 pela altura, com origem no canto superior esquerdo. `z` e
  `visibility` são enviados exatamente como o MediaPipe devolve. Pontos fora
  do quadro podem ter valores um pouco abaixo de 0 ou acima de 1, e isso é
  normal.
- **Parte não detectada → `null`.** O servidor preenche com zeros, igual ao
  treino. Também vale omitir o campo.
- **Use a imagem NÃO espelhada.** Os vídeos de treino foram gravados sem
  espelhar (`scripts/06_record_dataset.py` grava o frame original e só
  espelha a pré-visualização). Rode o MediaPipe no frame como a câmera o
  entrega e espelhe **apenas o preview** mostrado ao usuário.
  - Se a plataforma só fornecer o frame espelhado (comum na câmera frontal),
    desespelhe a imagem antes do MediaPipe. Converter os landmarks depois
    exige `x → 1 - x`, trocar `left_hand ↔ right_hand` **e** trocar os pares
    esquerda/direita da pose. Evite isso.
- **Esquerda/direita = do sinalizador.** É a convenção do Holistic: o ombro
  esquerdo é o ponto 11 da pose, e `left_hand` é a mão do mesmo lado. Num
  frame não espelhado da câmera frontal, a mão esquerda do sinalizador
  aparece do lado **direito** da imagem (x maior).
- **Números finitos** (sem `NaN`/`Infinity`). Arredondar para 4 casas
  decimais é suficiente (1/10000 da imagem) e reduz bastante o payload.

---

## `POST /predict_sequence`: reconhecimento de palavras

Corpo:

```json
{
  "frames": [
    {
      "pose":       [[0.61, 0.55, -0.21, 0.99], "... 33 pontos"],
      "face":       [[0.50, 0.31, -0.01], "... 468 ou 478 pontos"],
      "left_hand":  null,
      "right_hand": [[0.38, 0.62, -0.08], "... 21 pontos"]
    }
  ]
}
```

Limites:

| Regra | Valor |
|---|---|
| Frames por requisição | **10 a 60** |
| Tamanho máximo do corpo | **2 MB** (60 frames com 4 casas decimais ≈ 0,8 MB) |
| Mãos | pelo menos um frame com `left_hand` ou `right_hand` |

**Recomendação de captura: a sequência deve cobrir cerca de 4 s, a duração
dos clipes de treino.** No treino, cada clipe de 4 s a 30 fps é reamostrado
para 30 frames, o que dá cerca de **7,5 fps**. O servidor também reamostra
uniformemente para os 30 frames do modelo (`num_frames`), então o que importa
é o **tempo coberto**, não a taxa de captura. 30 frames capturados a 30 fps
cobrem só 1 s e chegam ao modelo como um sinal 4 vezes mais rápido.

- **Recomendado:** amostrar 1 frame a cada ~133 ms (7,5 fps) e enviar
  janelas de 30 frames (4 s), igual ao treino.
- **Alternativa:** 15 fps com janelas de 60 frames (também 4 s).
- Com janela deslizante, envie de novo a cada ~1 s, e não a cada frame.

Exemplo completo: `tests/fixtures/predict_sequence.json`.

Resposta `200`:

```json
{
  "predictions": [
    {"word": "Obrigado", "confidence": 0.91, "rank": 1},
    {"word": "Oi",       "confidence": 0.05, "rank": 2},
    {"word": "Sim",      "confidence": 0.02, "rank": 3}
  ],
  "top1_word": "Obrigado",
  "top1_confidence": 0.91,
  "is_confident": true,
  "frames_received": 30
}
```

`is_confident` vale `true` quando `top1_confidence >= 0.25`. A lista de
palavras que o modelo conhece vem de `GET /classes`.

Erros (sempre `{"error": "<mensagem>"}`):

| Status | Quando |
|---|---|
| 400 | JSON inválido, `frames` fora dos limites, ponto com shape/tipo errado (a mensagem diz qual frame e campo) |
| 413 | corpo maior que 2 MB |
| 422 | payload válido, mas nenhuma mão detectada em nenhum frame |
| 503 | modelo LSTM não carregado no servidor |

---

## `POST /predict_letter_landmarks`: reconhecimento de letras

Corpo, com um único frame da primeira mão detectada:

```json
{ "hand": [[0.42, 0.61], "... 21 pontos"], "target_letter": "A", "handedness": "right" }
```

`hand: null` indica que nenhuma mão foi detectada. A resposta é `200` com
`correct: false` e `predicted_letter: "Nenhuma mao detectada"`.

Resposta `200`:

```json
{
  "success": true,
  "correct": true,
  "predicted_letter": "A",
  "target_letter": "A",
  "box": [0.31, 0.42, 0.55, 0.80],
  "handedness": "right",
  "confidence": 0.97,
  "message": "Gesto Correto!"
}
```

- `box` é `[x_min, y_min, x_max, y_max]` da mão em **coordenadas
  normalizadas** (0–1), nas mesmas coordenadas enviadas pelo cliente. A
  antiga `/predict_letter` devolvia o box em pixels.
- **`handedness` (opcional): `"left"` ou `"right"`, a mão do sinalizador.**
  O modelo de letras atual aprendeu o formato da mão **esquerda** (confirmado
  no celular em 22/10/2026). Por isso o servidor espelha a mão direita
  (`x → 1 − x`) antes de classificar. Sem o campo, ele assume a direita.
  - Os clientes continuam enviando a imagem **não** espelhada.
  - No `HandLandmarker` do `@mediapipe/tasks-vision` 0.10.22, com frames não
    espelhados, `handedness[0][0].categoryName` já é o lado real
    (`"Right"` → `"right"`), medido com vídeos do dataset. A documentação do
    MediaPipe diz que o rótulo assume imagem espelhada, então confira se
    mudar de versão.
  - O modelo de **palavras** não precisa disso, porque recebe cada mão na sua
    posição (`left_hand` / `right_hand`).
- Confiança abaixo de 0,8 → `predicted_letter: "Movimento Invalido"`.

Erros (sempre `{"success": false, "message": "<mensagem>"}`): `400` para
payload inválido e `503` para modelo de letras não carregado.

---

## Mapeamento a partir do MediaPipe Tasks

O modelo foi treinado com o **Holistic legado** (Python `mediapipe==0.10.9`).
Os SDKs atuais usam o MediaPipe **Tasks**:

| Tasks | Campo do contrato |
|---|---|
| `HolisticLandmarker`: `poseLandmarks[0]` | `pose`, com `[x, y, z, visibility]` de cada landmark |
| `HolisticLandmarker`: `faceLandmarks[0]` (478 pontos) | `face` (pode enviar os 478) |
| `HolisticLandmarker`: `leftHandLandmarks[0]` / `rightHandLandmarks[0]` | `left_hand` / `right_hand` |
| `HandLandmarker`: `landmarks[0]` | `hand` (letras) |
| lista vazia (nada detectado) | `null` |

- **Web, Android e iOS.** Todos têm `HolisticLandmarker` no MediaPipe Tasks
  (`@mediapipe/tasks-vision`, `com.google.mediapipe:tasks-vision`, pod
  `MediaPipeTasksVision`), e cada landmark traz `visibility`. Prefira o
  Holistic.
- **Se for preciso combinar** `PoseLandmarker` + `FaceLandmarker` +
  `HandLandmarker`: o rótulo de lateralidade do `HandLandmarker` assume
  imagem espelhada, então **não confie nele**. Atribua cada mão ao punho mais
  próximo da pose (ponto 15 = punho esquerdo, 16 = punho direito).
- **`visibility`.** O `HolisticLandmarker` sempre devolve esse valor (0% de
  ausência na validação), e ele fica próximo do legado. Envie como vem.

## Formato do vetor no servidor

O servidor monta o vetor no formato que o modelo carregado espera, lido do
shape de entrada de `best_model.keras` e exposto em `GET /health` como
`lstm_keypoint_dim`:

| `lstm_keypoint_dim` | Formato |
|---|---|
| 1662 | legado: face completa (468), **sem** normalização espacial. É o modelo versionado hoje. |
| 438 | atual: face reduzida a lábios + sobrancelhas, **com** `normalize_spatial`. Vale após retreinar. |

Para o cliente o contrato é o mesmo nos dois casos.

## Validação e pendências

- [x] **Validação numérica (passo 2.2.2)**, em
      [`VALIDACAO_MEDIAPIPE_TASKS.md`](VALIDACAO_MEDIAPIPE_TASKS.md). O contrato
      está correto: a lateralidade é consistente e face e mãos ficam
      praticamente iguais ao legado. Mas o **modelo atual** perde acurácia com
      landmarks do Tasks (top-1 cai de 93,8% para 85,2%), principalmente por
      causa do z da pose.
- [ ] Retreinar o LSTM com landmarks extraídos pelo MediaPipe Tasks, no
      formato de 438 dimensões, avaliando uma versão sem o z da pose.
