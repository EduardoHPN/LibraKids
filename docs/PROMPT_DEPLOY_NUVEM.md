# Preparar o LibraKids para rodar em container na nuvem e atender o app mobile

Você vai preparar o backend do LibraKids (Flask + TensorFlow + MediaPipe, em `src/libras/api/`) para rodar num container na nuvem e servir uma API consumida por um aplicativo de celular e pelo frontend Angular (`Frontend/`). O trabalho está dividido em 6 fases, que devem ser executadas **na ordem**, uma de cada vez.

## Regras gerais (valem para todas as fases)

- **Antes de começar cada fase**, leia os arquivos envolvidos e me mostre um plano curto do que vai mudar. Só implemente depois que eu aprovar.
- **Ao final de cada fase**: rode `pytest`, mostre o resultado, e faça **um commit** com mensagem em português no padrão do repositório (`feat: ...`, `fix: ...`, `chore: ...`, `refactor: ...`). Não faça `push`.
- Pare e me pergunte sempre que houver um **CHECKPOINT** abaixo, ou quando precisar de uma decisão que não está aqui.
- **Nunca** escreva credenciais reais (senha do Supabase, `SECRET_KEY`, connection string) em arquivos versionados nem me peça para colá-las no chat. Use `.env` (já ignorado pelo git? confira e adicione ao `.gitignore` se não estiver) e mantenha um `.env.example` com valores fictícios.
- Mantenha o estilo do código existente: docstrings e comentários em português, `from __future__ import annotations`, type hints, blueprints separados por domínio.
- Para cada mudança de comportamento, adicione ou ajuste testes em `tests/`. Use um `create_app(test_config=...)` com SQLite em memória nos testes, para não depender do Supabase.
- Versões fixadas que não devem ser alteradas sem me consultar: Python 3.10, `tensorflow==2.12.0`, `mediapipe==0.10.9`, `numpy==1.23.5`, `protobuf==3.20.3`.

## Contexto que já foi levantado (não precisa redescobrir)

- `src/libras/api/app.py` usa SQLite em `instance/site.db` e chama `db.create_all()`. O `SECRET_KEY` já lê `LIBRAS_SECRET_KEY`, mas cai num default inseguro (`"dev-only-change-me"`). `CORS(app, supports_credentials=True)` está liberado para qualquer origem.
- `src/libras/api/sign_routes.py` tem **estado global compartilhado entre todos os usuários**: um único `_holistic` (MediaPipe), um único `_predictor` (`StreamPredictor`, com buffer de 30 frames) e um único `_last_prediction`. Dois usuários simultâneos misturam frames no mesmo buffer. Esse estado também impede rodar mais de um worker ou réplica.
- `/predict` e `/predict_letter` recebem **um frame JPEG em base64 por requisição** e rodam o MediaPipe no servidor. No celular isso é lento e caro.
- `src/libras/utils/mediapipe_holistic.py`: `extract_keypoints(results)` monta o vetor `pose 33×4 + face (subset FACE_SUBSET_INDICES) 60×3 + mão esquerda 21×3 + mão direita 21×3` e aplica `normalize_spatial` (centraliza no meio dos ombros e escala pela largura dos ombros). `normalize_sequence(frames, target)` ajusta para `num_frames` (30, em `configs/default.yaml`). O módulo importa `cv2` e `mediapipe` no topo.
- `src/libras/api/letter_routes.py`: o modelo de letras usa os 21 landmarks da mão, só `x, y`, **sem normalização** (`flatten` → shape `(1, 42)`), com limiar de confiança 0.8. Ele espera `artifacts/models/letters/keras_landmarks_model.h5`, mas **`*.h5` está no `.gitignore`**, então esse arquivo não existe no repositório. `/record_data` grava imagens em disco (`data/raw_custom/letters`).
- `src/libras/api/auth.py`: o token é opaco (`secrets.token_hex(32)`), salvo em texto puro na coluna `User.token`. `/logout` **não invalida o token**. Nenhuma rota de predição verifica o token.
- `src/libras/config.py`: `PROJECT_ROOT = Path(__file__).resolve().parents[2]`, e `load_config()` usa o caminho relativo `"configs/default.yaml"`. **Se o pacote for instalado sem `-e` (em `site-packages`), os caminhos de modelo e config quebram.**
- `requirements.txt` está em **UTF-16** e mistura dependências de treino (matplotlib, seaborn, pandas, kagglehub, pytest) com as de runtime.
- Frontend (Angular 20): `Frontend/src/app/services/auth.service.ts` tem `baseUrl = 'http://localhost:5001/'` (com barra no fim, o que gera `//register`) e usa `withCredentials: true` (desnecessário, a auth não usa cookies). `Frontend/src/app/services/libras.service.ts` tem `apiUrl = 'http://localhost:5001'`. Não existe pasta `src/environments/`.
- `artifacts/models/test_data.npz` (13 MB) está versionado, mas não é necessário em produção.

---

## Fase 1: Migrar SQLite para PostgreSQL (Supabase)

1. Adicione `psycopg[binary]` (psycopg 3), `Flask-Migrate` e `python-dotenv` às dependências `api` do `pyproject.toml`.
2. Em `create_app`:
   - Leia a URL do banco de `DATABASE_URL`. Se ela começar com `postgres://`, converta para `postgresql+psycopg://`; se começar com `postgresql://`, troque o driver para `postgresql+psycopg://`.
   - Se `DATABASE_URL` não estiver definida, use o SQLite atual como fallback **só para desenvolvimento local**, com um aviso no log.
   - Configure `SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True, "pool_recycle": 300}`.
   - Aceite um parâmetro `test_config: dict | None = None` que sobrescreve a configuração (usado nos testes).
   - Carregue o `.env` com `python-dotenv` apenas quando o arquivo existir (em produção as variáveis vêm do provedor).
3. Troque `db.create_all()` por migrações com Flask-Migrate/Alembic. Gere a migração inicial a partir do model `User` atual (pasta `migrations/` versionada). Documente no README os comandos `flask --app run_backend db upgrade` e `flask --app run_backend db migrate -m "..."`.
4. Crie `scripts/migrate_sqlite_to_postgres.py`, que copia os usuários de `instance/site.db` para o banco apontado por `DATABASE_URL` sem duplicar (`username` é único). Os hashes bcrypt são copiados como estão.
5. Crie `.env.example` com `DATABASE_URL=postgresql://postgres.<ref>:<senha>@<host-do-pooler>:5432/postgres?sslmode=require`.
6. **Observações sobre o Supabase** (coloque também no README):
   - Use a connection string do **Session pooler** (porta 5432 no host `*.pooler.supabase.com`). A conexão direta (`db.<ref>.supabase.co`) é só IPv6 e costuma falhar em containers. O Transaction pooler (porta 6543) não suporta prepared statements, que o psycopg 3 usa.
   - Exija `sslmode=require`.
7. Testes: registro, login e `/me` funcionando com SQLite em memória via `test_config`.

**CHECKPOINT**: me peça para criar o projeto no Supabase e colocar a `DATABASE_URL` no meu `.env` local. Depois rode `db upgrade` contra o Supabase e o script de migração dos usuários, e confirme com um `GET /health` e um login.

---

## Fase 2: Rodar o MediaPipe no celular e deixar a API stateless

### 2.1 Trazer o aplicativo para a minha branch

O aplicativo mobile foi feito por outra pessoa e está em outra branch.

**CHECKPOINT**: rode `git fetch --all`, liste as branches remotas (existem, por exemplo, `origin/eduardo`, `origin/rodrigo-fronted`, `origin/rodrigo-backend`, `origin/develop`) e **me pergunte qual delas tem o aplicativo**. Não escolha sozinho.

Depois que eu responder:
1. Confirme que estou na branch `lucas` e que o working tree está limpo. Se houver arquivos não commitados ou não rastreados (por exemplo, testes novos em `tests/`), me mostre e pergunte se devo commitá-los antes.
2. Mostre um resumo do que vem na outra branch (`git log lucas..<branch> --oneline` e `git diff --stat lucas...<branch>`).
3. Faça `git merge <branch>` (sem rebase, sem squash). Se houver conflitos, **me mostre cada conflito e proponha a resolução antes de aplicar**. Não descarte o código de nenhum dos lados sem me consultar.
4. Depois do merge, descubra o stack do app (Flutter, React Native/Expo, Kotlin, Swift...), onde ele faz as chamadas HTTP para o backend e como captura a câmera. Me mostre esse resumo.

### 2.2 Definir o contrato de dados entre app e API

O modelo foi treinado com o **MediaPipe Holistic legado** (Python `mediapipe==0.10.9`), então o app precisa enviar landmarks no mesmo formato e na mesma topologia.

1. Escreva `docs/API_CONTRACT.md` com o formato exato do payload. Recomendação: o app envia **landmarks brutos** (sem normalizar) e o servidor faz a seleção do subset da face, o `normalize_spatial` e o `normalize_sequence`. Assim existe uma única fonte de verdade para o pré-processamento.

   Formato sugerido para `POST /predict_sequence`:
   ```json
   {
     "frames": [
       {
         "pose":       [[x, y, z, visibility], ...] | null,   // 33 pontos
         "face":       [[x, y, z], ...] | null,               // 468 pontos (malha completa)
         "left_hand":  [[x, y, z], ...] | null,               // 21 pontos
         "right_hand": [[x, y, z], ...] | null                // 21 pontos
       }
     ]
   }
   ```
   Para `POST /predict_letter_landmarks`: `{"hand": [[x, y], ...21], "target_letter": "A"}`.

   Documente também: coordenadas normalizadas de 0 a 1 pela largura e altura da imagem, a convenção de mão esquerda/direita do Holistic (atenção com câmera frontal espelhada) e os limites (mínimo e máximo de frames por requisição).

2. **Atenção à compatibilidade dos modelos no celular**: os SDKs mobile atuais usam o MediaPipe **Tasks** (Pose/Hand/Face Landmarker ou Holistic Landmarker, conforme a plataforma), e não o Holistic legado. As saídas são parecidas, mas não idênticas: a malha da face pode ter 478 pontos em vez de 468, e a pose tem `visibility` em campo separado. Documente o mapeamento para o formato acima e **valide numericamente**: pegue alguns vídeos de `data/` (ou grave alguns), extraia os landmarks pelo pipeline Python atual e pelo mesmo pipeline usado no app (se possível, pela versão Python do MediaPipe Tasks, com os mesmos modelos `.task`), e compare os vetores finais e a acurácia do LSTM. Me mostre a diferença antes de seguir.

### 2.3 Refatorar o backend

1. Separe as funções puras de `mediapipe_holistic.py` (`normalize_spatial`, `normalize_sequence`, constantes de dimensão, `FACE_SUBSET_INDICES`, `REGION_SLICES`) num módulo **sem import de `cv2` nem `mediapipe`**, por exemplo `src/libras/features/keypoints.py`. Crie lá uma função `keypoints_from_arrays(pose, face, left_hand, right_hand) -> np.ndarray` que gera exatamente o mesmo vetor que `extract_keypoints(results)`. Mantenha `extract_keypoints` reexportando ou delegando para essa função, para não quebrar os scripts de treino. Adicione um teste que compara as duas funções com o mesmo input.
2. Crie `POST /predict_sequence` **stateless**: recebe a sequência inteira, converte, aplica `normalize_sequence` para `num_frames` e chama `model.predict` uma vez. A resposta mantém os campos que o frontend já usa (`predictions` com `word`/`confidence`/`rank`, `top1_word`, `top1_confidence`, `is_confident`). Valide shape e tipos, e retorne 400 com mensagem clara para payload inválido.
3. Crie `POST /predict_letter_landmarks` **stateless**, com a mesma resposta de `/predict_letter` (o `box` pode ser omitido ou calculado a partir dos landmarks).
4. Carregue o modelo LSTM e o de letras **uma vez por processo**, em leitura, sem buffer global. O `StreamPredictor` continua existindo para o `scripts/05_run_webcam.py`, mas a API não deve depender do buffer dele.
5. Defina `MAX_CONTENT_LENGTH` no Flask (por exemplo, 2 MB).
6. As rotas antigas baseadas em imagem (`/predict`, `/clear`, `/predict_letter`) continuam com estado global.

   **CHECKPOINT**: me pergunte qual destas opções seguir:
   - (a) Migrar também o Angular para rodar o MediaPipe no navegador (`@mediapipe/tasks-vision`), usar os novos endpoints e **remover** as rotas antigas e o MediaPipe do servidor. Essa é a recomendada, porque deixa a imagem Docker bem menor (ver Fase 3).
   - (b) Manter as rotas antigas por enquanto, marcadas como deprecated, com o buffer isolado por usuário (dicionário indexado pelo id do usuário autenticado) e um `threading.Lock` em volta do MediaPipe. Nesse caso, documente que elas só funcionam com 1 worker e 1 réplica.
7. `/record_data` grava em disco, e o disco do container é efêmero. Ative essa rota só quando `LIBRAS_ENABLE_RECORD_DATA=1`; em produção ela fica desligada por padrão.

### 2.4 Ajustar o app mobile

1. Integre o MediaPipe no app usando o SDK oficial da plataforma descoberta no passo 2.1.
2. Acumule os frames no próprio app. Para palavras, mande a sequência quando tiver aproximadamente 30 frames, ou quando o usuário terminar o sinal. Para letras, mande um frame de landmarks da mão.
3. Mostre o progresso do buffer na UI localmente, sem chamadas ao servidor a cada frame.
4. Remova o envio de imagens em base64.

---

## Fase 3: Imagem Docker enxuta

1. **Dependências**:
   - Converta `requirements.txt` para **UTF-8 sem BOM**.
   - Separe em `requirements.txt` (runtime da API), `requirements-train.txt` (treino, notebooks, dataset) e `requirements-dev.txt` (pytest). Mantenha o `pyproject.toml` coerente (extras `api`, `train`, `dev`).
   - Fora do runtime ficam matplotlib, seaborn, pandas, kagglehub, tqdm e pytest.
   - Inclua no runtime: `gunicorn`, `psycopg[binary]`, `Flask-Migrate`. Use `tensorflow-cpu==2.12.0` em vez de `tensorflow`, porque não há GPU no container e o pacote é menor. Mantenha `scikit-learn`, porque o `label_encoder.pkl` é um `LabelEncoder` e o `joblib.load` precisa dele.
   - **OpenCV e MediaPipe**:
     - Se na Fase 2 escolhi a opção (a), o servidor **não precisa mais** de `mediapipe` nem de `opencv`. Remova os dois do runtime e confirme que nenhum módulo importado pela API os importa (por isso a separação do 2.3.1).
     - Se escolhi a opção (b), troque `opencv-python` por `opencv-python-headless`. Atenção: `mediapipe==0.10.9` depende de `opencv-contrib-python`, que puxa a versão com GUI. Instale `opencv-contrib-python-headless` na mesma versão e garanta que a versão não-headless não fique instalada (verifique com `pip list` dentro da imagem). Se não der, instale `libgl1` e `libglib2.0-0` via apt e documente o motivo.
2. **Caminhos**: faça `config.py` aceitar a variável de ambiente `LIBRAS_ROOT` para sobrescrever `PROJECT_ROOT`, e faça `load_config()` resolver o caminho padrão a partir de `PATHS["configs"]` em vez de um caminho relativo ao diretório atual. No Dockerfile, defina `LIBRAS_ROOT=/app`.
3. **Modelo de letras ausente** (`*.h5` ignorado pelo git).

   **CHECKPOINT**: me pergunte onde está o `keras_landmarks_model.h5`. Opções:
   - (a) Adicionar uma exceção no `.gitignore` (`!artifacts/models/letters/keras_landmarks_model.h5`) e versioná-lo, se for pequeno.
   - (b) Baixá-lo de um storage (por exemplo, Supabase Storage) no build ou no start do container.

   Avise também que a imagem precisa de `best_model.keras`, `metadata.json` e `letters/label_encoder.pkl`.
4. **`.dockerignore`**: exclua `.git`, `.venv`, `__pycache__`, `*.pyc`, `Frontend/`, a pasta do app mobile, `notebooks/`, `data/`, `docs/`, `tests/`, `scripts/` (exceto o que for necessário em runtime), `artifacts/logs/`, `artifacts/reports/`, `artifacts/models/test_data.npz`, `artifacts/models/olds/`, `instance/`, `.env` e `*.db`.
5. **Dockerfile** multi-stage, baseado em `python:3.10-slim`:
   - Stage de build: cria um venv em `/opt/venv` e instala `requirements.txt` com `--no-cache-dir`.
   - Stage final: copia o venv, `src/`, `configs/`, `artifacts/models/` (só os arquivos necessários), `migrations/` e `run_backend.py`. Instala o pacote com `pip install --no-deps .`, ou usa `PYTHONPATH=/app/src` (escolha um e justifique; o importante é que `LIBRAS_ROOT` aponte para `/app`).
   - Roda com um usuário não-root.
   - `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3`.
   - Usa a porta da variável `PORT` (Cloud Run e similares injetam essa variável), com default 8080.
   - `CMD` com gunicorn: `gunicorn --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY:-1} --threads 4 --timeout 120 run_backend:app` (em formato shell ou via script de entrypoint, para expandir as variáveis).
   - Se usar script de entrypoint `.sh`, crie um `.gitattributes` com `*.sh text eol=lf`, porque estou no Windows e CRLF quebra o script no Linux.
   - Inclua um `HEALTHCHECK` chamando `/health` (com `python -c "import urllib.request..."`, para não precisar instalar curl).
   - Rode as migrações (`flask db upgrade`) num passo separado documentado, não automaticamente a cada start. Se preferir automatizar, me consulte antes.
6. Remova o `print` de "rodando em localhost:5001" do caminho do gunicorn: mantenha `app.run` só no `if __name__ == "__main__"` (já é assim, apenas confirme).
7. **Validação** (rode e me mostre a saída):
   - `docker build -t librakids-api .` e `docker images librakids-api` (meta: bem abaixo de 2 GB; com a opção (a) da Fase 2 deve ficar perto de 1 GB).
   - `docker run --rm -p 8080:8080 --env-file .env librakids-api`
   - `curl http://localhost:8080/health`, que deve mostrar os dois modelos carregados.
   - Uma chamada a `/predict_sequence` com um payload de exemplo salvo em `tests/fixtures/`.

---

## Fase 4: Exigir token de login nas rotas de predição

1. Em `auth.py`, crie um decorator `login_required` que lê `Authorization: Bearer <token>`, busca o usuário, coloca em `flask.g.current_user` e responde `401 {"message": "Nao autenticado."}` se o token estiver ausente ou for inválido. Reaproveite essa lógica no `/me`.
2. Aplique o decorator em `/predict_sequence`, `/predict_letter_landmarks`, `/record_data` e, se ainda existirem, `/predict`, `/clear` e `/predict_letter`. Deixe públicas: `/health`, `/classes`, `/register` e `/login`.
3. Corrija o `/logout` para **invalidar o token** (`user.token = None`).
4. Armazene no banco apenas o **hash SHA-256 do token**, não o token em texto puro. O cliente continua recebendo o token original. Crie uma migração Alembic para isso. Tokens antigos deixam de valer, o que é aceitável: os usuários fazem login de novo.
5. Adicione expiração do token (coluna `token_expires_at`, validade configurável via `LIBRAS_TOKEN_TTL_HOURS`, default 168h = 7 dias).
6. **Angular**: crie um `HttpInterceptor` funcional (Angular 20, `provideHttpClient(withInterceptors([...]))` em `app.config.ts`) que adiciona o header `Authorization` a partir do token salvo pelo `AuthService`. Quando receber 401, ele limpa o token e redireciona para o login. Remova os `withCredentials: true`.
7. **App mobile**: envie o mesmo header em todas as chamadas autenticadas e trate o 401 do mesmo jeito.
8. Testes: cada rota protegida retorna 401 sem token, 401 com token inválido, 401 com token expirado, e sucesso com token válido. O logout invalida o token.

---

## Fase 5: Configuração por variáveis de ambiente (`LIBRAS_SECRET_KEY` e afins)

1. Centralize a leitura de variáveis de ambiente da API num único lugar (por exemplo, `src/libras/api/settings.py`), com valores default seguros para desenvolvimento.
2. Defina `LIBRAS_ENV` (`development` | `production`, default `development`). Em `production`, a aplicação deve **falhar no start** com mensagem clara se `LIBRAS_SECRET_KEY` ou `DATABASE_URL` não estiverem definidas. Só em `development` pode usar o default `dev-only-change-me` e o fallback SQLite.
3. Troque o `CORS(app, supports_credentials=True)` aberto por `CORS(app, origins=<lista>)`, lendo de `CORS_ORIGINS` (separadas por vírgula). Em desenvolvimento, o default é `http://localhost:4200`. Remova o `supports_credentials`, porque a auth é por header e não por cookie. O app mobile nativo não é afetado por CORS.
4. Atualize o `.env.example` com todas as variáveis: `LIBRAS_ENV`, `LIBRAS_SECRET_KEY`, `DATABASE_URL`, `CORS_ORIGINS`, `LIBRAS_TOKEN_TTL_HOURS`, `LIBRAS_ENABLE_RECORD_DATA`, `PORT`, `WEB_CONCURRENCY`. Inclua um comentário de como gerar a chave: `python -c "import secrets; print(secrets.token_hex(32))"`.
5. Documente no README a tabela de variáveis (nome, obrigatória em produção?, default, descrição).
6. Testes: `create_app` em modo `production` sem `LIBRAS_SECRET_KEY` levanta erro; com as variáveis definidas, sobe normalmente.

---

## Fase 6: URL da API configurável (Angular e app) e HTTPS em produção

1. **Angular**:
   - Rode `ng generate environments` dentro de `Frontend/` para criar `src/environments/environment.ts` (produção) e `environment.development.ts`, com o `fileReplacements` no `angular.json`.
   - Defina `apiUrl`: em desenvolvimento, `http://localhost:5001`; em produção, um placeholder `https://SEU-SERVICO.run.app` que eu vou trocar.
   - Troque as URLs fixas em `auth.service.ts` e `libras.service.ts` por `environment.apiUrl` e corrija a barra dupla (`'http://localhost:5001/'` + `'/register'`). O ideal é criar uma constante ou um `InjectionToken` `API_URL` usado pelos dois serviços.
   - Rode `ng build` e confirme que o bundle de produção usa a URL HTTPS.
2. **App mobile**: use o mecanismo padrão do stack descoberto na Fase 2, por exemplo:
   - Flutter: `--dart-define=API_URL=...` lido com `String.fromEnvironment`.
   - Expo/React Native: `EXPO_PUBLIC_API_URL` ou `react-native-config`, com `.env` por ambiente.
   - Android nativo: `buildConfigField` por `buildType`/flavor.

   Não deixe `localhost` fixo no código.
3. **HTTPS e desenvolvimento local no celular**:
   - Produção usa **apenas HTTPS** (o provedor de nuvem fornece o certificado).
   - Para testar no emulador Android contra o backend local, use `http://10.0.2.2:5001`. Permita cleartext **só no build de debug** (Android: `network_security_config` em `src/debug/`; iOS: exceção de ATS só na configuração Debug). **Nunca** habilite cleartext ou `NSAllowsArbitraryLoads` no build de release. Verifique e me mostre essas configurações.
4. Documente no README como apontar o Angular e o app para o backend local e para o de produção.

---

## Fechamento

Depois das 6 fases:
1. Rode `pytest`, `ng build` no `Frontend/` e o build do app mobile (ou o mais próximo possível de um build, como `flutter analyze` ou `npx tsc --noEmit`). Rode também o `docker build` + `docker run` + chamadas de teste da Fase 3, agora com token. Me mostre os resultados.
2. Me entregue um resumo: commits criados, variáveis de ambiente necessárias, passos manuais que ficaram comigo (criar o projeto no Supabase, subir o `.h5`, escolher o provedor de nuvem) e qualquer pendência ou risco encontrado, em especial a diferença numérica entre o MediaPipe do celular e o do treino.
3. **Não faça deploy nem push.** O deploy na nuvem (Cloud Run, Azure Container Apps ou similar, com `min-instances=1` para evitar a espera do carregamento do TensorFlow) fica para um próximo passo.
