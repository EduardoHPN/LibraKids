# Progresso: LibraKids na nuvem + app mobile

Estado da execução do roteiro [`PROMPT_DEPLOY_NUVEM.md`](PROMPT_DEPLOY_NUVEM.md), com as decisões já tomadas e o motivo de cada uma. Atualize este arquivo ao concluir cada fase. Ele é a fonte de verdade para retomar o trabalho em qualquer computador ou sessão do Claude Code.

**Última atualização:** 2026-10-22, com as Fases 1 e 2 concluídas. A próxima é a Fase 3 (imagem Docker).

## Repositórios

| Repositório | Onde | Branch de trabalho | Conteúdo |
|---|---|---|---|
| **LibraKids** | GitHub | `lucas` | API Flask (`src/libras/api/`), modelos, treino, frontend Angular (`Frontend/`) |
| **mobile-libras** | Bitbucket (`new-bank/mobile-libras`) | `lucas/mediapipe-on-device` | App Expo / React Native. O repositório é de outra pessoa: as mudanças vão por pull request para a `master` dela |

Os dois ficam em pastas irmãs, por exemplo `C:\Users\<voce>\LibraKids` e `C:\Users\<voce>\mobile-libras`.

## Como retomar em outro computador

1. **Clonar os dois repositórios** nas branches de trabalho:
   ```bash
   git clone -b lucas <url-github-librakids> LibraKids
   git clone -b lucas/mediapipe-on-device <url-bitbucket-mobile-libras> mobile-libras
   ```
2. **Backend (LibraKids).** Use Python 3.10 ou 3.11; o 3.11 foi testado e instala os pins do TensorFlow 2.12 e do MediaPipe 0.10.9.
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .[api,dev]
   ```
   Crie o `.env` a partir do `.env.example`, com a `DATABASE_URL` do **Session pooler** do Supabase. A senha não está em lugar nenhum do repositório: pegue-a no painel do Supabase. O banco já tem as migrações aplicadas, então rodar `flask --app run_backend db upgrade` de novo não altera nada.

   Para conferir, suba a API com `python run_backend.py` e abra `GET http://localhost:5001/health`. Ele deve mostrar `lstm_loaded: true`, `lstm_keypoint_dim: 405` e `letter_loaded: true`.
3. **App (mobile-libras).**
   ```bash
   npm install
   ```
   Crie o `.env.development.local` (não versionado) com `EXPO_PUBLIC_API_URL=http://<IP-do-PC>:5001`. Depois rode `npx expo start` e abra no Expo Go (veja [Testar o app no celular](#testar-o-app-no-celular)).
4. **Claude Code.** A memória e o histórico de conversas ficam só na máquina em que foram criados. Comece a sessão com:
   > Leia `docs/PROGRESSO.md` e `docs/PROMPT_DEPLOY_NUVEM.md` e continue de onde parou. O app está em `../mobile-libras`.

   Abra com `claude --add-dir ..\mobile-libras` para ele ter acesso aos dois repositórios.

Dados que **não** estão no git e só fazem falta para retreinar: os vídeos do dataset (338 vídeos, 11 classes), `data/landmarks/` (cache de landmarks) e `data/processed_custom_tasks*/`. Veja [`RETREINO_TASKS.md`](RETREINO_TASKS.md).

## Estado por fase

### Fase 1: PostgreSQL (Supabase). Concluída

- A API usa `DATABASE_URL` (psycopg 3), com o SQLite como fallback só em desenvolvimento. As migrações usam Flask-Migrate. Commit `831552d49`.
- O banco do Supabase já tem as migrações aplicadas, com a tabela `user` e a `alembic_version`.
- A **Data API (PostgREST) está desligada** no Supabase de propósito: o acesso ao banco é só pelo backend. Com ela ligada, a tabela `user` (hashes de senha e tokens) ficaria exposta pela chave pública do projeto.
- No plano Free, o Supabase pausa o projeto depois de cerca de 1 semana sem uso. Se a API der erro de conexão, use **Restore project** no painel.

### Fase 2: MediaPipe no cliente e API stateless. Concluída

**Backend (LibraKids)**
- Contrato de dados em [`API_CONTRACT.md`](API_CONTRACT.md): os clientes enviam **landmarks brutos**, e o servidor faz a seleção da face, a normalização e o ajuste para 30 frames. Assim existe uma única fonte de verdade para o pré-processamento.
- Rotas stateless `POST /predict_sequence` (palavras) e `POST /predict_letter_landmarks` (letras), sem estado global. Commit `d5d616c8f`.
- O servidor aceita os formatos 1662, 438 e 405 e escolhe pelo shape do modelo instalado.
- **Opção (a) do checkpoint 2.3.6:** o Angular também roda o MediaPipe no navegador (`@mediapipe/tasks-vision`). As rotas antigas, baseadas em imagem, e o MediaPipe/OpenCV saíram do servidor. Commit `c642e778b`.

**Modelo de palavras (retreino)**
- Validação numérica em [`VALIDACAO_MEDIAPIPE_TASKS.md`](VALIDACAO_MEDIAPIPE_TASKS.md): o modelo antigo, treinado com o Holistic legado, perdia acurácia com landmarks do MediaPipe Tasks, e o principal motivo era o z da pose.
- Retreinado no Colab com landmarks do Tasks. A variante escolhida foi a **sem o z da pose (405 dimensões)**:

  | Variante | Top-1 | Top-3 |
  |---|---|---|
  | Antigo (1662, Holistic legado) | 83,8% | 97,1% |
  | Tasks com z (438) | 86,8% | 92,6% |
  | **Tasks sem z (405), em uso** | **89,7%** | **97,1%** |

  Ela foi a melhor nas duas métricas e depende menos da versão do MediaPipe em cada cliente. Commit `dbaef6bc3`.
- Fluxo do Colab: o notebook é autocontido e usa TF 2.21 / Keras 3. Ele exporta um `weights_raw.npz`, e `scripts/legacy/load_weights_npz.py` reconstrói o modelo localmente no Keras 2.12 (lê `num_frames` e `keypoint_dim` do `metadata.json`). Depois, o `scripts/04_evaluate.py` deve reproduzir a acurácia do Colab.
- Ressalvas: todos os vídeos são de **um único sinalizador**, e o teste tem só 68 vídeos (cada erro vale cerca de 1,5 ponto).

**Modelo de letras**
- `artifacts/models/letters/keras_landmarks_model.h5` (580 KB) agora está versionado, com uma exceção no `.gitignore`. Commit `5cca35d87`. Isso resolve o checkpoint 3.3 do roteiro.
- Reconhece **21 letras estáticas**. Faltam H, J, K, X e Z, que envolvem movimento.
- O modelo foi treinado com o MediaPipe Hands legado (x e y crus) e **aprendeu a mão esquerda**. Os clientes enviam `handedness`, e o servidor espelha só a mão direita. O HandLandmarker do Tasks 0.10.22, com frames não espelhados, rotula o lado real da mão, ao contrário do que diz a documentação do MediaPipe. Commits: `f2f540831` (servidor) e `3b2df3048` (Angular).

**App (mobile-libras)**
- Integração com a API: cliente HTTP, login, cadastro, logout e `/me`, com o token no `expo-secure-store`. Commit `13d83a2`.
- MediaPipe no aparelho via **WebView com `@mediapipe/tasks-vision` 0.10.22-rc**, o mesmo runtime do Angular. Não exige código nativo e roda no Expo Go. Commits: `59c29db` (alfabeto), `3711c06` (palavras, captura de 4 s a 7,5 amostras/s, o mesmo ritmo do treino) e `0548ae0` (`handedness` nas letras).
- **Testado no celular (Android, Expo Go) em 2026-10-22:** login, alfabeto e palavras funcionaram.

### Fases 3 a 6. Não iniciadas

Observações já levantadas para a Fase 3:
- O servidor não precisa de `mediapipe` nem de `opencv` (opção (a)). Use `tensorflow-cpu==2.12.0`.
- O `label_encoder.pkl` das letras foi salvo com scikit-learn 1.7.1, mas o projeto usa 1.3.2. Isso gera um `InconsistentVersionWarning` ao carregar. Avalie regravá-lo ou alinhar a versão.
- No Windows 11 Home, o Docker Desktop precisa do WSL 2.

## Próximos passos

1. **Fase 3:** imagem Docker enxuta (veja o roteiro).
2. **Fases 4 a 6:** token nas rotas de predição, variáveis de ambiente e URLs configuráveis com HTTPS. O app já lê a URL de `EXPO_PUBLIC_API_URL` (`.env.development` e `.env.production`).
3. **Telas restantes do app** (Início, Histórico, Perfil etc.). As que guardam dados, como histórico e progresso, precisam de endpoints e tabelas novas no Supabase. Trate-as como uma fase própria.
4. **Deploy** na nuvem, fora do roteiro atual: Cloud Run ou Azure Container Apps, com `min-instances=1`.
5. Quando a parte do app estiver pronta, abra um **pull request no Bitbucket** da `lucas/mediapipe-on-device` para a `master`, para revisão da dona do repositório.
6. Melhoria de modelo: gravar **outros sinalizadores**. É o maior ganho de acurácia esperado.

## Testar o app no celular

- Use o **Expo Go do SDK 57**. Se a Play Store oferecer uma versão mais nova, instale o APK do SDK 57 em `expo.dev/go` e desligue a atualização automática.
- O celular e o PC precisam estar na **mesma rede**, marcada como **Privada** no Windows, com o firewall liberando a porta **5001** (API) e, se o Expo não conectar, a **8081** (Metro), só no perfil Privado.
- **Não marque redes de trabalho ou públicas como Privadas.** Nesses lugares, use o **hotspot do celular**.
- O IP do PC muda conforme a rede. Antes do `npx expo start`, atualize o `.env.development.local` do app e defina `REACT_NATIVE_PACKAGER_HOSTNAME=<IP>`.
- Se aparecer "Failed to download remote update" no primeiro carregamento, é timeout enquanto o Metro monta o bundle. Um Reload resolve.
