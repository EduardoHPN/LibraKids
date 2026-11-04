# Progresso: LibraKids na nuvem + app mobile

Estado da execução do roteiro [`PROMPT_DEPLOY_NUVEM.md`](PROMPT_DEPLOY_NUVEM.md), com as decisões já tomadas e o motivo de cada uma. Atualize este arquivo ao concluir cada fase. Ele é a fonte de verdade para retomar o trabalho em qualquer computador ou sessão do Claude Code.

**Última atualização:** 2026-11-04, com as Fases 1 a 4 concluídas, faltando o teste da Fase 4 no celular. A próxima é a Fase 5 (variáveis de ambiente e CORS).

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
2. **Backend (LibraKids).** Use Python 3.11: o SQLAlchemy 2.1.5 exige 3.11 ou mais novo.
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e .[api,dev]     # só a API e os testes
   pip install -e .[train,dev]   # também treino, notebooks e dataset (MediaPipe, OpenCV etc.)
   ```
   Use um venv novo: um `.venv` anterior à Fase 3 tem o pacote `tensorflow`, e instalar estas dependências nele colocaria o `tensorflow-cpu` junto, sendo que os dois instalam o mesmo módulo.

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
- O banco do Supabase já tem as migrações aplicadas, com as tabelas `user`, `user_session` e `alembic_version`.
- A **Data API (PostgREST) está desligada** no Supabase de propósito: o acesso ao banco é só pelo backend. Com ela ligada, as tabelas `user` (hashes de senha) e `user_session` (hashes de token) ficariam expostas pela chave pública do projeto.
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

### Fase 3: imagem Docker. Concluída

Como construir e rodar a imagem está no [README](../README.md#imagem-docker-da-api).

**Decisões que substituem o roteiro**
- **`python:3.11-slim`, não 3.10.** O SQLAlchemy 2.1.5 exige 3.11 ou mais novo, e o `.venv` testado já usa 3.11.
- **Sem `LIBRAS_ROOT`.** A imagem usa `PYTHONPATH=/app/src`, sem instalar o pacote: o `PROJECT_ROOT` (`parents[2]`) já resolve para `/app`.
- **`configs/` fica fora da imagem.** A API não usa `load_config`.
- **`.dockerignore` por lista de liberação,** em vez de lista de exclusão: um arquivo novo na raiz, como o `.env`, fica fora por padrão.
- **CMD em formato shell com `exec`,** sem script de entrypoint. Por isso não há `.gitattributes`.

**Dependências:** `requirements.txt` (runtime, UTF-8 sem BOM, com `tensorflow-cpu` e sem `mediapipe`/`opencv`), `requirements-train.txt` e `requirements-dev.txt`, espelhados nos extras `api`, `train` e `dev` do `pyproject.toml`. O `label_encoder.pkl` foi regravado com scikit-learn 1.3.2, mantendo as 21 classes na mesma ordem.

**Validação em 2026-11-04** (Docker 29.9.2, com 3,9 GB de RAM no WSL 2):

| Verificação | Resultado |
|---|---|
| `docker build` sem cache | Passou, em cerca de 9 min |
| `GET /health` | 200, `lstm_loaded` e `letter_loaded` verdadeiros, `lstm_keypoint_dim` 405; responde de 8 a 14 s depois do `docker run` |
| `POST /predict_sequence` com `tests/fixtures/predict_sequence.json` | 200, "Sim" com confiança 0,83, entre 0,33 e 0,76 s por chamada |
| `POST /predict_letter_landmarks` com uma mão sintética | 200, com a resposta completa. Como a mão não é real, o resultado foi "Movimento Invalido" (confiança 0,75, abaixo do limiar de 0,8) |
| Memória com 1 worker | Cerca de 320 MiB depois de 3 predições |
| `HEALTHCHECK`, com a porta padrão e com `PORT=9000` | `healthy` nos dois casos |
| Conteúdo da imagem | Usuário não-root, `/app` só leitura; sem `.env`, `configs/`, `test_data.npz`, `weights_raw.npz`, `mediapipe`, `opencv` e pacote `tensorflow` |
| Sem `DATABASE_URL` | Não sobe (`PermissionError` em `/app/instance`), como esperado |
| `LIBRAS_ENABLE_RECORD_DATA=1` | Não sobe (`No module named 'cv2'`), como esperado |
| `flask db upgrade` na imagem | Passou com SQLite temporário |
| `flask db current` no Supabase, pela imagem | Conectou, na revisão `4dc8d8585ded (head)`. Antes de restaurar o projeto pausado, o pooler respondia `tenant/user not found` |

**Tamanho da imagem:** o `docker images` mostra 2,16 GB, mas esse número soma as camadas compactadas e descompactadas. São **474 MB compactados** (o que vai para o registry) e cerca de **1,7 GB descompactados**. A meta do roteiro ("bem abaixo de 2 GB") foi cumprida só em parte: fica abaixo de 2 GB, mas não "bem abaixo". Os maiores pacotes são `tensorflow` (885 MB) e `scipy` (119 MB).

**`jax` e `jaxlib` foram removidos da imagem** no estágio de build, o que economiza 306 MB. O TensorFlow os declara como dependência, mas a API não precisa deles: o import vem de `tensorflow/lite/python/util.py` e é opcional. Com a remoção, a predição dá o mesmo resultado e a suíte, rodada num container sem os dois, termina com 84 testes aprovados e 9 pulados. Efeitos colaterais:
- O `pip check` na imagem acusa `tensorflow-cpu 2.12.0 requires jax`. É esperado.
- Se um dia a imagem precisar converter modelos para TFLite ou mudar a versão do TensorFlow, reveja esse corte.

**Pendências**
- O `create_app` imprime usuário e host do banco (sem a senha) no log, que na nuvem fica guardado. Vale rever na Fase 5.
- O `/health` responde 200 mesmo se um modelo falhar ao carregar, e o `HEALTHCHECK` só confere essa resposta. Para saber se os modelos carregaram, leia o JSON.

### Fase 4: token nas rotas de predição. Concluída, falta o teste no celular

O comportamento das rotas está em [`API_CONTRACT.md`](API_CONTRACT.md#autenticação).

**Decisões que substituem o roteiro**
- **Tabela `user_session` em vez de colunas em `user`:** cada login tem um token próprio (`token_hash` único). Com um token por usuário, entrar no site derrubaria o app, e vice-versa. As sessões vencidas do usuário são apagadas no login dele, sem tarefa agendada.
- **Rotas protegidas:** `/predict_sequence`, `/predict_letter_landmarks`, `/record_data`, `/me` e `/logout`. As rotas de imagem citadas no roteiro já não existem.
- **Validade fixa:** `LIBRAS_TOKEN_TTL_HOURS` (padrão de 168 h) conta a partir do login, sem renovação por uso. O usuário entra de novo uma vez por semana.

**Clientes**
- **Angular:** interceptor funcional que envia o token só às rotas da API, sem `/login` e `/register`, para o 401 de senha errada continuar aparecendo no formulário. O `logout()` agora chama `POST /logout`. A URL ficou em `services/api.config.ts`, ainda fixa em `localhost:5001` até a Fase 6.
- **App:** o `apiRequest` avisa a sessão quando o servidor recusa um token enviado, e o `Stack.Protected` volta ao login. Commit `aa7eb8a` no `mobile-libras`, sem push.
- Nos dois, uma resposta atrasada do token antigo não derruba um login mais novo, e 500 ou falta de rede não encerram a sessão.

**Migração `7c1e9a4b2f60`, aplicada no Supabase em 2026-11-04:** cria `user_session` e remove `user.token`. Os tokens antigos deixaram de valer. A impressão digital de ids, nomes e hashes de senha ficou igual antes e depois.

**Validação em 2026-11-04**

| Verificação | Resultado |
|---|---|
| `pytest` (cada rota protegida sem token, com token inválido, expirado, depois do logout e válido) | 191 aprovados, nenhum pulado |
| Migração no Postgres 17 descartável (ida e volta, `CASCADE`, schema igual aos models) | Passou |
| `ng build` | Passou, com aviso de bundle inicial acima do limite (948 kB para 500 kB) |
| `ng test` (spec do interceptor) | 16 aprovados |
| App: `tsc --noEmit` | Sem erros nos arquivos alterados. Há 4 erros de `ColorValue` em `src/app/(tabs)/_layout.tsx`, arquivo que esta fase não alterou |
| Imagem contra o Supabase: `/register`, `/login`, as duas predições, `/logout` e 401 depois | Todas as chamadas como esperado; palavras com "Sim", confiança 0,83. Usuário de teste removido |
| Memória com 1 worker | 355 MiB depois de 40 predições de letras e 11 de palavras |

**Latência: as predições agora consultam o banco.** Medianas com o container nesta máquina:

| Banco | Letras | Palavras |
|---|---|---|
| Postgres no mesmo Docker | 19 ms | 382 ms |
| Supabase (129 ms de ida e volta) | 597 ms | 965 ms |

Validar o token custa cerca de 4 idas e voltas ao banco (565 ms aqui), então o custo acompanha a distância: **no deploy, coloque a API na mesma região do Supabase**. Enquanto a API roda longe do banco, as letras saem a cada 600 ms ou mais, em vez de 500 ms.

**Pendências**
- **Testar no celular** com o container na porta 8080, incluindo letras com as duas mãos ([roteiro](#testar-o-app-no-celular)). A volta ao login no app só foi testada no cliente HTTP, com `fetch` simulado.
- Com o banco fora do ar ou pausado, as predições respondem 500. A sessão continua, mas o reconhecimento para.
- O `mobile-libras` não tem ESLint, e o `npm run lint` (`expo lint`) tenta instalá-lo e altera `package.json` e `package-lock.json`. Desfaça essa alteração se ela não for intencional.

### Fases 5 e 6. Não iniciadas

## Próximos passos

1. **Testar a Fase 4 no celular.**
2. **Fases 5 e 6:** variáveis de ambiente, CORS e URLs configuráveis com HTTPS. O app já lê a URL de `EXPO_PUBLIC_API_URL` (`.env.development` e `.env.production`).
3. **Telas restantes do app** (Início, Histórico, Perfil etc.). As que guardam dados, como histórico e progresso, precisam de endpoints e tabelas novas no Supabase. Trate-as como uma fase própria.
4. **Deploy** na nuvem, fora do roteiro atual: Cloud Run ou Azure Container Apps, com `min-instances=1`, na mesma região do Supabase.
5. Quando a parte do app estiver pronta, abra um **pull request no Bitbucket** da `lucas/mediapipe-on-device` para a `master`, para revisão da dona do repositório.
6. Melhoria de modelo: gravar **outros sinalizadores**. É o maior ganho de acurácia esperado.

## Testar o app no celular

- Use o **Expo Go do SDK 57**. Se a Play Store oferecer uma versão mais nova, instale o APK do SDK 57 em `expo.dev/go` e desligue a atualização automática.
- O celular e o PC precisam estar na **mesma rede**, marcada como **Privada** no Windows, com o firewall liberando a porta **5001** (API) e, se o Expo não conectar, a **8081** (Metro), só no perfil Privado.
- **Não marque redes de trabalho ou públicas como Privadas.** Nesses lugares, use o **hotspot do celular**.
- O IP do PC muda conforme a rede. Antes do `npx expo start`, atualize o `.env.development.local` do app e defina `REACT_NATIVE_PACKAGER_HOSTNAME=<IP>`.
- Se aparecer "Failed to download remote update" no primeiro carregamento, é timeout enquanto o Metro monta o bundle. Um Reload resolve.
- **Com o container:** rode `docker run --rm -p 8080:8080 --env-file .env librakids-api`, use a porta **8080** na `EXPO_PUBLIC_API_URL` e libere essa porta no firewall, só no perfil Privado.
- **Para testar o 401:** com o app aberto, apague as sessões da conta no SQL Editor do Supabase (`DELETE FROM user_session WHERE user_id = (SELECT id FROM "user" WHERE username = '<usuario>');`) e faça uma letra. O app deve voltar ao login.
