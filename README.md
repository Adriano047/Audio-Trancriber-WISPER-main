# IA Local de Transcrição e Conversação por Voz

Projeto de uma IA local capaz de receber áudio, transcrevê-lo para texto, enviar o conteúdo para um modelo de linguagem local e transformar a resposta da IA novamente em áudio.

O projeto foi desenvolvido com foco em processamento local e na construção de uma conversa contextual por voz.

## Funcionalidades

- 🎙️ Gravação de áudio pelo microfone.
- 📝 Transcrição de áudio utilizando Whisper/Faster-Whisper.
- 🤖 Comunicação com um modelo de linguagem local através do Ollama.
- 🧠 Manutenção do histórico da conversa para preservar o contexto durante a sessão.
- 🔊 Conversão da resposta da IA em áudio utilizando XTTS v2.
- 💾 Salvamento das transcrições e respostas em arquivos.
- 📁 Processamento de arquivos de áudio já existentes.
- 🗣️ Possibilidade de utilizar a conversa por voz através do microfone.

## Fluxo do projeto

```text
                  ┌───────────────┐
                  │     Áudio     │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │    Whisper     │
                  │  Transcrição  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │    Histórico  │
                  │  da conversa  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │    Ollama     │
                  │   LLM local   │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │     XTTS      │
                  │  Texto → Voz  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │     Áudio     │
                  └───────────────┘
```

## Tecnologias

- **Python**
- **Faster-Whisper** — transcrição de áudio.
- **Ollama** — execução/comunicação com o modelo de linguagem local.
- **XTTS v2** — síntese de voz.
- **SoundDevice** — gravação do microfone.
- **SoundFile** — leitura e gravação de arquivos WAV.
- **python-dotenv** — carregamento das configurações através de variáveis de ambiente.

## Estrutura

```text
.
├── transcribe_file.py
├── transcribe_microphone.py
├── audio_reference/
├── audio_outputs/
├── transcriptions/
├── .env
└── README.md
```

### `transcribe_file.py`

Contém a lógica principal do processamento:

- carregamento do modelo Whisper;
- transcrição de arquivos;
- gerenciamento do histórico da conversa;
- comunicação com o Ollama;
- geração da resposta em áudio através do XTTS;
- salvamento dos resultados.

### `transcribe_microphone.py`

Responsável pelo fluxo de conversação utilizando o microfone:

```text
Microfone
   ↓
Gravação
   ↓
Whisper
   ↓
Ollama
   ↓
XTTS
   ↓
Resposta em áudio
```

Ele reutiliza as funções do `transcribe_file.py` em vez de duplicar a implementação.

## Configuração

As configurações do projeto são obtidas através de variáveis de ambiente.

Exemplo de `.env`:

```env
WHISPER_MODEL_PATH=small
OLLAMA_MODEL=nome_modelo
OLLAMA_BASE_URL=http://localhost:11434
TOKEN_KEY=seu_token
```

> Não versione o arquivo `.env` quando ele contiver credenciais ou informações privadas.

## Histórico da conversa

Durante a execução, o projeto mantém uma lista de mensagens seguindo a estrutura de conversação utilizada pelo modelo:

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]
```

Isso permite que a IA receba as mensagens anteriores e mantenha contexto.

A persistência de conversas encerradas e a possibilidade de continuar uma conversa posteriormente fazem parte de uma etapa futura da aplicação, especialmente relacionada à interface de usuário.

## Processamento de arquivos

O projeto também permite trabalhar diretamente com arquivos de áudio.

O fluxo é:

```text
Arquivo de áudio
      ↓
Faster-Whisper
      ↓
Transcrição
      ↓
Histórico / IA
      ↓
Resposta
```

## Conversação pelo microfone

O script de microfone permite transformar o projeto em uma conversa por voz.

De forma simplificada:

```text
Usuário fala
     ↓
Microfone
     ↓
Arquivo WAV temporário
     ↓
Whisper
     ↓
Texto
     ↓
Ollama
     ↓
Resposta da IA
     ↓
XTTS
     ↓
Áudio da resposta
```

O histórico é mantido entre as interações enquanto o programa permanece em execução.

## TTS e carregamento do modelo

O XTTS é um modelo pesado e não precisa ser carregado quando a funcionalidade de voz não será utilizada.

A estratégia recomendada é utilizar carregamento tardio (lazy loading):

```python
tts = None

def audio_response(...):
    global tts

    if tts is None:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    # geração do áudio
```

Dessa forma:

- o modelo não é carregado apenas por importar o módulo;
- o modelo é carregado somente quando a geração de áudio for necessária;
- depois de carregado, ele permanece em memória e pode ser reutilizado nas próximas respostas.

## Estado atual

O projeto atualmente possui os principais componentes necessários para uma IA local de conversação por voz:

- [x] Gravação pelo microfone
- [x] Transcrição com Whisper/Faster-Whisper
- [x] Comunicação com LLM local
- [x] Histórico contextual durante a sessão
- [x] Geração de voz com XTTS
- [x] Processamento de arquivos de áudio
- [ ] Persistência completa de conversas encerradas
- [ ] Interface gráfica da aplicação
- [ ] Seleção de dispositivos de entrada pelo parâmetro `--source`

As funcionalidades marcadas como pendentes dependem da evolução da aplicação e, principalmente, da futura interface de usuário.

## Objetivo

O objetivo do projeto é construir uma aplicação de IA local capaz de oferecer uma experiência de conversa por voz:

> **falar → transcrever → compreender o contexto → responder → falar**

mantendo o processamento principal local e permitindo posteriormente a integração com uma interface de IA.
