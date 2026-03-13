# 🎤 Transcrição de Voz com IA (Português Brasileiro)

Este projeto grava áudio do microfone, transcreve automaticamente utilizando **Whisper (faster-whisper)** e pode gerar respostas de **IA local usando Ollama**.  
Opcionalmente, a resposta da IA também pode ser reproduzida em **áudio (TTS)**.

O sistema funciona como um **chat de voz simples local**, permitindo conversar com um modelo de linguagem usando apenas voz.

---

# 🚀 Funcionalidades

- 🎤 Grava áudio diretamente do microfone
- 📝 Transcrição automática em **Português (pt-BR)**
- 🤖 Resposta automática usando **modelo local via Ollama**
- 🔊 Conversão da resposta em **voz**
- 💾 Salvamento da conversa em:
  - `.txt`
  - `.json`
- 🔁 Conversação contínua (loop de gravação)

---

# 🧠 Tecnologias Utilizadas

- **faster-whisper** – transcrição de áudio
- **Ollama** – modelo de linguagem local
- **gTTS** – geração de voz
- **sounddevice** – captura de áudio
- **soundfile** – manipulação de áudio
- **numpy** – processamento de áudio

---

# 📦 Instalação

Clone o projeto:

```bash
git clone https://github.com/Adriano047/Audio-Trancriber-WISPER-main.git
cd seu-projeto
```
```bash
# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate.fish  # ou: source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```


## Uso

### 1. Transcrever arquivo de áudio

Transcreve arquivos de áudio (mp3, ogg, wav, etc.) para texto:

```bash
# Transcrever um arquivo
python transcribe_file.py audio.mp3

# Transcrição com resposta da IA:
python transcribe_file.py audio.mp3 --ia-response

# Transcrição com resposta da IA em formato de voz:
python transcribe_file.py audio.mp3 --ia-response --chat-voz 

# Especificar arquivo de saída
python transcribe_file.py audio.ogg --output resultado.txt --json resultado.json
```

**Saídas:**
- `transcription.txt` - Texto com quebras de linha a cada 10 palavras
- `transcription.json` - Metadados (idioma, probabilidade, texto)

### 2. Gravar do microfone e transcrever

Grava áudio do microfone por um tempo definido e transcreve:

```bash

# Gravar 10 segundos e transcrever
python transcribe_microphone.py --seconds 10

# Gravar 10s, transcrever e o modelo de Ia responder(texto):
python transcribe_microphone.py --seconds 10 --ia-response
    
# Gravar 10s, transcrever e o modelo de Ia responder(texto e audio):
python transcribe_microphone.py --seconds 10 --ia-response --chat-voz

# Apenas gravar (teste de áudio)
python transcribe_microphone.py --seconds 5 --test-only
```

**Saídas:**
- `captura.wav` - Arquivo de áudio gravado
- `transcription.txt` - Transcrição formatada
- `transcription.json` - Metadados completos

## Configuração

### Usar modelo Whisper já baixado

Para evitar downloads repetidos do modelo:

# ⚙️ Modelo Whisper

Por padrão o projeto usa o modelo:

small

O modelo será baixado automaticamente na primeira execução.

Se desejar utilizar um modelo local já baixado, é possível definir o caminho usando a variável de ambiente:

### Windows

set WHISPER_MODEL_PATH=C:\caminho\para\modelo

### Linux / Mac

export WHISPER_MODEL_PATH=/caminho/para/modelo

### Modelos disponíveis

- `tiny` - Mais rápido, menos preciso (~1GB)
- `base` - Bom equilíbrio (~1.5GB)
- `small` - **Padrão** - Boa precisão (~2.5GB)
- `medium` - Muito preciso, mais lento (~5GB)
- `large` - Máxima precisão, muito lento (~10GB)

3. **Testar captura sem transcrever:**
```bash
python transcribe_microphone.py --seconds 5 --test-only
# Verifique captura.wav em um player de áudio
```

### Erros de CUDA/GPU

Se aparecerem erros relacionados a CUDA/cuDNN, o projeto já está configurado para usar CPU por padrão (`compute_type="int8"`). Isso é mais estável em sistemas sem GPU dedicada.

### Download do modelo falha

Se o download falhar ou for interrompido:

1. Baixe manualmente de [Hugging Face](https://huggingface.co/guillaumekln/faster-whisper-small)
2. Configure `WHISPER_MODEL_PATH` apontando para a pasta do modelo
3. Rode os scripts normalmente

## Estrutura do projeto

```
AUDIO-TRANCRIBER-WISPER-MAIN/
|── transcriptions              # Onde é armazenado a saida dos audios
├── transcribe_file.py          # Transcreve arquivos de áudio
├── transcribe_microphone.py    # Grava do microfone e transcreve
├── requirements.txt             # Dependências Python
├── README.md                    # Este arquivo
├── env/                      # Ambiente virtual (não versionado)
└── *.mp3, *.ogg               # Arquivos de áudio de exemplo
```

## Formato de saída

### transcription.txt
```
Texto da transcrição com quebra de linha a
cada dez palavras para facilitar a leitura e
organização do conteúdo transcrito em formato de
texto simples...
```

### transcription.json
```json
{
  "language": "pt",
  "language_probability": 0.99,
  "text": "Texto completo da transcrição com quebras de linha..."
}
```

## Contribuindo

Sinta-se à vontade para abrir issues ou enviar pull requests com melhorias!

## 📜 Licença

Este projeto está sob a licença MIT.
