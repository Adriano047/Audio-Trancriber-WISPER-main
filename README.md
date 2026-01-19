# Audio Transcription Project

Projeto para transcrição de áudio para texto em português brasileiro usando **faster-whisper**.

## 📋 Requisitos Linux

- Python 3.13+
- PulseAudio/PipeWire (para gravação do microfone)
- `parec` (vem com `pipewire-pulse` ou `pulseaudio-utils`)

## Requisitos Windows
- Python 3.13+
- Dependências Python listadas em `requirements.txt`

> Observação: o arquivo `requirements.txt` foi criado durante a adaptação do projeto para Windows
> e reflete as dependências utilizadas e testadas nesse ambiente.

## 🚀 Instalação

```bash
# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate.fish  # ou: source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### Dependências principais:
- `faster-whisper` - Transcrição de áudio
- `soundfile` - Manipulação de arquivos de áudio
- `numpy` - Processamento numérico

## 📝 Uso

### 1. Transcrever arquivo de áudio

Transcreve arquivos de áudio (mp3, ogg, wav, etc.) para texto:

```bash
# Transcrever um arquivo
python transcribe_file.py audio.mp3

# Especificar arquivo de saída
python transcribe_file.py audio.ogg --output resultado.txt --json resultado.json
```

**Saídas:**
- `transcription.txt` - Texto com quebras de linha a cada 10 palavras
- `transcription.json` - Metadados (idioma, probabilidade, texto)

### 2. Gravar do microfone e transcrever

Grava áudio do microfone por um tempo definido e transcreve:

```bash
# Listar fontes de entrada disponíveis
python transcribe_microphone.py --list-sources

# Gravar 10 segundos e transcrever
python transcribe_microphone.py --seconds 10

# Usar fonte específica
python transcribe_microphone.py --seconds 15 --source "alsa_input.pci-0000_00_1f.3.analog-stereo"

# Apenas gravar (teste de áudio)
python transcribe_microphone.py --seconds 5 --test-only
```

**Saídas:**
- `captura.wav` - Arquivo de áudio gravado
- `transcription.txt` - Transcrição formatada
- `transcription.json` - Metadados completos

## ⚙️ Configuração

### Usar modelo Whisper já baixado

Para evitar downloads repetidos do modelo:

```bash
# Fish shell
set -x WHISPER_MODEL_PATH "/caminho/para/modelo"

# Bash
export WHISPER_MODEL_PATH="/caminho/para/modelo"
```

### Modelos disponíveis

- `tiny` - Mais rápido, menos preciso (~1GB)
- `base` - Bom equilíbrio (~1.5GB)
- `small` - **Padrão** - Boa precisão (~2.5GB)
- `medium` - Muito preciso, mais lento (~5GB)
- `large` - Máxima precisão, muito lento (~10GB)

## 🔧 Solução de problemas

### Microfone não captura áudio

1. **Verificar se o microfone está ativo:**
```bash
pactl list sources short
```

2. **Ajustar volume de entrada:**
```bash
pavucontrol  # Interface gráfica
# ou
pactl set-source-volume @DEFAULT_SOURCE@ 100%
```

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

## 📦 Estrutura do projeto

```
MyProject PY/
├── transcribe_file.py          # Transcreve arquivos de áudio
├── transcribe_microphone.py    # Grava do microfone e transcreve
├── requirements.txt             # Dependências Python
├── README.md                    # Este arquivo
├── .venv/                       # Ambiente virtual (não versionado)
└── *.mp3, *.ogg               # Arquivos de áudio de exemplo
```

## 📄 Formato de saída

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

## 🤝 Contribuindo

Sinta-se à vontade para abrir issues ou enviar pull requests com melhorias!

## 📜 Licença

Este projeto está sob a licença MIT.
