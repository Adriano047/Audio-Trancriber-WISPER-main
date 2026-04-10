"""
Uso:
    python transcribe_file.py audio.mp3
    python transcribe_file.py audio.mp3 --ia-response
    python transcribe_file.py audio.mp3 --ia-response --chat-voz 
    
"""
# import logging
# import TTS
from gtts import gTTS
import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from urllib import error, request

from faster_whisper import WhisperModel

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL").rstrip("/")
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest",
)
WHISPER_MODEL_REPOS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# Memoria
chat_history = [
    {"role": "system", "content": "Você é um assistente de voz útil, objetivo e natural."}
]

# Formatar texto
def wrap_text(text: str, words_per_line: int = 10) -> str:
    words = text.split()
    lines = [
        " ".join(words[i:i + words_per_line])
        for i in range(0, len(words), words_per_line)
    ]
    return "\n".join(lines)

#  Carregar o modelo de Transcrição
def _huggingface_cache_dir() -> Path:
    if os.getenv("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    if os.getenv("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _resolve_cached_whisper_model(model_path: str) -> Path | None:
    repo_id = WHISPER_MODEL_REPOS.get(model_path)
    if repo_id is None:
        return None

    repo_cache_dir = _huggingface_cache_dir() / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshot_candidates = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot_dir in snapshot_candidates:
        required_files = [
            snapshot_dir / "config.json",
            snapshot_dir / "model.bin",
            snapshot_dir / "tokenizer.json",
        ]
        if all(path.exists() for path in required_files):
            return snapshot_dir

    return None


def load_whisper_model(model_path: str) -> WhisperModel:
    print(f"Carregando modelo Whisper: {model_path}")

    explicit_model_path = Path(model_path)
    if explicit_model_path.exists():
        return WhisperModel(str(explicit_model_path), device="cpu", compute_type="int8")

    cached_snapshot = _resolve_cached_whisper_model(model_path)
    if cached_snapshot is not None:
        print(f"Usando cache local do Whisper: {cached_snapshot}")
        return WhisperModel(str(cached_snapshot), device="cpu", compute_type="int8")

    try:
        return WhisperModel(model_path, device="cpu", compute_type="int8")
    except Exception as exc:
        raise RuntimeError(
            "Nao foi possivel carregar o modelo Whisper. "
            "Defina WHISPER_MODEL_PATH para uma pasta local do modelo ou baixe o modelo "
            f"'{model_path}' em uma rede sem proxy autenticado."
        ) from exc

#  Transcrevi o audio em texto
def transcribe_audio_file(model: WhisperModel, audio_path: Path) -> tuple[str, dict]:
    print(f"Transcrevendo: {audio_path}")

    start = time.perf_counter()

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language="pt",
        task="transcribe",
        condition_on_previous_text=False,
        initial_prompt="Termos possíveis: Oxygeni Hub, AcademIA, Incode Tech School."
    )

    texts = [segment.text.strip() for segment in segments]
    full_text = " ".join(texts)
    wrapped = wrap_text(full_text)

    end = time.perf_counter()
    print(f"Tempo de transcrição: {end - start:.2f}s")

    data = {
        "language": info.language,
        "language_probability": info.language_probability,
        "User": wrapped,
    }

    return wrapped, data

# Chamar o mdelo de IA
def response_ia(response_text: str):
    global chat_history

    chat_history.append({"role": "user", "content": response_text})

    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": chat_history,
            "stream": True,
        }
    ).encode("utf-8")
    req = request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=180) as response:
            print(f"Chamando Ollama em {OLLAMA_BASE_URL} com streaming...")
            print("IA: ", end="", flush=True)
            ai_chunks = []
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                response_data = json.loads(line)
                message = response_data.get("message", {})
                chunk = message.get("content", "")
                if chunk:
                    ai_chunks.append(chunk)
                    print(chunk, end="", flush=True)

            print()
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Falha ao chamar Ollama em {OLLAMA_BASE_URL}: HTTP {exc.code} - {details}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Falha ao conectar ao Ollama em {OLLAMA_BASE_URL}: {exc.reason}"
        ) from exc

    ai_text = "".join(ai_chunks).strip()

    chat_history.append({"role": "assistant", "content": ai_text})
    return ai_text


#  retira as mensagens de configuração do terminal
# logging.getLogger("TTS").setLevel(logging.ERROR)



def play_audio_file(audio_path: Path) -> None:
    ctypes.windll.winmm.mciSendStringW(
        f'open "{audio_path}" type mpegvideo alias voz', None, 0, None
    )
    ctypes.windll.winmm.mciSendStringW("play voz", None, 0, None)

    status = ctypes.create_unicode_buffer(255)
    while True:
        ctypes.windll.winmm.mciSendStringW("status voz mode", status, 255, None)
        if status.value != "playing":
            break
        time.sleep(0.1)

    ctypes.windll.winmm.mciSendStringW("close voz", None, 0, None)


# o modelo de voz age
def audio_response(text_ia: str, output_path: Path, play_audio: bool = True) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gTTS(text_ia, lang="pt").save(str(output_path))
    except Exception as exc:
        raise RuntimeError(
            "Nao foi possivel gerar o audio da resposta. "
            "O gTTS depende de acesso aos servicos do Google para sintetizar a voz."
        ) from exc

    print(f"Audio da IA salvo em: {output_path}")

    if play_audio:
        play_audio_file(output_path)

    return output_path
    
def main() -> int:
    parser = argparse.ArgumentParser(description="Transcreve arquivo de áudio para pt-BR")
    parser.add_argument("audio_file", type=Path, help="Arquivo de áudio para transcrever")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Arquivo de saída (padrão: transcription.txt)"
    )
    parser.add_argument(
        "--ia-response",
        action="store_true",
        help="Após transcrever o modelo de ia responde"
    )
    parser.add_argument(
        "--chat-voz",
        action="store_true",
        help="A resposta da ia vira em audio"
    )
    parser.add_argument(
        "--audio-output",
        type=Path,
        default=None,
        help="Arquivo MP3 para salvar a resposta em audio"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Arquivo JSON de saída (padrão: transcription.json)"
    )
    args = parser.parse_args()
    
    # Valida arquivo de entrada
    if not args.audio_file.exists():
        print(f"✗ Arquivo não encontrado: {args.audio_file}")
        return 1
    
    # Modelo (usa variável de ambiente se definida)
    model_path = os.getenv("WHISPER_MODEL_PATH", "small") 
    
    # modelo de voz
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2",progress_bar=False, gpu=False) 
    try:
        print("=" * 60)

        whisper_model = load_whisper_model(model_path)

        text, data = transcribe_audio_file(whisper_model, args.audio_file)
        text = f"User:\n{text}"
        output_path = args.output or Path("transcription.txt")
        json_path = args.json or Path("transcription.json")
        if args.ia_response:
            ai_text = response_ia(text)
            data["IA"] = ai_text
            
            text += f"\n\nIA:\n{ai_text}"
            if args.chat_voz:
                audio_output = args.audio_output or output_path.with_name(f"{output_path.stem}_ia.mp3")
                saved_audio = audio_response(ai_text, audio_output)
                data["IA_audio_file"] = str(saved_audio)

        print("=" * 60)

        output_path.write_text(text, encoding="utf-8")

        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(text)
        print("✓ Arquivos salvos com sucesso")

        return 0
    
    except Exception as exc:
        print(f"Erro: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
