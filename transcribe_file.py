"""
Uso:
    python transcribe_file.py audio.mp3
    python transcribe_file.py audio.mp3 --ia-response
    python transcribe_file.py audio.mp3 --ia-response --chat-voz 
    
"""
# import logging
# import TTS
from gtts import gTTS
import ollama 
import argparse
import json
import os
import sys
from tempfile import NamedTemporaryFile
import ctypes
import time
from pathlib import Path
from faster_whisper import WhisperModel

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
def load_whisper_model(model_path: str) -> WhisperModel:
    print(f"Carregando modelo Whisper: {model_path}")
    return WhisperModel(model_path, device="cpu", compute_type="float32")

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

    response = ollama.chat(
        model="qwen3:0.6b",
        messages=chat_history
    )

    ai_text = response["message"]["content"]

    chat_history.append({"role": "assistant", "content": ai_text})
    return ai_text


#  retira as mensagens de configuração do terminal
# logging.getLogger("TTS").setLevel(logging.ERROR)



# o modelo de voz age
def audio_response(text_ia: str):
    with NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        gTTS(text_ia, lang="pt-br").save(f.name)
        filename = f.name
        # tts.tts_to_file(text=text_ia, file_path=f.name,  language="pt", speaker="female-en-5")
        # filename = f.name

    ctypes.windll.winmm.mciSendStringW(
        f'open "{filename}" type mpegvideo alias voz', None, 0, None
    )
    ctypes.windll.winmm.mciSendStringW("play voz", None, 0, None)

    status = ctypes.create_unicode_buffer(255)
    while True:
        ctypes.windll.winmm.mciSendStringW("status voz mode", status, 255, None)
        if status.value != "playing":
            break
        time.sleep(0.1)

    ctypes.windll.winmm.mciSendStringW("close voz", None, 0, None)
    os.remove(filename)
    
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
        if args.ia_response:
            ai_text = response_ia(text)
            data["IA"] = ai_text
            
            text += f"\n\nIA:\n{ai_text}"
            if args.chat_voz:
               
                audio_response(ai_text)

        print("=" * 60)

        output_path = Path("transcription.txt")
        output_path.write_text(text, encoding="utf-8")
        json_path = args.json or Path("transcription.json")

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
