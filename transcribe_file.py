"""
Transcreve um arquivo de áudio específico para português brasileiro.

Uso:
    python transcribe_file.py audio.mp3
    python transcribe_file.py audio.ogg --output saida.txt
    python transcribe_file.py audio.mp3 --ia-response
    python transcribe_file.py audio.mp3 --ia-response --chat-voz 
    
    
    # Usar modelo já baixado:
    set -x WHISPER_MODEL_PATH "/caminho/do/modelo"
    python transcribe_file.py audio.mp3
"""

import argparse
import json
import os
import sys
from gtts import gTTS
from tempfile import NamedTemporaryFile
import ctypes
import time


from pathlib import Path

from faster_whisper import WhisperModel
from config import GEMINI_API_KEY
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

def transcribe_audio_file(
    audio_path: Path,
    model_path: str = "small",
    language: str = "pt",
) -> tuple[str, dict]:
    """Transcreve arquivo de áudio usando faster-whisper."""
    print(f"Carregando modelo: {model_path}")
    model = WhisperModel(model_path, device="cpu", compute_type="int8")
    
    print(f"Transcrevendo: {audio_path}")
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=language,
        task="transcribe",
        # initial_prompt serve apenas para dar contexto linguístico ao Whisper.
        # Ele NÃO cria memória nem regras fixas, apenas influencia a transcrição.
        # Deve ser sempre texto puro (str). NÃO misturar com tokens (int).
        initial_prompt=(
            "Oxygeni, Hub"
        ),
        condition_on_previous_text=False
    )
  
    print(f"Idioma detectado: {info.language} (probabilidade: {info.language_probability:.2f})")
   
    # Coleta todos os segmentos
    texts = []
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
       
        texts.append(segment.text)
    
    
    full_text = " ".join(texts)
    # Quebra de linha a cada 10 palavras
    words = full_text.split()
    wrapped_lines = [" ".join(words[i:i + 10]) for i in range(0, len(words), 10)]
    wrapped_text = "\n".join(wrapped_lines)
    data = {
        "language": info.language,
        "language_probability": info.language_probability,
        "User": wrapped_text,
    }
    
    return wrapped_text, data

def response_ia(response_text: str, response_dict: dict) -> tuple[str, dict]:
    #criando um prompt
    prompt = f"""
        Persona:
        Responda em até 6 frases.
        Use frases curtas
        tom neutro e natural
        Evite jargões tecnicos
        Evite metáforas abstratas em tarefas práticas
        Mantenha o foco do tema da pergunta.

        Pergunta:
        {response_text}
    """
    
    # Realizando o Input para a IA
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    response_dict["IA"] = response.text
    return response.text, response_dict

def audio_response(text_ia: str):
    if isinstance(text_ia, set):
        text_ia = " ".join(text_ia)

    text_ia = str(text_ia)

    with NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        gTTS(text_ia, lang="pt-br").save(f.name)
        nome = f.name

    ctypes.windll.winmm.mciSendStringW(
        f'open "{nome}" type mpegvideo alias voz', None, 0, None
    )
    ctypes.windll.winmm.mciSendStringW("play voz", None, 0, None)

    status = ctypes.create_unicode_buffer(255)
    while True:
        ctypes.windll.winmm.mciSendStringW(
            "status voz mode", status, 255, None
        )
        if status.value != "playing":
            break
        time.sleep(0.1)

    ctypes.windll.winmm.mciSendStringW("close voz", None, 0, None)
    os.remove(nome)
    
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
    
    # Define saídas
    output_txt = args.output or Path("transcription.txt")
    output_json = args.json or Path("transcription.json")
    
    # Modelo (usa variável de ambiente se definida)
    model_path = os.getenv("WHISPER_MODEL_PATH", "small")
    
    try:
        print("="*60)
        text, data = transcribe_audio_file(args.audio_file, model_path)
        
    except Exception as exc:
        print(f"\n✗ Erro ao transcrever: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    # chamando a IA para responder
    if args.ia_response:
        try:
            backup = "User:" + text
            text, data = response_ia(text, data)
            print(text)
            if args.chat_voz:
                try:
                    audio_response(text)
                except Exception as exc:
                    print(f"Erro ao transformar em audio: {exc}")
                    import traceback
                    traceback.print_exc()
                    return 1
            text = backup + "\nIa: " + text
            
        except Exception as exc:
            print(f"Erro ao chamar a Ia: {exc}")
            import traceback
            traceback.print_exc()
            return 1
    print("="*60)
    # Salva arquivos
    output_txt.write_text(text, encoding="utf-8")
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"\n✓ Transcrição salva em:")
    print(f"  - {output_txt}")
    print(f"  - {output_json}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
