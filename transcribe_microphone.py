"""
Grava áudio do microfone e transcreve para português brasileiro.

Requisitos:
    pip install soundfile

Uso:
    # Gravar 10s e transcrever:
    python transcribe_microphone.py
    
    # Gravar 10s, transcrever e o modelo de Ia responder(texto):
    python transcribe_microphone.py --ia-response
    
    # Gravar 10s, transcrever e o modelo de Ia responder(texto e audio):
    python transcribe_microphone.py --ia-response --chat-voz 
    
    # Apenas gravar (teste):
    python transcribe_microphone.py --test-only
"""
# from TTS.api import TTS
import keyboard
import argparse
import json
import os
import sys
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
from transcribe_file import load_whisper_model, transcribe_audio_file, response_ia, audio_response


#para windows - Teste
def record_audio_windows(output_wav: Path) -> bool:
    fs = 16000
    channels = 1
    frames = []

    def callback(indata, frame_count, time, status):
        if status:
            print(status)
        frames.append(indata.copy())

    print("Pressione ENTER para iniciar a gravação ou s para Encerrar")
    tecla = keyboard.read_key().lower()
    if tecla == "s":
        return False
    input()

    print("🎤 Gravando... Pressione ENTER novamente para parar")

    with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
        input()  # Espera ENTER para parar

    # Junta todos os pedaços gravados
    audio = np.concatenate(frames, axis=0)
    sf.write(output_wav, audio, fs)

    print(f"✓ Áudio salvo em {output_wav}")
    return True

def main() -> int:
    parser = argparse.ArgumentParser(description="Grava do microfone e transcreve (pt-BR)")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Nome da fonte de entrada (veja --list-sources)"
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Lista fontes de entrada disponíveis e sai"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Apenas grava WAV, sem transcrever"
    )
    parser.add_argument(
        "--ia-response",
        action="store_true",
        help="Após transcrever o modelo de ia responde"
    )
    parser.add_argument(
        "--chat-voz",
        action="store_true",
        help="Após transcrever o modelo de ia responde"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("captura.wav"),
        help="Arquivo WAV de saída"
    )
    args = parser.parse_args()
    model_path = os.getenv("WHISPER_MODEL_PATH", "small")
    whisper_model = load_whisper_model(model_path)
    
    # modelo de voz
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2",progress_bar=False, gpu=False) 
    
    #criando os arquivos de saida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path("transcriptions")
    folder.mkdir(exist_ok=True)

    output_txt = folder / f"transcription_{timestamp}.txt"
    output_json = folder / f"transcription_{timestamp}.json"
    
    
    conversation_data = [] 
    turn_index = 1
    # Gravação
    print("="*70)
    while True: 
        
        success =  record_audio_windows(args.output)
        print("="*70)
    
        if not success:
            print("Encerrando o programa...")
            return 1
    
        print(f"\n✓ Áudio salvo: {args.output}")
    
        # Modo teste: para por aqui
        if args.test_only:
            print("\nModo teste: transcrição ignorada. Verifique o arquivo WAV.")
            return 0
    
        # Transcrição
        try:
            text, data = transcribe_audio_file(whisper_model, args.output)
        
        except Exception as exc:
            print(f"\n✗ Erro ao transcrever: {exc}")
            import traceback
            traceback.print_exc()
            return 1
        
        text = f"User:\n{text}"
        # chamando a IA para responder
        if args.ia_response:
            try:
                ai_text = response_ia(text)
                data["IA"] = ai_text
                text += f"\n\nIA:\n{ai_text}"

                if args.chat_voz:
                    audio_output = folder / f"response_{timestamp}_{turn_index:03d}.mp3"
                    saved_audio = audio_response(ai_text, audio_output)
                    data["IA_audio_file"] = str(saved_audio)
                    
            except Exception as exc:
                print(f"Erro ao chamar a Ia: {exc}")
                import traceback
                traceback.print_exc()
                return 1
            
        # Salva resultados em txt
        with open(output_txt, "a", encoding="utf-8") as f:
            f.write(text + "\n" + "="*70 + "\n")
        
        # Salva resultados em json
        conversation_data.append(data)
        output_json.write_text(
        json.dumps(conversation_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # printando no terminal
        print("\n" + "="*70)
        print("Transcrição:")
        print(text)
        print("="*70)
        print(f"\n✓ Salvos: {output_txt}, {output_json}")
        turn_index += 1
    

if __name__ == "__main__":
    sys.exit(main())
