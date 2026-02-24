import os, time
"""
Grava áudio do microfone e transcreve para português brasileiro.

Requisitos:
    pip install soundfile

Uso:
    # Listar dispositivos de entrada:
    python transcribe_microphone.py --list-sources
    
    # Gravar 10s e transcrever:
    python transcribe_microphone.py --seconds 10
    
    # Gravar 10s, transcrever e o modelo de Ia responder(texto):
    python transcribe_microphone.py --seconds 10 --ia-response
    
    # Gravar 10s, transcrever e o modelo de Ia responder(texto e audio):
    python transcribe_microphone.py --seconds 10 --ia-response --chat-voz 
    
    # Usar fonte específica:
    python transcribe_microphone.py --seconds 10 --source "alsa_input.pci-0000_00_1f.3.analog-stereo"
    
    # Apenas gravar (teste):
    python transcribe_microphone.py --seconds 5 --test-only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from transcribe_file import load_whisper_model, transcribe_audio_file, response_ia, audio_response

def list_audio_sources():
    """Lista fontes de entrada de áudio disponíveis."""
    result = subprocess.run(
        ["pactl", "list", "sources", "short"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("✗ Erro ao listar fontes. Certifique-se de que PulseAudio/PipeWire está rodando.")
        return False
    
    print("\nFontes de entrada disponíveis:")
    print("="*70)
    
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            idx, name = parts[0], parts[1]
            status = parts[-1] if len(parts) > 4 else ""
            # Mostra apenas entradas reais (não monitores)
            if "input" in name or "source" in name.lower():
                marker = "→" if status == "RUNNING" else " "
                print(f"{marker} {idx}: {name} [{status}]")
    
    print("="*70)
    print("Dica: Use o nome completo com --source")
    return True

# para linux----------------------------------------------------------------------- 
# def record_audio_parec(duration_sec: float, source: str | None, output_wav: Path) -> bool:
#     """Grava áudio usando parec (PulseAudio/PipeWire)."""
#     cmd = [
#         "parec",
#         "--format=s16le",
#         "--rate=16000",
#         "--channels=1"
#     ]
    
#     if source:
#         cmd.extend(["--device", source])
    
#     print(f"\nGravando por {duration_sec:.1f}s...")
#     print(f"Fonte: {source if source else 'padrão do sistema'}")
#     print("\n🎤 FALE AGORA!\n")
    
#     try:
#         proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         raw_audio, stderr = proc.communicate(timeout=duration_sec + 1)
#     except subprocess.TimeoutExpired:
#         proc.kill()
#         raw_audio, stderr = proc.communicate()
#     except FileNotFoundError:
#         print("✗ 'parec' não encontrado. Instale: sudo pacman -S pipewire-pulse")
#         return False
    
#     if stderr and b"error" in stderr.lower():
#         print(f"Avisos do parec:\n{stderr.decode()}")
    
#     # Converte PCM s16le para numpy float32
#     expected_samples = int(duration_sec * 16000)
#     raw_bytes = raw_audio[:expected_samples * 2]  # 2 bytes por sample
#     audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
#     # Salva WAV
#     import soundfile as sf
#     sf.write(output_wav, audio_np, 16000, subtype="PCM_16")
    
#     # Diagnóstico
#     rms = np.sqrt(np.mean(audio_np**2))
#     peak = np.max(np.abs(audio_np))
    
#     print(f"✓ Gravação concluída!")
#     print(f"  RMS: {rms:.6f} | Pico: {peak:.6f}")
    
#     if rms < 0.001:
#         print("\n⚠️  Áudio muito silencioso! Possíveis causas:")
#         print("  - Microfone não está selecionado como entrada padrão")
#         print("  - Volume de entrada muito baixo")
#         print("  - Use 'pavucontrol' para configurar entrada")
#         return False
    
#     return True

#para windows - Teste
def record_audio_windows(duration_sec: float, output_wav: Path) -> bool:
    import sounddevice as sd
    import soundfile as sf

    fs = 16000
    channels = 1
    print(f"\nGravando por {duration_sec:.1f}s...")
    print("🎤 FALE AGORA!")

    recording = sd.rec(int(duration_sec * fs), samplerate=fs, channels=channels)
    sd.wait()

    sf.write(output_wav, recording, fs)
    print(f"✓ Áudio salvo em {output_wav}")
    return True

def main() -> int:
    parser = argparse.ArgumentParser(description="Grava do microfone e transcreve (pt-BR)")
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Duração da gravação em segundos"
    )
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
    
    # Lista fontes e sai
    if args.list_sources:
        return 0 if list_audio_sources() else 1
    
    # Gravação
    print("="*70)
    # success = record_audio_parec(args.seconds, args.source, args.output)
    success =  record_audio_windows(args.seconds, args.output)
    print("="*70)
    
    if not success:
        return 1
    
    print(f"\n✓ Áudio salvo: {args.output}")
    
    # Modo teste: para por aqui
    if args.test_only:
        print("\nModo teste: transcrição ignorada. Verifique o arquivo WAV.")
        return 0
    
    # Transcrição
    model_path = os.getenv("WHISPER_MODEL_PATH", "small")
    
    try:
        whisper_model = load_whisper_model(model_path)

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
                audio_response(ai_text)
        except Exception as exc:
            print(f"Erro ao chamar a Ia: {exc}")
            import traceback
            traceback.print_exc()
            return 1
    # Salva resultados
    output_txt = Path("transcription.txt")
    output_json = Path("transcription.json")
    
    output_txt.write_text(text, encoding="utf-8")
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n" + "="*70)
    print("Transcrição:")
    print(text)
    print("="*70)
    print(f"\n✓ Salvos: {output_txt}, {output_json}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
