import os
import sys
import argparse
import pysrt
from tts_arabic import tts
from pydub import AudioSegment
import tempfile
import scipy.io.wavfile as wav
import numpy as np
import random
import re
import logging
from pyarabic.araby import strip_tashkeel, strip_tatweel, normalize_hamza, normalize_ligature, strip_harakat

# Configure logging for real-time output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Suppress ONNX Runtime warnings about CUDA
os.environ['ONNXRUNTIME_EXECUTION_PROVIDER'] = 'CPUExecutionProvider'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Force stdout to flush immediately
class ForceFlush:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def flush(self):
        self.stream.flush()

sys.stdout = ForceFlush(sys.stdout)

def is_arabic(text):
    arabic_pattern = re.compile('[\u0600-\u06FF]+')
    return bool(arabic_pattern.search(text))

def preprocess_text(text):
    # Enhanced text preprocessing (same as before)
    text = re.sub(r'^Speaker_\d+:\s*', '', text.strip())
    
    # Enhanced normalization
    text = normalize_hamza(text)
    text = normalize_ligature(text)
    text = strip_harakat(text)
    text = strip_tatweel(text)
    
    # Improved character handling
    text = text.replace('ء', 'أ')
    text = text.replace('آ', 'ا')
    text = text.replace('إ', 'ا')
    
    text = re.sub(r'([^\u0621-\u063A\u0641-\u064A\u0660-\u0669a-zA-Z 0-9\.,!?])', '', text)
    
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def apply_audio_effects(audio_segment, quality_level='high'):
    """Apply enhanced audio effects for more natural sound"""
    # Dynamic range compression with smoother settings
    audio_segment = audio_segment.compress_dynamic_range(
        threshold=-20.0,
        ratio=2.0,
        attack=5.0,
        release=50.0
    )
    
    audio_segment = audio_segment.high_pass_filter(60)
    audio_segment = audio_segment.low_pass_filter(10000)
    audio_segment = audio_segment.normalize(-3)
    
    return audio_segment

def generate_tts_segment(text, speaker, pace, pitch_variation, denoise_strength):
    """Generate TTS with enhanced parameters"""
    actual_pace = pace + random.uniform(-0.02, 0.02)
    pitch_mul = 1.0 + random.uniform(-pitch_variation/2, pitch_variation/2)
    
    try:
        wave_data = tts(
            text,
            speaker=speaker,
            pace=actual_pace,
            denoise=denoise_strength,
            pitch_mul=pitch_mul,
            vowelizer='shakkelha',
            cuda=False,  # Explicitly disable CUDA
            vocoder_id='hifigan',
            model_id='fastpitch',  # Missing comma added here
        )
        
        wave_data = wave_data.astype(np.float32)
        return wave_data
        
    except Exception as e:
        logger.error(f"Error in TTS generation: {str(e)}")
        return None

def process_srt(srt_file_path, output_audio_path, speaker=1, base_pace=0.9, pitch_variation=0.1, denoise_strength=0.005):
    logger.info("Starting SRT processing...")
    subs = pysrt.open(srt_file_path)
    full_audio = AudioSegment.silent(duration=0)
    temp_files = []
    cumulative_delay = 0
    total_lines = len(subs)
    
    for i, sub in enumerate(subs, 1):
        # Real-time progress logging
        logger.info(f"Processing line {i}/{total_lines} ({(i/total_lines*100):.1f}%)")
        
        original_text = sub.text.strip()
        if not original_text:
            logger.warning(f"Skipping empty line at index {i}")
            continue
        
        text = preprocess_text(original_text)
        if not text:
            logger.warning(f"Empty text after preprocessing: '{original_text}'")
            continue
        
        # Calculate timing
        start_time = sub.start.ordinal
        end_time = subs[i].start.ordinal if i < len(subs) else sub.end.ordinal
        available_duration = (end_time - start_time) / 1000
        
        # Generate TTS audio with immediate logging
        logger.info(f"Generating audio for: '{text}'")
        wave_data = generate_tts_segment(text, speaker, base_pace, pitch_variation, denoise_strength)
        if wave_data is None:
            continue
            
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_files.append(temp_filename)
            int_data = np.int16(wave_data * 32767)
            wav.write(temp_filename, 22050, int_data)
        
        # Process audio segment
        audio_segment = AudioSegment.from_wav(temp_filename)
        audio_segment = apply_audio_effects(audio_segment)
        
        # Handle timing
        actual_duration = len(audio_segment) / 1000
        if actual_duration > available_duration:
            speedup_factor = min(actual_duration / available_duration, 1.3)
            audio_segment = audio_segment.speedup(playback_speed=speedup_factor)
            actual_duration = len(audio_segment) / 1000
            logger.info(f"Applied speedup factor: {speedup_factor:.2f}")
        
        # Add silence for timing
        if start_time > len(full_audio):
            silence_duration = start_time - len(full_audio)
            full_audio += AudioSegment.silent(duration=silence_duration)
        
        full_audio += audio_segment
        
        # Detailed timing log
        logger.info(f"Line {i} completed - Duration: {actual_duration:.2f}s / {available_duration:.2f}s")
        sys.stdout.flush()  # Force flush after each line
    
    # Final processing
    logger.info("Applying final audio processing...")
    full_audio = apply_audio_effects(full_audio)
    
    # Export with higher quality settings
    logger.info(f"Exporting final audio to {output_audio_path}")
    full_audio.export(
        output_audio_path,
        format="wav",
        parameters=["-ar", "44100", "-sample_fmt", "s16"]
    )
    
    # Cleanup
    logger.info("Cleaning up temporary files...")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")

    logger.info("Processing completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Arabic SRT to natural-sounding audio using TTS")
    parser.add_argument("--speaker", type=int, default=1, choices=[0, 1, 2, 3], help="Speaker ID (0-3)")
    parser.add_argument("--pace", type=float, default=0.9, help="Base speaking pace")
    parser.add_argument("--pitch-variation", type=float, default=0.05, help="Pitch variation range")
    parser.add_argument("--denoise", type=float, default=0.003, help="Denoising strength")
    args = parser.parse_args()

    srt_file = os.environ.get('INPUT_SRT', '/app/srt/script.srt')
    output_file = os.environ.get('OUTPUT_AUDIO', '/app/output/output.wav')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    process_srt(srt_file, output_file, args.speaker, args.pace, args.pitch_variation, args.denoise)