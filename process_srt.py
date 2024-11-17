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
import subprocess  # Add this import
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
    # Enhanced text preprocessing
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
            cuda=False,
            vocoder_id='hifigan',
            model_id='fastpitch'
        )
        
        wave_data = wave_data.astype(np.float32)
        return wave_data
        
    except Exception as e:
        logger.error(f"Error in TTS generation: {str(e)}")
        return None

def process_srt_batch(subs_batch, batch_num, output_dir, speaker, base_pace, pitch_variation, denoise_strength):
    logger.info(f"Processing batch {batch_num}...")
    full_audio = AudioSegment.silent(duration=0)
    temp_files = []
    total_lines = len(subs_batch)
    
    for i, sub in enumerate(subs_batch, 1):
        logger.info(f"Processing line {i}/{total_lines} in batch {batch_num}")
        
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
        end_time = sub.end.ordinal
        available_duration = (end_time - start_time) / 1000
        
        # Generate TTS audio
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
    
    # Save batch output
    batch_output_path = os.path.join(output_dir, f"batch_{batch_num:04d}.wav")
    logger.info(f"Exporting batch {batch_num} to {batch_output_path}")
    full_audio.export(
        batch_output_path,
        format="wav",
        parameters=["-ar", "44100", "-sample_fmt", "s16"]
    )
    
    # Cleanup
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
    
    return batch_output_path

def merge_batch_files(batch_files, output_file):
    """Merge batch files using ffmpeg concat demuxer without re-encoding"""
    # Create temporary concat file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for file in batch_files:
            f.write(f"file '{os.path.abspath(file)}'\n")
    
    try:
        # Use ffmpeg concat demuxer
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',  # Copy without re-encoding
            output_file
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        os.remove(concat_file)

def process_srt(srt_file_path, output_audio_path, speaker=1, base_pace=0.9, pitch_variation=0.1, denoise_strength=0.005):
    logger.info("Starting SRT processing...")
    subs = pysrt.open(srt_file_path)
    batch_size = 10000
    total_subs = len(subs)
    
    # Calculate total duration from SRT file
    total_duration_ms = subs[-1].end.ordinal
    logger.info(f"Total SRT duration: {total_duration_ms/1000:.2f} seconds")
    
    # Create directories
    output_dir = os.path.dirname(output_audio_path)
    batch_dir = os.path.join(output_dir, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    
    processed_batch_files = []
    current_batch_files = []
    current_batch_start_time = 0
    
    # Process lines in batches
    for batch_start in range(0, total_subs, batch_size):
        batch_end = min(batch_start + batch_size, total_subs)
        batch_num = (batch_start // batch_size) + 1
        logger.info(f"Processing batch {batch_num} (lines {batch_start+1}-{batch_end})")
        
        # Process each line in the batch
        for i in range(batch_start, batch_end):
            sub = subs[i]
            original_text = sub.text.strip()
            if not original_text:
                continue
                
            text = preprocess_text(original_text)
            if not text:
                continue
            
            # Calculate timestamps
            start_time = sub.start.ordinal
            end_time = sub.end.ordinal
            available_duration = (end_time - start_time) / 1000
            
            # Calculate silence before this line
            if i == batch_start:  # First line in batch
                silence_before = start_time - current_batch_start_time
            elif i == 0:  # Very first line
                silence_before = start_time
            else:
                silence_before = start_time - subs[i-1].end.ordinal
            
            # Add final silence only to last line in file
            final_silence = 0
            if i == len(subs) - 1:
                final_silence = total_duration_ms - end_time
            
            # Generate and process audio
            wave_data = generate_tts_segment(text, speaker, base_pace, pitch_variation, denoise_strength)
            if wave_data is None:
                # Create silent audio segment including both the line duration AND the silence before
                total_silence_needed = available_duration + (silence_before / 1000)  # Convert silence_before to seconds
                logger.warning(f"TTS generation failed for line {i+1}, creating silence for duration: {total_silence_needed}s (including {silence_before/1000}s pre-silence)")
                audio_segment = AudioSegment.silent(duration=int((available_duration * 1000) + silence_before))
                actual_duration = total_silence_needed  # Set duration for logging
            else:
                # Process audio segment
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_filename = temp_file.name
                    int_data = np.int16(wave_data * 32767)
                    wav.write(temp_filename, 22050, int_data)
                
                audio_segment = AudioSegment.from_wav(temp_filename)
                audio_segment = apply_audio_effects(audio_segment)
                
                # Handle timing
                actual_duration = len(audio_segment) / 1000
                if actual_duration > available_duration:
                    speedup_factor = min(actual_duration / available_duration, 1.3)
                    audio_segment = audio_segment.speedup(playback_speed=speedup_factor)
                    actual_duration = len(audio_segment) / 1000
                
                os.remove(temp_filename)
            
            # Add silences (for both successful and failed generations)
            if silence_before > 0:
                audio_segment = AudioSegment.silent(duration=silence_before) + audio_segment
            if final_silence > 0:
                audio_segment = audio_segment + AudioSegment.silent(duration=final_silence)
            
            # Save line audio
            line_filename = os.path.join(batch_dir, f"batch_{batch_num:04d}_line_{i+1:04d}.wav")
            audio_segment.export(line_filename, format="wav", parameters=["-ar", "44100", "-sample_fmt", "s16"])
            current_batch_files.append(line_filename)
            
            logger.info(f"Line {i+1} completed - Duration: {actual_duration:.2f}s / {available_duration:.2f}s")
        
        # Merge this batch once it's complete
        if current_batch_files:
            batch_output = os.path.join(batch_dir, f"merged_batch_{batch_num:04d}.wav")
            logger.info(f"Merging batch {batch_num}...")
            merge_batch_files(current_batch_files, batch_output)
            processed_batch_files.append(batch_output)
            
            # Cleanup individual line files
            for file in current_batch_files:
                try:
                    os.remove(file)
                except Exception as e:
                    logger.warning(f"Failed to remove file {file}: {str(e)}")
            
            current_batch_files = []
            current_batch_start_time = subs[min(batch_end, len(subs)-1)].end.ordinal
            
            logger.info(f"Batch {batch_num} merged successfully")
    
    # Final merge of all batch files
    if len(processed_batch_files) > 1:
        logger.info("Performing final merge of all batches...")
        merge_batch_files(processed_batch_files, output_audio_path)
        
        # Cleanup batch files
        for file in processed_batch_files:
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"Failed to remove batch file {file}: {str(e)}")
    elif len(processed_batch_files) == 1:
        # Just rename the single batch file
        os.rename(processed_batch_files[0], output_audio_path)
    
    logger.info("All processing and merging completed successfully!")
    
    return output_audio_path

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
