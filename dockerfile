# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including git and FFmpeg
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install tts_arabic with HiFi-GAN support
RUN pip install git+https://github.com/nipponjo/tts_arabic.git

# Install additional required packages
RUN pip install pydub scipy numpy pyarabic

# Set default environment variables
ENV INPUT_SRT=/app/srt/script.srt
ENV OUTPUT_AUDIO=/app/output/output.wav
ENV SPEAKER=1
ENV PACE=0.9
ENV PITCH_VARIATION=0.05
ENV DENOISE=0.003
ENV USE_HIFIGAN=1

# Set environment variable to use CPU only
ENV ONNXRUNTIME_EXECUTION_PROVIDER=CPUExecutionProvider

# Make run.sh executable
RUN chmod +x /app/run.sh

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the shell script when the container launches
CMD ["/bin/bash", "/app/run.sh"]