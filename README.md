simple scripts that i have been using to turn my srt files into a timestamp accurate output tts audio file ,here is how to run your project:
to run, first install the code files locally (the 5 uploaded files are enough to run the script (character_map.py ,dockerfile ,process_srt.py ,requirements.txt ,run.sh)) build the dockerfile using (docker build -t arabic-tts .) and when finished ,you can run the script using (docker run -v C:\Users\(path to your srt file destination) -v C:\Users\(path to your output destination) -e SPEAKER=1 -e PACE=0.9 -e USE_HIFIGAN=1 arabic-tts

(settings can be tweaked based on your prefrences through either the script or the run command)
