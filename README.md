# audio-summary
Summarizes a given audio file using whisper (transcription) and BART(summarizer)

How to use:
Install Chocolatey: https://chocolatey.org/install, and then install FFMPEG. 
Add FFMPEG to PATH (Start > Environmental Variables > PATH > New > Path to FFMPEG here)

If using CUDA:
Make sure Cuda and CuDNN versions are compatible based on: https://www.tensorflow.org/install/source#gpu

Known Limitations:
The BART model is from HuggingFace, so it wasn't fine tuned. The only limit is it only summarizes up to 1024 tokens in a single prompt.

Notes:
Attempting to use faster-whisper instead of openai\whisper.
Implementing segmentation of text, therefore the 1024 token prompt limit would not be an issue. (Context preservation is a different topic in this case)
Will update soon.
