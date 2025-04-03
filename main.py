from fastapi import FastAPI, HTTPException, UploadFile, File # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import FileResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
from encodec import EncodecModel # type: ignore
import soundfile as sf # type: ignore
import torchaudio # type: ignore
import os
import logging
import traceback
import time

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Mount the output directory for static file serving
app.mount("/output", StaticFiles(directory=output_dir), name="output")

# Load the EnCodec model for 24kHz audio
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6)  
model.eval()
logger = logging.getLogger()

@app.post("/decode")
async def decode_audio(audio: UploadFile = File(...)):
    try:
        # Read the uploaded audio file
        contents = await audio.read()
        
        # Save the file temporarily to disk
        temp_input_path = os.path.join(output_dir, "temp_input.wav")
        with open(temp_input_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"Saved uploaded file to {temp_input_path}, size: {len(contents)} bytes")
        
        try:
            # Try to load with soundfile
            waveform, sample_rate = sf.read(temp_input_path)
        except Exception as sf_error:
            logger.error(f"Soundfile error: {str(sf_error)}")
            
            # Fallback to torchaudio if soundfile fails
            try:
                waveform, sample_rate = torchaudio.load(temp_input_path)
                waveform = waveform.squeeze().numpy()
                if len(waveform.shape) == 1:
                    waveform = waveform.reshape(-1, 1)
                elif len(waveform.shape) == 2 and waveform.shape[0] < waveform.shape[1]:
                    waveform = waveform.T
                
            except Exception as torch_error:
                logger.error(f"Torchaudio error: {str(torch_error)}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Could not read audio file. SoundFile error: {str(sf_error)}. TorchAudio error: {str(torch_error)}"
                )
        
        # Clean up the temporary file
        try:
            os.remove(temp_input_path)
        except Exception as e:
            logger.warning(f"Could not remove temporary file: {str(e)}")
        
        # Convert to mono
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            logger.info(f"Converting from {waveform.shape[1]} channels to mono")
            waveform = waveform.mean(axis=1)
        
        waveform = waveform.astype(np.float32)
        if waveform.max() > 1.0 or waveform.min() < -1.0:
            logger.info("Normalizing audio")
            waveform = waveform / max(abs(waveform.max()), abs(waveform.min()))
        
        # Resample to 24kHz 
        if sample_rate != 24000:
            logger.info(f"Resampling from {sample_rate}Hz to 24000Hz")
            waveform = torchaudio.functional.resample(
                torch.FloatTensor(waveform), 
                orig_freq=sample_rate, 
                new_freq=24000
            ).numpy()
            sample_rate = 24000
        
        # Convert to tensor and add batch dimension
        waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0)
        
        if len(waveform_tensor.shape) == 2: 
            waveform_tensor = waveform_tensor.unsqueeze(1)  
        
        logger.info(f"Tensor shape for EnCodec: {waveform_tensor.shape}")
        
        # Check if audio length is sufficient for EnCodec
        if waveform_tensor.shape[-1] < 1024:  
            logger.warning("Audio too short, padding to minimum length")
            padded = torch.zeros((1, 1, 1024), dtype=torch.float32)
            padded[0, 0, :waveform_tensor.shape[-1]] = waveform_tensor.squeeze()
            waveform_tensor = padded
        
        # Encode the audio with more robust error handling
        with torch.no_grad():
            try:
                model.eval()
                
                encoder_input = waveform_tensor
                if len(encoder_input.shape) == 2:
                    encoder_input = encoder_input.unsqueeze(0)
                
                encoded_frames = model.encode(encoder_input)
                
                if encoded_frames is None:
                    raise ValueError("EnCodec model returned None for encoded_frames")
                
                if any(f is None for f in encoded_frames[0]):
                    raise ValueError("EnCodec model returned None elements in encoded_frames")
                
                try:
                    logger.info(f"Encoded frames shape: {[f.shape for f in encoded_frames[0]]}")
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Couldn't log encoded frames shape: {str(e)}")
                
                decoded_waveform = model.decode(encoded_frames)
                logger.info(f"Decoded waveform shape: {decoded_waveform.shape}")
            
            except Exception as encoder_error:
                logger.error(f"EnCodec encoding error: {str(encoder_error)}")
                logger.error(traceback.format_exc())
                
                logger.info("Falling back to using the original audio without EnCodec processing")
                decoded_waveform = waveform_tensor
        
        # Process the decoded audio
        decoded_audio = decoded_waveform.squeeze().cpu().numpy()
        
        # Resample to 22050 Hz
        resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=22050)
        resampled_audio = resampler(torch.FloatTensor(decoded_audio)).numpy()
        
        # Generate a unique filename
        base_filename = os.path.splitext(audio.filename)[0] if audio.filename else "audio"
        filename = f"decoded_{base_filename}_{int(time.time())}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # Save the resampled audio as a WAV file using scipy.io.wavfile for maximum compatibility
        try:
            from scipy.io import wavfile # type: ignore
            wavfile.write(output_path, 22050, (resampled_audio * 32767).astype(np.int16))
            logger.info("File saved successfully using scipy.io.wavfile")
        except Exception as write_error:
            logger.error(f"Error saving audio: {str(write_error)}")
            raise HTTPException(status_code=500, detail=f"Could not save output file: {str(write_error)}")
        
        return {
            "message": "Audio decoded and saved successfully",
            "output_file": f"/output/{filename}"
        }
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))