from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
from pydantic import BaseModel
from pydub import AudioSegment
import logging

# Configurar logging para debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Burnout Analysis API")

# Configuração CORS SUPER permissiva para debug
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MUITO permissivo - para debug
    allow_credentials=False,  # Mudamos para False quando origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResponse(BaseModel):
    burnout_risk: str
    score: float

def extract_voice_features(audio_path: str) -> dict:
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        features = {}

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[magnitudes > np.max(magnitudes) * 0.1]
        features['pitch_mean'] = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
        features['pitch_std'] = np.std(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(mfccs.shape[0]):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])

        return features

    except Exception as e:
        raise Exception(f"Erro ao extrair características do áudio: {str(e)}")

def analyze_burnout_risk(features: dict) -> tuple[str, float]:
    risk_score = 0.0

    if features.get('pitch_mean', 0) < 150:
        risk_score += 0.2
    if features.get('pitch_std', 0) < 20:
        risk_score += 0.1
    if features.get('rms_mean', 0) < 0.02:
        risk_score += 0.15
    if features.get('zcr_mean', 0) > 0.1:
        risk_score += 0.1
    if features.get('spectral_centroid_mean', 0) < 1500:
        risk_score += 0.1

    variability_features = [
        features.get('pitch_std', 0),
        features.get('rms_std', 0),
        features.get('spectral_centroid_std', 0)
    ]
    avg_variability = np.mean(variability_features)
    if avg_variability < np.percentile(variability_features, 25):
        risk_score += 0.15

    mfcc_means = [features.get(f'mfcc_{i}_mean', 0) for i in range(13)]
    if np.mean(np.abs(mfcc_means)) < 5:
        risk_score += 0.1

    import random
    risk_score += random.uniform(-0.1, 0.1)

    risk_score = max(0.0, min(1.0, risk_score))

    if risk_score < 0.4:
        risk_level = "baixo"
    elif risk_score < 0.7:
        risk_level = "médio"
    else:
        risk_level = "alto"

    return risk_level, round(risk_score, 2)

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "Voice Burnout Analysis API is running", "cors": "enabled"}

@app.get("/health")
async def health_check():
    logger.info("Health check called")
    return {"status": "healthy", "version": "1.0.0", "cors": "enabled"}

# Endpoint para testar CORS especificamente
@app.get("/test-cors")
async def test_cors():
    return {"message": "CORS is working!", "timestamp": "2024-01-01"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_voice(file: UploadFile = File(...)):
    logger.info(f"Analyze endpoint called with file: {file.filename}")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")

    allowed_extensions = ['.wav', '.mp3', '.ogg', '.m4a', '.flac']
    file_extension = os.path.splitext(file.filename.lower())[1]
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de arquivo não suportado. Use: {', '.join(allowed_extensions)}"
        )

    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Arquivo muito grande. Máximo 10MB.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        converted_path = temp_file_path + "_converted.wav"
        audio = AudioSegment.from_file(temp_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(converted_path, format="wav")

        try:
            features = extract_voice_features(converted_path)
            risk_level, score = analyze_burnout_risk(features)
            logger.info(f"Analysis completed: {risk_level}, {score}")
            return AnalysisResponse(
                burnout_risk=risk_level,
                score=score
            )
        finally:
            for path in [temp_file_path, converted_path]:
                if os.path.exists(path):
                    os.unlink(path)

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar áudio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
