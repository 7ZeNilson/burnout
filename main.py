# main.py - VERSÃO CORRIGIDA
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
from pydantic import BaseModel

app = FastAPI(title="Voice Burnout Analysis API")

# Configurar CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://burnout-doi7.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResponse(BaseModel):
    burnout_risk: str
    score: float

def extract_voice_features(audio_path: str) -> dict:
    """
    Extrai características do áudio usando librosa
    """
    try:
        # Carrega o áudio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Características básicas
        features = {}
        
        # 1. Características espectrais
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # 2. Características de pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[magnitudes > np.max(magnitudes) * 0.1]
        features['pitch_mean'] = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
        features['pitch_std'] = np.std(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
        
        # 3. Zero crossing rate (indica vozeado vs não-vozeado)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 4. Energia RMS
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # 5. Tempo e ritmo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # 6. MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(mfccs.shape[0]):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        return features
        
    except Exception as e:
        raise Exception(f"Erro ao extrair características do áudio: {str(e)}")

def analyze_burnout_risk(features: dict) -> tuple[str, float]:
    """
    Análise simplificada de burnout baseada em características da voz
    NOTA: Esta é uma implementação de exemplo. Em produção, você usaria um modelo ML treinado.
    """
    
    # Regras heurísticas baseadas em pesquisas sobre burnout e características vocais
    risk_score = 0.0
    
    # 1. Pitch (pessoas com burnout tendem a ter pitch mais baixo e menos variável)
    if features.get('pitch_mean', 0) < 150:  # Pitch baixo
        risk_score += 0.2
    if features.get('pitch_std', 0) < 20:    # Pouca variação no pitch
        risk_score += 0.1
    
    # 2. Energia vocal (fadiga vocal)
    if features.get('rms_mean', 0) < 0.02:   # Energia baixa
        risk_score += 0.15
    
    # 3. Taxa de cruzamento por zero (qualidade vocal)
    if features.get('zcr_mean', 0) > 0.1:    # Voz mais "aérea"
        risk_score += 0.1
    
    # 4. Centroide espectral (brilho da voz)
    if features.get('spectral_centroid_mean', 0) < 1500:  # Voz menos brilhante
        risk_score += 0.1
    
    # 5. Variabilidade geral (monotonia)
    variability_features = [
        features.get('pitch_std', 0),
        features.get('rms_std', 0),
        features.get('spectral_centroid_std', 0)
    ]
    avg_variability = np.mean(variability_features)
    if avg_variability < np.percentile(variability_features, 25):  # Baixa variabilidade
        risk_score += 0.15
    
    # 6. MFCCs (padrões de fala)
    mfcc_means = [features.get(f'mfcc_{i}_mean', 0) for i in range(13)]
    if np.mean(np.abs(mfcc_means)) < 5:  # Padrões de fala menos expressivos
        risk_score += 0.1
    
    # Adiciona um pouco de aleatoriedade para simular variabilidade natural
    import random
    risk_score += random.uniform(-0.1, 0.1)
    
    # Normaliza o score entre 0 e 1
    risk_score = max(0.0, min(1.0, risk_score))
    
    # Determina o nível de risco
    if risk_score < 0.4:
        risk_level = "baixo"
    elif risk_score < 0.7:
        risk_level = "médio"
    else:
        risk_level = "alto"
    
    return risk_level, round(risk_score, 2)

@app.get("/")
async def root():
    return {"message": "Voice Burnout Analysis API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_voice(file: UploadFile = File(...)):
    """
    Analisa um arquivo de áudio para detectar sinais de burnout
    """
    
    # Validação do arquivo
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")
    
    # Verifica extensão do arquivo
    allowed_extensions = ['.wav', '.mp3', '.ogg', '.m4a', '.flac']
    file_extension = os.path.splitext(file.filename.lower())[1]
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Formato de arquivo não suportado. Use: {', '.join(allowed_extensions)}"
        )
    
    # Verifica tamanho do arquivo (máximo 10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Arquivo muito grande. Máximo 10MB.")
    
    try:
        # Salva o arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extrai características da voz
            features = extract_voice_features(temp_file_path)
            
            # Analisa o risco de burnout
            risk_level, score = analyze_burnout_risk(features)
            
            return AnalysisResponse(
                burnout_risk=risk_level,
                score=score
            )
            
        finally:
            # Remove arquivo temporário
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar áudio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
