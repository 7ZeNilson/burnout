// Copie e cole este bloco inteiro no arquivo src/App.js

import React, { useState, useRef } from 'react';
import { Upload, Mic, Play, Pause, Trash2, Send, RotateCcw, Loader2, AlertCircle } from 'lucide-react';
import { format } from 'date-fns';

// Configuração da API (deve corresponder à porta do seu backend Python)
const API_BASE_URL = 'http://localhost:8000';

// Função para chamar a API real
const callVoiceAnalysisAPI = async (audioFile) => {
  const formData = new FormData();
  formData.append('file', audioFile, audioFile.name || 'recording.wav');

  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Erro na API: ${response.status} - ${errorText}`);
  }
  return await response.json();
};

function App() {
  const [appState, setAppState] = useState('welcome'); // welcome, recording, loading, result, error
  const [analyses, setAnalyses] = useState([]);
  const [currentResult, setCurrentResult] = useState(null);
  const [apiError, setApiError] = useState(null);

  const handleAnalyze = async ({ audioBlob, file }) => {
    setAppState('loading');
    setApiError(null);
    
    const inputType = file ? 'upload' : 'gravacao';
    const fileName = file ? file.name : null;
    const audioFile = file || audioBlob;

    try {
      const result = await callVoiceAnalysisAPI(audioFile);
      
      const newAnalysis = {
        id: Date.now(),
        risk_level: result.burnout_risk,
        score: result.score,
        inputType,
        fileName,
        created_date: new Date().toISOString()
      };
      
      setAnalyses(prev => [newAnalysis, ...prev]);
      setCurrentResult(result);
      setAppState('result');
    } catch (error) {
      console.error('Erro ao analisar áudio:', error);
      setApiError(error.message);
      setAppState('error');
    }
  };

  const handleStart = () => setAppState('recording');
  const handleAnalyzeAgain = () => {
    setCurrentResult(null);
    setApiError(null);
    setAppState('recording');
  };

  const renderContent = () => {
    switch (appState) {
      case 'welcome': return <WelcomeScreen onStart={handleStart} />;
      case 'recording': return <AnalysisRecorder onAnalyze={handleAnalyze} />;
      case 'loading': return <LoadingScreen />;
      case 'result': return <AnalysisResult result={currentResult} onAnalyzeAgain={handleAnalyzeAgain} />;
      case 'error': return <ErrorScreen error={apiError} onRetry={handleAnalyzeAgain} />;
      default: return <WelcomeScreen onStart={handleStart} />;
    }
  };

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={styles.headerTitle}>Análise de Burnout por Voz</h1>
      </header>
      <main style={styles.main}>
        {renderContent()}
        <AnalysisHistory analyses={analyses} />
      </main>
      <footer style={styles.footer}>
        <p>&copy; {new Date().getFullYear()} - Cuidando da sua saúde mental.</p>
      </footer>
    </div>
  );
}

// --- COMPONENTES ---

const WelcomeScreen = ({ onStart }) => (
  <div style={styles.card}>
    <h2 style={styles.title}>Sua voz é um reflexo do seu bem-estar.</h2>
    <p style={styles.subtitle}>
      Nossa ferramenta utiliza IA para analisar padrões na sua voz e identificar sinais precoces de burnout.
    </p>
    <button onClick={onStart} style={styles.buttonPrimary}>
      Iniciar Análise <Send size={18} style={{ marginLeft: 8 }} />
    </button>
  </div>
);

const LoadingScreen = () => (
  <div style={{ ...styles.card, ...styles.center }}>
    <Loader2 size={64} style={{ ...styles.icon, animation: 'spin 1s linear infinite' }} />
    <h2 style={styles.title}>Analisando sua voz...</h2>
    <p style={styles.subtitle}>Aguarde, estamos processando o áudio.</p>
  </div>
);

const ErrorScreen = ({ error, onRetry }) => (
  <div style={{ ...styles.card, backgroundColor: '#fff1f2', borderColor: '#fecaca' }}>
    <div style={{ ...styles.center, color: '#dc2626' }}>
      <AlertCircle size={48} style={styles.icon} />
      <h2 style={styles.title}>Ocorreu um Erro</h2>
    </div>
    <p style={{...styles.subtitle, color: '#b91c1c', backgroundColor: '#fee2e2', padding: 12, borderRadius: 6}}>
      <strong>Detalhes:</strong> {error || 'Não foi possível conectar à API. Verifique se o backend está rodando.'}
    </p>
    <button onClick={onRetry} style={styles.buttonSecondary}>
      Tentar Novamente <RotateCcw size={18} style={{ marginLeft: 8 }} />
    </button>
  </div>
);

const AnalysisResult = ({ result, onAnalyzeAgain }) => {
  const config = {
    baixo: { title: 'Risco Baixo', color: '#16a34a', bgColor: '#f0fdf4', borderColor: '#bbf7d0', icon: <Play /> },
    médio: { title: 'Risco Médio', color: '#f59e0b', bgColor: '#fffbeb', borderColor: '#fde68a', icon: <Pause /> },
    alto: { title: 'Risco Alto', color: '#dc2626', bgColor: '#fef2f2', borderColor: '#fecaca', icon: <AlertCircle /> },
  };
  const currentConfig = config[result.burnout_risk] || config.médio;

  return (
    <div style={{ ...styles.card, backgroundColor: currentConfig.bgColor, borderColor: currentConfig.borderColor }}>
      <h2 style={{ ...styles.title, color: currentConfig.color }}>{currentConfig.title} de Burnout</h2>
      <p style={styles.subtitle}>
        O score de risco calculado foi <strong>{result.score.toFixed(2)}</strong>.
      </p>
      <div style={{ margin: '24px 0' }}>
        <p style={{...styles.subtitle, fontStyle: 'italic'}}>Lembre-se: esta é uma ferramenta de triagem e não substitui o diagnóstico de um profissional de saúde qualificado.</p>
      </div>
      <button onClick={onAnalyzeAgain} style={styles.buttonSecondary}>
        Realizar Nova Análise <RotateCcw size={18} style={{ marginLeft: 8 }} />
      </button>
    </div>
  );
};


const AnalysisHistory = ({ analyses }) => {
  return (
    <div style={{ ...styles.card, marginTop: 24 }}>
      <h2 style={{...styles.title, marginBottom: 16 }}>Histórico de Análises</h2>
      {analyses.length === 0 ? (
        <p style={styles.subtitle}>Nenhuma análise realizada ainda.</p>
      ) : (
        <table style={styles.table}>
            <thead>
                <tr>
                    <th style={styles.th}>Data</th>
                    <th style={styles.th}>Origem</th>
                    <th style={styles.th}>Risco</th>
                    <th style={{...styles.th, textAlign: 'right'}}>Score</th>
                </tr>
            </thead>
            <tbody>
                {analyses.map(analysis => (
                    <tr key={analysis.id}>
                        <td style={styles.td}>{format(new Date(analysis.created_date), 'dd/MM/yyyy HH:mm')}</td>
                        <td style={styles.td}>{analysis.inputType}</td>
                        <td style={styles.td}>{analysis.risk_level}</td>
                        <td style={{...styles.td, textAlign: 'right'}}>{analysis.score.toFixed(2)}</td>
                    </tr>
                ))}
            </tbody>
        </table>
      )}
    </div>
  );
};


const AnalysisRecorder = ({ onAnalyze }) => {
  const [audioSource, setAudioSource] = useState(null);
  const [audioData, setAudioData] = useState(null); // File ou Blob
  const [error, setError] = useState('');
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        setError('Arquivo muito grande. Limite de 10MB.');
        return;
      }
      setError('');
      setAudioData(file);
      setAudioSource('upload');
    }
  };

  const handleRecord = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = event => {
        audioChunksRef.current.push(event.data);
      };
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioData(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };
      mediaRecorderRef.current.start();
      setAudioSource('record');
    } catch (err) {
      setError('Permissão do microfone negada. Verifique as configurações do seu navegador.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
  };
  
  const reset = () => {
    setAudioSource(null);
    setAudioData(null);
    setError('');
  };

  const isRecording = mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording';

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>Como você se sente hoje?</h2>
      <p style={styles.subtitle}>Grave um áudio ou envie um arquivo de voz.</p>
      
      {error && <p style={{color: 'red'}}>{error}</p>}
      
      {!audioSource && (
        <div style={styles.buttonGroup}>
          <label style={{ ...styles.buttonSecondary, cursor: 'pointer' }}>
            <Upload size={18} style={{ marginRight: 8 }}/> Enviar Arquivo
            <input type="file" accept="audio/*" onChange={handleFileChange} style={{ display: 'none' }} />
          </label>
          <button onClick={handleRecord} style={styles.buttonSecondary}>
            <Mic size={18} style={{ marginRight: 8 }}/> Gravar com Microfone
          </button>
        </div>
      )}

      {audioSource === 'record' && (
        <div style={styles.center}>
          {isRecording ? (
            <button onClick={stopRecording} style={{...styles.buttonPrimary, backgroundColor: '#dc2626'}}>
              <Pause size={18} style={{ marginRight: 8 }}/> Parar Gravação
            </button>
          ) : (
            <p style={styles.subtitle}>Gravação concluída!</p>
          )}
        </div>
      )}

      {audioData && (
        <div style={{ marginTop: 24, borderTop: '1px solid #e2e8f0', paddingTop: 24 }}>
          <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 16}}>
            <button onClick={() => onAnalyze({ audioBlob: audioData, file: audioSource === 'upload' ? audioData : null })} style={styles.buttonPrimary}>
              <Send size={18} style={{ marginRight: 8 }}/> Enviar para Análise
            </button>
            <button onClick={reset} style={{...styles.buttonSecondary, backgroundColor: '#fee2e2', color: '#b91c1c', borderColor: '#fecaca'}}>
              <Trash2 size={18} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};


// --- ESTILOS CSS-in-JS ---

const styles = {
  page: { display: 'flex', flexDirection: 'column', minHeight: '100vh', backgroundColor: '#f1f5f9' },
  header: { backgroundColor: 'white', padding: '16px 32px', borderBottom: '1px solid #e2e8f0' },
  headerTitle: { margin: 0, color: '#1e293b', fontSize: 24 },
  main: { flex: 1, width: '100%', maxWidth: 800, margin: '32px auto', padding: '0 16px' },
  footer: { textAlign: 'center', padding: '16px', color: '#64748b', fontSize: 14 },
  card: { backgroundColor: 'white', padding: 32, borderRadius: 12, border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)', textAlign: 'center' },
  title: { margin: '0 0 12px 0', color: '#1e293b', fontSize: 28 },
  subtitle: { margin: 0, color: '#64748b', lineHeight: 1.6 },
  buttonGroup: { display: 'flex', gap: 16, justifyContent: 'center', margin: '24px 0' },
  buttonPrimary: { backgroundColor: '#3b82f6', color: 'white', border: 'none', padding: '12px 24px', borderRadius: 8, fontSize: 16, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', fontWeight: 600 },
  buttonSecondary: { backgroundColor: 'white', color: '#475569', border: '1px solid #cbd5e1', padding: '12px 24px', borderRadius: 8, fontSize: 16, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', fontWeight: 600 },
  icon: { marginBottom: 16, color: '#3b82f6' },
  center: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' },
  table: { width: '100%', borderCollapse: 'collapse', marginTop: 16 },
  th: { borderBottom: '2px solid #e2e8f0', padding: '12px 8px', textAlign: 'left', color: '#475569', fontWeight: 600 },
  td: { borderBottom: '1px solid #e2e8f0', padding: '12px 8px', color: '#334155' }
};

export default App;