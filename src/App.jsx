import { useState } from 'react'
import kopernikLogo from './assets/kopernik.svg'
import './App.css'
import '@material/web/button/elevated-button.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/filled-tonal-button.js';
import '@material/web/button/outlined-button.js';
import '@material/web/button/text-button.js';
import '@material/web/switch/switch.js';
import FileDropzone from './dropzone.jsx'
import { flagEmoji, predictLocal } from './onnx/inference.js'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const INFERENCE_MODE = import.meta.env.VITE_INFERENCE_MODE
  || (API_BASE_URL ? "backend" : "onnx");

function parseBackendResult(data) {
  const country = Object.entries(data.country_predictions?.predictions || {}).map(
    ([label, value]) => {
      const match = label.match(/\(([A-Z]{2}|UNDEFINED)\)\s*$/);
      const code = match ? match[1] : "";
      return {
        code,
        name: label.replace(/\s*\([^)]*\)\s*$/, ""),
        label,
        prob: Number(value) / 100,
        flag: flagEmoji(code),
      };
    },
  );
  const region = Object.entries(data.region_predictions?.predictions || {}).map(
    ([name, value]) => ({
      name,
      prob: Number(value) / 100,
      emoji: "🌍",
    }),
  );
  return { country, region, provider: "server", elapsedMs: null };
}

function App() {
  const [pretty, setPretty] = useState(1)
  const [result, setResult] = useState(null)
  const [pic, setImage] = useState()
  const [origUrl, setOrigUrl] = useState()
  const [fullscreenImage, setFullscreenImage] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [status, setStatus] = useState("")
  const [error, setError] = useState("")

  async function form(image, prettyValue) {
    try {
      setError("");
      setResult(null);
      setIsLoading(true);
      setStatus(INFERENCE_MODE === "onnx" ? "Loading models..." : "Sending image to server...");

      if (INFERENCE_MODE === "onnx") {
        const data = await predictLocal(image, prettyValue % 2, setStatus);
        setResult(data);
      } else {
        const formData = new FormData();
        formData.append('image', image);
        formData.append('pretty', prettyValue % 2);
        formData.append('act', 'run');

        const response = await fetch(`${API_BASE_URL}/api/inference`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setResult(parseBackendResult(data));
      }
    } catch (err) {
      console.error("inference error:", err);
      setError(err?.message || "Request failed");
    } finally {
      setIsLoading(false);
      setStatus("");
    }
  }

  const modeLabel = INFERENCE_MODE === "onnx"
    ? "static ONNX (WebGPU/WASM)"
    : "server inference";

  return (
    <>
    <div className='body'>
    <div className='header'>
      <div className='logotitle'>
      <p className="kopernik-title">Kopernik</p>
      <a href="https://github.com/FLMxN" target="_blank">
          <img src={kopernikLogo} className="logo kopernik" alt="Kopernik logo" />
        </a>
      </div>
      <div className='links'>
      <md-outlined-button onClick={() => window.open("https://github.com/FLMxN/kopernik/blob/main/README.md", "_blank")}>readme<svg slot="icon" viewBox="0 0 48 48"><path d="M9 42q-1.2 0-2.1-.9Q6 40.2 6 39V9q0-1.2.9-2.1Q7.8 6 9 6h13.95v3H9v30h30V25.05h3V39q0 1.2-.9 2.1-.9.9-2.1.9Zm10.1-10.95L17 28.9 36.9 9H25.95V6H42v16.05h-3v-10.9Z"/></svg></md-outlined-button>
      <md-outlined-button onClick={() => window.open("https://github.com/FLMxN/kopernik/blob/main/LICENSE", "_blank")}>license<svg slot="icon" viewBox="0 0 48 48"><path d="M9 42q-1.2 0-2.1-.9Q6 40.2 6 39V9q0-1.2.9-2.1Q7.8 6 9 6h13.95v3H9v30h30V25.05h3V39q0 1.2-.9 2.1-.9.9-2.1.9Zm10.1-10.95L17 28.9 36.9 9H25.95V6H42v16.05h-3v-10.9Z"/></svg></md-outlined-button>
      </div>
    </div>
    <div className="output">
      {result ? (
        <div>
          <h2>Inference results</h2>
          <p style={{ opacity: 0.65, marginTop: '-4px' }}>
            {result.provider}
            {result.elapsedMs != null ? ` · ${(result.elapsedMs / 1000).toFixed(2)} s` : ''}
          </p>

          {origUrl && (
            <div className="output-original">
              <img
                src={origUrl}
                alt="original"
                className="output-image output-image-clickable"
                onClick={() => setFullscreenImage(origUrl)}
              />
            </div>
          )}

          <div className="output-results-row">
            <div className="output-panel">
              <h3>Country</h3>
              <div className="output-table-card">
                <table>
                  <tbody>
                    {result.country.map((item, index) => (
                      <tr key={`${item.code}-${item.name}-${index}`}>
                        <td>
                          <span style={{ marginRight: '8px' }}>{item.flag}</span>
                          {item.name || item.label}
                          {item.code ? ` (${item.code})` : ''}
                        </td>
                        <td style={{ paddingLeft: '12px', fontWeight: 700 }}>
                          {(item.prob * 100).toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="output-panel">
              <h3>Region</h3>
              <div className="output-table-card">
                <table>
                  <tbody>
                    {result.region.map((item, index) => (
                      <tr key={`${item.name}-${index}`}>
                        <td>
                          <span style={{ marginRight: '8px' }}>{item.emoji || '🌍'}</span>
                          {item.name}
                        </td>
                        <td style={{ paddingLeft: '12px', fontWeight: 700 }}>
                          {(item.prob * 100).toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      ) : (
        origUrl ? (
          <div className="output-original">
            <img
              src={origUrl}
              alt="original"
              className="output-image output-image-clickable"
              onClick={() => setFullscreenImage(origUrl)}
            />
          </div>
        ) : (
          <pre></pre>
        )
      )}
    </div>
    <div className='Dropzone'>
    <FileDropzone onFile={file => {
      setImage(file);
      setOrigUrl(URL.createObjectURL(file));
      setResult(null);
      setError("");
    }}></FileDropzone>
    </div>
    <div className='run'>
      <label className='prettyswitch'>
          <span className='prettyswitch-label'>pretty</span>
          <md-switch
            selected={!!pretty}
            onClick={() => setPretty(prev => 1 - prev)}>
          </md-switch>
        </label>
      {isLoading && (
        <progress className="inference-progress" max="100" value={undefined}></progress>
      )}
      {isLoading && (
        <span style={{ opacity: 0.7, fontSize: '0.8rem' }}>
          {status || 'Working...'}
        </span>
      )}
      {error && (
        <span style={{ color: '#ff6b6b', fontSize: '0.8rem', maxWidth: '520px' }}>
          {error}
        </span>
      )}
  <md-elevated-button onClick={() => form(pic, pretty)} disabled={!pic || isLoading}>
        {isLoading ? 'Running...' : 'run'}
        <svg slot="icon" viewBox="0 0 48 48"><path d="M6 40V8l38 16Zm3-4.65L36.2 24 9 12.5v8.4L21.1 24 9 27Zm0 0V12.5 27Z"/></svg>
        </md-elevated-button>
      <span style={{ opacity: 0.45, fontSize: '0.75rem' }}>{modeLabel}</span>
    </div>
    {fullscreenImage && (
      <div className="image-modal" onClick={() => setFullscreenImage(null)}>
        <button
          type="button"
          className="image-modal-close"
          onClick={() => setFullscreenImage(null)}
        >
          close
        </button>
        <img
          src={fullscreenImage}
          alt="fullscreen preview"
          className="image-modal-content"
          onClick={e => e.stopPropagation()}
        />
      </div>
    )}
    </div>
    </>
  )
}

export default App
