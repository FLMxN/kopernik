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

let res;
let image_name = "lol.png";

function App() {
  const [pretty, setPretty] = useState(1)
  const [output, setOutput] = useState()
  const [pic, setImage] = useState()
  const [origUrl, setOrigUrl] = useState()
  const [fullscreenImage, setFullscreenImage] = useState(null)

  async function form(image, pretty) {
  try {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('pretty', pretty % 2);
    formData.append('act', 'run');

    const response = await fetch("http://127.0.0.1:8000/inference", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("response:", data);
    setOutput(data)
  } catch (err) {
    console.error("fetch error:", err);
  }
}

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
      {output ? (
        <div>
          <h2>Inference results</h2>

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
            {output.country_predictions?.gradcam && (
              <div className="output-panel">
                <img
                  src={`http://127.0.0.1:8000${output.country_predictions.gradcam}`}
                  alt="country gradcam"
                  className="output-image output-image-clickable"
                  onClick={() => setFullscreenImage(`http://127.0.0.1:8000${output.country_predictions.gradcam}`)}
                />
                {output.country_predictions?.predictions && (
                  <div className="output-table-card">
                    <table>
                      <tbody>
                        {Object.entries(output.country_predictions.predictions).slice(0, 8).map(([key, value]) => (
                          <tr key={key}>
                            <td>{key}</td>
                            <td style={{ paddingLeft: '12px', fontWeight: 700 }}>{value.toFixed(2)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {output.region_predictions?.gradcam && (
              <div className="output-panel">
                <img
                  src={`http://127.0.0.1:8000${output.region_predictions.gradcam}`}
                  alt="region gradcam"
                  className="output-image output-image-clickable"
                  onClick={() => setFullscreenImage(`http://127.0.0.1:8000${output.region_predictions.gradcam}`)}
                />
                {output.region_predictions?.predictions && (
                  <div className="output-table-card">
                    <table>
                      <tbody>
                        {Object.entries(output.region_predictions.predictions).slice(0, 8).map(([key, value]) => (
                          <tr key={key}>
                            <td>{key}</td>
                            <td style={{ paddingLeft: '12px', fontWeight: 700 }}>{value.toFixed(2)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      ) : (
        <pre></pre>
      )}
    </div>
    <div className='Dropzone'>
    <FileDropzone onFile={file => { setImage(file); setOrigUrl(URL.createObjectURL(file)); }}></FileDropzone>
    </div>
    <div className='run'>
      <label className='prettyswitch'>
          pretty
          <md-switch
            selected={!!pretty}
            onClick={() => setPretty(prev => 1 - prev)}>
          </md-switch>
        </label>
  <md-elevated-button onClick={() => form(pic, pretty)} disabled={!pic}>
        run
        <svg slot="icon" viewBox="0 0 48 48"><path d="M6 40V8l38 16Zm3-4.65L36.2 24 9 12.5v8.4L21.1 24 9 27Zm0 0V12.5 27Z"/></svg>
        </md-elevated-button>
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
