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
  const [pretty, setPretty] = useState(0)
  const [output, setOutput] = useState()
  const [pic, setImage] = useState()

  async function form(image, pretty) {
  try {
    const requestBody = new URLSearchParams({
      image: image,
      pretty: pretty%2,
      act: "run"
    });

    const response = await fetch("http://127.0.0.1:8000/inference", {
      method: "POST",
      body: requestBody.toString(),
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
    <div className='Dropzone'>
    <FileDropzone onFile={file => setImage(file)}></FileDropzone>
    </div>
    <div className='run'>
      <label className='prettyswitch'>
          pretty
          <md-switch
            selected={!!pretty}
            onClick={() => setPretty(prev => 1 - prev)}>
          </md-switch>
        </label>
      <md-elevated-button onClick={() => form(pic, pretty)}>
        run
        <svg slot="icon" viewBox="0 0 48 48"><path d="M6 40V8l38 16Zm3-4.65L36.2 24 9 12.5v8.4L21.1 24 9 27Zm0 0V12.5 27Z"/></svg>
        </md-elevated-button>
    </div>
    <div className="output">
  <pre>
  {JSON.stringify(output)}
  </pre>
    </div>
    </div>
    </>
  )
}

export default App
