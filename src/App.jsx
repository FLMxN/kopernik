import { useState } from 'react'
import kopernikLogo from './assets/kopernik.svg'
import './App.css'
import '@material/web/button/elevated-button.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/filled-tonal-button.js';
import '@material/web/button/outlined-button.js';
import '@material/web/button/text-button.js';
import '@material/web/switch/switch.js';

let image_name = "lol.png";
const pretty = 1;

async function form(image_name, pretty) {
  try {
    const requestBody = new URLSearchParams({
      image: image_name,
      pretty: pretty%2,
      act: "run"
    });

    const response = await fetch("http://127.0.0.1:8000/inference", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: requestBody.toString(),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("response:", data);
  } catch (err) {
    console.error("fetch error:", err);
  }
}


function App() {
  const [pretty, setPretty] = useState(0)

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
      <md-outlined-button onClick={() => window.open("https://github.com/FLMxN/kopernik/blob/main/README.md", "_blank")}>readme</md-outlined-button>
      <md-outlined-button onClick={() => window.open("https://github.com/FLMxN/kopernik/blob/main/LICENSE", "_blank")}>license</md-outlined-button>
      </div>
    </div>
    <div className='run'>
      <label className='prettyswitch'>
          pretty
          <md-switch
            selected={!!pretty}
            onClick={() => setPretty(prev => 1 - prev)}>
          </md-switch>
        </label>
      <md-elevated-button onClick={() => form(image_name, pretty)}>run</md-elevated-button>
    </div>
    </div>
    </>
  )
}

export default App
