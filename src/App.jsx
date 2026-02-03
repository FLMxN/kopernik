import { useState } from 'react'
import kopernikLogo from './assets/kopernik.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
    <div className='header'>
      <div className='logotitle'>
      <p className="kopernik-title">Kopernik</p>
      <a href="https://github.com/FLMxN" target="_blank">
          <img src={kopernikLogo} className="logo kopernik" alt="Kopernik logo" />
        </a>
      </div>
      <div className='links'>
      <a href="https://github.com/FLMxN/Kopernik#readme" target="_blank">
      <p className="info">
        readme
      </p>
      </a>

      <a href="https://github.com/FLMxN/kopernik/blob/main/LICENSE" target="_blank">
      <p className="info">
        license
      </p>
      </a>
      </div>
    </div>
    </>
  )
}

export default App
