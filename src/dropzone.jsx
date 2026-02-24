import { useRef, useState } from "react";
import "@material/web/button/filled-button.js";
import '@material/web/button/outlined-button.js';
import './App.css'


function FileDropzone({ onFile }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);

    function onFileChange(e) {
    const uploadedFile = e.target.files[0];
    setFile(uploadedFile);
    }


  function handleFiles(files) {
    if (!files || !files[0]) return;
    onFile(files[0]);
  }

  return (
    <div
      onDragOver={e => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => {
        e.preventDefault();
        setDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      style={{
        borderRadius: "16px",
        padding: "2rem",
        textAlign: "center",
        border: dragging
          ? "2px solid var(--md-sys-color-primary)"
          : "2px dashed var(--md-sys-color-outline)",
        background: dragging
          ? "var(--md-sys-color-primary-container)"
          : "var(--md-sys-color-surface)",
        color: "var(--md-sys-color-on-surface)",
        transition: "all 120ms ease"
      }}
    >
      <input
        ref={inputRef}
        type="file"
        hidden
        onChange={e => handleFiles(e.target.files[0])}
      />

      <p className="dropzone-text">
        drag & drop a file here
      </p>

      <md-outlined-button
        onClick={() => inputRef.current.click()}
      >
        pick an image
      </md-outlined-button>
    </div>
  );
}

export default FileDropzone;
