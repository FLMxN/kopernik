import { useEffect, useRef, useState } from "react";
import "@material/web/button/filled-button.js";
import '@material/web/button/outlined-button.js';
import './App.css'


function FileDropzone({ onFile }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [notification, setNotification] = useState("");

  useEffect(() => {
    if (!notification) return undefined;

    const timeoutId = setTimeout(() => {
      setNotification("");
    }, 2200);

    return () => clearTimeout(timeoutId);
  }, [notification]);


  function handleFiles(fileInput) {
    const uploadedFile = fileInput?.[0] ?? fileInput;
    if (!uploadedFile) return;

    onFile(uploadedFile);
    setNotification(`Loaded ${uploadedFile.name}`);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  function handleInputChange(e) {
    handleFiles(e.target.files);
  }

  function openFilePicker() {
    if (!inputRef.current) return;

    inputRef.current.value = "";
    inputRef.current.click();
  }

  return (
    <div className="dropzone-shell">
      <div
        className="dropzone-panel"
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
          border: dragging
            ? "2px solid var(--md-sys-color-primary)"
            : "2px dashed var(--md-sys-color-outline)",
          background: dragging
            ? "var(--md-sys-color-primary-container)"
            : "var(--md-sys-color-surface)"
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={handleInputChange}
        />

        <p className="dropzone-text">
          drag & drop a file here
        </p>

        <md-outlined-button
          onClick={openFilePicker}
        >
          pick an image
        </md-outlined-button>
      </div>

      <div className={`dropzone-notification ${notification ? 'visible' : ''}`}>
        {notification}
      </div>
    </div>
  );
}

export default FileDropzone;
