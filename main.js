const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const { exec } = require("child_process");
const os = require("os");
const net = require("net");

let py = null;
let windowCreated = false;
let venvPython

function getOS() {
  switch (process.platform) {
    case "win32":
      return "Windows";
    case "darwin":
      return "MacOS";
    case "linux":
      return "Linux";
    default:
      return "Unknown OS";
  }
}

function waitForPort(
  port,
  host = "127.0.0.1",
  maxWaitMs = 120_000,   // 2 minutes, be generous
  intervalMs = 300
) {
  const start = Date.now();

  return new Promise((resolve, reject) => {
    const tryConnect = () => {
      const socket = new net.Socket();

      socket
        .once("connect", () => {
          socket.destroy();
          resolve();
        })
        .once("error", () => {
          socket.destroy();
          if (Date.now() - start >= maxWaitMs) {
            reject(new Error("streamlit startup timeout"));
          } else {
            setTimeout(tryConnect, intervalMs);
          }
        })
        .connect(port, host);
    };

    tryConnect();
  });
}

function killPythonTree() {
  if (!py || py.killed) return;

  try {
    if (os.platform() === "win32") {
      exec(`taskkill /PID ${py.pid} /T /F`);
    } else {
      process.kill(-py.pid, "SIGTERM"); // negative = process group

      setTimeout(() => {
        try {
          process.kill(-py.pid, "SIGKILL");
        } catch {}
      }, 2000);
    }
  } catch {}
}

function startPython() {
  if (getOS() === "Windows") {
    venvPython = path.join(__dirname, "venv", "Scripts", "pythonw.exe");
  } else {
    venvPython = path.join(__dirname, "venv", "bin", "python");
  }

  py = spawn(
    venvPython,
    ["run.py"],
    {
      env: {
        ...process.env,
        VIRTUAL_ENV: path.join(__dirname, "venv"),
        PATH:
          getOS() === "Windows"
            ? path.join(__dirname, "venv", "Scripts") + ";" + process.env.PATH
            : path.join(__dirname, "venv", "bin") + ":" + process.env.PATH
      },
      detached: true, 
      stdio: ["pipe", "pipe", "pipe"]
    }
  );

  py.unref();
}


function createWindow() {
  const win = new BrowserWindow({ width: 1000, height: 700,
    // titleBarStyle: 'hidden',
    // ...(process.platform !== 'darwin' ? { titleBarOverlay: true } : {})
  });
  // win.loadFile('front/index.html')
  win.loadURL("http://127.0.0.1:8501")
}

app.whenReady().then(() => {
  startPython();
  waitForPort(8501).then(createWindow);
  // createWindow();
});

app.on("before-quit", killPythonTree);
app.on("will-quit", killPythonTree);

process.on("exit", killPythonTree);
process.on("SIGINT", killPythonTree);
process.on("SIGTERM", killPythonTree);
process.on("uncaughtException", killPythonTree);
