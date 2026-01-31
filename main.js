const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

let py = null;
let windowCreated = false;
let venvPython;

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

function startPython() {
  // const exePath = path.join(process.resourcesPath, "backend", "main.exe");
  // pyProcess = spawn(exePath);

  // pyProcess.stdout.on("data", d => console.log("py:", d.toString()));
  // pyProcess.stderr.on("data", d => console.error("py err:", d.toString()));

  if (getOS() == "Windows") {
    venvPython = path.join(
    __dirname,
    "venv",
    "Scripts",
    "python.exe"
  );
  } 
  else {
      venvPython = path.join(
        __dirname,
        "venv",
        "bin",
        "python"
      );
    }

  py = spawn(
    venvPython,
    ["run.py"],
    {
      env: {
        ...process.env,
        VIRTUAL_ENV: path.join(__dirname, "venv"),
        PATH: path.join(__dirname, "venv", "Scripts") + ";" + process.env.PATH
      },
      stdio: ["pipe", "pipe", "pipe"]
    }
  );


}

function createWindow() {
  const win = new BrowserWindow({ width: 1000, height: 700 });
  win.loadURL("http://127.0.0.1:8501")
}

app.whenReady().then(() => {
  startPython();
  py.stdout.on('data', (data) => {
    const text = data.toString();
    // console.log("py:", text);
    if (!windowCreated && text.includes("URL")) {
      windowCreated = true;
      setTimeout(createWindow, 100);
    }
  });
  // setTimeout(createWindow, 4000);
});

app.on("window-all-closed", () => {
  if (py) {
    // py.stdin.write("SHUTDOWN\n");
    py.kill();
  }
  app.quit();
});
