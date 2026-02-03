const { app, BrowserWindow } = require("electron");
const net = require("net");
const { exec } = require("child_process");

function waitForPort(
  port,
  host = "127.0.0.1",
  maxWaitMs = 30000,
  intervalMs = 500
) {
  const start = Date.now();
  
  console.log(`Waiting for ${host}:${port} to be available...`);

  return new Promise((resolve, reject) => {
    const tryConnect = () => {
      const socket = new net.Socket();
      const timeout = setTimeout(() => {
        socket.destroy();
      }, 1000);

      socket
        .once("connect", () => {
          clearTimeout(timeout);
          socket.destroy();
          console.log(`Connected to ${host}:${port}`);
          resolve();
        })
        .once("error", (err) => {
          clearTimeout(timeout);
          socket.destroy();
          const elapsed = Date.now() - start;
          
          if (elapsed >= maxWaitMs) {
            console.error(`Timeout after ${elapsed}ms waiting for ${host}:${port}`);
            reject(new Error(`Port ${port} not available after ${maxWaitMs}ms`));
          } else {
            setTimeout(tryConnect, intervalMs);
          }
        })
        .connect(port, host);
    };

    tryConnect();
  });
}

function createWindow() {
  console.log("Creating Electron window...");
  const win = new BrowserWindow({
    width: 1000,
    height: 700,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  win.loadFile("src/loading.html");

  const urls = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
  ];

  let currentIndex = 0;
  
  const tryLoadUrl = () => {
    if (currentIndex >= urls.length) {
      console.error("All URLs failed to load");
      return;
    }
    
    const url = urls[currentIndex];
    console.log(`Trying to load: ${url}`);
    
    win.loadURL(url).catch((err) => {
      console.error(`Failed to load ${url}:`, err.message);
      currentIndex++;
      setTimeout(tryLoadUrl, 100);
    });
  };

  setTimeout(tryLoadUrl, 10000);

}

app.whenReady().then(() => {
  console.log("Electron app is ready");
  
  waitForPort(5173)
    .then(() => {
      console.log("Port 5173 is ready, creating window...");
      setTimeout(createWindow, 2000);
    })
    // .catch((err) => {
    //   console.error("Failed to start:", err.message);
    //   console.log("Trying to create window anyway...");
    //   createWindow();
    // });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
    exec("npx kill-port 5173")
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    setTimeout(createWindow, 2000);
  }
});