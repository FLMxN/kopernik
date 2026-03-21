const { app, BrowserWindow } = require("electron");
const net = require("net");
const http = require("http");
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

function waitForServer(
  url = "http://localhost:5173",
  maxWaitMs = 60000,
  intervalMs = 1000
) {
  const start = Date.now();
  
  console.log(`Waiting for server at ${url} to be ready...`);

  return new Promise((resolve, reject) => {
    const tryRequest = () => {
      const req = http.get(url, (res) => {
        if (res.statusCode === 200) {
          console.log(`Server at ${url} is ready`);
          resolve();
        } else {
          console.log(`Server responded with status ${res.statusCode}, retrying...`);
          setTimeout(tryRequest, intervalMs);
        }
      });

      req.on("error", (err) => {
        const elapsed = Date.now() - start;
        if (elapsed >= maxWaitMs) {
          console.error(`Timeout after ${elapsed}ms waiting for server at ${url}`);
          reject(new Error(`Server not ready after ${maxWaitMs}ms`));
        } else {
          setTimeout(tryRequest, intervalMs);
        }
      });

      req.setTimeout(5000, () => {
        req.destroy();
        const elapsed = Date.now() - start;
        if (elapsed >= maxWaitMs) {
          reject(new Error(`Server not ready after ${maxWaitMs}ms`));
        } else {
          setTimeout(tryRequest, intervalMs);
        }
      });
    };

    tryRequest();
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

  const url = "http://localhost:5173";
  let retryCount = 0;
  const maxRetries = 50; // Retry up to 50 times
  const retryDelay = 500; // 0.5 second between retries
  
  const tryLoadUrl = () => {
    if (retryCount >= maxRetries) {
      console.error(`Failed to load ${url} after ${maxRetries} retries`);
      return;
    }
    
    console.log(`Trying to load: ${url} (attempt ${retryCount + 1})`);
    
    win.loadURL(url).catch((err) => {
      console.error(`Failed to load ${url}:`, err.message);
      retryCount++;
      setTimeout(tryLoadUrl, retryDelay);
    });
  };

  tryLoadUrl(); // Start trying immediately

}

app.whenReady().then(() => {
  console.log("Electron app is ready");
  
  waitForPort(5173)
    .then(() => {
      console.log("Port 5173 is ready, waiting for server...");
      return waitForServer();
    })
    .then(() => {
      console.log("Server is ready, creating window...");
      createWindow();
    })
    .catch((err) => {
      console.error("Failed to start:", err.message);
      console.log("Trying to create window anyway...");
      createWindow();
    });
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