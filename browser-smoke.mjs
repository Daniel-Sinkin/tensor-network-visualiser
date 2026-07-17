import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));
const profile = await mkdtemp(path.join(os.tmpdir(), "tn-contraction-planner-"));
const mime = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
]);

const server = http.createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    if (pathname === "/favicon.ico") {
      response.writeHead(204);
      response.end();
      return;
    }
    const relative = pathname === "/" ? "contraction-planner.html" : pathname.slice(1);
    const target = path.resolve(root, relative);
    if (!target.startsWith(root)) throw new Error("invalid path");
    const body = await readFile(target);
    response.writeHead(200, { "Content-Type": mime.get(path.extname(target)) || "application/octet-stream" });
    response.end(body);
  } catch (_) {
    response.writeHead(404);
    response.end("not found");
  }
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const webPort = server.address().port;

const debugPort = await new Promise((resolve, reject) => {
  const probe = net.createServer();
  probe.once("error", reject);
  probe.listen(0, "127.0.0.1", () => {
    const port = probe.address().port;
    probe.close((error) => (error ? reject(error) : resolve(port)));
  });
});

const chrome = spawn(
  process.env.CHROME || "/usr/bin/google-chrome",
  [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${profile}`,
    "about:blank",
  ],
  { stdio: ["ignore", "ignore", "ignore"] }
);

let socket;
try {
  let page;
  const deadline = Date.now() + 10000;
  while (!page && Date.now() < deadline) {
    try {
      const pages = await fetch(`http://127.0.0.1:${debugPort}/json/list`).then((response) => response.json());
      page = pages.find((item) => item.type === "page");
    } catch (_) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }
  if (!page) throw new Error("Chrome did not expose a debuggable page");

  socket = new WebSocket(page.webSocketDebuggerUrl);
  const pending = new Map();
  const browserErrors = [];
  let nextId = 1;
  let resolvePageLoad;
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const handlers = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) handlers.reject(new Error(message.error.message));
      else handlers.resolve(message.result);
      return;
    }
    if (message.method === "Runtime.exceptionThrown")
      browserErrors.push(message.params.exceptionDetails.text);
    if (message.method === "Log.entryAdded" && message.params.entry.level === "error")
      browserErrors.push(message.params.entry.text);
    if (message.method === "Page.loadEventFired" && resolvePageLoad) {
      resolvePageLoad();
      resolvePageLoad = undefined;
    }
  });
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });

  function send(method, params = {}) {
    const id = nextId++;
    socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
  }

  async function evaluate(expression) {
    const response = await send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (response.exceptionDetails) throw new Error(response.exceptionDetails.text);
    return response.result.value;
  }

  await send("Page.enable");
  const pageLoaded = new Promise((resolve) => {
    resolvePageLoad = resolve;
  });
  await send("Page.navigate", {
    url: `http://127.0.0.1:${webPort}/contraction-planner.html`,
  });
  await pageLoaded;
  await send("Runtime.enable");
  await send("Log.enable");

  const initial = await evaluate(`new Promise((resolve, reject) => {
    const deadline = Date.now() + 3000;
    const poll = () => {
      if (document.querySelector("#metricFlops").textContent !== "—") return resolve({
        badge: document.querySelector("#validityBadge").textContent,
        tensorEditors: document.querySelectorAll(".tensor-editor").length,
        connections: document.querySelectorAll(".svg-shared-line").length,
        connectionLabels: [...document.querySelectorAll(".svg-wire-label")].map((node) => node.textContent),
        outputRank: document.querySelector("#outputRank").textContent,
        outputLegs: [...document.querySelectorAll("#outputLegs .output-leg")].map((node) => node.textContent),
        flops: document.querySelector("#metricFlops").textContent,
        macs: document.querySelector("#metricMacs").textContent,
        scratch: document.querySelector("#metricScratch").textContent,
        gemm: document.querySelector("#gemmEquation").textContent,
        permutationA: document.querySelector("#permutationA").textContent,
        permutationB: document.querySelector("#permutationB").textContent,
      });
      if (Date.now() > deadline) return reject(new Error("initial workbench did not render"));
      setTimeout(poll, 20);
    };
    poll();
  })`);
  assert.deepEqual(initial, {
    badge: "Valid contraction",
    tensorEditors: 2,
    connections: 2,
    connectionLabels: ["spin:3", "bond:5"],
    outputRank: "rank 3 · [2 × 7 × 11]",
    outputLegs: ["axis 0 · batch : 2", "axis 1 · left : 7", "axis 2 · out : 11"],
    flops: "18,480",
    macs: "2,310",
    scratch: "2.92 KiB",
    gemm: "(14 × 15) · (15 × 11) → (14 × 11)",
    permutationA: "axes [0, 3, 1, 2] → [batch, left, spin, bond] · reshape 14 × 15",
    permutationB: "axes [2, 1, 0] → [spin, bond, out] · reshape 15 × 11",
  });

  const renamed = await evaluate(`new Promise((resolve, reject) => {
    const names = document.querySelectorAll("#legsB .leg-name-input");
    names[2].value = "channel";
    names[2].dispatchEvent(new Event("input", { bubbles: true }));
    const deadline = Date.now() + 3000;
    const poll = () => {
      if (
        document.querySelectorAll(".svg-shared-line").length === 1 &&
        document.querySelector("#metricFlops").textContent === "55,440"
      ) {
        return resolve({
          connections: document.querySelectorAll(".svg-shared-line").length,
          labels: [...document.querySelectorAll(".svg-wire-label")].map((node) => node.textContent),
          outputRank: document.querySelector("#outputRank").textContent,
          outputLegs: [...document.querySelectorAll("#outputLegs .output-leg")].map((node) => node.textContent),
          flops: document.querySelector("#metricFlops").textContent,
        });
      }
      if (Date.now() > deadline) return reject(new Error("renamed leg did not update the workbench"));
      setTimeout(poll, 20);
    };
    poll();
  })`);
  assert.deepEqual(renamed, {
    connections: 1,
    labels: ["bond:5"],
    outputRank: "rank 5 · [2 × 3 × 7 × 11 × 3]",
    outputLegs: [
      "axis 0 · batch : 2",
      "axis 1 · spin : 3",
      "axis 2 · left : 7",
      "axis 3 · out : 11",
      "axis 4 · channel : 3",
    ],
    flops: "55,440",
  });

  const controls = await evaluate(`new Promise((resolve, reject) => {
    document.querySelector("#resetButton").click();
    const deadline = Date.now() + 3000;
    const waitForReset = () => {
      if (document.querySelector("#metricFlops").textContent === "18,480") {
        document.querySelector('input[name="arithmetic"][value="real"]').click();
        return waitForReal();
      }
      if (Date.now() > deadline) return reject(new Error("reset did not restore the example"));
      setTimeout(waitForReset, 20);
    };
    const waitForReal = () => {
      if (document.querySelector("#metricFlops").textContent === "4,620") {
        const realF32 = {
          flops: document.querySelector("#metricFlops").textContent,
          output: document.querySelector("#metricOutput").textContent,
          scratch: document.querySelector("#metricScratch").textContent,
        };
        document.querySelector('input[name="precision"][value="f64"]').click();
        return waitForF64(realF32);
      }
      if (Date.now() > deadline) return reject(new Error("real arithmetic did not update"));
      setTimeout(waitForReal, 20);
    };
    const waitForF64 = (realF32) => {
      if (document.querySelector("#metricOutput").textContent === "1.20 KiB") {
        return resolve({
          realF32,
          realF64: {
            flops: document.querySelector("#metricFlops").textContent,
            output: document.querySelector("#metricOutput").textContent,
            scratch: document.querySelector("#metricScratch").textContent,
          },
          realChecked: document.querySelector('input[name="arithmetic"][value="real"]').checked,
          f64Checked: document.querySelector('input[name="precision"][value="f64"]').checked,
        });
      }
      if (Date.now() > deadline) return reject(new Error("f64 precision did not update"));
      setTimeout(() => waitForF64(realF32), 20);
    };
    waitForReset();
  })`);
  assert.deepEqual(controls, {
    realF32: { flops: "4,620", output: "616 B", scratch: "1.46 KiB" },
    realF64: { flops: "4,620", output: "1.20 KiB", scratch: "2.92 KiB" },
    realChecked: true,
    f64Checked: true,
  });

  const invalid = await evaluate(`new Promise((resolve, reject) => {
    document.querySelector("#resetButton").click();
    const deadline = Date.now() + 3000;
    const edit = () => {
      const inputs = document.querySelectorAll("#legsB .leg-dim-input");
      if (inputs.length === 3 && document.querySelector("#metricFlops").textContent === "18,480") {
        inputs[2].value = "4";
        inputs[2].dispatchEvent(new Event("input", { bubbles: true }));
        return poll();
      }
      if (Date.now() > deadline) return reject(new Error("example did not reset before invalid edit"));
      setTimeout(edit, 20);
    };
    const poll = () => {
      const box = document.querySelector("#errorBox");
      if (!box.hidden) return resolve({
        error: box.textContent,
        badge: document.querySelector("#validityBadge").textContent,
        flops: document.querySelector("#metricFlops").textContent,
        output: document.querySelector("#outputRank").textContent,
      });
      if (Date.now() > deadline) return reject(new Error("invalid model did not report an error"));
      setTimeout(poll, 20);
    };
    edit();
  })`);
  assert.match(invalid.error, /mismatched dimensions 3 on A and 4 on B/);
  assert.equal(invalid.badge, "Invalid input");
  assert.equal(invalid.flops, "—");
  assert.equal(invalid.output, "not available");

  const visualizerLoaded = new Promise((resolve) => {
    resolvePageLoad = resolve;
  });
  await send("Page.navigate", {
    url: `http://127.0.0.1:${webPort}/tensor-network-visualiser.html`,
  });
  await visualizerLoaded;
  const visualizer = await evaluate(`({
    title: document.title,
    hasCanvas: document.querySelector("#canvas") instanceof HTMLCanvasElement,
    plannerHref: document.querySelector('a[href="contraction-planner.html"]')?.getAttribute("href"),
  })`);
  assert.deepEqual(visualizer, {
    title: "Tensor Network Visualiser",
    hasCanvas: true,
    plannerHref: "contraction-planner.html",
  });
  assert.deepEqual(browserErrors, []);

  console.log(JSON.stringify({ initial, renamed, controls, invalid, visualizer, browserErrors }, null, 2));
} finally {
  if (socket) socket.close();
  chrome.kill("SIGTERM");
  await new Promise((resolve) => server.close(resolve));
  await rm(profile, { recursive: true, force: true });
}
