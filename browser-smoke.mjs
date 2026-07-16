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
      const step = document.querySelector(".step-card");
      if (step) return resolve({
        badge: document.querySelector("#resultBadge").textContent,
        order: document.querySelector(".order-display").textContent,
        gemm: document.querySelector(".gemm-equation").textContent,
        matched: document.querySelector(".contracted-legs").textContent.replace(/\\s+/g, " ").trim(),
        stepCount: document.querySelectorAll(".step-card").length,
      });
      if (Date.now() > deadline) return reject(new Error("initial result did not render"));
      setTimeout(poll, 20);
    };
    poll();
  })`);
  assert.deepEqual(initial, {
    badge: "Manual order",
    order: "(A*B)",
    gemm: "(14 × 15) · (15 × 11) → (14 × 11)",
    matched: "Matched by name: spin:3, bond:5",
    stepCount: 1,
  });

  const optimized = await evaluate(`new Promise((resolve, reject) => {
    document.querySelector('[data-preset="chain"]').click();
    const deadline = Date.now() + 3000;
    const optimize = () => {
      const button = document.querySelector("#minFlopsButton");
      if (
        document.querySelector("#orderInput").value === "A*(B*C)" &&
        document.querySelector("#resultBadge").textContent === "Manual order" &&
        !button.disabled
      ) {
        button.click();
        return poll();
      }
      if (Date.now() > deadline) return reject(new Error("chain preset did not render"));
      setTimeout(optimize, 20);
    };
    const poll = () => {
      if (document.querySelector("#resultBadge").textContent === "Minimum FLOPs") {
        return resolve({
          order: document.querySelector(".order-display").textContent,
          inputOrder: document.querySelector("#orderInput").value,
          status: document.querySelector("#statusLine").textContent,
          metrics: [...document.querySelectorAll(".summary-card .metric-value")].map((node) => node.textContent),
          stepCount: document.querySelectorAll(".step-card").length,
        });
      }
      if (Date.now() > deadline) return reject(new Error("optimizer result did not render"));
      setTimeout(poll, 20);
    };
    optimize();
  })`);
  assert.equal(optimized.order, "((A*B)*C)");
  assert.equal(optimized.inputOrder, "((A*B)*C)");
  assert.match(optimized.status, /Exact: exact subset dynamic programming/);
  assert.equal(optimized.metrics[0], "18,000");
  assert.equal(optimized.metrics[1], "36,000");
  assert.equal(optimized.stepCount, 2);

  const invalid = await evaluate(`new Promise((resolve, reject) => {
    document.querySelector('[data-preset="matmul"]').click();
    const deadline = Date.now() + 3000;
    const edit = () => {
      const inputs = document.querySelectorAll(".leg-dim-input");
      if (inputs.length === 4 && document.querySelector("#resultBadge").textContent === "Manual order") {
        inputs[2].value = "4";
        inputs[2].dispatchEvent(new Event("input", { bubbles: true }));
        document.querySelector("#evaluateButton").click();
        return poll();
      }
      if (Date.now() > deadline) return reject(new Error("matrix preset did not render"));
      setTimeout(edit, 20);
    };
    const poll = () => {
      const box = document.querySelector("#errorBox");
      if (!box.hidden) return resolve(box.textContent);
      if (Date.now() > deadline) return reject(new Error("invalid model did not report an error"));
      setTimeout(poll, 20);
    };
    edit();
  })`);
  assert.match(invalid, /mismatched dimensions 3 on A and 4 on B/);

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

  console.log(JSON.stringify({ initial, optimized, invalid, visualizer, browserErrors }, null, 2));
} finally {
  if (socket) socket.close();
  chrome.kill("SIGTERM");
  await new Promise((resolve) => server.close(resolve));
  await rm(profile, { recursive: true, force: true });
}
