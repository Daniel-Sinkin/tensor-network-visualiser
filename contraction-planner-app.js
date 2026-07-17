(function () {
  "use strict";

  const model = window.ContractionModel;
  const storageKey = "tensor-network-visualiser.two-tensor-workbench.v2";
  const example = {
    arithmetic: "complex",
    precision: "f32",
    A: [
      { name: "batch", dim: "2" },
      { name: "spin", dim: "3" },
      { name: "bond", dim: "5" },
      { name: "left", dim: "7" },
    ],
    B: [
      { name: "out", dim: "11" },
      { name: "bond", dim: "5" },
      { name: "spin", dim: "3" },
    ],
  };

  let state;
  let updateFrame = 0;

  const diagram = document.getElementById("networkDiagram");
  const legsA = document.getElementById("legsA");
  const legsB = document.getElementById("legsB");
  const shapeA = document.getElementById("shapeA");
  const shapeB = document.getElementById("shapeB");
  const outputRank = document.getElementById("outputRank");
  const outputLegs = document.getElementById("outputLegs");
  const errorBox = document.getElementById("errorBox");
  const validityBadge = document.getElementById("validityBadge");
  const legRowTemplate = document.getElementById("legRowTemplate");

  const metricFlops = document.getElementById("metricFlops");
  const metricFlopsNote = document.getElementById("metricFlopsNote");
  const metricMacs = document.getElementById("metricMacs");
  const metricOutput = document.getElementById("metricOutput");
  const metricOutputNote = document.getElementById("metricOutputNote");
  const metricScratch = document.getElementById("metricScratch");
  const metricScratchNote = document.getElementById("metricScratchNote");
  const metricWorkspace = document.getElementById("metricWorkspace");

  const sharedLegs = document.getElementById("sharedLegs");
  const permutationA = document.getElementById("permutationA");
  const permutationB = document.getElementById("permutationB");
  const badgeA = document.getElementById("badgeA");
  const badgeB = document.getElementById("badgeB");
  const gemmEquation = document.getElementById("gemmEquation");

  function cloneExample() {
    return {
      arithmetic: example.arithmetic,
      precision: example.precision,
      A: example.A.map((leg) => ({ ...leg })),
      B: example.B.map((leg) => ({ ...leg })),
    };
  }

  function loadState() {
    try {
      const parsed = JSON.parse(localStorage.getItem(storageKey));
      if (!parsed || !Array.isArray(parsed.A) || !Array.isArray(parsed.B)) return cloneExample();
      return {
        arithmetic: parsed.arithmetic === "real" ? "real" : "complex",
        precision: parsed.precision === "f64" ? "f64" : "f32",
        A: parsed.A.map((leg) => ({ name: String(leg.name ?? ""), dim: String(leg.dim ?? "") })),
        B: parsed.B.map((leg) => ({ name: String(leg.name ?? ""), dim: String(leg.dim ?? "") })),
      };
    } catch (_) {
      return cloneExample();
    }
  }

  function persist() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(state));
    } catch (_) {
      return;
    }
  }

  function escapeMarkup(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function exactCount(value) {
    return value.toLocaleString("en-US");
  }

  function compactCount(value) {
    const text = value.toString();
    if (text.length <= 15) return exactCount(value);
    const tail = text.slice(1, 5).replace(/0+$/, "");
    return `${text[0]}${tail ? `.${tail}` : ""}e+${text.length - 1}`;
  }

  function formatBytes(value) {
    const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"];
    let unit = 0;
    let scale = 1n;
    while (unit + 1 < units.length && value >= scale * 1024n) {
      scale *= 1024n;
      unit++;
    }
    if (unit === 0) return `${exactCount(value)} B`;
    const whole = value / scale;
    const fraction = ((value % scale) * 100n) / scale;
    return `${whole}.${fraction.toString().padStart(2, "0")} ${units[unit]}`;
  }

  function shorten(value, limit) {
    const text = String(value);
    return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`;
  }

  function rawShape(legs) {
    if (!legs.length) return "rank 0 · scalar";
    const dims = legs.map((leg) => String(leg.dim).trim() || "?");
    return `rank ${legs.length} · [${dims.join(" × ")}]`;
  }

  function normalizedShape(legs) {
    if (!legs.length) return "rank 0 · scalar";
    return `rank ${legs.length} · [${legs.map((leg) => leg.dim).join(" × ")}]`;
  }

  function syncControls() {
    document.querySelectorAll('input[name="arithmetic"]').forEach((input) => {
      input.checked = input.value === state.arithmetic;
    });
    document.querySelectorAll('input[name="precision"]').forEach((input) => {
      input.checked = input.value === state.precision;
    });
  }

  function uniqueLegName() {
    const used = new Set(state.A.concat(state.B).map((leg) => leg.name));
    let index = 1;
    while (used.has(`leg_${index}`)) index++;
    return `leg_${index}`;
  }

  function renderLegEditor(tensorName, container) {
    const legs = state[tensorName];
    container.innerHTML = "";
    legs.forEach((leg, index) => {
      const row = legRowTemplate.content.firstElementChild.cloneNode(true);
      row.dataset.tensor = tensorName;
      row.dataset.axis = String(index);
      row.querySelector(".axis-index").textContent = String(index);

      const nameInput = row.querySelector(".leg-name-input");
      const dimInput = row.querySelector(".leg-dim-input");
      nameInput.value = leg.name;
      dimInput.value = leg.dim;
      nameInput.addEventListener("input", () => {
        leg.name = nameInput.value;
        scheduleUpdate();
      });
      dimInput.addEventListener("input", () => {
        leg.dim = dimInput.value;
        scheduleUpdate();
      });

      const up = row.querySelector(".move-leg-up");
      const down = row.querySelector(".move-leg-down");
      up.disabled = index === 0;
      down.disabled = index === legs.length - 1;
      up.addEventListener("click", () => moveLeg(tensorName, index, index - 1));
      down.addEventListener("click", () => moveLeg(tensorName, index, index + 1));
      row.querySelector(".remove-leg-button").addEventListener("click", () => {
        legs.splice(index, 1);
        renderEditors();
        scheduleUpdate();
      });
      container.appendChild(row);
    });
  }

  function renderEditors() {
    renderLegEditor("A", legsA);
    renderLegEditor("B", legsB);
    shapeA.textContent = rawShape(state.A);
    shapeB.textContent = rawShape(state.B);
  }

  function moveLeg(tensorName, from, to) {
    const legs = state[tensorName];
    if (to < 0 || to >= legs.length) return;
    [legs[from], legs[to]] = [legs[to], legs[from]];
    renderEditors();
    scheduleUpdate();
  }

  function addLeg(tensorName) {
    state[tensorName].push({ name: uniqueLegName(), dim: "2" });
    renderEditors();
    scheduleUpdate();
  }

  function scheduleUpdate() {
    persist();
    shapeA.textContent = rawShape(state.A);
    shapeB.textContent = rawShape(state.B);
    if (updateFrame) cancelAnimationFrame(updateFrame);
    updateFrame = requestAnimationFrame(() => {
      updateFrame = 0;
      updateWorkbench();
    });
  }

  function networkInput() {
    return [
      { name: "A", legs: state.A.map((leg) => ({ ...leg })) },
      { name: "B", legs: state.B.map((leg) => ({ ...leg })) },
    ];
  }

  function setMetric(element, value, exactValue) {
    element.textContent = value;
    element.title = exactValue == null ? "" : exactCount(exactValue);
  }

  function renderOutput(result) {
    const legs = result.tensor.legs;
    outputRank.textContent = normalizedShape(legs);
    if (!legs.length) {
      outputLegs.innerHTML = '<span class="output-leg scalar">scalar · storage shape [1]</span>';
      return;
    }
    outputLegs.innerHTML = legs
      .map((leg, index) => `<span class="output-leg">axis ${index} · ${escapeMarkup(leg.name)} : ${escapeMarkup(leg.dim)}</span>`)
      .join("");
  }

  function renderMetrics(result, step) {
    setMetric(metricFlops, compactCount(result.realFlops), result.realFlops);
    metricFlopsNote.textContent = result.arithmetic === "complex"
      ? "8 real FLOPs / complex MAC"
      : "2 real FLOPs / real MAC";
    setMetric(metricMacs, compactCount(result.totalMacs), result.totalMacs);
    metricMacs.title = exactCount(result.totalMacs);
    metricOutput.textContent = formatBytes(result.resultBytes);
    metricOutput.title = exactCount(result.resultBytes);
    metricOutputNote.textContent = `${exactCount(result.resultElements)} element${result.resultElements === 1n ? "" : "s"}`;
    metricScratch.textContent = formatBytes(step.permutationScratchBytes);
    metricScratch.title = exactCount(step.permutationScratchBytes);
    metricScratchNote.textContent = `${exactCount(step.permutationScratchElements)} elements · two operand buffers`;
    metricWorkspace.textContent = formatBytes(step.workspaceBytes);
    metricWorkspace.title = exactCount(step.workspaceBytes);
  }

  function renderLowering(step) {
    sharedLegs.textContent = step.shared.length
      ? step.shared.map((leg) => `${leg.name}:${leg.dim}`).join(", ")
      : "none · outer product (K = 1)";
    badgeA.textContent = step.identityA ? "identity" : "reorder";
    badgeA.classList.toggle("identity", step.identityA);
    badgeB.textContent = step.identityB ? "identity" : "reorder";
    badgeB.classList.toggle("identity", step.identityB);
    permutationA.textContent = `axes [${step.permutationA.join(", ")}] → [${step.permutedLegsA.join(", ")}] · reshape ${exactCount(step.M)} × ${exactCount(step.K)}`;
    permutationB.textContent = `axes [${step.permutationB.join(", ")}] → [${step.permutedLegsB.join(", ")}] · reshape ${exactCount(step.K)} × ${exactCount(step.N)}`;
    gemmEquation.textContent = `(${exactCount(step.M)} × ${exactCount(step.K)}) · (${exactCount(step.K)} × ${exactCount(step.N)}) → (${exactCount(step.M)} × ${exactCount(step.N)})`;
  }

  function axisY(axis, rank) {
    if (rank <= 1) return 118;
    return 78 + (axis * 80) / (rank - 1);
  }

  function nodeMarkup(kind, x, letter, role, shape, invalid) {
    const disabled = invalid ? " svg-invalid" : "";
    return `<g class="${disabled.trim()}">` +
      `<rect class="svg-tensor-node ${kind}" x="${x}" y="58" width="190" height="120" rx="13"></rect>` +
      `<text class="svg-node-role" x="${x + 95}" y="84">${escapeMarkup(role)}</text>` +
      `<text class="svg-node-letter" x="${x + 95}" y="123">${escapeMarkup(letter)}</text>` +
      `<text class="svg-node-shape" x="${x + 95}" y="151">${escapeMarkup(shorten(shape, 25))}</text>` +
      `</g>`;
  }

  function freeLegMarkup(leg, side, rank) {
    const y = axisY(leg.axis, rank);
    const label = shorten(`${leg.name}:${leg.dim}`, 19);
    if (side === "a") {
      return `<line class="svg-free-line a" x1="180" y1="${y}" x2="116" y2="${y}"></line>` +
        `<circle class="svg-free-port a" cx="180" cy="${y}" r="4"></circle>` +
        `<text class="svg-free-label a" x="107" y="${y + 4}">${escapeMarkup(label)}</text>`;
    }
    return `<line class="svg-free-line b" x1="820" y1="${y}" x2="884" y2="${y}"></line>` +
      `<circle class="svg-free-port b" cx="820" cy="${y}" r="4"></circle>` +
      `<text class="svg-free-label b" x="893" y="${y + 4}">${escapeMarkup(label)}</text>`;
  }

  function sharedLegMarkup(leg, index, count, rankA, rankB) {
    const yA = axisY(leg.axisA, rankA);
    const yB = axisY(leg.axisB, rankB);
    const labelY = count <= 1 ? 118 : 82 + (index * 72) / (count - 1);
    const label = shorten(`${leg.name}:${leg.dim}`, 22);
    const width = Math.min(176, Math.max(62, 22 + label.length * 7));
    return `<path class="svg-shared-line" d="M 370 ${yA} C 430 ${yA}, 450 ${labelY}, 500 ${labelY} C 550 ${labelY}, 570 ${yB}, 630 ${yB}"></path>` +
      `<circle class="svg-shared-port" cx="370" cy="${yA}" r="5"></circle>` +
      `<circle class="svg-shared-port" cx="630" cy="${yB}" r="5"></circle>` +
      `<rect class="svg-wire-label-bg" x="${500 - width / 2}" y="${labelY - 10}" width="${width}" height="20" rx="10"></rect>` +
      `<text class="svg-wire-label" x="500" y="${labelY + 4}">${escapeMarkup(label)}</text>`;
  }

  function renderDiagram(network, step) {
    const valid = Boolean(network && step);
    const tensorA = valid ? network.tensors[0] : { legs: state.A };
    const tensorB = valid ? network.tensors[1] : { legs: state.B };
    const shapeAText = valid ? normalizedShape(tensorA.legs) : rawShape(state.A);
    const shapeBText = valid ? normalizedShape(tensorB.legs) : rawShape(state.B);
    const resultShape = valid ? normalizedShape(step.resultLegs) : "waiting for valid input";
    const outputNames = valid && step.resultLegs.length
      ? shorten(step.resultLegs.map((leg) => `${leg.name}:${leg.dim}`).join(" · "), 68)
      : valid ? "scalar · storage [1]" : "correct the leg model above";

    let wires = "";
    if (valid) {
      wires += step.freeA.map((leg) => freeLegMarkup(leg, "a", tensorA.legs.length)).join("");
      wires += step.freeB.map((leg) => freeLegMarkup(leg, "b", tensorB.legs.length)).join("");
      if (step.shared.length) {
        wires += step.shared
          .map((leg, index) => sharedLegMarkup(leg, index, step.shared.length, tensorA.legs.length, tensorB.legs.length))
          .join("");
      } else {
        wires += '<line class="svg-flow-line" x1="390" y1="118" x2="610" y2="118" stroke-dasharray="6 7"></line>' +
          '<text class="svg-outer-label" x="500" y="108">no shared legs · outer product</text>';
      }
    }

    diagram.innerHTML =
      '<defs><marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#60738e"></path></marker></defs>' +
      '<path class="svg-flow-line" d="M 275 178 C 275 218, 414 218, 500 247"></path>' +
      '<path class="svg-flow-line" d="M 725 178 C 725 218, 586 218, 500 247"></path>' +
      '<circle class="svg-flow-junction" cx="500" cy="247" r="4"></circle>' +
      '<line class="svg-flow-line" x1="500" y1="251" x2="500" y2="282" marker-end="url(#arrowhead)"></line>' +
      wires +
      nodeMarkup("a", 180, "A", "left operand", shapeAText, false) +
      nodeMarkup("b", 630, "B", "right operand", shapeBText, false) +
      `<g class="${valid ? "" : "svg-invalid"}">` +
      '<rect class="svg-tensor-node c" x="400" y="288" width="200" height="90" rx="13"></rect>' +
      '<text class="svg-node-role" x="500" y="310">output · A × B</text>' +
      '<text class="svg-node-letter" x="500" y="342">C</text>' +
      `<text class="svg-node-shape" x="500" y="363">${escapeMarkup(shorten(resultShape, 31))}</text>` +
      `</g><text class="svg-output-legs" x="500" y="399">${escapeMarkup(outputNames)}</text>`;

    diagram.setAttribute(
      "aria-label",
      valid
        ? `Tensor A contracts with tensor B over ${step.shared.length} shared leg${step.shared.length === 1 ? "" : "s"} to produce tensor C.`
        : "Tensor A and tensor B are waiting for a valid leg model before producing tensor C."
    );
  }

  function clearComputedDisplay(error) {
    validityBadge.textContent = "Invalid input";
    validityBadge.classList.add("invalid");
    errorBox.hidden = false;
    errorBox.textContent = error && error.message ? error.message : String(error);
    outputRank.textContent = "not available";
    outputLegs.innerHTML = '<span class="output-leg scalar">C will appear when A and B are valid.</span>';
    [metricFlops, metricMacs, metricOutput, metricScratch, metricWorkspace].forEach((element) => {
      element.textContent = "—";
      element.title = "";
    });
    metricFlopsNote.textContent = "waiting for valid dimensions";
    metricOutputNote.textContent = "—";
    metricScratchNote.textContent = "two operand buffers";
    sharedLegs.textContent = "—";
    permutationA.textContent = "—";
    permutationB.textContent = "—";
    gemmEquation.textContent = "—";
    badgeA.textContent = "—";
    badgeB.textContent = "—";
    badgeA.classList.remove("identity");
    badgeB.classList.remove("identity");
    renderDiagram(null, null);
  }

  function updateWorkbench() {
    try {
      const network = model.normalizeNetwork(networkInput());
      const result = model.evaluateTree(network, "A*B", {
        arithmetic: state.arithmetic,
        precision: state.precision,
      });
      const step = result.steps[0];
      shapeA.textContent = normalizedShape(network.tensors[0].legs);
      shapeB.textContent = normalizedShape(network.tensors[1].legs);
      validityBadge.textContent = step.outerProduct ? "Valid outer product" : "Valid contraction";
      validityBadge.classList.remove("invalid");
      errorBox.hidden = true;
      errorBox.textContent = "";
      renderDiagram(network, step);
      renderOutput(result);
      renderMetrics(result, step);
      renderLowering(step);
    } catch (error) {
      clearComputedDisplay(error);
    }
  }

  function initialize() {
    state = loadState();
    syncControls();
    renderEditors();

    document.querySelectorAll('input[name="arithmetic"]').forEach((input) => {
      input.addEventListener("change", () => {
        if (!input.checked) return;
        state.arithmetic = input.value;
        scheduleUpdate();
      });
    });
    document.querySelectorAll('input[name="precision"]').forEach((input) => {
      input.addEventListener("change", () => {
        if (!input.checked) return;
        state.precision = input.value;
        scheduleUpdate();
      });
    });
    document.getElementById("addLegA").addEventListener("click", () => addLeg("A"));
    document.getElementById("addLegB").addEventListener("click", () => addLeg("B"));
    document.getElementById("resetButton").addEventListener("click", () => {
      state = cloneExample();
      syncControls();
      renderEditors();
      scheduleUpdate();
    });
    updateWorkbench();
  }

  initialize();
})();
