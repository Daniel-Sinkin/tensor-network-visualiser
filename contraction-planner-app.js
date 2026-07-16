(function () {
  "use strict";

  const model = window.ContractionModel;
  const storageKey = "tensor-network-visualiser.contraction-planner.v1";
  let nextUid = 1;
  let state;
  let lastMode = "manual";

  const presets = {
    matmul: {
      arithmetic: "complex",
      precision: "f32",
      order: "A*B",
      tensors: [
        { name: "A", legs: [{ name: "row", dim: "2" }, { name: "contract", dim: "3" }] },
        { name: "B", legs: [{ name: "contract", dim: "3" }, { name: "column", dim: "4" }] },
      ],
    },
    multileg: {
      arithmetic: "complex",
      precision: "f32",
      order: "A*B",
      tensors: [
        {
          name: "A",
          legs: [
            { name: "batch", dim: "2" },
            { name: "spin", dim: "3" },
            { name: "bond", dim: "5" },
            { name: "left", dim: "7" },
          ],
        },
        {
          name: "B",
          legs: [
            { name: "out", dim: "11" },
            { name: "bond", dim: "5" },
            { name: "spin", dim: "3" },
          ],
        },
      ],
    },
    chain: {
      arithmetic: "real",
      precision: "f64",
      order: "A*(B*C)",
      tensors: [
        { name: "A", legs: [{ name: "i", dim: "10" }, { name: "k", dim: "20" }] },
        { name: "B", legs: [{ name: "k", dim: "20" }, { name: "j", dim: "30" }] },
        { name: "C", legs: [{ name: "j", dim: "30" }, { name: "l", dim: "40" }] },
      ],
    },
  };

  const tensorList = document.getElementById("tensorList");
  const tensorTemplate = document.getElementById("tensorTemplate");
  const legTemplate = document.getElementById("legTemplate");
  const orderInput = document.getElementById("orderInput");
  const arithmeticSelect = document.getElementById("arithmeticSelect");
  const precisionSelect = document.getElementById("precisionSelect");
  const evaluateButton = document.getElementById("evaluateButton");
  const minFlopsButton = document.getElementById("minFlopsButton");
  const minPeakButton = document.getElementById("minPeakButton");
  const addTensorButton = document.getElementById("addTensorButton");
  const resultBody = document.getElementById("resultBody");
  const resultBadge = document.getElementById("resultBadge");
  const statusLine = document.getElementById("statusLine");
  const errorBox = document.getElementById("errorBox");

  function uid() {
    return nextUid++;
  }

  function materialize(source) {
    return {
      arithmetic: source.arithmetic || "complex",
      precision: source.precision || "f32",
      order: source.order || "",
      tensors: source.tensors.map((tensor) => ({
        uid: uid(),
        name: String(tensor.name),
        legs: tensor.legs.map((leg) => ({
          uid: uid(),
          name: String(leg.name),
          dim: String(leg.dim),
        })),
      })),
    };
  }

  function loadState() {
    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) return materialize(presets.multileg);
      const parsed = JSON.parse(raw);
      if (!parsed || !Array.isArray(parsed.tensors) || parsed.tensors.length < 2)
        return materialize(presets.multileg);
      return materialize(parsed);
    } catch (_) {
      return materialize(presets.multileg);
    }
  }

  function persist() {
    try {
      localStorage.setItem(
        storageKey,
        JSON.stringify({
          arithmetic: state.arithmetic,
          precision: state.precision,
          order: orderInput.value,
          tensors: state.tensors.map((tensor) => ({
            name: tensor.name,
            legs: tensor.legs.map((leg) => ({ name: leg.name, dim: leg.dim })),
          })),
        })
      );
    } catch (_) {
      return;
    }
  }

  function escapeHtml(value) {
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
    const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
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

  function metricCard(label, value, sub, exactValue) {
    const exact = exactValue && compactCount(exactValue) !== exactCount(exactValue)
      ? `<div class="metric-sub" title="${escapeHtml(exactCount(exactValue))}">exact: ${escapeHtml(exactCount(exactValue))}</div>`
      : "";
    return `<div class="summary-card"><div class="metric-label">${escapeHtml(label)}</div>` +
      `<div class="metric-value">${escapeHtml(value)}</div>` +
      `<div class="metric-sub">${escapeHtml(sub)}</div>${exact}</div>`;
  }

  function leftAssociatedOrder() {
    const names = state.tensors.map((tensor) => tensor.name || "T");
    return names.slice(1).reduce((left, right) => `(${left}*${right})`, names[0]);
  }

  function nextTensorName() {
    const used = new Set(state.tensors.map((tensor) => tensor.name));
    for (let code = 65; code <= 90; code++) {
      const name = String.fromCharCode(code);
      if (!used.has(name)) return name;
    }
    let index = 1;
    while (used.has(`T${index}`)) index++;
    return `T${index}`;
  }

  function uniqueLegName(prefix) {
    const used = new Set(state.tensors.flatMap((tensor) => tensor.legs.map((leg) => leg.name)));
    let index = 1;
    while (used.has(`${prefix}${index}`)) index++;
    return `${prefix}${index}`;
  }

  function markStale() {
    lastMode = "manual";
    errorBox.hidden = true;
    statusLine.textContent = "Input changed; evaluate or optimize the new network.";
    resultBadge.textContent = "Not evaluated";
    renderEmpty("The previous result is stale", "Evaluate the edited order or choose an optimizer.");
    persist();
  }

  function renderEmpty(title, body) {
    resultBody.innerHTML = `<div class="result-empty"><div><strong>${escapeHtml(title)}</strong>${escapeHtml(body)}</div></div>`;
  }

  function renderEditor() {
    tensorList.innerHTML = "";
    state.tensors.forEach((tensor, tensorIndex) => {
      const card = tensorTemplate.content.firstElementChild.cloneNode(true);
      card.dataset.tensorUid = tensor.uid;
      const nameInput = card.querySelector(".tensor-name-input");
      nameInput.value = tensor.name;
      nameInput.addEventListener("input", () => {
        tensor.name = nameInput.value;
        markStale();
      });

      const removeTensor = card.querySelector(".remove-tensor-button");
      removeTensor.disabled = state.tensors.length <= 2;
      removeTensor.addEventListener("click", () => {
        if (state.tensors.length <= 2) return;
        state.tensors.splice(tensorIndex, 1);
        orderInput.value = leftAssociatedOrder();
        state.order = orderInput.value;
        renderEditor();
        markStale();
      });

      const legList = card.querySelector(".leg-list");
      tensor.legs.forEach((leg, legIndex) => {
        const row = legTemplate.content.firstElementChild.cloneNode(true);
        row.dataset.legUid = leg.uid;
        const legName = row.querySelector(".leg-name-input");
        const legDim = row.querySelector(".leg-dim-input");
        legName.value = leg.name;
        legDim.value = leg.dim;
        legName.addEventListener("input", () => {
          leg.name = legName.value;
          markStale();
        });
        legDim.addEventListener("input", () => {
          leg.dim = legDim.value;
          markStale();
        });
        const up = row.querySelector(".move-leg-up");
        const down = row.querySelector(".move-leg-down");
        up.disabled = legIndex === 0;
        down.disabled = legIndex === tensor.legs.length - 1;
        up.addEventListener("click", () => {
          if (legIndex === 0) return;
          [tensor.legs[legIndex - 1], tensor.legs[legIndex]] = [
            tensor.legs[legIndex],
            tensor.legs[legIndex - 1],
          ];
          renderEditor();
          markStale();
        });
        down.addEventListener("click", () => {
          if (legIndex + 1 >= tensor.legs.length) return;
          [tensor.legs[legIndex], tensor.legs[legIndex + 1]] = [
            tensor.legs[legIndex + 1],
            tensor.legs[legIndex],
          ];
          renderEditor();
          markStale();
        });
        row.querySelector(".remove-leg-button").addEventListener("click", () => {
          tensor.legs.splice(legIndex, 1);
          renderEditor();
          markStale();
        });
        legList.appendChild(row);
      });

      card.querySelector(".add-leg-button").addEventListener("click", () => {
        tensor.legs.push({ uid: uid(), name: uniqueLegName("leg_"), dim: "2" });
        renderEditor();
        markStale();
      });
      tensorList.appendChild(card);
    });
  }

  function networkInput() {
    return state.tensors.map((tensor) => ({
      name: tensor.name,
      legs: tensor.legs.map((leg) => ({ name: leg.name, dim: leg.dim })),
    }));
  }

  function settings() {
    return { arithmetic: state.arithmetic, precision: state.precision };
  }

  function legText(legs) {
    if (!legs.length) return "scalar";
    return legs.map((leg) => `${leg.name}:${leg.dim}`).join(", ");
  }

  function legChips(legs) {
    if (!legs.length) return `<span class="leg-chip scalar-chip">scalar · storage [1]</span>`;
    return legs
      .map((leg) => `<span class="leg-chip">${escapeHtml(leg.name)} : ${escapeHtml(leg.dim)}</span>`)
      .join("");
  }

  function renderStep(step) {
    const shared = step.shared.length
      ? step.shared.map((leg) => `${leg.name}:${leg.dim}`).join(", ")
      : "none";
    const result = legText(step.resultLegs);
    const axesA = `[${step.permutationA.join(", ")}]`;
    const axesB = `[${step.permutationB.join(", ")}]`;
    const legsA = `[${step.permutedLegsA.join(", ")}]`;
    const legsB = `[${step.permutedLegsB.join(", ")}]`;
    const kind = step.outerProduct ? "outer product" : `${step.shared.length}-leg contraction`;
    return `<article class="step-card" data-step="${step.index}">` +
      `<div class="step-header"><div class="step-title">${step.index}. ${escapeHtml(step.leftExpression)} × ${escapeHtml(step.rightExpression)}</div>` +
      `<div class="step-kind${step.outerProduct ? " outer" : ""}">${escapeHtml(kind)}</div></div>` +
      `<div class="step-body"><div class="contracted-legs">Matched by name: <code>${escapeHtml(shared)}</code></div>` +
      `<div class="lowering-grid">` +
      `<div class="lowering-box"><div class="lowering-label"><span>Left operand</span><span class="perm-badge${step.identityA ? " identity" : ""}">${step.identityA ? "identity" : "reorder"}</span></div>` +
      `<div class="lowering-code">axes ${escapeHtml(axesA)} → ${escapeHtml(legsA)}<br>reshape → ${exactCount(step.M)} × ${exactCount(step.K)}</div></div>` +
      `<div class="lowering-box"><div class="lowering-label"><span>Right operand</span><span class="perm-badge${step.identityB ? " identity" : ""}">${step.identityB ? "identity" : "reorder"}</span></div>` +
      `<div class="lowering-code">axes ${escapeHtml(axesB)} → ${escapeHtml(legsB)}<br>reshape → ${exactCount(step.K)} × ${exactCount(step.N)}</div></div>` +
      `</div>` +
      `<div class="gemm-equation">(${exactCount(step.M)} × ${exactCount(step.K)}) · (${exactCount(step.K)} × ${exactCount(step.N)}) → (${exactCount(step.M)} × ${exactCount(step.N)})</div>` +
      `<div class="step-result">Result legs: <code>${escapeHtml(result)}</code></div>` +
      `<div class="step-metrics">` +
      `<div class="step-metric"><span>MAC positions</span><strong title="${escapeHtml(exactCount(step.macs))}">${escapeHtml(compactCount(step.macs))}</strong></div>` +
      `<div class="step-metric"><span>Real FLOPs</span><strong title="${escapeHtml(exactCount(step.realFlops))}">${escapeHtml(compactCount(step.realFlops))}</strong></div>` +
      `<div class="step-metric"><span>Output</span><strong>${escapeHtml(formatBytes(step.resultBytes))}</strong></div>` +
      `<div class="step-metric"><span>Permutation scratch</span><strong>${escapeHtml(formatBytes(step.permutationScratchBytes))}</strong></div>` +
      `</div></div></article>`;
  }

  function renderResult(result, badge, method) {
    resultBadge.textContent = badge;
    const arithmetic = result.arithmetic === "complex" ? "8 real FLOPs / complex MAC" : "2 real FLOPs / real MAC";
    resultBody.innerHTML =
      `<div class="order-display">${escapeHtml(result.order)}</div>` +
      `<div class="summary-grid">` +
      metricCard("Scalar MACs", compactCount(result.totalMacs), "multiply-accumulate positions", result.totalMacs) +
      metricCard("Real FLOPs", compactCount(result.realFlops), arithmetic, result.realFlops) +
      metricCard("Result storage", formatBytes(result.resultBytes), `${exactCount(result.resultElements)} elements`) +
      metricCard("Peak intermediate", formatBytes(result.peakIntermediateBytes), `${exactCount(result.peakIntermediateElements)} produced elements`) +
      metricCard("Peak permute scratch", formatBytes(result.peakPermutationScratchBytes), "left + right materializations") +
      metricCard("Peak step workspace", formatBytes(result.peakWorkspaceBytes), "permuted operands + result") +
      `</div>` +
      `<div class="result-shape"><div class="result-shape-title">Logical result legs · ${escapeHtml(result.arithmetic)} ${escapeHtml(result.precision)}</div>` +
      `<div class="leg-chips">${legChips(result.tensor.legs)}</div></div>` +
      `<div class="steps-heading">Pairwise lowering · ${result.steps.length} step${result.steps.length === 1 ? "" : "s"}</div>` +
      `<div class="step-list">${result.steps.map(renderStep).join("")}</div>` +
      `<p class="field-help">Axis numbers are zero-based. Scratch follows the current CUDA helper's two permuted operand buffers; an identity badge marks cases where the Julia helper can reshape without reordering. ${escapeHtml(method)}</p>`;
  }

  function clearError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function showError(error) {
    errorBox.textContent = error && error.message ? error.message : String(error);
    errorBox.hidden = false;
    statusLine.textContent = "The network was not evaluated.";
    resultBadge.textContent = "Invalid input";
    renderEmpty("Fix the highlighted model error", errorBox.textContent);
  }

  function setBusy(busy) {
    evaluateButton.disabled = busy;
    minFlopsButton.disabled = busy;
    minPeakButton.disabled = busy;
    addTensorButton.disabled = busy;
  }

  function run(mode) {
    clearError();
    setBusy(true);
    const label = mode === "manual" ? "Evaluating order…" : "Searching contraction orders…";
    statusLine.textContent = label;
    window.setTimeout(() => {
      try {
        const network = model.normalizeNetwork(networkInput());
        let result;
        let badge;
        let method;
        if (mode === "manual") {
          result = model.evaluateTree(network, orderInput.value, settings());
          badge = "Manual order";
          method = "Manual order evaluated exactly.";
          statusLine.textContent = `Evaluated ${result.steps.length} pairwise contraction${result.steps.length === 1 ? "" : "s"}.`;
        } else {
          const optimized = model.optimizeNetwork(network, settings());
          result = mode === "flops" ? optimized.minFlops : optimized.minPeak;
          badge = mode === "flops" ? "Minimum FLOPs" : "Minimum peak";
          method = `${optimized.exact ? "Exact" : "Heuristic"}: ${optimized.method}.`;
          orderInput.value = result.order;
          state.order = result.order;
          statusLine.textContent = method;
        }
        lastMode = mode;
        renderResult(result, badge, method);
        persist();
      } catch (error) {
        showError(error);
      } finally {
        setBusy(false);
      }
    }, 0);
  }

  function applyPreset(name) {
    state = materialize(presets[name]);
    orderInput.value = state.order;
    arithmeticSelect.value = state.arithmetic;
    precisionSelect.value = state.precision;
    renderEditor();
    persist();
    run("manual");
  }

  function initialize() {
    state = loadState();
    orderInput.value = state.order || leftAssociatedOrder();
    arithmeticSelect.value = state.arithmetic;
    precisionSelect.value = state.precision;
    renderEditor();

    document.querySelectorAll(".preset-button").forEach((button) => {
      button.addEventListener("click", () => applyPreset(button.dataset.preset));
    });
    addTensorButton.addEventListener("click", () => {
      const name = nextTensorName();
      state.tensors.push({
        uid: uid(),
        name,
        legs: [{ uid: uid(), name: uniqueLegName("leg_"), dim: "2" }],
      });
      orderInput.value = leftAssociatedOrder();
      state.order = orderInput.value;
      renderEditor();
      markStale();
    });
    orderInput.addEventListener("input", () => {
      state.order = orderInput.value;
      markStale();
    });
    arithmeticSelect.addEventListener("change", () => {
      state.arithmetic = arithmeticSelect.value;
      persist();
      run(lastMode);
    });
    precisionSelect.addEventListener("change", () => {
      state.precision = precisionSelect.value;
      persist();
      run(lastMode);
    });
    evaluateButton.addEventListener("click", () => run("manual"));
    minFlopsButton.addEventListener("click", () => run("flops"));
    minPeakButton.addEventListener("click", () => run("peak"));
    orderInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") run("manual");
    });
    run("manual");
  }

  initialize();
})();
