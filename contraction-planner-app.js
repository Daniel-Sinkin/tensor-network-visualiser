(function () {
  "use strict";

  const model = window.ContractionModel;
  const storageKey = "tensor-network-visualiser.two-tensor-workbench.v3";
  const example = {
    arithmetic: "complex",
    precision: "f32",
    variables: [
      { name: "D", value: "4" },
      { name: "d", value: "2" },
    ],
    A: [
      { name: "batch", dim: "d" },
      { name: "spin", dim: "d" },
      { name: "bond", dim: "D" },
      { name: "left", dim: "D" },
    ],
    B: [
      { name: "out", dim: "D" },
      { name: "bond", dim: "D" },
      { name: "spin", dim: "d" },
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
  const variableList = document.getElementById("variableList");
  const variableTemplate = document.getElementById("variableTemplate");
  const dimensionVariables = document.getElementById("dimensionVariables");

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
      variables: example.variables.map((variable) => ({ ...variable })),
      A: example.A.map((leg) => ({ ...leg })),
      B: example.B.map((leg) => ({ ...leg })),
    };
  }

  function loadState() {
    try {
      const parsed = JSON.parse(localStorage.getItem(storageKey));
      if (
        !parsed ||
        !Array.isArray(parsed.variables) ||
        !Array.isArray(parsed.A) ||
        !Array.isArray(parsed.B)
      ) return cloneExample();
      return {
        arithmetic: parsed.arithmetic === "real" ? "real" : "complex",
        precision: parsed.precision === "f64" ? "f64" : "f32",
        variables: parsed.variables.map((variable) => ({
          name: String(variable.name ?? ""),
          value: String(variable.value ?? ""),
        })),
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

  function uniqueVariableName() {
    const used = new Set(state.variables.map((variable) => variable.name));
    let index = 1;
    while (used.has(`v${index}`)) index++;
    return `v${index}`;
  }

  function renderVariableOptions() {
    dimensionVariables.innerHTML = state.variables
      .map((variable) => variable.name.trim())
      .filter((name) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(name))
      .map((name) => `<option value="${escapeMarkup(name)}"></option>`)
      .join("");
  }

  function renderVariables() {
    variableList.innerHTML = "";
    state.variables.forEach((variable, index) => {
      const chip = variableTemplate.content.firstElementChild.cloneNode(true);
      chip.dataset.variableIndex = String(index);
      const nameInput = chip.querySelector(".variable-name-input");
      const valueInput = chip.querySelector(".variable-value-input");
      nameInput.value = variable.name;
      valueInput.value = variable.value;
      nameInput.setAttribute("aria-label", `Variable ${index + 1} name`);
      valueInput.setAttribute("aria-label", `Value of variable ${variable.name || index + 1}`);
      nameInput.addEventListener("input", () => {
        variable.name = nameInput.value;
        valueInput.setAttribute("aria-label", `Value of variable ${variable.name || index + 1}`);
        renderVariableOptions();
        scheduleUpdate();
      });
      valueInput.addEventListener("input", () => {
        variable.value = valueInput.value;
        scheduleUpdate();
      });
      chip.querySelector(".variable-remove-button").addEventListener("click", () => {
        state.variables.splice(index, 1);
        renderVariables();
        scheduleUpdate();
      });
      variableList.appendChild(chip);
    });
    renderVariableOptions();
  }

  function addVariable() {
    const name = uniqueVariableName();
    state.variables.push({ name, value: "2" });
    renderVariables();
    scheduleUpdate();
    const input = variableList.lastElementChild?.querySelector(".variable-name-input");
    if (input) {
      input.focus();
      input.select();
    }
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

  function updateEditorAnnotations() {
    let variables;
    try {
      variables = model.normalizeVariables(state.variables);
    } catch (_) {
      variables = null;
    }
    for (const tensorName of ["A", "B"]) {
      const rows = document.querySelectorAll(`#legs${tensorName} .leg-row`);
      state[tensorName].forEach((leg, index) => {
        const row = rows[index];
        if (!row) return;
        const expansion = row.querySelector(".leg-expansion");
        const resolution = row.querySelector(".dimension-resolution");
        const template = String(leg.name);
        if (template.includes("$")) {
          try {
            const expanded = variables
              ? model.expandLegName(template, variables, `leg name of ${tensorName} axis ${index}`)
              : null;
            expansion.textContent = expanded == null ? "unresolved" : `→ ${expanded}`;
          } catch (_) {
            expansion.textContent = "unresolved";
          }
        } else expansion.textContent = "";

        const token = String(leg.dim).trim();
        if (token && !/^[1-9][0-9]*$/.test(token)) {
          try {
            const value = variables
              ? model.resolveDimension(token, variables, `dimension of ${tensorName}.${leg.name}`)
              : null;
            resolution.textContent = value == null ? "unresolved" : `= ${exactCount(value)}`;
          } catch (_) {
            resolution.textContent = "unresolved";
          }
        } else resolution.textContent = "";
      });
    }
  }

  function moveLeg(tensorName, from, to) {
    const legs = state[tensorName];
    if (to < 0 || to >= legs.length) return;
    [legs[from], legs[to]] = [legs[to], legs[from]];
    renderEditors();
    scheduleUpdate();
  }

  function addLeg(tensorName) {
    const defaultDimension = state.variables.some((variable) => variable.name === "d") ? "d" : "2";
    state[tensorName].push({ name: uniqueLegName(), dim: defaultDimension });
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

  function resolveModel() {
    const variables = model.normalizeVariables(state.variables);
    const resolvedInput = model.resolveNetworkVariables(networkInput(), variables);
    return {
      variables,
      network: model.normalizeNetwork(resolvedInput),
    };
  }

  function displayDimension(token, value, compact) {
    const text = String(token).trim();
    if (/^[1-9][0-9]*$/.test(text)) return exactCount(value);
    return compact ? `${text}=${exactCount(value)}` : `${text} (= ${exactCount(value)})`;
  }

  function displayContext(step) {
    const tokensA = state.A.map((leg) => String(leg.dim).trim());
    const tokensB = state.B.map((leg) => String(leg.dim).trim());
    return {
      tokensA,
      tokensB,
      outputTokens: step.freeA.map((leg) => tokensA[leg.axis]).concat(
        step.freeB.map((leg) => tokensB[leg.axis])
      ),
    };
  }

  function symbolicShape(legs, tokens, includeResolved) {
    if (!legs.length) return "rank 0 · scalar";
    const symbolic = `[${tokens.join(" × ")}]`;
    const resolved = `[${legs.map((leg) => leg.dim).join(" × ")}]`;
    const same = tokens.every((token, index) => token === legs[index].dim.toString());
    return `rank ${legs.length} · ${symbolic}${includeResolved && !same ? ` = ${resolved}` : ""}`;
  }

  function setMetric(element, value, exactValue) {
    element.textContent = value;
    element.title = exactValue == null ? "" : exactCount(exactValue);
  }

  function renderOutput(result, context) {
    const legs = result.tensor.legs;
    outputRank.textContent = symbolicShape(legs, context.outputTokens, true);
    if (!legs.length) {
      outputLegs.innerHTML = '<span class="output-leg scalar">scalar · storage shape [1]</span>';
      return;
    }
    outputLegs.innerHTML = legs
      .map((leg, index) => `<span class="output-leg">axis ${index} · ${escapeMarkup(leg.name)} : ${escapeMarkup(displayDimension(context.outputTokens[index], leg.dim, false))}</span>`)
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

  function renderLowering(step, context) {
    sharedLegs.textContent = step.shared.length
      ? step.shared.map((leg) => {
        const tokenA = context.tokensA[leg.axisA];
        const tokenB = context.tokensB[leg.axisB];
        const dimension = tokenA === tokenB
          ? displayDimension(tokenA, leg.dim, true)
          : `${tokenA}/${tokenB}=${exactCount(leg.dim)}`;
        return `${leg.name}:${dimension}`;
      }).join(", ")
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

  function freeLegMarkup(leg, side, rank, token) {
    const y = axisY(leg.axis, rank);
    const label = shorten(`${leg.name}:${displayDimension(token, leg.dim, true)}`, 22);
    if (side === "a") {
      return `<line class="svg-free-line a" x1="180" y1="${y}" x2="116" y2="${y}"></line>` +
        `<circle class="svg-free-port a" cx="180" cy="${y}" r="4"></circle>` +
        `<text class="svg-free-label a" x="107" y="${y + 4}">${escapeMarkup(label)}</text>`;
    }
    return `<line class="svg-free-line b" x1="820" y1="${y}" x2="884" y2="${y}"></line>` +
      `<circle class="svg-free-port b" cx="820" cy="${y}" r="4"></circle>` +
      `<text class="svg-free-label b" x="893" y="${y + 4}">${escapeMarkup(label)}</text>`;
  }

  function sharedLegMarkup(leg, index, count, rankA, rankB, tokenA, tokenB) {
    const yA = axisY(leg.axisA, rankA);
    const yB = axisY(leg.axisB, rankB);
    const labelY = count <= 1 ? 118 : 82 + (index * 72) / (count - 1);
    const dimension = tokenA === tokenB
      ? displayDimension(tokenA, leg.dim, true)
      : `${tokenA}/${tokenB}=${exactCount(leg.dim)}`;
    const label = shorten(`${leg.name}:${dimension}`, 25);
    const width = Math.min(176, Math.max(62, 22 + label.length * 7));
    return `<path class="svg-shared-line" d="M 370 ${yA} C 430 ${yA}, 450 ${labelY}, 500 ${labelY} C 550 ${labelY}, 570 ${yB}, 630 ${yB}"></path>` +
      `<circle class="svg-shared-port" cx="370" cy="${yA}" r="5"></circle>` +
      `<circle class="svg-shared-port" cx="630" cy="${yB}" r="5"></circle>` +
      `<rect class="svg-wire-label-bg" x="${500 - width / 2}" y="${labelY - 10}" width="${width}" height="20" rx="10"></rect>` +
      `<text class="svg-wire-label" x="500" y="${labelY + 4}">${escapeMarkup(label)}</text>`;
  }

  function renderDiagram(network, step, context) {
    const valid = Boolean(network && step);
    const tensorA = valid ? network.tensors[0] : { legs: state.A };
    const tensorB = valid ? network.tensors[1] : { legs: state.B };
    const shapeAText = valid ? symbolicShape(tensorA.legs, context.tokensA, false) : rawShape(state.A);
    const shapeBText = valid ? symbolicShape(tensorB.legs, context.tokensB, false) : rawShape(state.B);
    const resultShape = valid
      ? symbolicShape(step.resultLegs, context.outputTokens, false)
      : "waiting for valid input";
    const outputNames = valid && step.resultLegs.length
      ? shorten(step.resultLegs.map((leg, index) =>
        `${leg.name}:${displayDimension(context.outputTokens[index], leg.dim, true)}`
      ).join(" · "), 76)
      : valid ? "scalar · storage [1]" : "correct the leg model above";

    let wires = "";
    if (valid) {
      wires += step.freeA.map((leg) =>
        freeLegMarkup(leg, "a", tensorA.legs.length, context.tokensA[leg.axis])
      ).join("");
      wires += step.freeB.map((leg) =>
        freeLegMarkup(leg, "b", tensorB.legs.length, context.tokensB[leg.axis])
      ).join("");
      if (step.shared.length) {
        wires += step.shared
          .map((leg, index) => sharedLegMarkup(
            leg,
            index,
            step.shared.length,
            tensorA.legs.length,
            tensorB.legs.length,
            context.tokensA[leg.axisA],
            context.tokensB[leg.axisB]
          ))
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
    shapeA.title = "";
    shapeB.title = "";
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
    updateEditorAnnotations();
    try {
      const { network } = resolveModel();
      const result = model.evaluateTree(network, "A*B", {
        arithmetic: state.arithmetic,
        precision: state.precision,
      });
      const step = result.steps[0];
      const context = displayContext(step);
      shapeA.textContent = rawShape(state.A);
      shapeA.title = normalizedShape(network.tensors[0].legs);
      shapeB.textContent = rawShape(state.B);
      shapeB.title = normalizedShape(network.tensors[1].legs);
      validityBadge.textContent = step.outerProduct ? "Valid outer product" : "Valid contraction";
      validityBadge.classList.remove("invalid");
      errorBox.hidden = true;
      errorBox.textContent = "";
      renderDiagram(network, step, context);
      renderOutput(result, context);
      renderMetrics(result, step);
      renderLowering(step, context);
    } catch (error) {
      clearComputedDisplay(error);
    }
  }

  function initialize() {
    state = loadState();
    syncControls();
    renderVariables();
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
    document.getElementById("addVariableButton").addEventListener("click", addVariable);
    document.getElementById("resetButton").addEventListener("click", () => {
      state = cloneExample();
      syncControls();
      renderVariables();
      renderEditors();
      scheduleUpdate();
    });
    updateWorkbench();
  }

  initialize();
})();
