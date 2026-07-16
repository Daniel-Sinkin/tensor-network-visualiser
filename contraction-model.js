(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.ContractionModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const EXACT_TENSOR_LIMIT = 14;
  const TENSOR_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

  class PlannerError extends Error {
    constructor(message) {
      super(message);
      this.name = "PlannerError";
    }
  }

  function fail(message) {
    throw new PlannerError(message);
  }

  function product(values) {
    let out = 1n;
    for (const value of values) out *= value;
    return out;
  }

  function parseDimension(value, context) {
    if (typeof value === "bigint") {
      if (value < 1n) fail(`${context} must be a positive integer`);
      return value;
    }
    if (typeof value === "number") {
      if (!Number.isSafeInteger(value) || value < 1)
        fail(`${context} must be a positive safe integer or an integer string`);
      return BigInt(value);
    }
    const text = String(value == null ? "" : value).trim();
    if (!/^[1-9][0-9]*$/.test(text)) fail(`${context} must be a positive integer`);
    return BigInt(text);
  }

  function normalizeArithmetic(arithmetic) {
    const value = arithmetic || "complex";
    if (value !== "real" && value !== "complex")
      fail(`unknown arithmetic domain '${value}'`);
    return value;
  }

  function normalizePrecision(precision) {
    const value = precision || "f32";
    if (value !== "f32" && value !== "f64") fail(`unknown precision '${value}'`);
    return value;
  }

  function flopFactor(arithmetic) {
    return normalizeArithmetic(arithmetic) === "complex" ? 8n : 2n;
  }

  function scalarBytes(arithmetic, precision) {
    const domain = normalizeArithmetic(arithmetic);
    const bits = normalizePrecision(precision);
    const realBytes = bits === "f32" ? 4n : 8n;
    return domain === "complex" ? 2n * realBytes : realBytes;
  }

  function normalizeNetwork(input) {
    if (!Array.isArray(input) || input.length < 2) fail("enter at least two tensors");
    const names = new Set();
    const occurrences = new Map();
    const tensors = input.map((source, tensorIndex) => {
      const name = String(source && source.name != null ? source.name : "").trim();
      if (!TENSOR_NAME_RE.test(name))
        fail(`tensor ${tensorIndex + 1} needs a unique identifier such as A or tensor_1`);
      if (names.has(name)) fail(`tensor name '${name}' is duplicated`);
      names.add(name);
      if (!Array.isArray(source.legs)) fail(`tensor ${name} has no leg list`);
      const localNames = new Set();
      const legs = source.legs.map((sourceLeg, legIndex) => {
        const legName = String(sourceLeg && sourceLeg.name != null ? sourceLeg.name : "").trim();
        if (!legName) fail(`tensor ${name}, leg ${legIndex + 1} needs a name`);
        if (localNames.has(legName))
          fail(`tensor ${name} repeats leg '${legName}'; self-traces are not supported yet`);
        localNames.add(legName);
        const dim = parseDimension(sourceLeg.dim, `dimension of ${name}.${legName}`);
        const occurrence = { tensorIndex, legIndex, tensorName: name, dim };
        if (!occurrences.has(legName)) occurrences.set(legName, []);
        occurrences.get(legName).push(occurrence);
        return { name: legName, dim };
      });
      return { name, expression: name, legs };
    });

    for (const [legName, holders] of occurrences) {
      if (holders.length > 2)
        fail(`leg '${legName}' occurs ${holders.length} times; hyperedges are not supported yet`);
      if (holders.length === 2 && holders[0].dim !== holders[1].dim) {
        fail(
          `leg '${legName}' has mismatched dimensions ` +
            `${holders[0].dim} on ${holders[0].tensorName} and ` +
            `${holders[1].dim} on ${holders[1].tensorName}`
        );
      }
    }

    return {
      tensors,
      tensorByName: new Map(tensors.map((tensor) => [tensor.name, tensor])),
      occurrences,
    };
  }

  function legElements(legs) {
    return product(legs.map((leg) => leg.dim));
  }

  function isIdentityPermutation(permutation) {
    return permutation.every((axis, index) => axis === index);
  }

  function buildPairPlan(left, right) {
    const rightAxis = new Map(right.legs.map((leg, axis) => [leg.name, axis]));
    const shared = [];
    const sharedNames = new Set();
    for (let axisA = 0; axisA < left.legs.length; axisA++) {
      const legA = left.legs[axisA];
      if (!rightAxis.has(legA.name)) continue;
      const axisB = rightAxis.get(legA.name);
      const legB = right.legs[axisB];
      if (legA.dim !== legB.dim)
        fail(
          `leg '${legA.name}' has mismatched dimensions ${legA.dim} and ${legB.dim}`
        );
      shared.push({ name: legA.name, dim: legA.dim, axisA, axisB });
      sharedNames.add(legA.name);
    }

    const freeA = left.legs
      .map((leg, axis) => ({ name: leg.name, dim: leg.dim, axis }))
      .filter((leg) => !sharedNames.has(leg.name));
    const freeB = right.legs
      .map((leg, axis) => ({ name: leg.name, dim: leg.dim, axis }))
      .filter((leg) => !sharedNames.has(leg.name));
    const permutationA = freeA.map((leg) => leg.axis).concat(shared.map((leg) => leg.axisA));
    const permutationB = shared.map((leg) => leg.axisB).concat(freeB.map((leg) => leg.axis));
    const permutedLegsA = freeA.concat(shared).map((leg) => leg.name);
    const permutedLegsB = shared.concat(freeB).map((leg) => leg.name);
    const M = product(freeA.map((leg) => leg.dim));
    const K = product(shared.map((leg) => leg.dim));
    const N = product(freeB.map((leg) => leg.dim));
    const resultLegs = freeA.concat(freeB).map((leg) => ({ name: leg.name, dim: leg.dim }));
    const resultElements = M * N;
    const leftElements = M * K;
    const rightElements = K * N;
    const expression = `(${left.expression}*${right.expression})`;

    return {
      leftExpression: left.expression,
      rightExpression: right.expression,
      expression,
      shared,
      freeA,
      freeB,
      permutationA,
      permutationB,
      permutedLegsA,
      permutedLegsB,
      identityA: isIdentityPermutation(permutationA),
      identityB: isIdentityPermutation(permutationB),
      M,
      K,
      N,
      macs: M * K * N,
      leftElements,
      rightElements,
      permutationScratchElements: leftElements + rightElements,
      resultElements,
      resultLegs,
      storageShape: resultLegs.length ? resultLegs.map((leg) => leg.dim) : [1n],
      outerProduct: shared.length === 0,
      tensor: { name: expression, expression, legs: resultLegs },
    };
  }

  function tokenizeOrder(source) {
    const text = String(source == null ? "" : source);
    const tokens = [];
    let offset = 0;
    while (offset < text.length) {
      const ch = text[offset];
      if (/\s/.test(ch)) {
        offset++;
        continue;
      }
      if (ch === "(" || ch === ")" || ch === "*") {
        tokens.push({ type: ch, text: ch, offset });
        offset++;
        continue;
      }
      if (/[A-Za-z_]/.test(ch)) {
        const start = offset++;
        while (offset < text.length && /[A-Za-z0-9_]/.test(text[offset])) offset++;
        tokens.push({ type: "name", text: text.slice(start, offset), offset: start });
        continue;
      }
      fail(`unexpected '${ch}' at position ${offset + 1} in contraction order`);
    }
    return tokens;
  }

  function parseOrder(source) {
    const tokens = tokenizeOrder(source);
    if (!tokens.length) fail("enter a contraction order such as (A*B)");
    let cursor = 0;

    function primary() {
      const token = tokens[cursor];
      if (!token) fail("contraction order ended early");
      if (token.type === "name") {
        cursor++;
        return { type: "leaf", name: token.text };
      }
      if (token.type === "(") {
        cursor++;
        const node = expression();
        if (!tokens[cursor] || tokens[cursor].type !== ")")
          fail(`missing ')' in contraction order`);
        cursor++;
        return node;
      }
      fail(`expected a tensor name at position ${token.offset + 1}`);
    }

    function expression() {
      let node = primary();
      while (tokens[cursor] && tokens[cursor].type === "*") {
        cursor++;
        node = { type: "pair", left: node, right: primary() };
      }
      return node;
    }

    const tree = expression();
    if (cursor !== tokens.length) {
      const token = tokens[cursor];
      fail(`unexpected '${token.text}' at position ${token.offset + 1}`);
    }
    return tree;
  }

  function orderExpression(tree) {
    if (tree.type === "leaf") return tree.name;
    return `(${orderExpression(tree.left)}*${orderExpression(tree.right)})`;
  }

  function treeLeaves(tree, out) {
    if (tree.type === "leaf") out.push(tree.name);
    else {
      treeLeaves(tree.left, out);
      treeLeaves(tree.right, out);
    }
    return out;
  }

  function maxBigInt(...values) {
    let out = values[0];
    for (let index = 1; index < values.length; index++)
      if (values[index] > out) out = values[index];
    return out;
  }

  function evaluateTree(networkInput, treeInput, options) {
    const network = networkInput && networkInput.tensorByName
      ? networkInput
      : normalizeNetwork(networkInput);
    const tree = typeof treeInput === "string" ? parseOrder(treeInput) : treeInput;
    const leaves = treeLeaves(tree, []);
    const counts = new Map();
    for (const name of leaves) counts.set(name, (counts.get(name) || 0) + 1);
    for (const name of counts.keys())
      if (!network.tensorByName.has(name)) fail(`contraction order references unknown tensor ${name}`);
    for (const tensor of network.tensors) {
      const count = counts.get(tensor.name) || 0;
      if (count === 0) fail(`contraction order omits tensor ${tensor.name}`);
      if (count > 1) fail(`contraction order uses tensor ${tensor.name} ${count} times`);
    }

    function visit(node) {
      if (node.type === "leaf") {
        const tensor = network.tensorByName.get(node.name);
        return {
          tensor,
          steps: [],
          totalMacs: 0n,
          peakIntermediateElements: 0n,
          peakPermutationScratchElements: 0n,
          peakWorkspaceElements: 0n,
        };
      }
      const left = visit(node.left);
      const right = visit(node.right);
      const plan = buildPairPlan(left.tensor, right.tensor);
      return {
        tensor: plan.tensor,
        steps: left.steps.concat(right.steps, [plan]),
        totalMacs: left.totalMacs + right.totalMacs + plan.macs,
        peakIntermediateElements: maxBigInt(
          left.peakIntermediateElements,
          right.peakIntermediateElements,
          plan.resultElements
        ),
        peakPermutationScratchElements: maxBigInt(
          left.peakPermutationScratchElements,
          right.peakPermutationScratchElements,
          plan.permutationScratchElements
        ),
        peakWorkspaceElements: maxBigInt(
          left.peakWorkspaceElements,
          right.peakWorkspaceElements,
          plan.permutationScratchElements + plan.resultElements
        ),
      };
    }

    const result = visit(tree);
    const settings = options || {};
    const arithmetic = normalizeArithmetic(settings.arithmetic);
    const precision = normalizePrecision(settings.precision);
    const factor = flopFactor(arithmetic);
    const bytes = scalarBytes(arithmetic, precision);
    result.order = orderExpression(tree);
    result.tree = tree;
    result.arithmetic = arithmetic;
    result.precision = precision;
    result.flopFactor = factor;
    result.realFlops = result.totalMacs * factor;
    result.scalarBytes = bytes;
    result.resultElements = legElements(result.tensor.legs);
    result.resultBytes = result.resultElements * bytes;
    result.peakIntermediateBytes = result.peakIntermediateElements * bytes;
    result.peakPermutationScratchBytes = result.peakPermutationScratchElements * bytes;
    result.peakWorkspaceBytes = result.peakWorkspaceElements * bytes;
    result.steps.forEach((step, index) => {
      step.index = index + 1;
      step.realFlops = step.macs * factor;
      step.resultBytes = step.resultElements * bytes;
      step.permutationScratchBytes = step.permutationScratchElements * bytes;
      step.workspaceBytes = (step.permutationScratchElements + step.resultElements) * bytes;
    });
    return result;
  }

  function popcount(value) {
    let count = 0;
    for (let x = value; x; x &= x - 1) count++;
    return count;
  }

  function subsetData(network, mask) {
    const legs = [];
    for (const [name, holders] of network.occurrences) {
      let inside = null;
      let count = 0;
      for (const holder of holders) {
        if (mask & (1 << holder.tensorIndex)) {
          count++;
          inside = holder;
        }
      }
      if (count === 1) legs.push({ name, dim: inside.dim });
    }
    return { legs, elements: legElements(legs) };
  }

  function unionElements(left, right) {
    const dims = new Map();
    for (const leg of left.legs) dims.set(leg.name, leg.dim);
    for (const leg of right.legs) dims.set(leg.name, leg.dim);
    return product(dims.values());
  }

  function leafTree(name) {
    return { type: "leaf", name };
  }

  function exactOptimize(network, options) {
    const n = network.tensors.length;
    const count = 1 << n;
    const subsets = new Array(count);
    for (let mask = 1; mask < count; mask++) subsets[mask] = subsetData(network, mask);
    const bestFlops = new Array(count);
    const bestPeak = new Array(count);
    for (let index = 0; index < n; index++) {
      const mask = 1 << index;
      const entry = {
        cost: 0n,
        peak: 0n,
        tree: leafTree(network.tensors[index].name),
      };
      bestFlops[mask] = entry;
      bestPeak[mask] = entry;
    }

    for (let bits = 2; bits <= n; bits++) {
      for (let mask = 1; mask < count; mask++) {
        if (popcount(mask) !== bits) continue;
        const low = mask & -mask;
        let selectedFlops = null;
        let selectedPeak = null;
        for (let sub = (mask - 1) & mask; sub; sub = (sub - 1) & mask) {
          if (!(sub & low)) continue;
          const rest = mask ^ sub;
          if (!rest) continue;
          const pairMacs = unionElements(subsets[sub], subsets[rest]);
          const outputElements = subsets[mask].elements;
          const flopCandidate = {
            cost: bestFlops[sub].cost + bestFlops[rest].cost + pairMacs,
            peak: maxBigInt(bestFlops[sub].peak, bestFlops[rest].peak, outputElements),
            tree: { type: "pair", left: bestFlops[sub].tree, right: bestFlops[rest].tree },
          };
          if (
            !selectedFlops ||
            flopCandidate.cost < selectedFlops.cost ||
            (flopCandidate.cost === selectedFlops.cost && flopCandidate.peak < selectedFlops.peak)
          )
            selectedFlops = flopCandidate;

          const peakCandidate = {
            cost: bestPeak[sub].cost + bestPeak[rest].cost + pairMacs,
            peak: maxBigInt(bestPeak[sub].peak, bestPeak[rest].peak, outputElements),
            tree: { type: "pair", left: bestPeak[sub].tree, right: bestPeak[rest].tree },
          };
          if (
            !selectedPeak ||
            peakCandidate.peak < selectedPeak.peak ||
            (peakCandidate.peak === selectedPeak.peak && peakCandidate.cost < selectedPeak.cost)
          )
            selectedPeak = peakCandidate;
        }
        bestFlops[mask] = selectedFlops;
        bestPeak[mask] = selectedPeak;
      }
    }

    const full = count - 1;
    return {
      exact: true,
      method: "exact subset dynamic programming",
      minFlops: evaluateTree(network, bestFlops[full].tree, options),
      minPeak: evaluateTree(network, bestPeak[full].tree, options),
    };
  }

  function greedyTree(network, objective) {
    let states = network.tensors.map((tensor) => ({
      tensor,
      tree: leafTree(tensor.name),
      peak: 0n,
      cost: 0n,
    }));
    while (states.length > 1) {
      let selected = null;
      for (let leftIndex = 0; leftIndex < states.length; leftIndex++) {
        for (let rightIndex = leftIndex + 1; rightIndex < states.length; rightIndex++) {
          const left = states[leftIndex];
          const right = states[rightIndex];
          const plan = buildPairPlan(left.tensor, right.tensor);
          const candidate = {
            leftIndex,
            rightIndex,
            plan,
            immediate: plan.macs,
            output: plan.resultElements,
            expression: `(${orderExpression(left.tree)}*${orderExpression(right.tree)})`,
          };
          const candidateKey = objective === "peak"
            ? [candidate.output, candidate.immediate, candidate.expression]
            : [candidate.immediate, candidate.output, candidate.expression];
          const selectedKey = selected && (objective === "peak"
            ? [selected.output, selected.immediate, selected.expression]
            : [selected.immediate, selected.output, selected.expression]);
          if (
            !selected ||
            candidateKey[0] < selectedKey[0] ||
            (candidateKey[0] === selectedKey[0] && candidateKey[1] < selectedKey[1]) ||
            (candidateKey[0] === selectedKey[0] &&
              candidateKey[1] === selectedKey[1] &&
              candidateKey[2] < selectedKey[2])
          )
            selected = candidate;
        }
      }
      const left = states[selected.leftIndex];
      const right = states[selected.rightIndex];
      const merged = {
        tensor: selected.plan.tensor,
        tree: { type: "pair", left: left.tree, right: right.tree },
        peak: maxBigInt(left.peak, right.peak, selected.plan.resultElements),
        cost: left.cost + right.cost + selected.plan.macs,
      };
      states = states.filter(
        (_, index) => index !== selected.leftIndex && index !== selected.rightIndex
      );
      states.push(merged);
    }
    return states[0].tree;
  }

  function optimizeNetwork(networkInput, options) {
    const network = networkInput && networkInput.tensorByName
      ? networkInput
      : normalizeNetwork(networkInput);
    if (network.tensors.length <= EXACT_TENSOR_LIMIT)
      return exactOptimize(network, options || {});
    return {
      exact: false,
      method: `deterministic greedy heuristic above ${EXACT_TENSOR_LIMIT} tensors`,
      minFlops: evaluateTree(network, greedyTree(network, "flops"), options || {}),
      minPeak: evaluateTree(network, greedyTree(network, "peak"), options || {}),
    };
  }

  return {
    EXACT_TENSOR_LIMIT,
    PlannerError,
    buildPairPlan,
    evaluateTree,
    flopFactor,
    normalizeNetwork,
    optimizeNetwork,
    orderExpression,
    parseDimension,
    parseOrder,
    scalarBytes,
  };
});
