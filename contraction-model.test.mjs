import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const model = require("./contraction-model.js");

const matrixChain = [
  { name: "A", legs: [{ name: "i", dim: 10 }, { name: "k", dim: 20 }] },
  { name: "B", legs: [{ name: "k", dim: 20 }, { name: "j", dim: 30 }] },
  { name: "C", legs: [{ name: "j", dim: 30 }, { name: "l", dim: 40 }] },
];

test("pair lowering follows free-left/shared and shared/free-right ordering", () => {
  const network = model.normalizeNetwork([
    {
      name: "A",
      legs: [
        { name: "batch", dim: 2 },
        { name: "shared_y", dim: 3 },
        { name: "shared_x", dim: 5 },
        { name: "a_free", dim: 7 },
      ],
    },
    {
      name: "B",
      legs: [
        { name: "b_free", dim: 11 },
        { name: "shared_x", dim: 5 },
        { name: "out", dim: 13 },
        { name: "shared_y", dim: 3 },
      ],
    },
  ]);
  const plan = model.buildPairPlan(network.tensors[0], network.tensors[1]);
  assert.deepEqual(plan.shared.map((leg) => leg.name), ["shared_y", "shared_x"]);
  assert.deepEqual(plan.permutationA, [0, 3, 1, 2]);
  assert.deepEqual(plan.permutationB, [3, 1, 0, 2]);
  assert.deepEqual(plan.permutedLegsA, ["batch", "a_free", "shared_y", "shared_x"]);
  assert.deepEqual(plan.permutedLegsB, ["shared_y", "shared_x", "b_free", "out"]);
  assert.equal(plan.M, 14n);
  assert.equal(plan.K, 15n);
  assert.equal(plan.N, 143n);
  assert.equal(plan.macs, 30030n);
  assert.deepEqual(plan.resultLegs, [
    { name: "batch", dim: 2n },
    { name: "a_free", dim: 7n },
    { name: "b_free", dim: 11n },
    { name: "out", dim: 13n },
  ]);
});

test("a complete contraction is represented as a logical scalar with one storage element", () => {
  const result = model.evaluateTree(
    [
      { name: "A", legs: [{ name: "x", dim: 2 }, { name: "y", dim: 3 }] },
      { name: "B", legs: [{ name: "y", dim: 3 }, { name: "x", dim: 2 }] },
    ],
    "A*B",
    { arithmetic: "real", precision: "f64" }
  );
  assert.deepEqual(result.tensor.legs, []);
  assert.deepEqual(result.steps[0].storageShape, [1n]);
  assert.equal(result.totalMacs, 6n);
  assert.equal(result.realFlops, 12n);
  assert.equal(result.resultBytes, 8n);
});

test("manual orders are binary trees and must use every tensor once", () => {
  const result = model.evaluateTree(matrixChain, "A*(B*C)", {
    arithmetic: "complex",
    precision: "f32",
  });
  assert.equal(result.order, "(A*(B*C))");
  assert.equal(result.totalMacs, 32000n);
  assert.equal(result.realFlops, 256000n);
  assert.deepEqual(result.tensor.legs, [
    { name: "i", dim: 10n },
    { name: "l", dim: 40n },
  ]);
  assert.throws(() => model.evaluateTree(matrixChain, "A*B"), /omits tensor C/);
  assert.throws(() => model.evaluateTree(matrixChain, "A*(B*A)"), /uses tensor A 2 times/);
  assert.throws(() => model.evaluateTree(matrixChain, "A*(B*D)"), /unknown tensor D/);
});

test("the exact optimizer finds the matrix-chain minimum", () => {
  const optimized = model.optimizeNetwork(matrixChain, {
    arithmetic: "real",
    precision: "f32",
  });
  assert.equal(optimized.exact, true);
  assert.equal(optimized.minFlops.order, "((A*B)*C)");
  assert.equal(optimized.minFlops.totalMacs, 18000n);
  assert.equal(optimized.minFlops.realFlops, 36000n);
  assert.equal(optimized.minPeak.peakIntermediateElements, 400n);
});

test("minimum FLOPs and minimum peak intermediate are independent objectives", () => {
  const network = [
    { name: "A", legs: [{ name: "e0", dim: 6 }, { name: "f0", dim: 4 }] },
    {
      name: "B",
      legs: [
        { name: "e1", dim: 6 },
        { name: "e2", dim: 6 },
        { name: "f1", dim: 5 },
      ],
    },
    {
      name: "C",
      legs: [
        { name: "e0", dim: 6 },
        { name: "e1", dim: 6 },
        { name: "e3", dim: 6 },
        { name: "f2", dim: 5 },
      ],
    },
    { name: "D", legs: [{ name: "f3", dim: 2 }] },
    {
      name: "E",
      legs: [
        { name: "e2", dim: 6 },
        { name: "e3", dim: 6 },
        { name: "f4", dim: 3 },
      ],
    },
  ];
  const optimized = model.optimizeNetwork(network, {
    arithmetic: "real",
    precision: "f32",
  });
  assert.equal(optimized.minFlops.totalMacs, 18960n);
  assert.equal(optimized.minFlops.peakIntermediateElements, 720n);
  assert.equal(optimized.minPeak.totalMacs, 21840n);
  assert.equal(optimized.minPeak.peakIntermediateElements, 600n);
});

test("outer products and dimensions beyond Number precision remain exact", () => {
  const huge = "9007199254740993";
  const result = model.evaluateTree(
    [
      { name: "A", legs: [{ name: "left", dim: huge }] },
      { name: "B", legs: [{ name: "right", dim: 3 }] },
    ],
    "A*B",
    { arithmetic: "complex", precision: "f64" }
  );
  assert.equal(result.steps[0].outerProduct, true);
  assert.equal(result.totalMacs, 27021597764222979n);
  assert.equal(result.realFlops, 216172782113783832n);
  assert.equal(result.resultBytes, 432345564227567664n);
});

test("registered variables resolve dimensions and interpolate leg names", () => {
  const variables = model.normalizeVariables([
    { name: "D", value: "4" },
    { name: "d", value: "2" },
    { name: "row", value: "3" },
    { name: "col", value: "4" },
  ]);
  assert.equal(model.resolveDimension("D", variables), 4n);
  assert.equal(model.resolveDimension("9007199254740993", variables), 9007199254740993n);
  assert.equal(model.parseIntegerExpression("2 * (col - 1)", variables), 6n);
  assert.equal(
    model.expandLegName("r$row,c$(col)-$(col+1)", variables),
    "r3,c4-5"
  );
  assert.equal(model.expandLegName("cash$$$(row-4)", variables), "cash$-1");

  const resolved = model.resolveNetworkVariables(
    [
      { name: "A", legs: [{ name: "r$row,c$(col)-$(col+1)", dim: "D" }] },
      { name: "B", legs: [{ name: "r3,c4-5", dim: "4" }] },
    ],
    variables
  );
  assert.equal(resolved[0].legs[0].name, "r3,c4-5");
  assert.equal(resolved[0].legs[0].dim, 4n);
  assert.deepEqual(model.normalizeNetwork(resolved).tensors[1].legs[0], {
    name: "r3,c4-5",
    dim: 4n,
  });
});

test("invalid variable registries, references, and interpolation expressions fail loudly", () => {
  assert.throws(
    () => model.normalizeVariables([{ name: "D", value: 4 }, { name: "D", value: 2 }]),
    /duplicated/
  );
  assert.throws(() => model.normalizeVariables([{ name: "D", value: "0" }]), /positive integer/);
  assert.throws(() => model.normalizeVariables([{ name: "not-valid", value: 2 }]), /identifier/);
  const variables = model.normalizeVariables([{ name: "col", value: 4 }]);
  assert.throws(() => model.resolveDimension("D", variables), /unknown variable 'D'/);
  assert.throws(() => model.resolveDimension("col+1", variables), /positive integer or registered/);
  assert.throws(() => model.expandLegName("r$row", variables), /unknown variable 'row'/);
  assert.throws(() => model.expandLegName("c$(col+1", variables), /unclosed/);
  assert.throws(() => model.expandLegName("c$(col/3)", variables), /exact integer division/);
  assert.throws(() => model.expandLegName("c$(col%0)", variables), /divides by zero/);
});

test("invalid leg models fail loudly", () => {
  assert.throws(
    () =>
      model.normalizeNetwork([
        { name: "A", legs: [{ name: "x", dim: 2 }, { name: "x", dim: 2 }] },
        { name: "B", legs: [] },
      ]),
    /self-traces/
  );
  assert.throws(
    () =>
      model.normalizeNetwork([
        { name: "A", legs: [{ name: "x", dim: 2 }] },
        { name: "B", legs: [{ name: "x", dim: 3 }] },
      ]),
    /mismatched dimensions/
  );
  assert.throws(
    () =>
      model.normalizeNetwork([
        { name: "A", legs: [{ name: "x", dim: 2 }] },
        { name: "B", legs: [{ name: "x", dim: 2 }] },
        { name: "C", legs: [{ name: "x", dim: 2 }] },
      ]),
    /hyperedges/
  );
  assert.throws(
    () =>
      model.normalizeNetwork([
        { name: "A", legs: [{ name: "x", dim: "2.5" }] },
        { name: "B", legs: [] },
      ]),
    /positive integer/
  );
});

test("order parser accepts left associativity and rejects malformed expressions", () => {
  assert.equal(model.orderExpression(model.parseOrder("A*B*C")), "((A*B)*C)");
  assert.equal(model.orderExpression(model.parseOrder("A*(B*C)")), "(A*(B*C))");
  assert.throws(() => model.parseOrder("A*(B*C"), /missing/);
  assert.throws(() => model.parseOrder("A+B"), /unexpected '\+'/);
});
