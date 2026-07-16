# Tensor Network Visualiser

This repository contains two browser tools:

- `contraction-planner.html` models abstract tensor contractions from ordered, string-named legs.
  It evaluates a supplied pairwise order or searches for a minimum-FLOP or
  minimum-peak-intermediate order.
- `tensor-network-visualiser.html` is the existing interactive tensor-network canvas. Its
  numerical MPS/MPO contraction action can use the optional Julia server in `server.jl`.

## Contraction planner

Open `contraction-planner.html` directly in a browser. No build, package install, or backend is
required. A local web server is also convenient:

```sh
python3 -m http.server 8765
```

Then visit <http://127.0.0.1:8765/contraction-planner.html>.

Each tensor has a unique identifier and an ordered list of legs. A leg is an exact,
case-sensitive string name and a positive integer dimension. A name appearing on two different
tensors identifies a contracted pair; a name appearing once is a free leg. The first release
rejects self-traces and names appearing on more than two tensors.

Manual orders use tensor identifiers, `*`, and parentheses. Whitespace is ignored, and an
unparenthesized expression is left-associative:

```text
A * B * C
A * (B * C)
((A * B) * C)
```

For every binary step, the planner shows the actual reshape-to-matrix-multiply lowering:

```text
left  [free-left, shared]  -> M x K
right [shared, free-right] -> K x N
result [free-left, free-right]
```

Shared axes are ordered by their position on the left operand. The right operand is permuted to
the same shared-name order. This mirrors the current cuQuantumNaturalfPEPS Julia and CUDA
contraction helpers and makes the required axis permutations explicit.

The scalar MAC count for a step is `M*K*N`. The UI reports that exact integer separately from
conventional real FLOPs: two FLOPs per real MAC or eight per complex MAC. Choosing `f32` or `f64`
changes storage and workspace estimates, but not the arithmetic count. Counts use JavaScript
`BigInt`, so large dimensions are not rounded.

Both optimization objectives consider all binary contraction trees. Search is exact subset
dynamic programming through 14 input tensors. Larger networks use a deterministic greedy
fallback and are clearly labeled heuristic. The detailed scope and acceptance criteria are in
`DESIGN.md`.

## Tests

The model tests require Node.js:

```sh
node --test contraction-model.test.mjs
```

The end-to-end smoke test additionally requires Google Chrome (override its executable with the
`CHROME` environment variable):

```sh
node browser-smoke.mjs
```

The smoke test opens the real page in headless Chrome, checks a multi-leg permutation, verifies
the exact matrix-chain optimum and FLOP count, and confirms dimension mismatches surface as UI
errors.

## Optional Julia contraction server

The visual canvas can call its existing local Julia endpoint for numerical MPS/MPO contractions:

```sh
julia --project=. server.jl
```

It listens on `http://127.0.0.1:8754`. The abstract contraction planner is independent of this
server and does not execute tensor values or CUDA kernels.
