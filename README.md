# Tensor Network Visualiser

This repository contains two browser tools:

- `contraction-planner.html` is a focused two-tensor contraction workbench. It connects matching
  string-named legs, displays the output tensor, and reports the exact reshape and cost live.
- `tensor-network-visualiser.html` is the existing interactive tensor-network canvas. Its
  numerical MPS/MPO contraction action can use the optional Julia server in `server.jl`.

## Two-tensor contraction workbench

Open `contraction-planner.html` directly in a browser. No build, package install, or backend is
required. A local web server is also convenient:

```sh
python3 -m http.server 8765
```

Then visit <http://127.0.0.1:8765/contraction-planner.html>.

The workbench always contains input tensors A and B and output tensor C. Edit the ordered leg
rows beneath A and B:

- a leg name appearing on both tensors is contracted and connected in the diagram;
- a name appearing on only one tensor remains free and becomes an axis of C;
- names are exact and case-sensitive after trimming;
- dimensions are arbitrary-size positive integers or registered variables;
- shared names must have the same dimension on both sides.

Leg rows may be added, removed, and reordered, but the workflow remains one fixed contraction,
`A * B`. C updates immediately and orders its axes as free A legs followed by free B legs.

The controls above the panel select real or complex arithmetic and f32 or f64 storage. The
variable registry starts with `D = 4` and `d = 2`. Press `+` to register another variable, then
edit its case-sensitive identifier and positive integer value. A dimension field accepts either
an integer literal or a registered name such as `D`.

Leg names can interpolate the same variables:

```text
$row                  direct substitution
$(col + 1)            integer expression
r$row,c$(col)-$(col+1)
```

For `row = 3` and `col = 4`, the last template expands to `r3,c4-5`. The raw template remains in
the editor, while expanded names determine duplicate detection, shared-leg connections, and C's
axis names. `$$` produces a literal dollar sign.

Interpolation expressions support integer literals, registered variables, parentheses, unary
`+`/`-`, and binary `+`, `-`, `*`, `/`, and `%`. Division must be exact. Dimension fields remain
deliberately simpler: they accept a literal or one variable name, not an expression.

Every valid edit updates:

- the shared-leg connections and output dimensions;
- scalar multiply-accumulate positions and conventional real FLOPs;
- output storage, two-operand permutation scratch, and per-step workspace;
- the exact A and B axis permutations;
- the `M x K` by `K x N` matrix multiplication.

The lowering matches the current cuQuantumNaturalfPEPS Julia and CUDA helpers:

```text
A [free-A, shared] -> M x K
B [shared, free-B] -> K x N
C [free-A, free-B]
```

Shared axes are ordered by their position on A, and B is permuted to the same name order. One real
MAC is reported as two real FLOPs and one complex MAC as eight. Counts use JavaScript `BigInt`, so
large dimensions are not rounded.

The reusable model in `contraction-model.js` still contains the tested manual-order evaluator and
multi-tensor exact/heuristic optimizer. They are deliberately not exposed in this first fixed
workflow. The current interaction contract is in `DESIGN.md`.

## Tests

The model tests require Node.js:

```sh
node --test contraction-model.test.mjs
```

The end-to-end test additionally requires Google Chrome. Override the executable with `CHROME` if
needed:

```sh
node browser-smoke.mjs
```

It opens the real page in headless Chrome and checks the fixed A/B editors, default and newly
registered variables, `$` interpolation, automatic connection changes, C's axes, cost totals,
arithmetic and precision controls, invalid dimensions, and the link back to the existing canvas.

## Optional Julia contraction server

The visual canvas can call its existing local Julia endpoint for numerical MPS/MPO contractions:

```sh
julia --project=. server.jl
```

It listens on `http://127.0.0.1:8754`. The abstract contraction workbench is independent of this
server and does not execute tensor values or CUDA kernels.
