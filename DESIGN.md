# Two-tensor contraction workbench

## Workflow

The primary browser tool models one contraction, `A * B`. The number of tensors and their names
are fixed. The user edits ordered `(leg name, dimension)` rows beneath A and B; leg rows may be
added, removed, or reordered.

The central panel updates immediately:

- A and B are visible tensor nodes;
- equal, case-sensitive leg names are joined by labeled lines;
- unmatched legs remain free;
- the output tensor `C = A * B` is visible below the inputs with its ordered leg names and
  dimensions;
- invalid or incomplete input is reported in the panel instead of producing a cost.

Real/complex arithmetic, f32/f64 precision, and a leg-variable registry sit above the panel. There
is no contraction-order input and no optimizer in this first workflow. The tested multi-tensor
planning logic remains available in `contraction-model.js` for a later interface.

## Leg variables

The registry starts with `D = 4` and `d = 2`. Its `+` control registers another variable with a
unique editable identifier and positive integer value. Variables may be edited or removed.

A dimension field accepts either a positive integer literal or the exact, case-sensitive name of
a registered variable. Variable values are resolved before calling the numerical contraction
model; changing a value therefore updates C, FLOPs, and storage immediately. Symbolic names remain
visible on tensor shapes, connections, and output axes, alongside their resolved values where that
helps interpretation.

Variable identifiers use `[A-Za-z_][A-Za-z0-9_]*`. Values are positive integer literals; variable
aliases and dimension expressions are not part of this version. Duplicate names, unknown names,
and non-positive or non-integer values invalidate the live contraction with a specific message.

Leg names are templates. `$name` inserts a registered variable directly, and `$(expression)`
inserts an integer expression using registered variables. For example, with `row = 3` and
`col = 4`, `r$row,c$(col)-$(col+1)` expands to `r3,c4-5`. Expanded names—not the raw templates—are
used for duplicate detection, shared-leg matching, output axes, and connection labels. The editor
keeps the template visible and shows its expansion.

Interpolation expressions are parsed without `eval` and support integer literals, variables,
parentheses, unary `+`/`-`, and binary `+`, `-`, `*`, `/`, and `%`. Division must be exact and
division by zero is invalid. An interpolated result may be zero or negative because it labels an
index; registered variable values and tensor dimensions remain strictly positive.

## Lowering and cost

The workbench uses the same pairwise convention as the cuQuantumNaturalfPEPS Julia and CUDA
helpers:

1. retain free axes of A in their original order;
2. order shared axes by their appearance on A;
3. permute A to `[free-A, shared]` and reshape it to `M x K`;
4. permute B to `[shared, free-B]` and reshape it to `K x N`;
5. matrix-multiply and reshape C to `[free-A, free-B]`.

The display reports the exact axis permutations, `M`, `K`, `N`, scalar MAC positions, real FLOPs,
output storage, two-operand permutation scratch, and per-step workspace. A real MAC counts as two
real FLOPs and a complex MAC as eight. Counts use arbitrary-precision integers.

## Model boundary

- Leg names are exact strings after trimming and may occur at most once per tensor.
- A shared name must have the same positive integer dimension on A and B.
- Dangling legs, multiple shared legs, complete scalar contractions, and outer products work.
- Self-traces, tensor values, decompositions, and CUDA execution remain out of scope.

## Acceptance

- Opening `contraction-planner.html` directly produces a usable two-tensor workbench.
- Editing a name or dimension updates connections, C, and all totals without an Evaluate button.
- Multiple shared legs show distinct labeled connections and the correct right-hand permutation.
- Arithmetic and precision controls update FLOP/storage totals live.
- `D = 4` and `d = 2` exist initially; pressing `+` registers an editable variable.
- Dimension fields accept registered variables, and variable edits propagate through the diagram,
  C, and every cost metric.
- `$name` and `$(integer expression)` expand inside leg names and determine live connections.
- Duplicate, unknown, and invalid variables fail visibly without losing the symbolic input.
- The layout remains usable on desktop and narrow screens.
- Model unit tests and a headless browser interaction test cover the lowering, live edits, controls,
  invalid dimensions, and the link back to the existing visual canvas.
