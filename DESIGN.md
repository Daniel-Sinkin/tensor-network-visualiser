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

Real/complex arithmetic and f32/f64 precision controls sit above the panel. There is no
contraction-order input and no optimizer in this first workflow. The tested multi-tensor planning
logic remains available in `contraction-model.js` for a later interface.

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
- The layout remains usable on desktop and narrow screens.
- Model unit tests and a headless browser interaction test cover the lowering, live edits, controls,
  invalid dimensions, and the link back to the existing visual canvas.
