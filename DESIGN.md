# Browser contraction planner

## Deliverable

The first planner is a static browser utility for two or more abstract tensors. A tensor is a
name plus an ordered list of string-named legs and positive integer dimensions. Equal leg names
on two tensors identify contracted axes; unmatched names remain free.

The utility evaluates a parenthesized binary contraction order and finds minimum-FLOP and
minimum-peak-intermediate orders. It reports every pairwise lowering using the same convention as
the cuQuantumNaturalfPEPS Julia and CUDA helpers:

1. retain free axes of the left tensor in their original order;
2. order shared axes by their appearance on the left tensor;
3. permute the left tensor to `[free-left, shared]` and reshape it to `M x K`;
4. permute the right tensor to `[shared, free-right]` and reshape it to `K x N`;
5. multiply the matrices and reshape the result to `[free-left, free-right]`.

The cost display separates scalar multiply-accumulate positions from conventional real FLOPs.
One real MAC is two real FLOPs and one complex MAC is eight real FLOPs. Precision changes storage
bytes, not the arithmetic count. Counts use arbitrary-precision integers.

## Initial boundary

- Leg names are exact, case-sensitive strings after trimming.
- A leg name may occur once or twice globally and at most once on a tensor.
- A twice-occurring leg must have the same dimension on both tensors.
- Dangling legs and outer products are supported.
- Self-traces, hyperedges, truncating decompositions, tensor values, and CUDA execution are not.
- Exact subset dynamic programming is used through 14 tensors; larger inputs use a deterministic
  greedy order and are labeled heuristic.

## Acceptance

- The default two-matrix example can be edited and evaluated without a server.
- Multiple shared legs produce the expected permutations, `M`, `K`, `N`, result order, and cost.
- A manual binary expression must use every tensor exactly once.
- Minimum-FLOP and minimum-peak orders are independently available.
- Every step shows matched legs, axis permutations, matrix shapes, result legs, MACs, real FLOPs,
  intermediate size, and implementation-shaped permutation scratch.
- Invalid dimensions, duplicate names, dimension mismatches, traces, and hyperedges fail loudly.
- The numerical model has automated Node tests and the browser has a headless interaction smoke
  test.
