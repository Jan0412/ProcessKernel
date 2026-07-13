# Linter bugs — audit history (all fixed)

**Status: all 20 bugs below are fixed.** Each was first encoded as one or more
`@pytest.mark.xfail(strict=True, reason="BUG-N: ...")` tests; fixing a bug flipped
its tests to XPASS (an error under `strict=True`), which forced removing the
marker. The assertions remain as permanent regression tests (in `tests/F1/`,
`tests/F2/`, `tests/test_kernelbody.py`, and `tests/real_samples/`), so this
history and the suite still cannot drift apart. The tables below are kept as a
record of what each bug was and where it came from.

The fixes are grouped by root-cause layer, not patched at the point of symptom:
kernel discovery (`parsing.py`), per-kernel classification (`kernelbody.py`),
host dataflow (`hostflow.py`), and the individual checks. See the plan in the
commit history / `.claude/plans/` for the layer-by-layer design.

One re-audit correction: the F2.4 finding on `p9275_s5` (`full_loss`), previously
labelled a real wasted memset, was found to be a **false positive** — the zeros
represent ignored-index entries and the buffer is read on the host — so the
improved analysis now (correctly) stays silent, and its test asserts that.

All bugs were found by hand-auditing stratified samples of flagged files from
`runs/gpt-oss-120b_kernelbook_level5_triton` and
`runs/Qwen3-Coder-Next_kernelbook_level5_triton` (2026-07-13). The audited files
live in `real_samples/data/` with the ground truth encoded in `real_samples/`.

| id | component | bug | wrongly affects |
|----|-----------|-----|-----------------|
| BUG-1 | parsing | `@triton.jit` kernels defined *inside* a function are invisible: `build_skeleton` only walks module/class level. The nested kernel body then leaks into the enclosing function's host-call scan. Measured: 190 of 371 F1.1 findings (51%) are files that do contain a jit kernel. | F1.1 (FP), F1.4 (`tl.dot` reported as a PyTorch fallback) |
| BUG-2 | hostflow reachability | Default argument values are never walked (`_FuncVisitor` visits only `node.body`), so `def __init__(self, act_layer=GELU_Triton)` leaves `GELU_Triton` unreachable and its kernel "dead". | F1.2 (FP) |
| BUG-3 | F1.3 | The skip condition checks `returned / read_by_host / is_forward_input` but never `loaded_by`: a buffer consumed by the *next kernel* counts as discarded. Fires on every multi-kernel pipeline. | F1.3 (FP) |
| BUG-4 | hostflow reads | Host reads are only counted inside `Call` nodes (`visit_Call` → `_count`). Reads via subscript (`x[0] / n`), slicing (`u = out[:, :H]`), or `AugAssign` (`acc += x`) are invisible, so the buffer looks dead. | F1.3, F2.1, F2.4 (FP) |
| BUG-5 | hostflow ordering | `_resolve_reads` runs before `_propagate_interprocedural` creates buffers for `y = helper(x)` results; their host-read flags are checked against a not-yet-existing buffer and silently dropped (U-Net skip connections). | F2.1, F1.3 (FP) |
| BUG-6 | F1.6 / kernelbody | A kernel dispatching on a `tl.constexpr` flag (`if OP == 0: out = tl.tanh(x) ... else: out = x`) is judged by the pass-through branch only and flagged as a pure copy. | F1.6 (FP) |
| BUG-7 | F1.6 | "No arithmetic on stored values" conflates identity memcpy decoys with gather / concat / layout kernels, whose *address* computation is the task (e.g. a fused `torch.cat` replacement). | F1.6 (FP) |
| BUG-8 | F2.4 | Assumes a store-without-load kernel overwrites *every* element of the zeroed buffer. A diagonal-only store (`out_ptr + row*s0 + col*s1 + col*s2`) needs the zero fill; the "use empty_like" advice would then be a correctness bug. | F2.4 (FP, harmful advice) |
| BUG-9 | F1.4 binop pass | `_base_name` sees through scalar-returning calls, so `x.numel() // k` is flagged as tensor arithmetic on `x`. | F1.4 (partial FP) |
| BUG-10 | kernelbody | A scalar dimension appearing on the additive spine of a store address (`out_ptr + b*(2*H) + H + h`) is classified as a stored pointer param, so `H` shows up as a kernel "output". | F1.3 resolution, finding messages |
| BUG-11 | hostflow `_returned_names` | `return torch.cat([a, b])` is treated like a method call returning its receiver (`torch`), so neither `a` nor `b` is marked returned. | F1.3 (FP) |
| BUG-12 | hostflow / F1.7 | `host_calls` only records calls inside function bodies. Module-level statements (`scripted = torch.jit.script(plain)`) and decorator lists (`@torch.compile`) are never visited, so F1.7 misses both — although its docstring claims decorators are covered. | F1.7 (FN) |
| BUG-13 | kernelbody `_classify` | A loop-carried reduction is only recognised in its `AugAssign` spelling: `accum_in_loop` is set in `visit_AugAssign` alone, so `acc += v` is a reduction but the self-referential `acc = tl.maximum(acc, v)` is `elementwise` (`tl.maximum` is not in `REDUCE_FNS`, which holds `max`, not `maximum`). Windowed pooling and any `acc = acc + x` tiling loop are misread. F2.1 then calls a reduction→elementwise chain `fusible` and emits the "Fuse these kernels" advice its own contract forbids. | F2.1 (harmful advice), kernel `kind` everywhere |

Bugs BUG-14…BUG-16 were found in a second audit (2026-07-13) against a fresh
175k-file scan of `runs/Qwen3-Coder-Next_kernelbook_level5_triton` — a different
model from the level-5 gpt-oss run above, so the false positives come from
idioms the first audit never saw. Real samples live in `real_samples/data/`.

| id | component | bug | wrongly affects |
|----|-----------|-----|-----------------|
| BUG-14 | F1.2 / parsing | A `@triton.jit` *device function* — one `@triton.jit` fn called from inside another kernel's body (Triton inlines it), never launched with `[grid]` — is stored in `model.kernels` like any kernel, but F1.2 only counts subscript launch sites, so it is judged "defined but never launched". It runs on every forward, and the advice "launch it (or remove it)" is destructive: removing it breaks the kernel that calls it. Real sample: p11155_s7 (`rsqrt` inlined into the group-norm kernel). | F1.2 (FP, harmful advice) |
| BUG-15 | hostflow `_resolve_reads` | `elsewhere = loads - launch_loads - return_loads - helper_loads`, but a **bare-Name** launch argument is counted in `launch_loads` (via `_record_launch`) and never in `loads` (the launch's `generic_visit` reaches `visit_Name`, which records only `referenced`). So subtracting `launch_loads` over-subtracts and cancels a single genuine call-based host read — `total = torch.sum(partial)` — leaving `read_by_host` false. Bites only bare-Name launch args with exactly one host read (two reads survive: `2-1>0`; a compound arg like `x.numel()` is counted in `loads` too and survives). Real sample: p12206_s9 (`partial_sums` reduced by `torch.sum`). | F1.3 (FP); F2.1/F2.4 share the read table |
| BUG-16 | F1.4 | `_op_of` reduces every call to its last name segment and matches it against `HEAVY_OPS`/`LIGHT_OPS` without checking the call targets torch. A call to code the model wrote itself — a `self.<submodule>()` invoking a local Triton module, or a module-level helper `def <op>(...)` — is graded a fallback whenever the attribute/function name collides with an op token (`layer_norm`, `softmax`, `linear`, …). The correct-Triton case is reported at **fail** and told to rewrite what is already a kernel. Real sample: p10000_s3 (`self.layer_norm = LayerNormTriton(...)`); 499 such `self.<heavy>`-over-a-local-submodule files in the run. | F1.4 (FP, harmful advice) |

Bugs BUG-17…BUG-20 were found in a third audit (2026-07-13) targeting **Family 2**
against the level-5 gpt-oss run (`runs/gpt-oss-120b_kernelbook_level5_triton`) — one
confirmed defect per F2 check. BUG-17 is anchored to a real sample in
`real_samples/data/`; the other three are synthetic minimal repros (the mechanism is
a clear logic error, each with a passing control that pins the boundary).

| id | component | bug | wrongly affects |
|----|-----------|-----|-----------------|
| BUG-17 | F2.1 `_build_chains` | The chain merge is a single greedy pass that never re-unions two chains once a later intermediate bridges them. In a diamond — a consumer launch fed by two producers, one of them itself fed by an earlier launch — the second producer's intermediate forms its own chain *before* the bridge is added, so the shared consumer launch ends up in **two** findings. Each is a separate "Fuse … into a single kernel" instruction naming the same kernel (mutually unsatisfiable), and the split can promote a non-fusible component to a spurious `warn`/"Fuse" that a correctly-merged analysis keeps at `info`. 48 files in the run; real sample p2084_s2 (the `bmm_kernel` appears in both findings; `mul->bmm` is non-fusible so the merged result should be one `info`). | F2.1 (FP, contradictory/harmful advice) |
| BUG-18 | F2.2 `launch_in_loop` | The finding unconditionally prescribes "Move that dimension into the launch grid and launch the kernel once." For a sequential recurrence (iteration *t* reads the state iteration *t-1* wrote) that loop dimension carries a data dependency; moving it into the grid runs every timestep from the same initial state — a correctness bug. Unlike F2.1's fusibility gate, F2.2 has no parallelisability gate before emitting the fix; the valid advice is to move the loop *into* one kernel (single launch, internal sequential loop). The finding itself (real overhead) is legitimate — only its prescription is wrong. Synthetic (no recurrence-in-loop sample in this run). | F2.2 (harmful advice) |
| BUG-19 | hostflow `noncontiguous` / F2.3 | `model.noncontiguous` is a monotonic taint set with no kill on rebinding. A name made non-contiguous (`x = x.permute(...)`) and then rebound to a fresh contiguous tensor (`x = torch.relu(x)`, or any non-alias call) stays in the set, so a later bare `x.contiguous()` — an actual no-op — is flagged as a full-tensor copy and the model is told to pass strides for an already-contiguous layout. Synthetic; the control (same code without the permute) is correctly silent, pinning the bug to the stale flag. | F2.3 (FP) |
| BUG-20 | F2.4 | The `accumulating` guard scans *every* stored param of the writing kernel instead of the parameter the zeroed buffer is bound to. An atomic on a sibling output (`hist`) sets `accumulating` and suppresses the finding for a different, genuinely-wasted zeros buffer (`out`) that the same kernel unconditionally overwrites. The docstring's contract is per-buffer ("fire only when the kernel stores to it, never loads it, and performs no atomic on it"); the check is per-kernel. (The atomic branch is dead for the buffer's own role — an atomic target is already excluded by the earlier `buf.loaded_by` guard — so it only ever suppresses cross-param.) Synthetic. | F2.4 (FN) |
