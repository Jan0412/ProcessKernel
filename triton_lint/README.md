# triton_lint

Deterministic, GPU-free static analysis for LLM-generated Triton kernels.

The linter answers two questions about a generated solution, one per check family:

- **Family 1 — Did it cheat?** Does the file actually do the work in Triton, or does
  it hand the computation back to PyTorch (or never run its kernel at all)?
- **Family 2 — Is it provably slow?** Which memory transactions does the code pay
  for that a well-written kernel would not — expressed in **bytes and microseconds**,
  not vague counts.

Every finding carries a human-readable, *actionable* `message` naming the exact op,
line, tensor, and fix. That is deliberate: the findings are designed to be fed back
into an LLM prompt so the model can repair its own kernel (self-refinement loop), and
"your code is bad" is not a repairable instruction — "`tmp` round-trips through HBM
for 33.6 MB; fuse `exp_kernel` into `scale_kernel`" is.

The analysis is pure stdlib `ast`. No GPU, no `torch` import, no compilation — it
runs on a login node at roughly 1 ms per file, and a full 175k-file run folder scans
in a few minutes.

---

## Quick start

```bash
# lint one generated kernel file
python -m triton_lint check runs/<run_name>/level_6_problem_1_sample_7_kernel.py

# scan a whole run folder (multiprocessing, JSONL output)
python -m triton_lint scan runs/<run_name> --out findings.jsonl --workers 32

# smoke test on the first 500 files, restricted to two checks
python -m triton_lint scan runs/<run_name> --limit 500 --checks F1.2,F2.1 --out /tmp/f.jsonl

# join findings with eval_results.json + baseline timings and print
# per-check hit rates split by correctness and speedup
python -m triton_lint report runs/<run_name> --findings findings.jsonl
```

Programmatic use:

```python
from triton_lint import analyze_file, analyze_source

report = analyze_file("level_6_problem_1_sample_7_kernel.py")
for finding in report.findings:
    print(finding.check_id, finding.severity, finding.message)
```

Tests: `uv run --group dev pytest tests/`.

---

## How it works

`analyze_source` runs a five-stage pipeline; every check consumes only the resulting
`ModuleModel` and returns `list[Finding]`:

1. **parsing.py** — `ast.parse` with recovery (0-byte and truncated files degrade to
   `parse_status: "partial"`/`"empty"` instead of crashing a batch); finds every
   `@triton.jit` kernel (including `@triton.autotune` stacked above it, the bare
   `@jit` form, and the `@triton.jit()` call form); resolves the entry point
   (`ModelNew` → `Model` → `Model = X` alias → sole `nn.Module`).
2. **kernelbody.py** — per-kernel *argument-role table*: for each parameter, is it a
   `tl.store` target (output), a `tl.load` source (input), touched by `tl.atomic_*`
   (accumulator), or `tl.constexpr`? Pointer identification follows only the
   *additive spine* of address expressions (`x_ptr + base + c * (H * W)` → pointers
   are `x_ptr`, `base`; `c`, `H`, `W` are offset arithmetic). Also classifies each
   kernel's **kind**: `tl.dot` → *matmul*; `tl.sum/max/min` or a loop-carried
   accumulator → *reduction*; every stored value traces to a load with no arithmetic
   → *copy*; otherwise *elementwise*.
3. **hostflow.py** — launch sites (the `kernel[grid](...)` subscript-call), loop
   nesting depth, reachability worklist from the entry point (through helper
   functions, custom `nn.Module` submodules, and `torch.autograd.Function` classes,
   to a fixpoint), alias map (`view`/`reshape`/`permute`/... create second names for
   the same storage), the buffer table (which launches store/load each tensor,
   whether it is returned or read by host code), and interprocedural propagation so
   an intermediate crossing a helper-function boundary is still tracked.
4. **shapes.py** — input shapes from the file's `get_inputs()`, falling back to the
   KernelBench reference problem's `get_inputs()` when the generation dropped it
   (only ~31% keep it). Shapes propagate through `empty_like`/`zeros_like`/`empty`
   to byte counts. When a shape cannot be resolved, findings are emitted **without**
   byte estimates — never with a guessed number.
5. **checks/** — the registry runs every registered check (or the `--checks`
   subset), each in its own try/except so no single file kills a batch.

### Run-folder joins

A run folder is self-sufficient; nothing depends on the reranker or its datasets:

- `generation_config.yaml` → level (`pseudo_level`), model, backend.
- `level_{L}_problem_{P}_sample_{S}_kernel.py` → the generated kernels.
- `eval_results.json` → per-sample `compiled` / `correctness` / `runtime`.
- `KernelBench/level{L}/{P}_*.py` → the reference PyTorch model and its
  `get_inputs()` (used for shape inference fallback).
- `timing/{GPU}/baseline_time_torch.json` → eager baseline;
  `speedup = baseline.mean / runtime`.

---

## Severity levels

| severity | meaning |
|---|---|
| `fail` | the solution does not do what the task asked (cheat) or has a provable, serious performance defect |
| `warn` | a real, provable cost, actionable but not disqualifying |
| `info` | context/cost information where the fix is not provably safe, or evidence is weaker |

---

# Family 1 — fallback / fake work (cheating)

A generation can pass every correctness test and still be a wrapper around cuDNN.
These checks catch that. Detection is purely static, which is sufficient here
because the LLM is not an adversary trying to evade analysis — it writes
`torch.conv2d(...)` in plain sight because it does not know better.

Full per-check descriptions with paper references also live in
[checks/family1/CHECKS.txt](checks/family1/CHECKS.txt); each check is one file in
[checks/family1/](checks/family1/).

---

## F1.1 `no_triton_kernel` — severity: **fail**

**Detects:** the file contains zero functions decorated with `@triton.jit`.

**What it tries to capture** — the degenerate failure. The task was "write a Triton
kernel" and no Triton kernel exists; the model solved the problem in plain PyTorch:

```python
# BAD: no @triton.jit anywhere in the file
class ModelNew(nn.Module):
    def forward(self, x, y):
        return x + y          # the whole "solution"
```

**Guards:** handles all decorator spellings — `@triton.autotune(...)` stacked above
`@triton.jit`, bare `@jit` from `from triton import jit`, and the `@triton.jit()`
call form — so a real kernel is never miscounted as absent.

**Reference:** AutoTriton (arXiv:2507.05687) — the rule-based part of its RL reward
assigns 0 to any generation lacking `@triton.jit`. This check *is* that rule.

---

## F1.2 `dead_kernel` — severity: **fail** (info when the entry point is unresolvable)

**Detects:** a `@triton.jit` kernel exists, but no launch site for it is reachable
from the entry point (`ModelNew.forward`), following calls transitively through
helper functions to a fixpoint.

**What it tries to capture** — the model writes a beautiful kernel, satisfying any
"must contain `@triton.jit`" rule, and then never calls it:

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK: tl.constexpr):
    ...  # a perfectly plausible kernel

class ModelNew(nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)     # BAD: the kernel above is never launched
```

This is not hypothetical. A real sample from our runs
(`level_6_problem_1_sample_7_kernel.py`) defines `linear_embedding_kernel`, builds
`grid` and `BLOCK_SIZE`, never launches the kernel, and calls `torch.matmul` with
the comment *"For simplicity in this constrained example we'll use a more direct
approach."*

**Guards / deviations:** we anchor on actual launch *sites* (`kernel[grid](...)`
nodes), not on referenced names, which is strictly stronger than the published
gate — a kernel merely *mentioned* in dead code does not count as live.
Reachability follows helpers, custom `nn.Module` submodules
(`self.norm = LayerNormTriton(...)`), and `torch.autograd.Function.apply`, so
legitimate indirection does not fire. Without a resolvable entry point,
unreachability cannot be proven and the finding is downgraded to `info`.

**Reference:** "Fine-Tuning GPT-5 for GPU Kernel Generation" (arXiv:2602.11000),
Static Reachability Analysis; the failure mode itself is documented in AutoTriton
(arXiv:2507.05687).

---

## F1.3 `discarded_output` — severity: **fail**

**Detects:** a kernel *is* launched, but none of the tensors it writes (its
`tl.store` targets) flows into `forward`'s return value or is read by host code.

**What it tries to capture** — the subtle sibling of F1.2. Reachability analysis
passes this file happily, because the kernel does run; only dataflow catches that
its result is thrown away:

```python
def forward(self, x):
    out = torch.empty_like(x)
    relu_kernel[(n,)](x, out, x.numel())   # out is the kernel's store target...
    return torch.relu(x)                   # BAD: ...and is discarded; torch did the work
```

**Guards:** conservative by construction. It needs the argument-role table (which
kernel params are store targets) plus host-side alias resolution, and it fires only
when at least one store-target buffer was actually resolved — an unresolvable
argument yields silence, not a false alarm. In-place kernels (which load and store
the same tensor) do not fire.

**Reference:** extension beyond the reachability gate of arXiv:2602.11000, which
accepts the code above. The runtime analogue is Dr. Kernel / KernelGYM
(arXiv:2602.05885), which instruments Triton's launch path to detect kernels that
"execute no code" — doing it statically needs no GPU.

---

## F1.4 `torch_fallback` — severity: **fail** (heavy op) / **warn** (light op or operator arithmetic)

**Detects:** calls to PyTorch *compute* operators in host code reachable from
`forward`.

**What it tries to capture** — the headline cheating pattern. The model writes a
Triton kernel for the easy part and quietly leaves the expensive part to PyTorch:

```python
def forward(self, x):
    x = torch.conv2d(x, self.weight)   # BAD: cuDNN does the real work (HEAVY -> fail)
    out = torch.empty_like(x)
    relu_kernel[grid](x, out, x.numel(), BLOCK=1024)   # the "Triton solution"
    return out
```

**Design — an allowlist, not a blocklist.** Three op classes:

- `PLUMBING_OPS` (never fire): allocation (`torch.empty`, `empty_like`, ...), shape
  and layout metadata (`.view`, `.stride`, `.numel`, `.permute`, ...), launch
  plumbing (`triton.cdiv`, ...). Legitimate host-wrapper code *must* call these.
  Memory movers (`.contiguous`, `.clone`, `torch.cat`) are also excluded here —
  they are not *cheating* and are Family 2's business (F2.3/F2.4).
- `HEAVY_OPS` (**fail**): the ops that dominate a task's FLOPs — `matmul`/`mm`/
  `bmm`/`einsum`/`linear`, all `conv*`/`conv_transpose*`,
  `scaled_dot_product_attention`, `softmax`, `*_norm`, pooling, `sort`/`topk`/
  `cumsum`. Any `F.*` / `nn.functional.*` call is treated as compute.
- `LIGHT_OPS` (**warn**): real compute, but cheap — `relu`, `exp`, `sum`, `mean`,
  `clamp`, ...

Severity is FLOP-graded because falling back on `conv2d` and falling back on
`clamp` are not the same crime. This grading is a static analogue of Dr. Kernel's
runtime profiling ratio PR = T_generated / T_total.

**Second pass — operator arithmetic.** `a + b` and `a @ b` are `ast.BinOp` nodes,
not calls, so a call scanner misses them entirely:

```python
def forward(self, x, y):
    z = my_kernel_output_plus_something(x)
    return z * self.scale + y     # BAD: two PyTorch kernel launches, zero torch.* calls
```

This pass fires (at `warn`, or `fail` when the operator is `@`) only when an operand
provably traces back to a `forward` input or to a tensor a kernel wrote — as close
to type inference as static analysis gets without executing anything.

**Guards:** `__init__` is excluded entirely (weight preparation with PyTorch is
legitimate); plumbing never fires; the BinOp pass requires the taint evidence above,
so loop-index arithmetic and grid computations stay silent.

**References:** AutoTriton (arXiv:2507.05687) documents exactly this failure mode
(Triton ReLU + PyTorch convolution). TritonRL (arXiv:2510.17891) describes — in one
sentence, without released code — a rule-based linter that "detects actual calls of
Triton kernels and flags reliance on PyTorch modules"; this check fills that hole.

---

## F1.5 `nn_module_call` — severity: **fail** (heavy module) / **warn** (other)

**Detects:** an `nn.*` module constructed in `__init__` **and invoked as a
callable** in `forward` — the module-level form of F1.4.

**What it tries to capture:**

```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        x = self.conv(x)            # BAD: cuDNN computed the answer (fail)
        out = torch.empty_like(x)
        bias_kernel[grid](x, out, x.numel(), BLOCK=1024)
        return out
```

**The false-positive guard that matters — weight holders.** Keeping an `nn.Conv2d`
purely to *own the weights* is completely legitimate, and very common in genuine
solutions:

```python
self.conv = nn.Conv2d(3, 16, 3)                       # fine: weight holder
conv_kernel[grid](x, self.conv.weight, self.conv.bias, out, ...)   # fine
```

The check is therefore "is a constructed module ever **called**", never "is one
constructed". Attribute reads (`.weight`, `.bias`) are not calls and structurally
cannot fire.

**Second guard — inert modules.** `nn.Dropout` is an identity at eval time and
launches nothing; `nn.Identity`, `nn.Flatten`, `nn.Unflatten` are free. Calling
them is not cheating, and they are excluded (`INERT_MODULES`). Heavy modules
(`Linear`, `Conv*`, `*Norm*`, `MultiheadAttention`, pooling, `Softmax`,
`Sequential`, ...) grade the finding to `fail`; anything else is `warn`.

**Reference:** same as F1.4 — AutoTriton, TritonRL.

---

## F1.6 `passthrough_kernel` — severity: **fail**

**Detects:** a launched kernel whose every stored value traces straight back to a
`tl.load` with **no arithmetic in between** — i.e. the kernel is a memcpy.

**What it tries to capture:**

```python
@triton.jit
def my_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(out_ptr + offs, v, mask=offs < n)   # BAD: computes nothing

def forward(self, x):
    y = torch.sigmoid(x)                # torch does the real work
    out = torch.empty_like(y)
    my_kernel[grid](y, out, y.numel(), BLOCK=1024)   # "uses Triton"
    return out
```

This is the shape a model produces when it is satisfying a *checker* rather than
solving the problem. Expect it to be rare in a baseline but to **appear once a
feedback loop is running** — a model told "you must launch a Triton kernel" will
learn to launch a decoy. That is precisely why this check exists before the loop
starts: so we can detect the loop inducing it.

**Reference:** Dr. Kernel (arXiv:2602.05885) — reward hacking via kernels that
execute no meaningful code, and "lazy optimization": AutoTriton reaches 30.6%
Fast@1 on KernelBench Level-2 but only 9.2% Fast@1.2. The gap *is* lazy
optimization.

---

## F1.7 `compile_offload` — severity: **fail**

**Detects:** `torch.compile`, `torch.jit.script`, `torch.jit.trace`,
`torch._dynamo`, or `torch._inductor` anywhere in the file.

**What it tries to capture** — letting TorchInductor generate the kernel is a
fallback wearing a hat:

```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._fn = torch.compile(lambda x, y: x @ y + y)   # BAD: Inductor writes the kernel

    def forward(self, x, y):
        return self._fn(x, y)
```

**Note:** this check scans the *whole module*, not just reachable code — a
`@torch.compile` decorator executes at class-definition time and routes the
computation away from Triton regardless of reachability.

**Reference:** fallback variant of F1.4; the broader anti-cheat taxonomy is
SOL-ExecBench (NVIDIA, arXiv:2603.19173), which found 589 of 4,075 agent
submissions (14.5%) engaged in some form of reward hacking.

---

### What Family 1 cannot catch

robust-kbench (Sakana AI, arXiv:2509.14279) documents "cheating" kernels achieving
fake 50–120× speedups by **hardcoding outputs for the benchmark's test inputs** or
exploiting fixed weights. A kernel that memorises the test inputs looks like
perfectly clean Triton; no static analysis will catch it — that requires
randomised-input testing at runtime. Family 1 is one signal among several and must
never be the only gate.

---

# Family 2 — memory traffic / fusion (provably slow)

Not "did it cheat" but "is it slow, and *provably* so". Everything here is expressed
in **bytes and microseconds**, never bare counts, because that is what makes the
feedback repairable: "you have too many kernels" can be satisfied by fusing the
*wrong* pair — which breaks correctness — while "`tmp` costs 33.6 MB of HBM
traffic; fuse `exp_kernel` into `scale_kernel`" names a specific transaction and a
specific, provably safe fix.

**The soundness argument.** Triton has **no cross-launch fusion pass** — unlike
TorchInductor, nothing merges `k1[grid](...)` and `k2[grid](...)`. So a tensor
written by one launch and read by the next *provably* makes a full round trip
through HBM. These checks read off memory transactions that are guaranteed to
happen; they do not estimate.

**The fusibility gate (why we stay silent more often than we could).** KernelBenchX
(arXiv:2605.04956) found 72% of fusion tasks fail across all evaluated methods, and
that iterative refinement raises compile rate (52.3% → 68.8%) while *lowering*
average speedup (1.58× → 1.44×). Telling a model to fuse an unfusible pair is how
you reproduce that degradation. So fusion *suggestions* are emitted only for
provably compatible producer→consumer kernel kinds:

| producer → consumer | suggested? | why |
|---|---|---|
| elementwise → elementwise | yes | inline the second computation |
| elementwise → reduction | yes | the reduction transforms values before reducing |
| copy → anything | yes | a copy fuses into whatever consumes it |
| matmul → elementwise | yes | epilogue fusion |
| reduction → elementwise | **no** | softmax-shaped; legal only if the reduced axis fits one block, which we do not try to prove |
| reduction → reduction | **no** | over different axes: not fusible at all |

For unsuggested pairs the cost is still reported (as `info`) — just without an
instruction that could be wrong.

Per-check descriptions with references also live in
[checks/family2/CHECKS.txt](checks/family2/CHECKS.txt); shared cost constants
(achievable HBM bandwidth 1.6 TB/s ≈ 80% of A100 peak; ~5 µs per launch) are in
[checks/family2/_common.py](checks/family2/_common.py).

---

## F2.1 `dead_intermediate` — severity: **warn** (fusible) / **info** (cost only)

**Detects:** a buffer allocated in host code, written by exactly **one** launch,
read by one or more **other** launches, and never returned or read by host code.

**What it tries to capture:**

```python
def forward(self, x):
    tmp = torch.empty_like(x)
    exp_kernel[grid](x, tmp, n)       # tmp is written to HBM...
    out = torch.empty_like(x)
    scale_kernel[grid](tmp, out, n)   # ...and immediately read back
    return out                        # tmp never escapes
```

`tmp` round-trips through HBM — 2 × numel × itemsize bytes — for nothing. A fused
kernel keeps the value in registers:

```python
@triton.jit
def exp_scale_kernel(x_ptr, out_ptr, scale, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.exp(tl.load(x_ptr + offs, mask=offs < n))
    tl.store(out_ptr + offs, v * scale, mask=offs < n)   # tmp never existed
```

The finding reports the byte cost and — only when the producer/consumer kinds are
in the fusible table above — the concrete instruction "fuse `exp_kernel` →
`scale_kernel` into a single kernel".

**Chains merge.** `k1 → t1 → k2 → t2 → k3` (all elementwise) becomes **one**
finding — "fuse these three" — not three separate suggestions the model would apply
piecemeal.

**False-positive guards:**
- A buffer that is *also* returned (multi-output models) is not dead — it must be
  materialised.
- A kernel that both loads and stores the same buffer is an in-place accumulator,
  not a producer.
- Aliases are unioned first: `t2 = t1.view(...)` is a second name for the same
  storage; without the union, `t1` would wrongly appear "read by only one kernel".
- Intermediates cross helper-function boundaries in real generations (the dominant
  shape is one wrapper helper per kernel); interprocedural propagation in
  `hostflow.py` handles this, so `y = helper_a(x); return helper_b(y)` is analysed
  exactly like the inline form.
- An intermediate smaller than L2 (~40 MB on A100) may never reach HBM, so the
  penalty is an upper bound; the byte count is emitted as data rather than
  hard-thresholded.

**References:** Liger-Kernel (arXiv:2410.10989) — fusion eliminating intermediate
materialisation gives ~20% training throughput and ~60% memory reduction; that is
the quantitative justification. KernelEvolve (arXiv:2512.23236) — the Conv2d case
pays a full tensor pass per auxiliary kernel. KernelBenchX (arXiv:2605.04956) — the
reason the suggestion is gated.

---

## F2.2 `launch_overhead` / `launch_in_loop` — severity: **fail** (in loop) / **warn** or **info** (count)

Two findings from one analysis.

**(a) `launch_in_loop` — fail.** A kernel launched from a host-side Python
`for`/`while` loop:

```python
def forward(self, x):          # x: (32, N)
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        row_kernel[grid](x[i], out[i], x.shape[1], BLOCK=1024)   # BAD: 32 serialised launches
    return out
```

N launches, fully serialised, N × ~5 µs of pure overhead — and it means the model
failed to put that dimension in the grid, which is the entire point of a grid:

```python
    row_kernel[(x.shape[0], triton.cdiv(N, BLOCK))](x, out, N, x.stride(0), BLOCK=1024)
```

The finding names the loop variable, depth, and line. High precision, trivially
actionable.

**(b) `launch_count` — warn/info, fires at ≥ 3 reachable launches.** Each launch
costs ~5 µs. On a typical KernelBench Level-1 problem — a 1024×1024 fp32
elementwise op, 4 MB in / 4 MB out — the memory time is ~3 µs, so four launches is
~20 µs of overhead against ~3 µs of real work: the solution is **launch-bound**.
When input shapes are known and overhead exceeds the memory-transfer time, the
finding is upgraded to `warn` and says so explicitly.

**Deliberately not a lint on the count.** The count is a cost term and context, not
an instruction — the actionable instruction comes from F2.1, which names a specific
provably-safe pair to fuse. What F2.2 contributes is the *regime*: it tells the
model **why** fusing matters here, making a proper fusion (rather than a cosmetic
one) far more likely.

**Reference:** KernelEvolve (arXiv:2512.23236) — the Conv2d case study, where the
PyTorch workaround launches four auxiliary kernels and the two-kernel fused Triton
solution wins end-to-end.

---

## F2.3 `layout_churn` — severity: **warn**

**Detects:** host-code operations that launch a **hidden** kernel costing a full
tensor pass. The key distinction the model usually does not know:

| free (metadata only) | costs a full pass + a launch |
|---|---|
| `.view` `.reshape` `.permute` `.transpose` `.squeeze` `.unsqueeze` `.expand` slicing | `.contiguous()` **on a non-contiguous tensor**, `.clone()`, `.to(dtype)` casts, `torch.cat`/`stack`, `.repeat` |

**What it tries to capture:**

```python
def forward(self, x):                     # x: (B, C, T)
    xt = x.permute(0, 2, 1)               # free: a stride trick
    xt = xt.contiguous()                  # BAD: hidden kernel, 2 x numel x itemsize bytes
    my_kernel[grid](xt, out, ...)
```

The fix is what Triton's API exists for — kernels take **stride arguments**
precisely so the host never materialises a transposed copy:

```python
    my_kernel[grid](x, out, x.stride(0), x.stride(2), x.stride(1), ...)  # index the permuted layout directly
```

Similarly `.to(torch.float32)` (a cast — device moves like `.to('cuda')` are
excluded) should be a per-element cast on load inside the kernel,
`tl.load(...).to(tl.float32)`, not a whole-tensor pass in host code.

**Why this is the most underrated check in the suite:** it catches solutions a
`@triton.jit` counter scores as perfectly clean. KernelEvolve's Conv2d case — the
motivating example for kernel counting — consists of `unsqueeze` (free), a *layout
conversion* (a real kernel, a full pass), the conv, and `squeeze` (free). None of
the four is a `@triton.jit` function. The expensive kernels are often the ones with
no decorator on them.

**Guards:**
- `.contiguous()` fires **only** when the receiver is provably non-contiguous —
  either the chained form `x.permute(...).contiguous()` (caught structurally) or
  the bound form `xt = x.permute(...); xt.contiguous()` (caught via the layout set
  that `hostflow` maintains for permute/transpose/`.T`/`expand`/`as_strided`
  results). `.contiguous()` on an already-contiguous tensor is a no-op and stays
  silent.
- `.to(...)` distinguishes dtype casts (fire) from device moves (silent) by
  inspecting the arguments.
- Byte costs (2 × nbytes, read + write) are attached when the tensor's shape is
  known.

**Reference:** KernelEvolve (arXiv:2512.23236), the layout-conversion kernel in the
Conv2d case study.

---

## F2.4 `zeroed_overwritten_buffer` — severity: **warn**

**Detects:** a buffer allocated with `torch.zeros`/`zeros_like` that is a store
target of a kernel which never loads it and performs no atomic on it.

**What it tries to capture:**

```python
def forward(self, x, y):
    out = torch.zeros_like(x)        # BAD: pays for a full memset kernel...
    add_kernel[grid](x, y, out, x.numel(), BLOCK=1024)   # ...then overwrites every element
    return out
```

`torch.zeros` launches a memset — a full write pass, numel × itemsize bytes,
entirely wasted because the kernel overwrites every element anyway.
`torch.empty_like(x)` gives the identical result for free.

**The false-positive guard that matters:** if the kernel accumulates —
`tl.atomic_add` on that pointer, or it loads the buffer back — the zero-init is
**required**, and "fixing" it would introduce a correctness bug:

```python
    hist = torch.zeros(K, device=x.device)          # fine: zero-init is required
    histogram_kernel[grid](x, hist, n, BLOCK=1024)  # uses tl.atomic_add(hist_ptr + b, 1)
```

The check therefore consults the argument-role table first and fires only when the
kernel *stores* to the parameter, never *loads* it, and performs no *atomic* on it.

Two more notes: a store mask like `offs < n` does **not** disqualify the finding
(the mask only prevents out-of-bounds writes; every in-bounds element is still
written), and `nn.Parameter(torch.zeros(...))` in `__init__` is structurally
excluded because buffer collection only walks host code reachable from `forward`.

**Reference:** no direct paper precedent — a novel check applying Liger-Kernel's
memory-traffic principle (a wasted full-tensor pass) to allocation.

---

### What Family 2 composes into

The reason everything is in bytes is that the costs **sum**:

```
total_bytes = essential_io            (reference model's inputs/outputs)
            + wasted_intermediates    (F2.1)
            + layout_copies           (F2.3)
            + zero_init               (F2.4)

T_static   ≈ max(FLOPs / peak_compute, total_bytes / achievable_BW)
            + n_launches × launch_overhead              (F2.2)
```

giving `memory_efficiency = essential_bytes / total_bytes ∈ (0, 1]` — a static,
deterministic, *graded* speed score. It is the static counterpart of SOL-ExecBench's
runtime T_SOL roofline bound: they compute the ideal time from the reference; we
compute the actual cost from the generated code, with no GPU in the loop.
The per-file `summary` exposes `wasted_bytes_lower_bound` for exactly this purpose.

### What Family 2 cannot catch

- **Compute-bound inefficiency.** A bad `BLOCK_M`, low occupancy, poor tensor-core
  utilisation — invisible. A dumb matmul and a great matmul look identical here.
- **Unresolvable shapes.** Findings are emitted without byte estimates rather than
  guessed; a finding with a wrong number is worse than one with no number.
- **L2 residency.** An intermediate smaller than L2 may never reach HBM; byte costs
  are upper bounds on the waste.

---

## Output format

`scan` writes one JSON object per kernel file:

```json
{
  "run_name": "...", "level": 6, "problem_id": 42, "sample_id": 6,
  "path": "...", "parse_status": "ok",
  "findings": [
    {"check_id": "F2.1", "severity": "warn",
     "message": "`tmp` is written by one kernel and immediately read by the next...",
     "data": {"intermediates": ["tmp"], "kernels": ["exp_kernel", "scale_kernel"],
              "fusible": true, "bytes": 33554432, "lineno": 21}}
  ],
  "summary": {"n_kernels": 2, "n_launches": 2, "launches_in_loop": 0,
              "wasted_bytes_lower_bound": 33554432,
              "n_fail": 0, "n_warn": 1, "n_info": 0}
}
```

`message` is the LLM-facing feedback text; `data` holds the machine-readable fields.
`report` joins these rows with `eval_results.json` and the eager baselines and
prints per-check hit rates split by correctness and speedup — the evidence for
whether each check actually predicts a wrong or slow kernel. `report.rows()` returns
the joined rows for notebook use (`pd.DataFrame(rows(...))`).

## Does it predict anything? (empirical sanity check)

From a full scan of `runs/Qwen3-Coder-30B-A3B-Instruct_kernelbook_level6_triton`
(175,350 files; 74,610 evaluated samples, 13,646 correct):

- **68.7% of *correct* kernels trip a Family 1 check** — passing the correctness
  test says little about having actually used Triton.
- Family 1 findings correlate with *higher* correctness and *lower* speedup
  (F1.2-flagged: 24.3% correct vs 13.2% clean; cheating-and-correct median speedup
  0.61× vs 0.75× genuine, and only 9.0% vs 35.9% beat eager) — the signature of
  reward hacking: falling back to PyTorch is a reliable way to pass a test.
- Family 2 findings predict badness on both axes: F2.2-flagged samples are 3.3%
  correct vs 21.5% clean (median speedup 0.37× vs 0.63×); F2.1-flagged are 5.9% vs
  20.1% correct.

## Extending

Add a check by dropping one file into a family package:

```python
from ...model import Finding, ModuleModel
from .. import register

@register("F2.5", "my_check", "warn")
def check(model: ModuleModel) -> list[Finding]:
    ...
```

Import it from the family's `__init__.py`, add an entry to that family's
`CHECKS.txt` (description + paper reference), and add true-positive plus
false-positive-guard tests in `tests/`. A future Family 3 is a new subfolder with
the same pattern.
