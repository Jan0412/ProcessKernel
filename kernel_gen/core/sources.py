"""Dataset -> ``list[Problem]``. Three functions, no base class.

KernelBench and KernelBook are never both loaded in one process -- an arm picks one
with ``--dataset`` -- so a common interface would exist purely to be implemented
twice and dispatched over once. The abstraction that does earn its keep is
:class:`~kernel_gen.core.model.Problem` itself: everything downstream (prompting,
shape inference, linting) sees only ``ref_arch_src`` and never learns which dataset
it came from.

The two HF loaders differ in exactly one interesting way. A KernelBench row *is* a
KernelBench reference. A KernelBook row is an arbitrary ``nn.Module`` under its own
name, and has to be rewritten into one (class ``Model``, positional
``get_init_inputs``) before the evaluator could ever instantiate it -- so loading can
fail, and rows get skipped.

**The third loader exists because that rewrite is not the only one.** ``eval_run``
scores a generation against the *staged* reference on disk, ``KernelBench/level{L}/``,
which ``convert_kernelbook.py`` produced with ``--scale`` and a smoke test. Converting
the same row again here, in-process and unscaled, yields a *different* reference: the
model was prompted with ``torch.rand([4, 4])`` while eval instantiated
``torch.rand([2048, 2048])``. That is not a hypothetical -- 16,425 of the 17,071 staged
level-6 references carry scaled dims, and the linter's shape-dependent F2 checks read
their byte estimates off this same ``ref_arch_src`` (see ``core/critics.py``), so the
feedback folded into the next round was computed against shapes that never existed.

:func:`load_local_problems` closes that hole by construction rather than by two
conversions agreeing: point ``--ref-dir`` at the staged level dir and the prompt, the
linter's shapes and the eval reference are the same bytes. It cannot be reconstructed
by passing a ``ScaleConfig`` here instead -- the staging run picks a scale *per row*
from a smoke-test fallback ladder, so reproducing it would mean re-running that ladder
on a GPU at generation time.
"""

from __future__ import annotations

import os
import re

from .model import Problem
from .text import parse_int_spec, problem_id_from_name

KERNELBOOK_SPLIT = "train"  # KernelBook ships a single split

# A staged level dir names its files ``<problem_id>_<Name>.py``, and the id is the
# primary key joining generation to eval. KernelBench's LocalKernelBenchDataset parses the
# same id as ``int(name.split("_")[0])``, so the underscore is load-bearing in both, and it
# is what keeps problem 1 from matching ``10_Foo.py``.
_LEVEL_FILE = re.compile(r"^(\d+)_.*\.py$")


def _select(dataset, spec: str | None, all_rows: bool) -> list[int]:
    if all_rows:
        return list(range(len(dataset)))
    if spec:
        return parse_int_spec(spec)
    raise ValueError("Provide --problems/--rows or --all.")


def load_kernelbench_problems(
    dataset_name: str,
    level: int,
    spec: str | None = None,
    all_rows: bool = False,
) -> list[Problem]:
    """KernelBench level ``level``. ``spec`` indexes the split, not the problem id."""
    from datasets import load_dataset

    split = f"level_{level}"
    print(f"Loading dataset {dataset_name} split={split} …")
    dataset = load_dataset(dataset_name, split=split)

    problems: list[Problem] = []
    for index in _select(dataset, spec, all_rows):
        try:
            row = dataset[index]
        except IndexError:
            print(f"[WARN] index {index} not in level {level}, skipping")
            continue
        name = row.get("name", f"problem_{index:04d}.py")
        problems.append(
            Problem(
                level=level,
                problem_id=problem_id_from_name(name, index),
                name=name,
                ref_arch_src=row["code"],
            )
        )
    return problems


def load_kernelbook_problems(
    dataset_name: str,
    level: int,
    spec: str | None = None,
    all_rows: bool = False,
    max_src_chars: int = 24000,
) -> list[Problem]:
    """KernelBook rows, converted to KernelBench-style references.

    ``level`` is the pseudo-level (5/6) -- it only ever appears in filenames, and it
    must match whatever ``convert_kernelbook.py`` was told when it staged the local
    reference dir that eval reads. The row index is the problem id.

    Rows are dropped when they are too long to prompt with (``max_src_chars``) or when
    the conversion cannot find a module to rewrite. Both are load-time facts about the
    corpus, not failures of a run, so they print and move on.
    """
    from datasets import load_dataset

    from kernel_gen.kernelbook_convert import ConversionError, convert_row

    print(f"Loading dataset {dataset_name} split={KERNELBOOK_SPLIT} …")
    dataset = load_dataset(dataset_name, split=KERNELBOOK_SPLIT)

    problems: list[Problem] = []
    skipped_size = skipped_convert = 0
    for index in _select(dataset, spec, all_rows):
        try:
            row = dataset[index]
        except IndexError:
            print(f"[WARN] row {index} out of range, skipping")
            continue

        module_name = row.get("module_name") or row.get("entry_point") or ""
        python_code = row.get("python_code") or ""

        if len(python_code) > max_src_chars:
            skipped_size += 1
            continue
        try:
            ref_arch_src = convert_row(python_code, module_name)
        except ConversionError as exc:
            print(f"[SKIP convert] row {index} ({module_name}): {exc}")
            skipped_convert += 1
            continue

        problems.append(
            Problem(
                level=level,
                problem_id=index,
                name=module_name or f"row_{index}",
                ref_arch_src=ref_arch_src,
            )
        )

    if skipped_size or skipped_convert:
        print(
            f"KernelBook: skipped {skipped_size} rows over {max_src_chars} chars, "
            f"{skipped_convert} that would not convert"
        )
    return problems


def index_level_dir(ref_dir: str) -> dict[int, str]:
    """Map ``problem_id -> filename`` for a staged level dir.

    Two ids colliding would be silently catastrophic rather than merely wrong: eval
    resolves the reference with ``find_ref``, which returns whatever the glob yields
    first, so generation and eval could pick *different* files for the same id and
    reintroduce exactly the prompt/eval divergence this loader exists to remove. The
    staging script keys filenames on the dataset row index so it cannot happen -- which
    is precisely why a violation means something is wrong upstream, and the run should
    stop instead of quietly picking one.
    """
    if not os.path.isdir(ref_dir):
        raise ValueError(f"--ref-dir {ref_dir!r} is not a directory")

    index: dict[int, str] = {}
    for entry in os.scandir(ref_dir):
        if not entry.is_file():
            continue
        match = _LEVEL_FILE.match(entry.name)
        if match is None:
            continue  # manifest.json, conversion_stats.json, stray archives
        problem_id = int(match.group(1))
        if problem_id in index:
            raise ValueError(
                f"{ref_dir} has two references for problem {problem_id}: "
                f"{index[problem_id]!r} and {entry.name!r}. eval's find_ref would pick "
                f"one by glob order, so generation could prompt from the other."
            )
        index[problem_id] = entry.name

    if not index:
        raise ValueError(f"--ref-dir {ref_dir!r} holds no <id>_<Name>.py reference files")
    return index


def load_local_problems(
    ref_dir: str,
    level: int,
    spec: str | None = None,
    all_rows: bool = False,
    max_src_chars: int = 24000,
) -> list[Problem]:
    """References read straight from a staged level dir -- the files eval scores.

    ``spec`` selects by *problem id*, which for a KernelBook level is the dataset row
    index, so ``--problems 0-999`` still shards the same way. Unlike the HF loaders it
    is not an index into a dense array: the staging run drops rows that would not
    convert or failed the smoke test, so a requested id may simply not be here. Level 6
    is missing ~1,090 of the ids in ``0-18161``, and those gaps are a *fact about the
    corpus*, already reflected in what eval can score -- so they are counted and
    reported, never an error.

    ``max_src_chars`` is applied to the converted reference rather than to a raw
    KernelBook ``python_code``, because this string is what the prompt actually carries
    and the token budget is the reason the flag exists.
    """
    index = index_level_dir(ref_dir)

    if all_rows:
        wanted = sorted(index)
    elif spec:
        wanted = parse_int_spec(spec)
    else:
        raise ValueError("Provide --problems/--rows or --all.")

    # --problems means dataset indices for the HF loaders and problem ids here; say which.
    print(
        f"Loading references from {ref_dir} ({len(index)} staged), "
        f"selecting by problem id …"
    )

    problems: list[Problem] = []
    missing = skipped_size = 0
    for problem_id in wanted:
        name = index.get(problem_id)
        if name is None:
            missing += 1
            continue
        with open(os.path.join(ref_dir, name), encoding="utf-8") as fh:
            ref_arch_src = fh.read()
        if len(ref_arch_src) > max_src_chars:
            skipped_size += 1
            continue
        problems.append(
            Problem(
                level=level,
                problem_id=problem_id,
                name=name,
                ref_arch_src=ref_arch_src,
            )
        )

    if missing or skipped_size:
        print(
            f"{ref_dir}: {missing} of the {len(wanted)} requested ids are not staged "
            f"(dropped at conversion or by the smoke test), {skipped_size} over "
            f"{max_src_chars} chars"
        )
    return problems


def shard_range(ref_dir: str, shard: int, nshards: int) -> str:
    """The ``--problems`` range for one task of a sharded array run.

    A staged dir's ids are sparse, so cutting the id *range* into equal parts does not
    cut the *work* into equal parts. This cuts the sorted id list into equal chunks and
    returns the contiguous range spanning the requested chunk.

    Widening a chunk to its spanning range is safe, and the reason is worth stating
    because it is what keeps shards disjoint: chunk ``k`` ends at ``ids[hi-1]`` and
    chunk ``k+1`` begins at ``ids[hi]``, which is strictly greater, so the two integer
    intervals cannot overlap. Every id between them is by construction one that is not
    staged -- the loader counts it as absent and skips it. Union of the ranges therefore
    covers every problem exactly once.

    A range rather than a comma list because ``--problems`` is persisted into
    ``generation_config.yaml``, which ``triton_lint/runs.py`` reads with a flat
    ``key: value`` scanner -- it must stay one short scalar.
    """
    if nshards < 1:
        raise ValueError(f"nshards must be >= 1, got {nshards}")
    if not 0 <= shard < nshards:
        raise ValueError(f"shard {shard} out of range for {nshards} shards")

    ids = sorted(index_level_dir(ref_dir))
    if nshards > len(ids):
        raise ValueError(f"{nshards} shards over only {len(ids)} problems in {ref_dir}")

    lo = shard * len(ids) // nshards
    hi = (shard + 1) * len(ids) // nshards
    chunk = ids[lo:hi]
    return f"{chunk[0]}-{chunk[-1]}"


def load_problems(dataset: str, ref_dir: str | None = None, **kwargs) -> list[Problem]:
    """Dispatch on ``--ref-dir`` first, then on the ``--dataset`` flag.

    ``--ref-dir`` overrides the row source but NOT ``--dataset``, which still decides
    whether the run records ``level`` or ``pseudo_level`` (see ``artifacts.write_config``
    -- writing the wrong one repoints every downstream filename lookup). So the dataset
    name is still validated here even when it no longer selects a loader.
    """
    if dataset not in ("kernelbench", "kernelbook"):
        raise ValueError(f"unknown dataset {dataset!r}")

    if ref_dir:
        kwargs.pop("dataset_name", None)  # nothing is fetched; the dir is the source
        return load_local_problems(ref_dir=ref_dir, **kwargs)

    if dataset == "kernelbench":
        kwargs.pop("max_src_chars", None)
        return load_kernelbench_problems(**kwargs)
    return load_kernelbook_problems(**kwargs)
