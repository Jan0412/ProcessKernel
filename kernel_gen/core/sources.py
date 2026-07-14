"""Dataset -> ``list[Problem]``. Two functions, no base class.

KernelBench and KernelBook are never both loaded in one process -- an arm picks one
with ``--dataset`` -- so a common interface would exist purely to be implemented
twice and dispatched over once. The abstraction that does earn its keep is
:class:`~kernel_gen.core.model.Problem` itself: everything downstream (prompting,
shape inference, linting) sees only ``ref_arch_src`` and never learns which dataset
it came from.

The two differ in exactly one interesting way. A KernelBench row *is* a KernelBench
reference. A KernelBook row is an arbitrary ``nn.Module`` under its own name, and has
to be rewritten into one (class ``Model``, positional ``get_init_inputs``) before the
evaluator could ever instantiate it -- so loading can fail, and rows get skipped.
"""

from __future__ import annotations

from .model import Problem
from .text import parse_int_spec, problem_id_from_name

KERNELBOOK_SPLIT = "train"  # KernelBook ships a single split


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


def load_problems(dataset: str, **kwargs) -> list[Problem]:
    """Dispatch on the ``--dataset`` flag."""
    if dataset == "kernelbench":
        kwargs.pop("max_src_chars", None)
        return load_kernelbench_problems(**kwargs)
    if dataset == "kernelbook":
        return load_kernelbook_problems(**kwargs)
    raise ValueError(f"unknown dataset {dataset!r}")
