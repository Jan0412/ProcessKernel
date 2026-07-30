# Real-sample regression tests

The kernels in `data/` are copies of generated files from
`runs/gpt-oss-120b_kernelbook_level5_triton`, hand-audited on 2026-07-13 as a
stratified sample of linter findings (3-5 flagged files per check, plus one
known-clean file). Each test encodes the audit's ground truth:

- **plain tests** assert findings the audit confirmed as real (true positives)
  or confirmed absences;
- **`xfail(strict=True)` tests** assert the *correct* behaviour on files where
  the audit found a false positive — they document open bugs (see
  `../BUGS.md`) and start erroring the moment a fix lands, so the marker (and
  the bug-register entry) must be removed together with the fix.

Run folders are transient; the copies here make the ground truth permanent.
