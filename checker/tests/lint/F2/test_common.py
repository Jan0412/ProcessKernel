"""The cost model shared by the Family-2 checks."""

from __future__ import annotations

from checker.lint.checks.family2._common import fmt_bytes, fmt_time, transfer_time


class TestFormatters:
    def test_fmt_bytes(self):
        assert fmt_bytes(512) == "512 B"
        assert fmt_bytes(2048) == "2.0 KB"
        assert fmt_bytes(3 << 20) == "3.0 MB"

    def test_fmt_time(self):
        assert fmt_time(5e-6) == "5.0 us"
        assert fmt_time(2e-3) == "2.0 ms"

    def test_transfer_time_scales_with_bytes(self):
        assert transfer_time(1_600_000_000) == 1e-3  # 1.6 GB at 1.6 TB/s
