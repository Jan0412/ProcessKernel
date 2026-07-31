"""The Check/Registry contract.

These pin the three conventions the base class exists to make structural: the check id is
stamped once, ``lineno`` is a real argument whose key is absent when there is no line, and
a registry is an instance rather than a module global.
"""

from __future__ import annotations

import pytest

from checker.core.check import Check, Registry
from checker.core.model import ModuleModel


def model() -> ModuleModel:
    return ModuleModel(path="<test>", source="")


class Noisy(Check):
    check_id = "X1.0"
    name = "noisy"
    severity = "fail"

    def run(self, model):
        return [self.finding("something is wrong", lineno=7, kind="bad")]


class Quiet(Check):
    check_id = "X1.1"
    name = "quiet"

    def run(self, model):
        return []


class TestFinding:
    def test_stamps_the_check_id_from_the_class(self):
        finding = Noisy().finding("m")
        assert finding.check_id == "X1.0"

    def test_lineno_lands_in_data(self):
        assert Noisy().finding("m", lineno=42).data["lineno"] == 42

    def test_the_lineno_key_is_absent_when_there_is_no_line(self):
        # F1.1 is a whole-file finding. Inventing line 1 would put it on real source that
        # has nothing to do with it, and inspect_trace joins on exactly this key.
        assert "lineno" not in Noisy().finding("m", n_kernels=0).data

    def test_payload_order_is_the_caller_order(self):
        # The summary is serialised with json.dumps, so key order is observable on disk.
        # No shipped check writes lineno first -- six write it last -- so finding() must
        # not hoist it.
        assert list(Noisy().finding("m", kernel="k", lineno=1, entry="e").data) == [
            "kernel",
            "lineno",
            "entry",
        ]

    def test_severity_falls_back_to_the_class_default(self):
        assert Noisy().finding("m").severity == "fail"
        assert Quiet().finding("m").severity == "warn"

    def test_severity_is_overridable_per_finding(self):
        # Five of the F1/F2 findings decide severity from what they found.
        assert Noisy().finding("m", severity="info").severity == "info"

    def test_extra_keywords_become_the_data_payload(self):
        data = Noisy().finding("m", kind="fusible", bytes=64).data
        assert data == {"kind": "fusible", "bytes": 64}


class TestSubclassContract:
    def test_a_check_without_an_id_fails_at_import(self):
        with pytest.raises(TypeError, match="check_id"):

            class Anonymous(Check):
                name = "anonymous"

                def run(self, model):
                    return []

    def test_a_check_without_a_name_fails_at_import(self):
        with pytest.raises(TypeError, match="name"):

            class Unnamed(Check):
                check_id = "X9.9"

                def run(self, model):
                    return []

    def test_run_is_abstract(self):
        class Abstract(Check):
            check_id = "X9.8"
            name = "abstract"

        with pytest.raises(TypeError):
            Abstract()


class TestRegistry:
    def test_add_stores_an_instance_and_returns_the_class(self):
        registry = Registry("t")
        returned = registry.add(Noisy)
        assert returned is Noisy
        assert isinstance(registry.checks[0], Noisy)

    def test_check_ids_reflect_registration_order(self):
        registry = Registry("t")
        registry.add(Quiet)
        registry.add(Noisy)
        assert registry.check_ids == ["X1.1", "X1.0"]

    def test_run_collects_findings_from_every_check(self):
        registry = Registry("t")
        registry.add(Noisy)
        registry.add(Quiet)
        assert [f.check_id for f in registry.run(model())] == ["X1.0"]

    def test_only_filters_by_check_id(self):
        registry = Registry("t")
        registry.add(Noisy)
        assert registry.run(model(), only={"X1.1"}) == []
        assert len(registry.run(model(), only={"X1.0"})) == 1

    def test_a_raising_check_is_noted_not_fatal(self):
        class Boom(Check):
            check_id = "X2.0"
            name = "boom"

            def run(self, model):
                raise RuntimeError("check exploded")

        registry = Registry("t")
        registry.add(Boom)
        registry.add(Noisy)
        subject = model()

        findings = registry.run(subject)

        assert [f.check_id for f in findings] == ["X1.0"]
        assert subject.notes == ["X2.0 raised RuntimeError: check exploded"]

    def test_findings_sort_by_id_then_line(self):
        class Multi(Check):
            check_id = "X0.1"
            name = "multi"

            def run(self, model):
                return [self.finding("b", lineno=9), self.finding("a", lineno=2)]

        registry = Registry("t")
        registry.add(Noisy)
        registry.add(Multi)

        findings = registry.run(model())

        assert [(f.check_id, f.data.get("lineno")) for f in findings] == [
            ("X0.1", 2),
            ("X0.1", 9),
            ("X1.0", 7),
        ]

    def test_a_finding_with_no_line_sorts_as_zero(self):
        # The old run_checks sorted on data.get("lineno", 0); F1.1 relies on it.
        class Whole(Check):
            check_id = "X1.0"
            name = "whole"

            def run(self, model):
                return [self.finding("late", lineno=5), self.finding("first")]

        registry = Registry("t")
        registry.add(Whole)

        assert [f.message for f in registry.run(model())] == ["first", "late"]

    def test_two_registries_do_not_share_checks(self):
        # The module-global CHECKS list is exactly what this replaces.
        first, second = Registry("first"), Registry("second")
        first.add(Noisy)
        assert second.check_ids == []
        assert first.check_ids == ["X1.0"]
