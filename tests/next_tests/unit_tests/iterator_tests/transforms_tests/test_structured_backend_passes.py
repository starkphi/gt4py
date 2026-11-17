# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Hypothesis tests for structured_backend_passes.py.

Each test verifies a specific claim about dead/live branches, redundancies,
or simplification opportunities. Run BEFORE applying simplifications:
  - Green (pass) = branch is live / behaviour is as documented
  - Red (fail) = hypothesis confirmed dead / bug found

See the plan for the full list of hypotheses H1–H8.
"""

import copy
from typing import Any
from unittest.mock import patch

import pytest

from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import type_specifications as it_ts
from gt4py.next.type_system import type_specifications as ts

from gt4py.next.iterator.transforms.structured_backend_passes import (
    _domain_with_vertical_dims_of,
    _split_mixed_entity_tuple_setat,
    _minus_one,
    _offset_add,
    _offset_sub,
    _entity_cartesian_bounds,
    _mapping_based_axis_bounds,
    _cartesian_axis_bounds,
    _pick_symbolic_int,
    _pick_size_param,
    _offset_int_value,
    SetAtRemapper,
    StructuredTypeRemapper,
    SymbolicSizeInliner,
    NeighborReductionUnroller,
)
# _build_edge_validity_masked_expr is a method on SetAtRemapper, call via instance
_build_edge_validity_masked_expr = SetAtRemapper()._build_edge_validity_masked_expr


# ── Shared test helpers ───────────────────────────────────────────────────────

def _make_minimal_mapping(n_edges: int = 4, horizontal_start: int = 1) -> dict[str, Any]:
    """Minimal symbolic_domain_sizes with all 3 kolors active after horizontal_start."""
    edge_to_ijk = [(0, 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 2)][:n_edges]
    return {
        "use_horizontal_start_mapping": True,
        "edge_to_ijk": edge_to_ijk,
        "horizontal_start": horizontal_start,
        "max_i": 2,
        "max_j": 2,
    }


def _make_edge_domain() -> ir.Expr:
    edge_axis = ir.AxisLiteral(value="Edge", kind=common.DimensionKind.HORIZONTAL)
    return im.call("unstructured_domain")(
        im.call("named_range")(edge_axis, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=4))
    )


def _make_cartesian_edge_domain() -> ir.Expr:
    IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
    JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
    Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
    return im.call("cartesian_domain")(
        im.named_range(IDim, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=2)),
        im.named_range(JDim, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=2)),
        im.named_range(Kolor, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=3)),
    )


# ── H1: _minus_one Literal branch ────────────────────────────────────────────

class TestH1MinusOneLiteralBranch:
    """H1: Does _minus_one hit the ir.Literal branch?

    Claim: After StructuredTypeRemapper all integer bounds are ir.OffsetLiteral.
    The ir.Literal branch in _minus_one (lines 635-640) is never reached.
    If the test passes, the Literal branch IS live and must be kept.
    If the test fails, the branch is dead and can be removed.
    """

    def test_minus_one_with_offset_literal(self):
        """OffsetLiteral input always works — this is the primary path."""
        result = _minus_one(ir.OffsetLiteral(value=5))
        assert result == ir.OffsetLiteral(value=4)

    def test_minus_one_with_offset_literal_zero(self):
        result = _minus_one(ir.OffsetLiteral(value=0))
        assert result == ir.OffsetLiteral(value=-1)

    def test_minus_one_literal_branch_is_reachable(self):
        """H1: Call _minus_one with ir.Literal — does it hit the Literal branch?

        Expected: the Literal branch IS hit (returns ir.Literal, not ir.FunCall).
        If this fails (returns im.minus()), the Literal branch is dead → remove it.
        """
        lit = ir.Literal(value="5", type=ts.ScalarType(kind=ts.ScalarKind.INT32))
        result = _minus_one(lit)
        # If Literal branch is live: result is ir.Literal(value="4")
        # If Literal branch is dead: result is FunCall(minus, ...)
        assert isinstance(result, ir.Literal), (
            "H1 CONFIRMED: _minus_one's Literal branch IS dead — "
            "it falls through to im.minus(). Safe to remove Literal branch."
        )
        assert result.value == "4"

    def test_minus_one_symref_falls_to_minus(self):
        """Non-literal, non-OffsetLiteral input falls to im.minus — expected."""
        ref = im.ref("some_var")
        result = _minus_one(ref)
        assert cpm.is_call_to(result, "minus")


# ── H2: _entity_cartesian_bounds double call ─────────────────────────────────

class TestH2EntityCartesianBoundsDoubleCall:
    """H2: _entity_cartesian_bounds calls _mapping_based_axis_bounds twice for Cell.

    Claim: Line 758 re-invokes _mapping_based_axis_bounds with identical args as line 752.
    Expected: call count = 2 (before fix). After fix: call count = 1.
    """

    def test_cell_with_mapping_calls_mapping_twice(self):
        """H2: Counts how many times _mapping_based_axis_bounds is called."""
        sds = _make_minimal_mapping()
        # Cell mapping needs cell_to_ijk (not edge_to_ijk)
        cell_to_ijk = [(0, 0, 0), (0, 0, 0), (0, 0, 1)]
        sds["cell_to_ijk"] = cell_to_ijk
        sds["horizontal_start_cell"] = 1

        call_count = {"n": 0}
        original = _mapping_based_axis_bounds

        def counting(entity, axis, sds_arg, kolor=None):
            call_count["n"] += 1
            return original(entity, axis, sds_arg, kolor=kolor)

        with patch(
            "gt4py.next.iterator.transforms.structured_backend_passes._mapping_based_axis_bounds",
            side_effect=counting,
        ):
            _entity_cartesian_bounds("Cell", "IDim", set(), sds)

        assert call_count["n"] == 1, (
            f"H2 FIXED: _mapping_based_axis_bounds called {call_count['n']} times "
            "(was 2 before S1 fix, now 1 — caching working correctly)"
        )

    def test_edge_with_mapping_call_count(self):
        """H2b: For Edge, same double-call happens (for consistency check)."""
        sds = _make_minimal_mapping()
        call_count = {"n": 0}
        original = _mapping_based_axis_bounds

        def counting(entity, axis, sds_arg, kolor=None):
            call_count["n"] += 1
            return original(entity, axis, sds_arg, kolor=kolor)

        with patch(
            "gt4py.next.iterator.transforms.structured_backend_passes._mapping_based_axis_bounds",
            side_effect=counting,
        ):
            _entity_cartesian_bounds("Edge", "IDim", set(), sds)

        # For Edge, the Cell-specific branch does NOT fire, so only 1 call expected
        assert call_count["n"] == 1, (
            f"Edge entity should call _mapping_based_axis_bounds once; got {call_count['n']}"
        )


# ── H3: SetAtRemapper.visit_FunCall — cartesian_domain+entity check ──────────

class TestH3SetAtRemapperCartesianEntityDead:
    """H3: After StructuredTypeRemapper, no cartesian_domain contains entity axes.

    Claim: The second clause in SetAtRemapper.visit_FunCall's or-condition
    (checking cartesian_domain with Edge/Cell/Vertex axes) is dead code.
    """

    def test_unstructured_domain_is_converted(self):
        """H3a: unstructured_domain(Edge, K) in as_fieldop body IS converted by SetAtRemapper."""
        sds = _make_minimal_mapping()
        edge_axis = ir.AxisLiteral(value="Edge", kind=common.DimensionKind.HORIZONTAL)
        K = common.Dimension("K", kind=common.DimensionKind.VERTICAL)
        unstr_domain = im.call("unstructured_domain")(
            im.call("named_range")(edge_axis, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=4)),
            im.call("named_range")(K, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=5)),
        )
        lifted = im.as_fieldop(im.lambda_("it")(im.deref("it")), unstr_domain)(im.ref("vn"))
        program = ir.Program(
            id="test", function_definitions=[], declarations=[],
            params=[im.sym("vn"), im.sym("out")],
            body=[ir.SetAt(expr=lifted, domain=_make_edge_domain(), target=im.ref("out"))],
        )
        result = SetAtRemapper.apply(StructuredTypeRemapper.apply(program, symbolic_domain_sizes=sds), symbolic_domain_sizes=sds)
        # The as_fieldop domain should now be cartesian_domain, not unstructured_domain
        for stmt in result.body:
            body_expr = stmt.expr
            if cpm.is_applied_as_fieldop(body_expr):
                fieldop_domain = body_expr.fun.args[1] if len(body_expr.fun.args) > 1 else None
                if fieldop_domain is not None:
                    assert cpm.is_call_to(fieldop_domain, "cartesian_domain"), (
                        "as_fieldop body domain should be cartesian after SetAtRemapper"
                    )
                    # And it should NOT contain entity axes
                    for nr in fieldop_domain.args:
                        if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
                            axis = nr.args[0].value if hasattr(nr.args[0], "value") else None
                            assert axis not in {"Edge", "Cell", "Vertex"}, (
                                f"H3 FALSIFIED: cartesian_domain still has entity axis '{axis}' "
                                "after SetAtRemapper — the dead-code check IS needed"
                            )

    def test_cartesian_domain_with_entity_axis_never_input_to_setat_remapper(self):
        """H3b: cartesian_domain with Edge axis is never a natural input to SetAtRemapper.

        In the full pipeline, StructuredTypeRemapper runs first, converting field types.
        Domain expressions are only converted in SetAtRemapper or in as_fieldop body visits.
        A node like cartesian_domain(named_range(Edge,...)) is not produced by StructuredTypeRemapper.
        """
        # Build a cartesian_domain with an Edge axis (unusual — should not occur naturally)
        edge_axis = ir.AxisLiteral(value="Edge", kind=common.DimensionKind.HORIZONTAL)
        weird_domain = im.call("cartesian_domain")(
            im.call("named_range")(edge_axis, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=4))
        )
        sds = _make_minimal_mapping()
        # If SetAtRemapper.visit_FunCall handles this, the output should be remapped
        remapper = SetAtRemapper()
        result = remapper.visit_FunCall(
            weird_domain,
            symbolic_domain_sizes=sds,
            program_param_ids=set(),
        )
        # Expected: the cartesian_domain+entity path IS handled (condition fires)
        # If it returns weird_domain unchanged, the check is dead
        assert cpm.is_call_to(result, "cartesian_domain")
        for nr in result.args:
            if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
                axis = nr.args[0].value if hasattr(nr.args[0], "value") else None
                assert axis not in {"Edge", "Cell", "Vertex"}, (
                    "H3b: cartesian_domain+entity check DID fire and converted the domain"
                )


# ── H4: _build_edge_validity_masked_expr None path ───────────────────────────

class TestH4BuildEdgeValidityMaskedExprNonePath:
    """H4: _build_edge_validity_masked_expr returns None only without mapping.

    In production, mapping is always present. The None path is only for test fixtures.
    """

    def _make_cartesian_domain_kolor3(self):
        IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
        Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
        return im.call("cartesian_domain")(
            im.named_range(IDim, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=2)),
            im.named_range(JDim, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=2)),
            im.named_range(Kolor, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=3)),
        )

    def test_returns_none_without_mapping(self):
        """H4a: Without mapping, returns None."""
        domain = self._make_cartesian_domain_kolor3()
        result = _build_edge_validity_masked_expr(im.ref("inp"), im.ref("out"), domain, {})
        assert result is None, "H4a: No mapping → should return None"

    def test_returns_none_with_horizontal_start_zero(self):
        """H4b: With mapping but horizontal_start=0, returns None (no interior edges)."""
        domain = self._make_cartesian_domain_kolor3()
        sds = _make_minimal_mapping(horizontal_start=0)
        result = _build_edge_validity_masked_expr(im.ref("inp"), im.ref("out"), domain, sds)
        assert result is None, "H4b: horizontal_start=0 → no interior edges → returns None"

    def test_returns_concat_where_with_valid_mapping(self):
        """H4c: With valid mapping (horizontal_start > 0 and all 3 kolors), returns concat_where."""
        domain = self._make_cartesian_domain_kolor3()
        sds = _make_minimal_mapping(horizontal_start=1)
        result = _build_edge_validity_masked_expr(im.ref("inp"), im.ref("out"), domain, sds)
        assert result is not None, "H4c: Valid mapping should produce concat_where"
        assert cpm.is_call_to(result, "concat_where"), (
            f"H4c: Expected concat_where, got {result}"
        )

    def test_none_return_for_non_3_kolor_domain(self):
        """H4d: Only works on 3-kolor (edge) domains. Cell 2-kolor → None."""
        IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
        Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
        cell_domain = im.call("cartesian_domain")(
            im.named_range(IDim, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=2)),
            im.named_range(JDim, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=2)),
            im.named_range(Kolor, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=2)),  # 2 kolors
        )
        sds = _make_minimal_mapping(horizontal_start=1)
        result = _build_edge_validity_masked_expr(im.ref("inp"), im.ref("out"), cell_domain, sds)
        assert result is None, "H4d: 2-kolor domain → function only handles 3-kolor → None"


# ── H5: SymbolicSizeInliner handles both Literal and OffsetLiteral indices ──

class TestH5SymbolicSizeInlinerIndexTypes:
    """H5: SymbolicSizeInliner correctly handles ir.Literal tuple indices.

    im.tuple_get() uses ir.Literal for the index, not ir.OffsetLiteral.
    The fix at line 1784 ensures both are handled.
    """

    def test_tuple_get_with_literal_index_is_inlined(self):
        """H5a: tuple_get(ir.Literal(0), get_domain_range(out, IDim)) → OffsetLiteral(0)."""
        IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        domain = im.call("cartesian_domain")(
            im.named_range(
                IDim,
                im.tuple_get(0, im.call("get_domain_range")(im.ref("out"), IDim)),
                im.tuple_get(1, im.call("get_domain_range")(im.ref("out"), IDim)),
            ),
        )
        program = ir.Program(
            id="t", function_definitions=[], declarations=[],
            params=[im.sym("out"), im.sym("max_i")],
            body=[ir.SetAt(expr=im.ref("out"), domain=domain, target=im.ref("out"))],
        )
        result = SymbolicSizeInliner.apply(program, symbolic_domain_sizes={})
        dom = result.body[0].domain
        assert cpm.is_call_to(dom, "cartesian_domain")
        for nr in dom.args:
            if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
                lo, hi = nr.args[1], nr.args[2]
                # Should be OffsetLiteral(0) and ref("max_i"), not tuple_get(...)
                assert not cpm.is_call_to(lo, "tuple_get"), (
                    "H5: Lower bound not inlined — Literal index fix missing"
                )
                assert not cpm.is_call_to(hi, "tuple_get"), (
                    "H5: Upper bound not inlined — Literal index fix missing"
                )

    def test_tuple_get_with_offset_literal_index_on_make_tuple_is_inlined(self):
        """H5b: tuple_get(OffsetLiteral(0), make_tuple(a, b)) → element a.

        im.tuple_get uses ir.Literal index (uncurried form). The OffsetLiteral path
        in the make_tuple case is an additional guard for unusual IR forms.
        After the S1 fix, both Literal and OffsetLiteral indices work for make_tuple.
        """
        IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        # Build tuple_get in the uncurried form (as im.tuple_get would produce)
        # with an OffsetLiteral index instead of the usual Literal
        make_tuple_expr = im.call("make_tuple")(ir.OffsetLiteral(value=0), im.ref("max_i"))
        # Uncurried: FunCall(fun=SymRef("tuple_get"), args=[OffsetLiteral(0), make_tuple(...)])
        tg = ir.FunCall(
            fun=ir.SymRef(id="tuple_get"),
            args=[ir.OffsetLiteral(value=0), make_tuple_expr],
        )
        result = SymbolicSizeInliner().visit_FunCall(
            tg, symbolic_domain_sizes={}, program_param_ids={"out", "max_i"}
        )
        # Should be OffsetLiteral(0) — the first element of make_tuple
        assert result == ir.OffsetLiteral(value=0), (
            f"H5b: OffsetLiteral index on make_tuple should be inlined; got {result}"
        )


# ── H6: NeighborReductionUnroller three paths are exclusive ──────────────────

class TestH6NeighborUnrollerPathsExclusive:
    """H6: Three paths in NeighborReductionUnroller.visit_FunCall are mutually exclusive.

    Path 1: as_fieldop + is_applied_shift (deref-shift pattern)
    Path 2: _extract_generic_reduce_inputs (generic reduce)
    Path 3: raw is_applied_shift (after generic_visit)
    """

    def test_path3_raw_shift_fires_for_v2e(self):
        """H6a: V2E shift fires path 3 (raw shift)."""
        testee = im.shift("V2E", 0)(im.ref("vert"))
        unroller = NeighborReductionUnroller()
        result = unroller.visit_FunCall(testee, current_domain=None, current_kolor=None)
        # V2E slot 0 should be a simple shift (no concat_where)
        assert not cpm.is_call_to(result, "concat_where"), (
            "H6a: V2E[0] should be a pure shift, not concat_where"
        )

    def test_path2_generic_reduce_fires_for_neighbors_reduce(self):
        """H6b: reduce(plus)(neighbors(V2E, it), ...) fires path 2 (generic reduce)."""
        K = common.Dimension("K", kind=common.DimensionKind.VERTICAL)
        Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
        IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
        domain = im.call("cartesian_domain")(
            im.named_range(IDim, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=2)),
            im.named_range(JDim, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=2)),
            im.named_range(Kolor, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1)),
        )
        vn_field = im.ref("vn")
        # Build: as_fieldop(λit. reduce(plus)(0.0)(deref(it)), domain)(neighbors(V2E, vn))
        reduce_stencil = im.lambda_("it")(
            im.call(im.call("reduce")(im.ref("plus"), im.literal("0.0", "float64")))(
                im.deref("it")
            )
        )
        neighbors_expr = im.as_fieldop(im.lambda_("x")(im.call("neighbors")(im.ref("V2E"), "x")), domain)(vn_field)
        testee = im.as_fieldop(reduce_stencil, domain)(neighbors_expr)
        unroller = NeighborReductionUnroller()
        # Should fire path 2 and unroll the reduction
        result = unroller.visit_FunCall(testee, current_domain=domain, current_kolor=None)
        # If path 2 fires: result is an as_fieldop with accumulated sum
        assert cpm.is_applied_as_fieldop(result), "H6b: path 2 should produce as_fieldop"


# ── H7: _pick_symbolic_int always receives integer values ────────────────────

class TestH7PickSymbolicIntAlwaysInt:
    """H7: _pick_symbolic_int is called only with dict values that are integers.

    Claim: No call site passes a SymRef or other non-integer symbolic value.
    If true, _pick_symbolic_int can be unified with _pick_size_param.
    """

    def test_pick_symbolic_int_with_int_value(self):
        """H7a: Works correctly with plain int."""
        sds = {"max_i": 16}
        result = _pick_symbolic_int(sds, "max_i", "nx")
        assert result == 16

    def test_pick_symbolic_int_with_str_int(self):
        """H7b: Works correctly with string-encoded int."""
        sds = {"max_i": "16"}
        result = _pick_symbolic_int(sds, "max_i")
        assert result == 16

    def test_pick_symbolic_int_returns_none_for_symref_value(self):
        """H7c: If dict value is a SymRef (non-integer), returns None gracefully."""
        sds = {"max_i": im.ref("some_sym")}
        result = _pick_symbolic_int(sds, "max_i")
        assert result is None, (
            "H7c: If a SymRef is stored, _pick_symbolic_int returns None — "
            "call sites must be confirmed to never have SymRef values"
        )

    def test_pick_size_param_vs_pick_symbolic_int_equivalence(self):
        """H7d: _pick_size_param + _offset_int_value is equivalent to _pick_symbolic_int for int values."""
        sds = {"max_i": 16}
        pids = set()
        int_result = _pick_symbolic_int(sds, "max_i")
        expr_result = _pick_size_param(pids, sds, "max_i")
        assert int_result == _offset_int_value(expr_result), (
            "H7d: Both helpers give the same result for int dict values"
        )


# ── H8: deepcopy in _offset_add/_offset_sub zero-path ───────────────────────

class TestH8DeepCopyInArithmeticHelpers:
    """H8: copy.deepcopy(lhs/rhs) in _offset_add/_offset_sub zero-path.

    Claim: When returning lhs/rhs unchanged, deepcopy is called unnecessarily.
    If the returned expression is always embedded in a new node, the copy is wasted.
    """

    def test_offset_add_both_offset_literals_uses_fast_path(self):
        """H8a: When both args are OffsetLiteral, the integer fast-path fires first.

        The zero-path (rv==0) is unreachable for OffsetLiteral inputs because rv is an int,
        so the lv+rv fast path fires first and returns a fresh OffsetLiteral.
        The deepcopy removal in the zero-path is correct but vacuously so for this case.
        """
        expr = ir.OffsetLiteral(value=5)
        result = _offset_add(expr, ir.OffsetLiteral(value=0))
        assert result == expr, "H8a: Result value should equal original"
        # Fast path creates a NEW OffsetLiteral — is not the same object
        assert isinstance(result, ir.OffsetLiteral)
        assert result.value == 5

    def test_offset_sub_both_offset_literals_uses_fast_path(self):
        """H8b: Same fast-path behavior for _offset_sub."""
        expr = ir.OffsetLiteral(value=5)
        result = _offset_sub(expr, ir.OffsetLiteral(value=0))
        assert result == expr
        assert isinstance(result, ir.OffsetLiteral)
        assert result.value == 5

    def test_offset_add_both_int_returns_fresh_node(self):
        """H8c: _offset_add(OffsetLiteral, OffsetLiteral) creates a new OffsetLiteral — no copy."""
        a = ir.OffsetLiteral(value=3)
        b = ir.OffsetLiteral(value=4)
        result = _offset_add(a, b)
        assert result == ir.OffsetLiteral(value=7)
        # No deepcopy in this path — result is always a fresh OffsetLiteral
        assert result is not a and result is not b

    def test_offset_add_zero_lhs_returns_rhs_copy(self):
        """H8d: _offset_add(0, expr) also returns a copy of rhs."""
        rhs = ir.OffsetLiteral(value=7)
        result = _offset_add(ir.OffsetLiteral(value=0), rhs)
        assert result == rhs
        assert result is not rhs, "H8d: deepcopy in zero-lhs path"


# ── Integration: full pipeline with mapping ───────────────────────────────────

class TestFullPipelineIntegration:
    """Integration tests verifying the full pipeline produces correct structured IR."""

    def test_edge_setat_splits_into_3_kolors_with_mapping(self):
        """After full pipeline with mapping: edge SetAt → 3 per-kolor SetAts."""
        sds = _make_minimal_mapping(horizontal_start=1)
        program = ir.Program(
            id="t", function_definitions=[], declarations=[],
            params=[im.sym("inp"), im.sym("out")],
            body=[ir.SetAt(expr=im.ref("inp"), domain=_make_edge_domain(), target=im.ref("out"))],
        )
        result = StructuredTypeRemapper.apply(program, symbolic_domain_sizes=sds)
        result = SetAtRemapper.apply(result, symbolic_domain_sizes=sds)
        assert len(result.body) == 3, (
            f"Expected 3 per-kolor SetAts, got {len(result.body)}"
        )
        kolor_slices = set()
        for stmt in result.body:
            dom = stmt.domain
            if cpm.is_call_to(dom, "cartesian_domain"):
                for nr in dom.args:
                    if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
                        if hasattr(nr.args[0], "value") and nr.args[0].value == "Kolor":
                            lo = nr.args[1].value if isinstance(nr.args[1], ir.OffsetLiteral) else None
                            hi = nr.args[2].value if isinstance(nr.args[2], ir.OffsetLiteral) else None
                            if lo is not None and hi is not None:
                                kolor_slices.add((lo, hi))
        assert kolor_slices == {(0, 1), (1, 2), (2, 3)}, (
            f"Expected kolor slices {{(0,1),(1,2),(2,3)}}, got {kolor_slices}"
        )

    def test_vertex_setat_produces_single_structured_setat(self):
        """Vertex SetAt → 1 structured SetAt (1 kolor)."""
        vertex_axis = ir.AxisLiteral(value="Vertex", kind=common.DimensionKind.HORIZONTAL)
        vertex_domain = im.call("unstructured_domain")(
            im.call("named_range")(vertex_axis, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=4))
        )
        sds = _make_minimal_mapping(horizontal_start=1)
        sds["vertex_to_ij"] = [(0, 0), (0, 0), (0, 1), (1, 0)]
        program = ir.Program(
            id="t", function_definitions=[], declarations=[],
            params=[im.sym("inp"), im.sym("out")],
            body=[ir.SetAt(expr=im.ref("inp"), domain=vertex_domain, target=im.ref("out"))],
        )
        result = StructuredTypeRemapper.apply(program, symbolic_domain_sizes=sds)
        result = SetAtRemapper.apply(result, symbolic_domain_sizes=sds)
        assert len(result.body) == 1, "Vertex SetAt should NOT be split (1 kolor)"
        dom = result.body[0].domain
        assert cpm.is_call_to(dom, "cartesian_domain"), "Vertex domain should be structured"


# ── Mixed-entity tuple SetAt (upstream combined programs) ─────────────────────

_K = common.Dimension("K", common.DimensionKind.VERTICAL)
_KH = common.Dimension("_StaggeredK", common.DimensionKind.VERTICAL)


def _udom(entity: str, vdim: common.Dimension) -> ir.Expr:
    return im.domain(
        common.GridType.UNSTRUCTURED,
        {common.Dimension(entity): (0, 10), vdim: (0, 5)},
    )


def test_split_mixed_entity_tuple_setat_groups_by_entity_and_vertical_dim():
    # let(a = x) in {a, y, z}  @ {Edge x K, Cell x _StaggeredK, Cell x _StaggeredK}
    expr = im.let("a", "x")(im.make_tuple("a", "y", "z"))
    stmt = ir.SetAt(
        expr=expr,
        domain=im.make_tuple(
            _udom("Edge", _K), _udom("Cell", _KH), _udom("Cell", _KH)
        ),
        target=im.make_tuple("o0", "o1", "o2"),
    )
    parts = _split_mixed_entity_tuple_setat(stmt)
    assert parts is not None and len(parts) == 2
    # Edge group keeps the let, single output -> no tuple
    assert parts[0].target == im.ref("o0")
    assert parts[0].expr == im.let("a", "x")("a")
    # Cell group: the let is dead there and dropped
    assert parts[1].target == im.make_tuple("o1", "o2")
    assert parts[1].expr == im.make_tuple("y", "z")


def test_split_mixed_entity_tuple_setat_keeps_uniform_setat():
    stmt = ir.SetAt(
        expr=im.make_tuple("y", "z"),
        domain=im.make_tuple(_udom("Cell", _K), _udom("Cell", _K)),
        target=im.make_tuple("o1", "o2"),
    )
    assert _split_mixed_entity_tuple_setat(stmt) is None


def test_domain_with_vertical_dims_of_flips_and_collapses_tuple():
    cdom = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension("IDim"): (0, 3), _KH: (1, 5)},
    )
    node = im.ref("f")
    node.type = ts.FieldType(
        dims=[common.Dimension("IDim"), _K], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    )
    result = _domain_with_vertical_dims_of(im.make_tuple(cdom, copy.deepcopy(cdom)), node)
    assert cpm.is_call_to(result, "cartesian_domain")
    axes = [nr.args[0].value for nr in result.args]
    assert axes == ["IDim", "K"]
    # untyped node: domain unchanged
    assert _domain_with_vertical_dims_of(cdom, im.ref("g")) is cdom


def test_collect_neighbor_tags_skips_the_field_read_by_neighbors():
    # C2E list over an edge field that is itself an E2C reduction
    inner = im.as_fieldop(im.lambda_("it")(im.neighbors("E2C", "it")))("cell_field")
    inner_red = im.as_fieldop(
        im.lambda_("l")(im.reduce("plus", im.literal_from_value(0.0))(im.deref("l")))
    )(inner)
    outer = im.as_fieldop(im.lambda_("it")(im.neighbors("C2E", "it")))(inner_red)
    assert NeighborReductionUnroller._collect_neighbor_tags(outer) == {"C2E"}
    # a list argument (not read via neighbors) is still followed
    mapped = im.as_fieldop(
        im.lambda_("a", "b")(im.map_list("multiplies")(im.deref("a"), im.deref("b")))
    )("sparse", outer)
    assert NeighborReductionUnroller._collect_neighbor_tags(mapped) == {"C2E"}
