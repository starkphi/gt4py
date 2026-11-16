# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ComposedShiftInliner pass.

Tests verify that chained connectivity patterns are correctly fused into a
single kernel with composed direct shifts:

  V2E ∘ E2C2V  →  vertex output reads vertex source with composed (IDim, JDim) shifts
  E2C ∘ C2E   →  edge output reads edge source with composed shifts (e2c2e-equivalent)
  C2E ∘ E2C   →  cell output reads cell source with composed shifts (c2e2c-equivalent)

Each test builds post-NeighborReductionUnroller IR (cartesian domains, fully
unrolled shifts) and applies ComposedShiftInliner to verify the result.
"""

import copy

import pytest

from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms.structured_backend_passes import (
    ComposedShiftInliner,
    _apply_shift_chain,
    _make_lifted_deref_shift,
)
from gt4py.next.iterator.transforms.map_dict import _kolor_slice


# ── Shared helpers ────────────────────────────────────────────────────────────

def _OL(v: int | str) -> ir.OffsetLiteral:
    return ir.OffsetLiteral(value=v)


def _kolor_threshold_cond(lo: int) -> ir.Expr:
    """Build cartesian_domain condition K[lo, ∞) — the inverse/threshold format produced
    after canonicalize_domain_argument + FuseAsFieldOp."""
    return im.call("cartesian_domain")(
        im.named_range(IDim, _OL(0), _OL(10)),
        im.named_range(JDim, _OL(0), _OL(10)),
        im.named_range(Kolor, _OL(lo), ir.InfinityLiteral.POSITIVE),
    )


IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)


def _cartesian_domain(kolor_lo: int, kolor_hi: int) -> ir.Expr:
    """Minimal cartesian domain with given Kolor range."""
    return im.call("cartesian_domain")(
        im.named_range(IDim, _OL(0), _OL(10)),
        im.named_range(JDim, _OL(0), _OL(10)),
        im.named_range(Kolor, _OL(kolor_lo), _OL(kolor_hi)),
    )


def _shift_spec(di: int, dj: int, dk: int) -> tuple:
    """Build unnormalized shift spec; _apply_shift_chain will normalize."""
    return (
        _OL("IDim"), _OL(di),
        _OL("JDim"), _OL(dj),
        _OL("Kolor"), _OL(dk),
    )


def _make_outer_access(param_name: str, di: int, dj: int, dk: int) -> ir.Expr:
    """Build deref(shift(di,dj,dk)(param_name)) for use inside an outer lambda body."""
    return im.deref(_apply_shift_chain(im.ref(param_name), _shift_spec(di, dj, dk)))


def _make_2branch_cw(src: ir.Expr, outer_domain: ir.Expr,
                     sh_k0: tuple, sh_k1: tuple) -> ir.Expr:
    """Build concat_where(K[0,1), leaf_k0, leaf_k1) for a 2-kolor inner field."""
    leaf_k0 = _make_lifted_deref_shift(copy.deepcopy(src), _shift_spec(*sh_k0), outer_domain)
    leaf_k1 = _make_lifted_deref_shift(copy.deepcopy(src), _shift_spec(*sh_k1), outer_domain)
    return im.concat_where(_kolor_slice(0, 1), leaf_k0, leaf_k1)


def _make_3branch_cw(src: ir.Expr, outer_domain: ir.Expr,
                     sh_k0: tuple, sh_k1: tuple, sh_k2: tuple) -> ir.Expr:
    """Build concat_where(K[0,1), leaf_k0, cw(K[1,2), leaf_k1, leaf_k2)) for 3-kolor."""
    leaf_k0 = _make_lifted_deref_shift(copy.deepcopy(src), _shift_spec(*sh_k0), outer_domain)
    leaf_k1 = _make_lifted_deref_shift(copy.deepcopy(src), _shift_spec(*sh_k1), outer_domain)
    leaf_k2 = _make_lifted_deref_shift(copy.deepcopy(src), _shift_spec(*sh_k2), outer_domain)
    inner_cw = im.concat_where(_kolor_slice(1, 2), leaf_k1, leaf_k2)
    return im.concat_where(_kolor_slice(0, 1), leaf_k0, inner_cw)


def _collect_outer_shifts(setat: ir.SetAt) -> set[tuple]:
    """Return all shifts found as lifted-deref-shift args of the outer as_fieldop."""
    assert cpm.is_applied_as_fieldop(setat.expr)
    result = set()
    for arg in setat.expr.args:
        sh = ComposedShiftInliner._get_lifted_shift(arg)
        if sh is not None:
            result.add(sh)
    return result


def _has_inner_on_domain(setat: ir.SetAt, kolor_lo: int, kolor_hi: int) -> bool:
    """Return True if any as_fieldop with given kolor range still appears in the SetAt."""
    for node in setat.expr.pre_walk_values().if_isinstance(ir.FunCall):
        if not cpm.is_applied_as_fieldop(node):
            continue
        dom = node.fun.args[1] if len(node.fun.args) > 1 else None
        k_range = ComposedShiftInliner._get_kolor_range(dom)
        if k_range == (kolor_lo, kolor_hi):
            return True
    return False


# ── Test _extract_kolor_branch_shifts helper ─────────────────────────────────

class TestExtractKolorBranchShifts:
    """Verify that _extract_kolor_branch_shifts parses the concat_where structure
    produced by _build_field_concat_where_from_branches."""

    def _leaf(self, di: int, dj: int, dk: int) -> ir.Expr:
        dom = _cartesian_domain(0, 1)
        return _make_lifted_deref_shift(im.ref("src"), _shift_spec(di, dj, dk), dom)

    def test_single_leaf_no_concat_where(self):
        leaf = self._leaf(1, 2, 0)
        result = ComposedShiftInliner._extract_kolor_branch_shifts(leaf)
        assert result == {0: (1, 2, 0)}

    def test_2kolor_tree(self):
        """concat_where(K[0,1), leaf_k0, leaf_k1) → {0: sh0, 1: sh1}"""
        leaf_k0 = self._leaf(0, 0, 0)
        leaf_k1 = self._leaf(0, 1, -1)
        cw = im.concat_where(_kolor_slice(0, 1), leaf_k0, leaf_k1)
        result = ComposedShiftInliner._extract_kolor_branch_shifts(cw)
        assert result == {0: (0, 0, 0), 1: (0, 1, -1)}

    def test_3kolor_tree(self):
        """concat_where(K[0,1), leaf_k0, cw(K[1,2), leaf_k1, leaf_k2)) → {0,1,2}"""
        leaf_k0 = self._leaf(0, 0, 0)
        leaf_k1 = self._leaf(0, 1, -1)
        leaf_k2 = self._leaf(1, 1, -2)
        inner_cw = im.concat_where(_kolor_slice(1, 2), leaf_k1, leaf_k2)
        outer_cw = im.concat_where(_kolor_slice(0, 1), leaf_k0, inner_cw)
        result = ComposedShiftInliner._extract_kolor_branch_shifts(outer_cw)
        assert result == {0: (0, 0, 0), 1: (0, 1, -1), 2: (1, 1, -2)}

    def test_non_lifted_shift_returns_none(self):
        """A plain SymRef (not a lifted-deref-shift) in a leaf → None."""
        not_a_leaf = im.ref("some_field")
        cw = im.concat_where(_kolor_slice(0, 1), not_a_leaf, self._leaf(0, 0, 0))
        result = ComposedShiftInliner._extract_kolor_branch_shifts(cw)
        assert result is None


# ── Test: V2E∘E(2-kolor)→V chain (vertex output ← edge intermediate ← vertex src) ──

class TestVertexEdgeVertexFusion:
    """2-kolor inner: vertex stencil fuses edge intermediate into direct vertex reads.

    Pattern:
      outer vertex domain [0,1) reads edge field at K=0 and K=1
      edge domain [0,2) reads vertex source via 2-kolor concat_where

    Composed connectivity: outer K=0 + inner k0 shift = (0,0,0)
                           outer K=1 + inner k1 shift = (0,1,0)
    Both dk_comp = 0, valid for vertex domain [0,1).
    """

    def _make_setat(self) -> ir.SetAt:
        vertex_dom = _cartesian_domain(0, 1)
        edge_dom = _cartesian_domain(0, 2)
        src = im.ref("v_src")

        # Inner edge as_fieldop: λ(vsrc) → deref(vsrc) applied to 2-kolor concat_where
        # kolor 0 → identity (0,0,0); kolor 1 → shift (0,+1,-1)
        inner_concat = _make_2branch_cw(src, vertex_dom, (0, 0, 0), (0, 1, -1))
        inner_asfop = im.as_fieldop(
            im.lambda_("vsrc")(im.deref(im.ref("vsrc"))),
            edge_dom,
        )(inner_concat)

        # Outer vertex stencil: reads edge field at K=0 and K=1
        outer_lambda = im.lambda_("inter")(
            im.plus(
                _make_outer_access("inter", 0, 0, 0),
                _make_outer_access("inter", 0, 0, 1),
            )
        )
        outer_asfop = im.as_fieldop(outer_lambda, vertex_dom)(inner_asfop)
        return ir.SetAt(expr=outer_asfop, domain=vertex_dom, target=ir.SymRef(id="out"))

    def test_intermediate_eliminated(self):
        """After fusion, the edge-domain intermediate must be gone."""
        result = ComposedShiftInliner.apply(self._make_setat())
        assert not _has_inner_on_domain(result, 0, 2), \
            "Edge intermediate [0,2) should have been eliminated"

    def test_two_composed_args(self):
        """Outer as_fieldop should have exactly 2 args (the 2 unique vertex reads)."""
        result = ComposedShiftInliner.apply(self._make_setat())
        assert cpm.is_applied_as_fieldop(result.expr)
        assert len(result.expr.args) == 2, \
            f"Expected 2 composed args, got {len(result.expr.args)}"

    def test_composed_shifts_are_correct(self):
        """Composed shifts: (0,0,0) and (0,1,0) — both at vertex kolor 0."""
        result = ComposedShiftInliner.apply(self._make_setat())
        shifts = _collect_outer_shifts(result)
        assert shifts == {(0, 0, 0), (0, 1, 0)}, \
            f"Unexpected composed shifts: {shifts}"

    def test_cse_collapses_duplicates(self):
        """If both outer shifts map to the same composed position, only 1 arg produced."""
        vertex_dom = _cartesian_domain(0, 1)
        edge_dom = _cartesian_domain(0, 2)
        src = im.ref("v_src")
        # Both inner kolor branches have identity shift → both outer accesses compose to (0,0,0)
        inner_concat = _make_2branch_cw(src, vertex_dom, (0, 0, 0), (0, 0, -1))
        inner_asfop = im.as_fieldop(im.lambda_("vsrc")(im.deref(im.ref("vsrc"))), edge_dom)(inner_concat)
        outer_lambda = im.lambda_("inter")(
            im.plus(
                _make_outer_access("inter", 0, 0, 0),  # K=0 → inner k0: (0,0,0) → composed (0,0,0)
                _make_outer_access("inter", 0, 0, 1),  # K=1 → inner k1: (0,0,-1) → composed (0,0,0)
            )
        )
        setat = ir.SetAt(
            expr=im.as_fieldop(outer_lambda, vertex_dom)(inner_asfop),
            domain=vertex_dom,
            target=ir.SymRef(id="out"),
        )
        result = ComposedShiftInliner.apply(setat)
        # Both compose to (0,0,0) → CSE collapses to 1 arg
        assert len(result.expr.args) == 1, \
            f"CSE should collapse duplicates to 1 arg, got {len(result.expr.args)}"


# ── Test: E2C∘C2E chain (edge output ← cell intermediate ← edge source) ─────

class TestEdgeCellEdgeFusion:
    """2-kolor inner: edge stencil fuses cell intermediate into direct edge reads.

    This simulates an E2C∘C2E composed connectivity (E2C2E-equivalent).

    Pattern:
      outer edge domain [0,1) reads cell field at K=0 and K=1
      cell domain [0,2) reads edge source via 2-kolor concat_where

    Composed:
      outer K=0 + inner k0 (0,0,0)  = (0,0,0),  dk_comp=0 ∈ [0,1) ✓
      outer K=1 + inner k1 (1,0,-1) = (1,0,0),  dk_comp=0 ∈ [0,1) ✓
    """

    def _make_setat(self) -> ir.SetAt:
        edge_dom = _cartesian_domain(0, 1)
        cell_dom = _cartesian_domain(0, 2)
        src = im.ref("e_src")

        # Inner cell as_fieldop: λ(esrc) → deref(esrc), 2-kolor concat_where
        # kolor 0 → (0,0,0); kolor 1 → (1,0,-1)
        inner_concat = _make_2branch_cw(src, edge_dom, (0, 0, 0), (1, 0, -1))
        inner_asfop = im.as_fieldop(
            im.lambda_("esrc")(im.deref(im.ref("esrc"))),
            cell_dom,
        )(inner_concat)

        # Outer edge stencil reads cell intermediate at K=0 and K=1
        outer_lambda = im.lambda_("cm")(
            im.plus(
                _make_outer_access("cm", 0, 0, 0),
                _make_outer_access("cm", 0, 0, 1),
            )
        )
        outer_asfop = im.as_fieldop(outer_lambda, edge_dom)(inner_asfop)
        return ir.SetAt(expr=outer_asfop, domain=edge_dom, target=ir.SymRef(id="out"))

    def test_intermediate_eliminated(self):
        """Cell intermediate [0,2) must be gone after fusion."""
        result = ComposedShiftInliner.apply(self._make_setat())
        assert not _has_inner_on_domain(result, 0, 2)

    def test_two_composed_args(self):
        result = ComposedShiftInliner.apply(self._make_setat())
        assert cpm.is_applied_as_fieldop(result.expr)
        assert len(result.expr.args) == 2

    def test_composed_shifts_are_e2c2e_equivalent(self):
        """Composed shifts: (0,0,0) and (1,0,0) — equivalent to E2C2E direct shifts."""
        result = ComposedShiftInliner.apply(self._make_setat())
        shifts = _collect_outer_shifts(result)
        assert shifts == {(0, 0, 0), (1, 0, 0)}, \
            f"Expected E2C2E-equivalent shifts, got {shifts}"

    def test_source_field_is_preserved(self):
        """The source field SymRef 'e_src' must still appear in the result args."""
        result = ComposedShiftInliner.apply(self._make_setat())
        src_found = any(
            isinstance(n, ir.SymRef) and n.id == "e_src"
            for n in result.expr.pre_walk_values().if_isinstance(ir.SymRef)
        )
        assert src_found, "Source field 'e_src' should be present in composed result"


# ── Test: C2E∘E2C chain (cell output ← edge intermediate ← cell source) ─────

class TestCellEdgeCellFusion:
    """3-kolor inner: cell stencil fuses edge intermediate into direct cell reads.

    This simulates a C2E∘E2C composed connectivity (C2E2C-equivalent).

    Pattern:
      outer cell domain [0,1) reads edge field at K=0 and K=2
      edge domain [0,3) reads cell source via 3-kolor concat_where

    Composed:
      outer K=0 + inner k0 (0,0,0)   = (0,0,0),   dk_comp=0 ∈ [0,1) ✓
      outer K=2 + inner k2 (-1,0,-2) = (-1,0,0),  dk_comp=0 ∈ [0,1) ✓
    """

    def _make_setat(self) -> ir.SetAt:
        cell_dom = _cartesian_domain(0, 1)
        edge_dom = _cartesian_domain(0, 3)
        src = im.ref("c_src")

        # Inner edge as_fieldop: λ(csrc) → deref(csrc), 3-kolor concat_where
        # kolor 0 → (0,0,0); kolor 1 → (0,-1,1) (not accessed by outer); kolor 2 → (-1,0,-2)
        inner_concat = _make_3branch_cw(src, cell_dom, (0, 0, 0), (0, -1, 1), (-1, 0, -2))
        inner_asfop = im.as_fieldop(
            im.lambda_("csrc")(im.deref(im.ref("csrc"))),
            edge_dom,
        )(inner_concat)

        # Outer cell stencil reads edge intermediate at K=0 and K=2
        outer_lambda = im.lambda_("em")(
            im.plus(
                _make_outer_access("em", 0, 0, 0),
                _make_outer_access("em", 0, 0, 2),
            )
        )
        outer_asfop = im.as_fieldop(outer_lambda, cell_dom)(inner_asfop)
        return ir.SetAt(expr=outer_asfop, domain=cell_dom, target=ir.SymRef(id="out"))

    def test_intermediate_eliminated(self):
        """Edge intermediate [0,3) must be gone after fusion."""
        result = ComposedShiftInliner.apply(self._make_setat())
        assert not _has_inner_on_domain(result, 0, 3)

    def test_two_composed_args(self):
        result = ComposedShiftInliner.apply(self._make_setat())
        assert cpm.is_applied_as_fieldop(result.expr)
        assert len(result.expr.args) == 2

    def test_composed_shifts_are_c2e2c_equivalent(self):
        """Composed shifts: (0,0,0) and (-1,0,0) — equivalent to C2E2C direct shifts."""
        result = ComposedShiftInliner.apply(self._make_setat())
        shifts = _collect_outer_shifts(result)
        assert shifts == {(0, 0, 0), (-1, 0, 0)}, \
            f"Expected C2E2C-equivalent shifts, got {shifts}"


# ── Test: V2E∘E2C2V chain with 3-kolor inner (vertex output) ─────────────────

class TestVertexEdgeVertexFusion3Kolor:
    """3-kolor inner: vertex stencil fuses edge intermediate into direct vertex reads.

    Models the rbf_nabla4 pattern (V2E outer, E2C2V inner).

    Pattern:
      outer vertex domain [0,1) reads edge field at K=0 and K=2
      edge domain [0,3) reads vertex source via 3-kolor concat_where

    Composed:
      outer K=0 + inner k0 (-1,+1,0) = (-1,+1,0), dk_comp=0 ∈ [0,1) ✓
      outer K=2 + inner k2 (+1,+1,-2) = (+1,+1,0), dk_comp=0 ∈ [0,1) ✓
    """

    def _make_setat(self) -> ir.SetAt:
        vertex_dom = _cartesian_domain(0, 1)
        edge_dom = _cartesian_domain(0, 3)
        src = im.ref("u_vert")

        # Inner edge as_fieldop: λ(vsrc) → deref(vsrc), 3-kolor concat_where
        # kolor 0 → (-1,+1,0); kolor 1 → (0,+1,-1); kolor 2 → (+1,+1,-2)
        inner_concat = _make_3branch_cw(src, vertex_dom, (-1, 1, 0), (0, 1, -1), (1, 1, -2))
        inner_asfop = im.as_fieldop(
            im.lambda_("vsrc")(im.deref(im.ref("vsrc"))),
            edge_dom,
        )(inner_concat)

        # Outer vertex stencil reads edge at K=0 and K=2
        outer_lambda = im.lambda_("inter")(
            im.plus(
                _make_outer_access("inter", 0, 0, 0),
                _make_outer_access("inter", 0, 0, 2),
            )
        )
        outer_asfop = im.as_fieldop(outer_lambda, vertex_dom)(inner_asfop)
        return ir.SetAt(expr=outer_asfop, domain=vertex_dom, target=ir.SymRef(id="out"))

    def test_intermediate_eliminated(self):
        result = ComposedShiftInliner.apply(self._make_setat())
        assert not _has_inner_on_domain(result, 0, 3)

    def test_composed_shifts_correct(self):
        """K=0 + k0(-1,+1,0) = (-1,+1,0); K=2 + k2(+1,+1,-2) = (+1,+1,0)."""
        result = ComposedShiftInliner.apply(self._make_setat())
        shifts = _collect_outer_shifts(result)
        assert shifts == {(-1, 1, 0), (1, 1, 0)}, \
            f"Expected V2E∘E2C2V composed shifts, got {shifts}"


# ── Test: no-op cases ─────────────────────────────────────────────────────────

class TestNoOpCases:
    """Verify that the pass is a strict no-op when conditions are not met."""

    def test_only_one_outer_shift_is_noop(self):
        """Fewer than 2 distinct outer shifts → no inlining."""
        vertex_dom = _cartesian_domain(0, 1)
        edge_dom = _cartesian_domain(0, 2)
        src = im.ref("src")
        inner_concat = _make_2branch_cw(src, vertex_dom, (0, 0, 0), (0, 1, -1))
        inner_asfop = im.as_fieldop(
            im.lambda_("vsrc")(im.deref(im.ref("vsrc"))), edge_dom
        )(inner_concat)
        # Only ONE outer shift (0,0,0) — below the 2-shift minimum
        outer_lambda = im.lambda_("inter")(
            im.deref(_apply_shift_chain(im.ref("inter"), (
                _OL("IDim"), _OL(0), _OL("JDim"), _OL(0), _OL("Kolor"), _OL(0)
            )))
        )
        setat = ir.SetAt(
            expr=im.as_fieldop(outer_lambda, vertex_dom)(inner_asfop),
            domain=vertex_dom,
            target=ir.SymRef(id="out"),
        )
        result = ComposedShiftInliner.apply(setat)
        # Should be unchanged: edge intermediate still present
        assert _has_inner_on_domain(result, 0, 2), \
            "Pass should be no-op when outer has <2 distinct shifts"

    def test_same_kolor_domain_is_noop(self):
        """Inner domain same as outer domain → no inlining (same entity)."""
        vertex_dom = _cartesian_domain(0, 1)
        src = im.ref("src")
        # Inner has same kolor range [0,1) as outer
        leaf = _make_lifted_deref_shift(src, (
            _OL("IDim"), _OL(0), _OL("JDim"), _OL(0), _OL("Kolor"), _OL(0)
        ), vertex_dom)
        inner_asfop = im.as_fieldop(
            im.lambda_("p")(im.deref(im.ref("p"))), vertex_dom
        )(leaf)
        outer_lambda = im.lambda_("inter")(
            im.plus(
                _make_outer_access("inter", 0, 0, 0),
                _make_outer_access("inter", 0, 1, 0),
            )
        )
        setat = ir.SetAt(
            expr=im.as_fieldop(outer_lambda, vertex_dom)(inner_asfop),
            domain=vertex_dom,
            target=ir.SymRef(id="out"),
        )
        result = ComposedShiftInliner.apply(setat)
        # Same kolor range → inner domains match → no-op
        assert result is setat or result == setat, \
            "Pass should be no-op when inner and outer have same kolor range"

    def test_invalid_composed_kolor_is_noop(self):
        """Composed kolor out of outer range → validity guard rejects, no-op."""
        vertex_dom = _cartesian_domain(0, 1)
        edge_dom = _cartesian_domain(0, 2)
        src = im.ref("src")
        # kolor 0 inner: dk=+1 → composed dk = 0+1 = 1, NOT in [0,1) → guard rejects
        # kolor 1 inner: dk=0  → composed dk = 1+0 = 1, NOT in [0,1) → guard rejects
        inner_concat = _make_2branch_cw(src, vertex_dom, (0, 0, 1), (0, 0, 0))
        inner_asfop = im.as_fieldop(
            im.lambda_("vsrc")(im.deref(im.ref("vsrc"))), edge_dom
        )(inner_concat)
        outer_lambda = im.lambda_("inter")(
            im.plus(
                _make_outer_access("inter", 0, 0, 0),
                _make_outer_access("inter", 0, 0, 1),
            )
        )
        setat = ir.SetAt(
            expr=im.as_fieldop(outer_lambda, vertex_dom)(inner_asfop),
            domain=vertex_dom,
            target=ir.SymRef(id="out"),
        )
        result = ComposedShiftInliner.apply(setat)
        # dk_comp = 0+1=1 and 1+0=1, both out of [0,1) → pass skips
        assert _has_inner_on_domain(result, 0, 2), \
            "Pass should be no-op when composed kolor is out of outer range"


# ── Helpers for inverse/threshold concat_where structure ─────────────────────
# After canonicalize_domain_argument + FuseAsFieldOp the conditions change from
# K[k,k+1) (exact/forward) to K[k,∞) (threshold/inverse):
#
#   2-kolor: concat_where(K[1,∞), leaf_k1, leaf_k0)
#   3-kolor: concat_where(K[1,∞),
#              concat_where(K[2,∞), leaf_k2, leaf_k1),
#              leaf_k0)

def _make_2branch_cw_threshold(src: ir.Expr, outer_domain: ir.Expr,
                                sh_k0: tuple, sh_k1: tuple) -> ir.Expr:
    """Build concat_where(K[1,∞), leaf_k1, leaf_k0) — inverse/threshold format."""
    leaf_k0 = _make_lifted_deref_shift(copy.deepcopy(src), _shift_spec(*sh_k0), outer_domain)
    leaf_k1 = _make_lifted_deref_shift(copy.deepcopy(src), _shift_spec(*sh_k1), outer_domain)
    return im.concat_where(_kolor_threshold_cond(1), leaf_k1, leaf_k0)


def _make_3branch_cw_threshold(src: ir.Expr, outer_domain: ir.Expr,
                                sh_k0: tuple, sh_k1: tuple, sh_k2: tuple) -> ir.Expr:
    """Build the 3-kolor threshold structure matching production IR from FuseAsFieldOp:
      concat_where(K[1,∞), concat_where(K[2,∞), leaf_k2, leaf_k1), leaf_k0)
    """
    leaf_k0 = _make_lifted_deref_shift(copy.deepcopy(src), _shift_spec(*sh_k0), outer_domain)
    leaf_k1 = _make_lifted_deref_shift(copy.deepcopy(src), _shift_spec(*sh_k1), outer_domain)
    leaf_k2 = _make_lifted_deref_shift(copy.deepcopy(src), _shift_spec(*sh_k2), outer_domain)
    inner_cw = im.concat_where(_kolor_threshold_cond(2), leaf_k2, leaf_k1)
    return im.concat_where(_kolor_threshold_cond(1), inner_cw, leaf_k0)


# ── Test _extract_kolor_branch_shifts with K[k,∞) conditions ──────────────────

class TestExtractKolorBranchShiftsThreshold:
    """Verify _extract_kolor_branch_shifts handles the K[k,∞) inverse/threshold format
    produced by canonicalize_domain_argument + FuseAsFieldOp in the fusion loop."""

    def _leaf(self, di: int, dj: int, dk: int) -> ir.Expr:
        dom = _cartesian_domain(0, 1)
        return _make_lifted_deref_shift(im.ref("src"), _shift_spec(di, dj, dk), dom)

    def test_2kolor_threshold(self):
        """concat_where(K[1,∞), leaf_k1, leaf_k0) → {0: sh0, 1: sh1}"""
        leaf_k0 = self._leaf(0, 0, 0)
        leaf_k1 = self._leaf(0, 1, -1)
        cw = im.concat_where(_kolor_threshold_cond(1), leaf_k1, leaf_k0)
        result = ComposedShiftInliner._extract_kolor_branch_shifts(cw)
        assert result == {0: (0, 0, 0), 1: (0, 1, -1)}

    def test_3kolor_threshold(self):
        """concat_where(K[1,∞), cw(K[2,∞), leaf_k2, leaf_k1), leaf_k0) → {0,1,2}"""
        leaf_k0 = self._leaf(0, 0, 0)
        leaf_k1 = self._leaf(0, 1, -1)
        leaf_k2 = self._leaf(1, 1, -2)
        inner_cw = im.concat_where(_kolor_threshold_cond(2), leaf_k2, leaf_k1)
        outer_cw = im.concat_where(_kolor_threshold_cond(1), inner_cw, leaf_k0)
        result = ComposedShiftInliner._extract_kolor_branch_shifts(outer_cw)
        assert result == {0: (0, 0, 0), 1: (0, 1, -1), 2: (1, 1, -2)}

    def test_non_lifted_shift_in_threshold_returns_none(self):
        """A plain SymRef (not a lifted-deref-shift) in a leaf → None."""
        not_a_leaf = im.ref("some_field")
        cw = im.concat_where(_kolor_threshold_cond(1), self._leaf(0, 0, 0), not_a_leaf)
        result = ComposedShiftInliner._extract_kolor_branch_shifts(cw)
        assert result is None

    def test_kolor_lo_from_cond_extracts_threshold(self):
        """_kolor_lo_from_cond returns lo for K[lo, ∞) condition."""
        assert ComposedShiftInliner._kolor_lo_from_cond(_kolor_threshold_cond(0)) == 0
        assert ComposedShiftInliner._kolor_lo_from_cond(_kolor_threshold_cond(1)) == 1
        assert ComposedShiftInliner._kolor_lo_from_cond(_kolor_threshold_cond(2)) == 2

    def test_kolor_lo_from_cond_rejects_exact_slice(self):
        """_kolor_lo_from_cond returns None for exact K[k,k+1) slice (non-infinity upper)."""
        from gt4py.next.iterator.transforms.map_dict import _kolor_slice
        assert ComposedShiftInliner._kolor_lo_from_cond(_kolor_slice(0, 1)) is None
        assert ComposedShiftInliner._kolor_lo_from_cond(_kolor_slice(1, 2)) is None


# ── Test ComposedShiftInliner with threshold inner structure ──────────────────

class TestVertexEdgeVertexFusionThreshold:
    """Verify ComposedShiftInliner fires when the inner concat_where uses K[k,∞) conditions
    (the actual production format produced after canonicalize_domain_argument).

    This mirrors TestVertexEdgeVertexFusion3Kolor but with threshold conditions.
    """

    def _make_setat(self) -> ir.SetAt:
        vertex_dom = _cartesian_domain(0, 1)
        edge_dom = _cartesian_domain(0, 3)
        src = im.ref("u_vert")

        # Inner edge as_fieldop with K[k,∞) concat_where:
        #   kolor 0 → (-1,+1, 0);  kolor 1 → (0,+1,-1);  kolor 2 → (+1,+1,-2)
        # Structure: concat_where(K[1,∞), cw(K[2,∞), leaf_k2, leaf_k1), leaf_k0)
        inner_concat = _make_3branch_cw_threshold(
            src, vertex_dom, (-1, 1, 0), (0, 1, -1), (1, 1, -2)
        )
        inner_asfop = im.as_fieldop(
            im.lambda_("vsrc")(im.deref(im.ref("vsrc"))),
            edge_dom,
        )(inner_concat)

        # Outer vertex stencil reads edge intermediate at K=0 and K=2
        outer_lambda = im.lambda_("inter")(
            im.plus(
                _make_outer_access("inter", 0, 0, 0),
                _make_outer_access("inter", 0, 0, 2),
            )
        )
        outer_asfop = im.as_fieldop(outer_lambda, vertex_dom)(inner_asfop)
        return ir.SetAt(expr=outer_asfop, domain=vertex_dom, target=ir.SymRef(id="out"))

    def test_intermediate_eliminated_threshold(self):
        """Edge intermediate [0,3) must be gone after fusion with threshold conditions."""
        result = ComposedShiftInliner.apply(self._make_setat())
        assert not _has_inner_on_domain(result, 0, 3), \
            "Edge intermediate should be eliminated even with K[k,∞) conditions"

    def test_composed_shifts_correct_threshold(self):
        """Same composed shifts as exact-format test: (-1,+1,0) and (+1,+1,0)."""
        result = ComposedShiftInliner.apply(self._make_setat())
        shifts = _collect_outer_shifts(result)
        assert shifts == {(-1, 1, 0), (1, 1, 0)}, \
            f"Expected V2E∘E2C2V composed shifts with threshold inner, got {shifts}"

    def test_two_args_after_fusion_threshold(self):
        """After fusion: outer as_fieldop should have exactly 2 composed-shift args."""
        result = ComposedShiftInliner.apply(self._make_setat())
        assert cpm.is_applied_as_fieldop(result.expr)
        assert len(result.expr.args) == 2


class TestEdgeCellEdgeFusionThreshold:
    """E2C∘C2E chain where inner concat_where uses K[k,∞) threshold conditions."""

    def _make_setat(self) -> ir.SetAt:
        edge_dom = _cartesian_domain(0, 1)   # single output kolor
        cell_dom = _cartesian_domain(0, 2)   # 2-kolor cell intermediate
        src = im.ref("c_src")

        # Inner cell as_fieldop: K[1,∞) threshold — kolor 0 → (0,0,0); kolor 1 → (-1,0,-1)
        inner_concat = _make_2branch_cw_threshold(src, edge_dom, (0, 0, 0), (-1, 0, -1))
        inner_asfop = im.as_fieldop(
            im.lambda_("csrc")(im.deref(im.ref("csrc"))),
            cell_dom,
        )(inner_concat)

        # Outer edge stencil reads cell intermediate at K=0 and K=1
        outer_lambda = im.lambda_("em")(
            im.plus(
                _make_outer_access("em", 0, 0, 0),
                _make_outer_access("em", 0, 0, 1),
            )
        )
        outer_asfop = im.as_fieldop(outer_lambda, edge_dom)(inner_asfop)
        return ir.SetAt(expr=outer_asfop, domain=edge_dom, target=ir.SymRef(id="out"))

    def test_intermediate_eliminated_threshold(self):
        result = ComposedShiftInliner.apply(self._make_setat())
        assert not _has_inner_on_domain(result, 0, 2), \
            "Cell intermediate should be eliminated with threshold conditions"

    def test_composed_shifts_correct_threshold(self):
        """K=0 + k0(0,0,0) = (0,0,0);  K=1 + k1(-1,0,-1) = (-1,0,0)."""
        result = ComposedShiftInliner.apply(self._make_setat())
        shifts = _collect_outer_shifts(result)
        assert shifts == {(0, 0, 0), (-1, 0, 0)}, \
            f"Expected E2C∘C2E composed shifts with threshold inner, got {shifts}"


# ── Test: multi-arg inner as_fieldop (formula inlining, rbf_nabla4 pattern) ──

class TestMultiArgFormulaInlining:
    """ComposedShiftInliner fuses an inner as_fieldop that has multiple per-kolor
    shifted vertex args (simulating the rbf_nabla4 nabla4 formula structure).

    Uses the actual V2E and E2C2V shifts from the parallelogram mesh:
      6 outer V2E shifts × 4 E2C2V slot groups (3 kolors each)
      = 24 compositions → 7 unique (di, dj, 0) vertex positions

    Inner formula (simplified, no scalar factors): u2 + u3 + u0 + u1
    The pass inlines this formula once per outer V2E slot.

    Per-j CSE (no cross-arg dedup): 4 unique composed shifts per slot × 4 slots = 16 params.
    All composed dk = 0 (valid vertex kolor-0 domain).
    """

    # 6 outer V2E shifts (di, dj, dk) — dk selects edge kolor
    _V2E_SHIFTS = [(-1, 0, 2), (-1, 0, 1), (0, -1, 0), (0, -1, 2), (0, 0, 1), (0, 0, 0)]

    # The 7 unique V2E∘E2C2V composed vertex positions (all dk=0)
    _UNIQUE_7 = {(0, 1, 0), (-1, 1, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (1, -1, 0), (0, 0, 0)}

    def _make_setat(self, include_scalar: bool = False) -> ir.SetAt:
        vertex_dom = _cartesian_domain(0, 1)
        edge_dom = _cartesian_domain(0, 3)
        u_src = im.ref("u_vert")

        # 4 per-kolor shifted vertex args (E2C2V slots 2, 3, 0, 1 — threshold format)
        cw_args = [
            _make_3branch_cw_threshold(u_src, vertex_dom, (-1, 1, 0), (0, 1, -1), (1, 1, -2)),
            _make_3branch_cw_threshold(u_src, vertex_dom, (1, 0, 0), (1, -1, -1), (0, 0, -2)),
            _make_3branch_cw_threshold(u_src, vertex_dom, (0, 0, 0), (0, 0, -1), (0, 1, -2)),
            _make_3branch_cw_threshold(u_src, vertex_dom, (0, 1, 0), (1, 0, -1), (1, 0, -2)),
        ]

        if include_scalar:
            # Add a plain edge-domain scalar arg (pnv1) — not per-kolor shifted
            inner_stencil = im.lambda_("u2", "u3", "u0", "u1", "pnv1")(
                im.plus(
                    im.plus(im.plus(im.plus(
                        im.deref(im.ref("u2")), im.deref(im.ref("u3"))),
                        im.deref(im.ref("u0"))), im.deref(im.ref("u1"))),
                    im.deref(im.ref("pnv1"))
                )
            )
            inner_args = cw_args + [im.ref("pnv1_field")]
        else:
            inner_stencil = im.lambda_("u2", "u3", "u0", "u1")(
                im.plus(im.plus(im.plus(
                    im.deref(im.ref("u2")), im.deref(im.ref("u3"))),
                    im.deref(im.ref("u0"))), im.deref(im.ref("u1")))
            )
            inner_args = cw_args

        inner_asfop = im.as_fieldop(inner_stencil, edge_dom)(*inner_args)

        # Outer vertex stencil: sum of 6 V2E-shifted accesses to edge intermediate
        accesses = [_make_outer_access("inter", di, dj, dk) for di, dj, dk in self._V2E_SHIFTS]
        body = accesses[0]
        for a in accesses[1:]:
            body = im.plus(body, a)
        outer_asfop = im.as_fieldop(im.lambda_("inter")(body), vertex_dom)(inner_asfop)
        return ir.SetAt(expr=outer_asfop, domain=vertex_dom, target=ir.SymRef(id="out"))

    def test_fires(self):
        """Edge-domain intermediate [0,3) is eliminated after fusion."""
        result = ComposedShiftInliner.apply(self._make_setat())
        assert not _has_inner_on_domain(result, 0, 3), \
            "Edge-domain intermediate should be eliminated for multi-arg inner"

    def test_7_unique_composed_positions(self):
        """Composed shifts cover exactly the 7 V2E2C2V vertex positions."""
        result = ComposedShiftInliner.apply(self._make_setat())
        shifts = _collect_outer_shifts(result)
        assert shifts == self._UNIQUE_7, \
            f"Expected 7 V2E2C2V positions, got {shifts}"

    def test_16_vertex_args(self):
        """Per-j CSE: 4 unique per E2C2V slot × 4 slots = 16 vertex read params."""
        result = ComposedShiftInliner.apply(self._make_setat())
        assert cpm.is_applied_as_fieldop(result.expr)
        n_lifted = sum(
            1 for arg in result.expr.args
            if ComposedShiftInliner._get_lifted_shift(arg) is not None
        )
        assert n_lifted == 16, f"Expected 16 vertex-read args, got {n_lifted}"

    def test_with_scalar_fires(self):
        """Pass fires even when inner has a plain edge-scalar arg alongside per-kolor args."""
        result = ComposedShiftInliner.apply(self._make_setat(include_scalar=True))
        assert not _has_inner_on_domain(result, 0, 3), \
            "Edge-domain intermediate should be eliminated with mixed (vertex+scalar) inner args"

    def test_with_scalar_22_lifted_args(self):
        """With 1 edge scalar: 16 vertex reads + 6 V2E-shifted scalar reads = 22 total."""
        result = ComposedShiftInliner.apply(self._make_setat(include_scalar=True))
        assert cpm.is_applied_as_fieldop(result.expr)
        n_lifted = sum(
            1 for arg in result.expr.args
            if ComposedShiftInliner._get_lifted_shift(arg) is not None
        )
        assert n_lifted == 22, f"Expected 22 lifted args (16 vertex + 6 scalar), got {n_lifted}"

    def test_no_minus_in_param_names(self):
        """Edge scalar param names must not contain '-' (invalid C identifier).
        Regression for the _enc() fix in _try_inline_intermediate."""
        result = ComposedShiftInliner.apply(self._make_setat(include_scalar=True))
        new_stencil = result.expr.fun.args[0]
        for param in new_stencil.params:
            assert "-" not in param.id, \
                f"Param name contains '-' (invalid C identifier): {param.id!r}"


# ── Test: let-binding path for tuple output ──────────────────────────────────

class TestLetBindingForTupleOutput:
    """When the outer stencil has a tuple output {u_out, v_out} that both access
    the intermediate at the SAME shifts, CSI should introduce let-bindings
    (FunCall(Lambda, args)) for the unique inlined values instead of duplicating
    the full expression.  DaCe's is_let handler then computes each nabla4 value
    once into a shared transient.
    """

    def _make_tuple_setat(self) -> ir.SetAt:
        """Outer stencil: vertex domain, tuple output {u, v} both summing inter[K=0]+inter[K=1].

        The intermediate is a 2-kolor edge field read at the same K=0 and K=1 shifts for
        both the u_out and v_out components → 4 total accesses, 2 unique → let-binding fires.
        """
        vertex_dom = _cartesian_domain(0, 1)
        edge_dom = _cartesian_domain(0, 2)
        src = im.ref("v_src")

        # Inner edge as_fieldop: kolor 0 → identity (0,0,0); kolor 1 → shift (0,+1,-1)
        inner_concat = _make_2branch_cw(src, vertex_dom, (0, 0, 0), (0, 1, -1))
        inner_asfop = im.as_fieldop(
            im.lambda_("vsrc")(im.deref(im.ref("vsrc"))),
            edge_dom,
        )(inner_concat)

        # Outer stencil: both u_out and v_out use inter at K=0 and K=1 (same shifts)
        outer_lambda = im.lambda_("inter")(
            im.make_tuple(
                im.plus(
                    _make_outer_access("inter", 0, 0, 0),  # nabla4 at K=0
                    _make_outer_access("inter", 0, 0, 1),  # nabla4 at K=1
                ),
                im.plus(
                    _make_outer_access("inter", 0, 0, 0),  # SAME nabla4 at K=0
                    _make_outer_access("inter", 0, 0, 1),  # SAME nabla4 at K=1
                ),
            )
        )
        outer_asfop = im.as_fieldop(outer_lambda, vertex_dom)(inner_asfop)
        return ir.SetAt(expr=outer_asfop, domain=vertex_dom, target=ir.SymRef(id="out"))

    def _make_single_output_setat(self) -> ir.SetAt:
        """Same setup but single-output outer stencil — no sharing, no let-binding."""
        vertex_dom = _cartesian_domain(0, 1)
        edge_dom = _cartesian_domain(0, 2)
        src = im.ref("v_src")
        inner_concat = _make_2branch_cw(src, vertex_dom, (0, 0, 0), (0, 1, -1))
        inner_asfop = im.as_fieldop(
            im.lambda_("vsrc")(im.deref(im.ref("vsrc"))),
            edge_dom,
        )(inner_concat)
        # Single output: inter[K=0] + inter[K=1] — each shift appears exactly once
        outer_lambda = im.lambda_("inter")(
            im.plus(
                _make_outer_access("inter", 0, 0, 0),
                _make_outer_access("inter", 0, 0, 1),
            )
        )
        outer_asfop = im.as_fieldop(outer_lambda, vertex_dom)(inner_asfop)
        return ir.SetAt(expr=outer_asfop, domain=vertex_dom, target=ir.SymRef(id="out"))

    def test_tuple_output_fires(self):
        """CSI fires for the tuple-output stencil (intermediate is eliminated)."""
        result = ComposedShiftInliner.apply(self._make_tuple_setat())
        assert not _has_inner_on_domain(result, 0, 2), \
            "Edge-domain intermediate should be eliminated for tuple-output stencil"

    def _csi_v_params_and_refs(self, setat: ir.SetAt) -> tuple[list[str], set[str]]:
        result = ComposedShiftInliner.apply(setat)
        assert cpm.is_applied_as_fieldop(result.expr)
        fused_lambda = result.expr.fun.args[0]
        assert isinstance(fused_lambda, ir.Lambda)
        params = [str(p.id) for p in fused_lambda.params if str(p.id).startswith("__csi_v")]
        refs = {
            str(n.id)
            for n in fused_lambda.expr.pre_walk_values().if_isinstance(ir.SymRef)
            if str(n.id).startswith("__csi_v")
        }
        return params, refs

    def test_tuple_output_shares_unique_shifts(self):
        """Both tuple components read inter[K=0] and inter[K=1]: 4 accesses, 2 unique
        composed shifts. CSI turns each unique shift into one lifted-deref-shift param
        (__csi_v<j>_<k>) of the fused stencil, so the value is computed once and both
        components reference the same param (no duplicated inner expression)."""
        params, refs = self._csi_v_params_and_refs(self._make_tuple_setat())
        assert len(params) == 2, f"Expected 2 shared __csi_v params, got {params}"
        assert refs == set(params), f"Body refs {refs} != params {params}"

    def test_tuple_output_same_params_as_single_output(self):
        """Duplicating the read in a second tuple component adds no new params."""
        tuple_params, _ = self._csi_v_params_and_refs(self._make_tuple_setat())
        single_params, _ = self._csi_v_params_and_refs(self._make_single_output_setat())
        assert tuple_params == single_params

    def test_single_output_no_let_bindings(self):
        """Single-output stencil: no sharing → body is NOT a let-binding."""
        result = ComposedShiftInliner.apply(self._make_single_output_setat())
        assert not _has_inner_on_domain(result, 0, 2), \
            "Edge-domain intermediate should still be eliminated for single-output"
        fused_lambda = result.expr.fun.args[0]
        assert isinstance(fused_lambda, ir.Lambda)
        assert not cpm.is_let(fused_lambda.expr), \
            "Single-output stencil body should NOT be wrapped in let-bindings"

    def test_tuple_output_no_residual_inner_symrefs(self):
        """The let-binding args should contain the inlined expressions, not refs to the
        eliminated intermediate param — i.e. the intermediate is fully gone from the result."""
        result = ComposedShiftInliner.apply(self._make_tuple_setat())
        # "inter" (the eliminated intermediate param name) should not appear anywhere
        all_symrefs = {
            n.id for n in result.expr.pre_walk_values().if_isinstance(ir.SymRef)
        }
        assert "inter" not in all_symrefs, \
            "Intermediate param 'inter' should be fully eliminated from the fused expression"
