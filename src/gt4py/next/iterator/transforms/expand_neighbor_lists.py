# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Structured backend: rewrite neighbor-list fields into per-slot fields.

The structured backend lowers a neighbor access only in its per-slot form
`as_fieldop(λ(it) → ·⟪E2C2Vₒ, iₒ⟫(it))(f)` (what `f(E2C2V[i])` gives). Helpers such as
`laplacian_on_edge` on gt4py/icon4py main build the whole neighbor list instead and index it
afterwards::

    let lst = as_fieldop(λ(a, b) → map_list(multiplies)(·a, ·b))(
                  as_fieldop(λ(it) → neighbors(E2C2Vₒ, it))(u), pnx)
    in ... as_fieldop(λ(it) → list_get(0, ·it))(lst) ...

This pass pushes each `list_get(i, ·)` of such a field down to its leaves::

    as_fieldop(λ(a, b) → multiplies(·a, ·b))(
        as_fieldop(λ(it) → ·⟪E2C2Vₒ, 0ₒ⟫(it))(u),
        as_fieldop(λ(it) → list_get(0, ·it))(pnx))

Rules for the slot `i` of a list-valued field expression:
  - `as_fieldop(λ(p…) → map_list(f)(·p…))(a…)`  →  `as_fieldop(λ(p…) → f(·p…))(slot(i, a)…)`
  - `as_fieldop(λ(it) → neighbors(o, it))(v)`    →  `as_fieldop(λ(it) → ·⟪o, i⟫(it))(v)`
  - `as_fieldop(λ(p) → make_const_list(·p))(v)`  →  `v`
  - anything else (e.g. a sparse parameter `Field[Edge, E2C2V]`) keeps `list_get(i, ·)`.
A field-level `let` whose bound value is one of the first three forms is inlined first, so the
`list_get`s in its body see the definition.

Only the shapes above are rewritten; `reduce` over neighbor lists is left to the existing
unrolling passes. Run with USE_STRUCTURED_BACKEND=1 only, before `CartesianDomainAndTypeRemapper`.
"""

from __future__ import annotations

from gt4py import eve
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda


def _stencil(node: itir.Expr) -> itir.Lambda | None:
    """Stencil lambda of an applied `as_fieldop`, None otherwise."""
    if cpm.is_applied_as_fieldop(node) and isinstance(node.fun.args[0], itir.Lambda):
        return node.fun.args[0]
    return None


def _rebuild(node: itir.FunCall, stencil: itir.Lambda, args: list[itir.Expr]) -> itir.FunCall:
    """`as_fieldop(stencil, <same domain>)(args)`."""
    domain = node.fun.args[1] if len(node.fun.args) > 1 else None
    return im.as_fieldop(stencil, domain)(*args)


def _derefs_of_params(args: list[itir.Expr], stencil: itir.Lambda) -> bool:
    return len(args) == len(stencil.params) and all(
        cpm.is_call_to(a, "deref") and cpm.is_ref_to(a.args[0], str(p.id))
        for a, p in zip(args, stencil.params)
    )


def _list_kind(node: itir.Expr) -> str | None:
    """Which list-producing form `node` is: 'map', 'neighbors', 'const' or None."""
    stencil = _stencil(node)
    if stencil is None:
        return None
    body = stencil.expr
    if cpm.is_applied_map(body) and _derefs_of_params(list(body.args), stencil):
        return "map"
    if (
        cpm.is_call_to(body, "neighbors")
        and len(stencil.params) == 1
        and cpm.is_ref_to(body.args[1], str(stencil.params[0].id))
    ):
        return "neighbors"
    if cpm.is_call_to(body, "make_const_list") and _derefs_of_params(list(body.args), stencil):
        return "const"
    return None


def _slot(index: int, node: itir.Expr) -> itir.Expr:
    """Field expression for slot `index` of the list-valued field expression `node`."""
    kind = _list_kind(node)
    if kind is None:
        return im.as_fieldop(im.lambda_("it")(im.list_get(index, im.deref("it"))))(node)
    assert isinstance(node, itir.FunCall)
    stencil = _stencil(node)
    assert stencil is not None
    body = stencil.expr
    if kind == "map":
        fun = body.fun.args[0]
        new_stencil = im.lambda_(*stencil.params)(im.call(fun)(*body.args))
        return _rebuild(node, new_stencil, [_slot(index, a) for a in node.args])
    if kind == "neighbors":
        offset = body.args[0]
        param = str(stencil.params[0].id)
        new_stencil = im.lambda_(param)(
            im.deref(im.shift(offset, itir.OffsetLiteral(value=index))(param))
        )
        return _rebuild(node, new_stencil, list(node.args))
    # const list: every slot is the broadcast value
    return node.args[0]


def _list_get_index(node: itir.Expr) -> int | None:
    """`i` if `node` is `as_fieldop(λ(it) → list_get(i, ·it))(x)` with a literal `i`."""
    stencil = _stencil(node)
    if stencil is None or len(stencil.params) != 1:
        return None
    body = stencil.expr
    if (
        cpm.is_call_to(body, "list_get")
        and isinstance(body.args[0], itir.Literal)
        and cpm.is_call_to(body.args[1], "deref")
        and cpm.is_ref_to(body.args[1].args[0], str(stencil.params[0].id))
    ):
        return int(body.args[0].value)
    return None


class ExpandNeighborLists(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """See module docstring."""

    @classmethod
    def apply(cls, program: itir.Program) -> itir.Program:
        return cls().visit(program)

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> itir.Node:
        # inline field-level lets that bind a list-producing field (before visiting the body,
        # so its `list_get`s see the definition)
        if cpm.is_let(node):
            eligible = [_list_kind(arg) is not None for arg in node.args]
            if any(eligible):
                return self.visit(inline_lambda(node, eligible_params=eligible), **kwargs)
        node = self.generic_visit(node, **kwargs)
        index = _list_get_index(node)
        if index is not None and _list_kind(node.args[0]) is not None:
            return _slot(index, node.args[0])
        return node
