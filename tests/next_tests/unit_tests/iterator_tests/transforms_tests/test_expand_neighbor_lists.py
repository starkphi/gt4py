# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.expand_neighbor_lists import ExpandNeighborLists


def _nbr(field: str, offset: str = "E2C2V") -> itir.Expr:
    return im.as_fieldop(
        im.lambda_("it")(im.call("neighbors")(itir.OffsetLiteral(value=offset), "it"))
    )(field)


def _map(fun: str, *args: itir.Expr | str) -> itir.Expr:
    params = [f"p{i}" for i in range(len(args))]
    return im.as_fieldop(
        im.lambda_(*params)(im.call(im.call("map_list")(fun))(*[im.deref(p) for p in params]))
    )(*args)


def _get(index: int, field: itir.Expr | str) -> itir.Expr:
    return im.as_fieldop(im.lambda_("it")(im.list_get(index, im.deref("it"))))(field)


def _slot_shift(index: int, field: str, offset: str = "E2C2V") -> itir.Expr:
    return im.as_fieldop(
        im.lambda_("it")(
            im.deref(im.shift(itir.OffsetLiteral(value=offset), itir.OffsetLiteral(value=index))("it"))
        )
    )(field)


def _apply(expr: itir.Expr) -> itir.Expr:
    return ExpandNeighborLists().visit(expr)


def test_neighbors_slot():
    assert _apply(_get(2, _nbr("u"))) == _slot_shift(2, "u")


def test_map_of_neighbors_and_sparse_field():
    # laplacian_on_edge: (u(E2C2V) * pnx)[E2C2VDim(1)]
    testee = _get(1, _map("multiplies", _nbr("u"), "pnx"))
    expected = im.as_fieldop(im.lambda_("p0", "p1")(im.call("multiplies")(im.deref("p0"), im.deref("p1"))))(
        _slot_shift(1, "u"), _get(1, "pnx")
    )
    assert _apply(testee) == expected


def test_nested_map():
    # (u(E2C2V) * pnx + v(E2C2V) * pny)[E2C2VDim(3)]
    testee = _get(3, _map("plus", _map("multiplies", _nbr("u"), "pnx"), _map("multiplies", _nbr("v"), "pny")))
    mul = lambda a, b: im.as_fieldop(
        im.lambda_("p0", "p1")(im.call("multiplies")(im.deref("p0"), im.deref("p1")))
    )(a, b)
    expected = im.as_fieldop(im.lambda_("p0", "p1")(im.call("plus")(im.deref("p0"), im.deref("p1"))))(
        mul(_slot_shift(3, "u"), _get(3, "pnx")), mul(_slot_shift(3, "v"), _get(3, "pny"))
    )
    assert _apply(testee) == expected


def test_let_bound_list_is_inlined():
    body = im.as_fieldop(im.lambda_("x", "y")(im.plus(im.deref("x"), im.deref("y"))))(
        _get(0, "lst"), _get(3, "lst")
    )
    testee = im.let("lst", _nbr("u"))(body)
    expected = im.as_fieldop(im.lambda_("x", "y")(im.plus(im.deref("x"), im.deref("y"))))(
        _slot_shift(0, "u"), _slot_shift(3, "u")
    )
    assert _apply(testee) == expected


def test_const_list_slot_is_the_value():
    const = im.as_fieldop(im.lambda_("p")(im.call("make_const_list")(im.deref("p"))))("a")
    assert _apply(_get(1, const)) == im.ref("a")


def test_untouched():
    # sparse parameter, let of a non-list field, non-literal index
    assert _apply(_get(0, "pnx")) == _get(0, "pnx")
    plain_let = im.let("t", im.as_fieldop(im.lambda_("p")(im.deref("p")))("a"))(_get(0, "t"))
    assert _apply(plain_let) == plain_let
    dyn = im.as_fieldop(im.lambda_("it")(im.list_get("i", im.deref("it"))))(_nbr("u"))
    assert _apply(dyn) == dyn
