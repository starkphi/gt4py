# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
import pytest
import functools
from gt4py.next import utils
from gt4py.next import common
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.transforms import (
    global_tmps,
    infer_domain,
    collapse_tuple as ct,
)
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator.ir_utils import (
    ir_makers as im,
    common_pattern_matcher as cpm,
    misc as ir_utils_misc,
)


IDim = common.Dimension(value="IDim")
JDim = common.Dimension(value="JDim")
KDim = common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL)
index_type = ts.ScalarType(kind=getattr(ts.ScalarKind, builtins.INTEGER_INDEX_BUILTIN.upper()))
float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
i_field_type = ts.FieldType(dims=[IDim], dtype=float_type)


def index_field_type_factory(dim):
    return ts.FieldType(dims=[dim], dtype=index_type)


def field_tuple_type_factory(field_type, n):
    return ts.TupleType(types=[field_type] * n)


i_field_tuple_type_factory = functools.partial(field_tuple_type_factory, i_field_type)


def program_factory(
    params: list[itir.Sym],
    body: list[itir.SetAt],
    declarations: Optional[list[itir.Temporary]] = None,
) -> itir.Program:
    return itir.Program(
        id="testee",
        function_definitions=[],
        params=params,
        declarations=declarations or [],
        body=body,
    )


def test_trivial(uids: utils.IDGeneratorPool):
    domain = im.domain("cartesian_domain", {IDim: (0, 1)})
    offset_provider = {}
    testee = program_factory(
        params=[im.sym("inp", i_field_type), im.sym("out", i_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop("deref", domain)(im.as_fieldop("deref", domain)("inp")),
                domain=domain,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)
    testee = infer_domain.infer_program(testee, offset_provider=offset_provider)

    expected = program_factory(
        params=[im.sym("inp", i_field_type), im.sym("out", i_field_type)],
        declarations=[itir.Temporary(id="__tmp_0", domain=domain, dtype=float_type)],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_0"), expr=im.as_fieldop("deref", domain)("inp"), domain=domain
            ),
            itir.SetAt(
                target=im.ref("out"), expr=im.as_fieldop("deref", domain)("__tmp_0"), domain=domain
            ),
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider, uids=uids)
    assert actual == expected


def test_trivial_let(uids: utils.IDGeneratorPool):
    domain = im.domain("cartesian_domain", {IDim: (0, 1)})
    offset_provider = {}
    testee = program_factory(
        params=[im.sym("inp", i_field_type), im.sym("out", i_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.let("tmp", im.as_fieldop("deref", domain)("inp"))(
                    im.as_fieldop("deref", domain)("tmp")
                ),
                domain=domain,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)
    testee = infer_domain.infer_program(testee, offset_provider=offset_provider)

    expected = program_factory(
        params=[im.sym("inp", i_field_type), im.sym("out", i_field_type)],
        declarations=[itir.Temporary(id="__tmp_0", domain=domain, dtype=float_type)],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_0"), expr=im.as_fieldop("deref", domain)("inp"), domain=domain
            ),
            itir.SetAt(
                target=im.ref("out"), expr=im.as_fieldop("deref", domain)("__tmp_0"), domain=domain
            ),
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider, uids=uids)
    assert actual == expected


@pytest.mark.parametrize(
    "projector_maker, inp_type, out_type",
    [
        (
            # let projector
            lambda inp: im.let("foo", inp)(
                im.make_tuple(im.tuple_get(1, "foo"), im.tuple_get(3, "foo"))
            ),
            i_field_tuple_type_factory(4),
            i_field_tuple_type_factory(2),
        ),
        (
            # tuple_get projector
            lambda inp: im.tuple_get(2, inp),
            i_field_tuple_type_factory(4),
            i_field_type,
        ),
        (
            # nested tuple_get projector
            lambda inp: im.tuple_get(1, im.tuple_get(2, inp)),
            field_tuple_type_factory(i_field_tuple_type_factory(4), 4),
            i_field_type,
        ),
        (
            lambda inp: im.make_tuple(im.tuple_get(1, inp), im.tuple_get(3, inp)),
            i_field_tuple_type_factory(4),
            i_field_tuple_type_factory(2),
        ),
    ],
)
def test_dont_extract_projector(projector_maker, inp_type, out_type, uids: utils.IDGeneratorPool):
    domain = im.domain("cartesian_domain", {IDim: (0, 1)})
    # this is a a scan, because we assert that we only extract projectors from scans
    scan = im.as_fieldop(
        im.call("scan")(im.lambda_("state", "inp")(im.call("deref")("inp")), True, 0), domain
    )(im.ref("inp", inp_type))

    testee = program_factory(
        params=[
            im.sym("inp", inp_type),
            im.sym("out", out_type),
        ],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=projector_maker(scan),
                domain=domain,
            )
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider={}, uids=uids)
    assert actual == testee  # did not extract from projector


def test_top_level_if(uids: utils.IDGeneratorPool):
    domain = im.domain("cartesian_domain", {IDim: (0, 1)})
    offset_provider = {}
    testee = program_factory(
        params=[
            im.sym("inp1", i_field_type),
            im.sym("inp2", i_field_type),
            im.sym("out", i_field_type),
        ],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.if_(
                    True,
                    im.as_fieldop("deref", domain)("inp1"),
                    im.as_fieldop("deref", domain)("inp2"),
                ),
                domain=domain,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)
    testee = infer_domain.infer_program(testee, offset_provider=offset_provider)

    expected = program_factory(
        params=[
            im.sym("inp1", i_field_type),
            im.sym("inp2", i_field_type),
            im.sym("out", i_field_type),
        ],
        declarations=[],
        body=[
            itir.IfStmt(
                cond=im.literal_from_value(True),
                true_branch=[
                    itir.SetAt(
                        target=im.ref("out"),
                        expr=im.as_fieldop("deref", domain)("inp1"),
                        domain=domain,
                    )
                ],
                false_branch=[
                    itir.SetAt(
                        target=im.ref("out"),
                        expr=im.as_fieldop("deref", domain)("inp2"),
                        domain=domain,
                    )
                ],
            )
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider, uids=uids)
    assert actual == expected


def test_nested_if(uids: utils.IDGeneratorPool):
    domain = im.domain("cartesian_domain", {IDim: (0, 1)})
    offset_provider = {}
    testee = program_factory(
        params=[
            im.sym("inp1", i_field_type),
            im.sym("inp2", i_field_type),
            im.sym("out", i_field_type),
        ],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop("deref", domain)(
                    im.if_(
                        True,
                        im.as_fieldop("deref", domain)("inp1"),
                        im.as_fieldop("deref", domain)("inp2"),
                    )
                ),
                domain=domain,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)
    testee = infer_domain.infer_program(testee, offset_provider=offset_provider)

    expected = program_factory(
        params=[
            im.sym("inp1", i_field_type),
            im.sym("inp2", i_field_type),
            im.sym("out", i_field_type),
        ],
        declarations=[itir.Temporary(id="__tmp_0", domain=domain, dtype=float_type)],
        body=[
            itir.IfStmt(
                cond=im.literal_from_value(True),
                true_branch=[
                    itir.SetAt(
                        target=im.ref("__tmp_0"),
                        expr=im.as_fieldop("deref", domain)("inp1"),
                        domain=domain,
                    )
                ],
                false_branch=[
                    itir.SetAt(
                        target=im.ref("__tmp_0"),
                        expr=im.as_fieldop("deref", domain)("inp2"),
                        domain=domain,
                    )
                ],
            ),
            itir.SetAt(
                target=im.ref("out"), expr=im.as_fieldop("deref", domain)("__tmp_0"), domain=domain
            ),
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider, uids=uids)
    assert actual == expected


def test_tuple_different_domain(uids: utils.IDGeneratorPool):
    domain01 = im.domain("cartesian_domain", {IDim: (0, 1)})
    domain12 = im.domain("cartesian_domain", {IDim: (1, 2)})
    offset_provider = {"I": IDim}

    def add_shifted(domain: itir.FunCall | None = None):
        return im.as_fieldop(
            im.lambda_("it1", "it2")(im.plus(im.deref("it1"), im.deref(im.shift("I", 1)("it2")))),
            domain,
        )

    params = [
        im.sym("cond", ts.ScalarType(kind=ts.ScalarKind.BOOL)),
        im.sym("inp1", i_field_type),
        im.sym("inp2", i_field_type),
        im.sym("out", i_field_type),
    ]
    testee = program_factory(
        params=params,
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.let(
                    "val",
                    im.if_("cond", im.make_tuple("inp1", "inp2"), im.make_tuple("inp2", "inp1")),
                )(add_shifted(None)(im.tuple_get(0, "val"), im.tuple_get(1, "val"))),
                domain=domain01,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)
    testee = infer_domain.infer_program(testee, offset_provider=offset_provider)

    expected = program_factory(
        params=params,
        declarations=[
            itir.Temporary(id="__tmp_0", domain=domain01, dtype=float_type),
            itir.Temporary(id="__tmp_1", domain=domain12, dtype=float_type),
        ],
        body=[
            itir.IfStmt(
                cond=im.ref("cond"),
                true_branch=[
                    itir.SetAt(
                        target=im.make_tuple(im.ref("__tmp_0")),
                        expr=im.make_tuple(im.ref("inp1")),
                        domain=domain01,
                    ),
                    itir.SetAt(
                        target=im.make_tuple(im.ref("__tmp_1")),
                        expr=im.make_tuple(im.ref("inp2")),
                        domain=domain12,
                    ),
                ],
                false_branch=[
                    itir.SetAt(
                        target=im.make_tuple(im.ref("__tmp_0")),
                        expr=im.make_tuple(im.ref("inp2")),
                        domain=domain01,
                    ),
                    itir.SetAt(
                        target=im.make_tuple(im.ref("__tmp_1")),
                        expr=im.make_tuple(im.ref("inp1")),
                        domain=domain12,
                    ),
                ],
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=add_shifted(domain01)(
                    im.ref("__tmp_0"),
                    im.ref("__tmp_1"),
                ),
                domain=domain01,
            ),
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider, uids=uids)
    actual = ct.CollapseTuple.apply(actual, uids=uids)
    assert actual == expected


def test_tuple_different_domain_nested(uids: utils.IDGeneratorPool):
    domain01 = im.domain("cartesian_domain", {IDim: (0, 1)})
    domain12 = im.domain("cartesian_domain", {IDim: (1, 2)})
    domainm10 = im.domain("cartesian_domain", {IDim: (-1, 0)})
    offset_provider = {"I": IDim}

    def add_shifted(domain: itir.FunCall | None = None):
        return im.as_fieldop(
            im.lambda_("it1", "it2", "it3")(
                im.plus(
                    im.deref("it1"),
                    im.plus(im.deref(im.shift("I", 1)("it2")), im.deref(im.shift("I", -1)("it3"))),
                )
            ),
            domain,
        )

    params = [
        im.sym("cond", ts.ScalarType(kind=ts.ScalarKind.BOOL)),
        im.sym("inp1", i_field_type),
        im.sym("inp2", i_field_type),
        im.sym("inp3", i_field_type),
        im.sym("inp4", i_field_type),
        im.sym("inp5", i_field_type),
        im.sym("out", i_field_type),
    ]
    testee = program_factory(
        params=params,
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.let(
                    "val",
                    im.if_(
                        "cond",
                        im.make_tuple(im.make_tuple("inp1", "inp2"), "inp3"),
                        im.make_tuple(im.make_tuple("inp4", "inp5"), "inp1"),
                    ),
                )(
                    add_shifted(None)(
                        im.tuple_get(0, im.tuple_get(0, "val")),
                        im.tuple_get(1, im.tuple_get(0, "val")),
                        im.tuple_get(1, "val"),
                    )
                ),
                domain=domain01,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)
    testee = infer_domain.infer_program(testee, offset_provider=offset_provider)

    expected = program_factory(
        params=params,
        declarations=[
            itir.Temporary(id="__tmp_0", domain=domain01, dtype=float_type),
            itir.Temporary(id="__tmp_1", domain=domain12, dtype=float_type),
            itir.Temporary(id="__tmp_2", domain=domainm10, dtype=float_type),
        ],
        body=[
            itir.IfStmt(
                cond=im.ref("cond"),
                true_branch=[
                    itir.SetAt(
                        target=im.make_tuple(im.ref("__tmp_0")),
                        expr=im.make_tuple(im.ref("inp1")),
                        domain=domain01,
                    ),
                    itir.SetAt(
                        target=im.make_tuple(im.ref("__tmp_1")),
                        expr=im.make_tuple(im.ref("inp2")),
                        domain=domain12,
                    ),
                    itir.SetAt(
                        target=im.make_tuple(im.ref("__tmp_2")),
                        expr=im.make_tuple(im.ref("inp3")),
                        domain=domainm10,
                    ),
                ],
                false_branch=[
                    itir.SetAt(
                        target=im.make_tuple(im.ref("__tmp_0")),
                        expr=im.make_tuple(im.ref("inp4")),
                        domain=domain01,
                    ),
                    itir.SetAt(
                        target=im.make_tuple(im.ref("__tmp_1")),
                        expr=im.make_tuple(im.ref("inp5")),
                        domain=domain12,
                    ),
                    itir.SetAt(
                        target=im.make_tuple(im.ref("__tmp_2")),
                        expr=im.make_tuple(im.ref("inp1")),
                        domain=domainm10,
                    ),
                ],
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=add_shifted(domain01)(
                    im.ref("__tmp_0"),
                    im.ref("__tmp_1"),
                    im.ref("__tmp_2"),
                ),
                domain=domain01,
            ),
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider, uids=uids)
    actual = ct.CollapseTuple.apply(actual, uids=uids)
    assert actual == expected


def test_domain_preservation(uids: utils.IDGeneratorPool):
    domain = im.domain("cartesian_domain", {IDim: (0, 2)})
    domain_tb = im.domain("cartesian_domain", {IDim: (0, 1)})
    domain_fb = im.domain("cartesian_domain", {IDim: (1, 2)})
    offset_provider = {}
    testee = program_factory(
        params=[im.sym("inp", i_field_type), im.sym("out", i_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.op_as_fieldop("plus")(  # no domain here
                    # smaller domain for the next two exprs, must not be re-inferred
                    im.as_fieldop("deref", domain_tb)("inp"),
                    im.as_fieldop("deref", domain_fb)("inp"),
                ),
                domain=domain,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)

    expected = program_factory(
        params=[im.sym("inp", i_field_type), im.sym("out", i_field_type)],
        declarations=[
            itir.Temporary(id="__tmp_0", domain=domain_tb, dtype=float_type),
            itir.Temporary(id="__tmp_1", domain=domain_fb, dtype=float_type),
        ],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_0"),
                expr=im.as_fieldop("deref", domain_tb)("inp"),
                domain=domain_tb,
            ),
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop("deref", domain_fb)("inp"),
                domain=domain_fb,
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=im.op_as_fieldop("plus", domain)("__tmp_0", "__tmp_1"),
                domain=domain,
            ),
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider, uids=uids)
    assert actual == expected


def test_non_scan_projector(uids: utils.IDGeneratorPool):
    domain = im.domain("cartesian_domain", {IDim: (0, 2)})
    offset_provider = {}
    stmt = itir.SetAt(
        target=im.ref("out"),
        expr=im.make_tuple(im.tuple_get(0, "inp")),
        domain=domain,
    )
    testee = program_factory(
        params=[
            im.sym("inp", ts.TupleType(types=[i_field_type, float_type])),
            im.sym("out", ts.TupleType(types=[i_field_type])),
        ],
        body=[stmt],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)

    # make sure the statement actually has a projector
    projector, expr = ir_utils_misc.extract_projector(stmt.expr)
    assert projector is not None

    actual = global_tmps.create_global_tmps(testee, offset_provider, uids=uids)
    assert actual == testee


def _assert_writes_within_targets(program: itir.Program, target_domains: dict[str, itir.FunCall]):
    """Assert that every `SetAt` writes exactly the domain its target was allocated / requested on."""
    target_domains = {
        **target_domains,
        **{decl.id: decl.domain for decl in program.declarations},
    }

    def check(stmts):
        for stmt in stmts:
            if isinstance(stmt, itir.IfStmt):
                check(stmt.true_branch)
                check(stmt.false_branch)
                continue
            targets = (
                stmt.target.args if cpm.is_call_to(stmt.target, "make_tuple") else [stmt.target]
            )
            for target in targets:
                assert isinstance(target, itir.SymRef)
                # compared as text: on this branch the intersection's literals can differ in integer type
                assert str(stmt.domain) == str(target_domains[str(target.id)]), (
                    f"'{target.id}' written on {stmt.domain}, "
                    f"but allocated / requested on {target_domains[str(target.id)]}."
                )

    check(program.body)


def _if_branch_program(body_expr: itir.Expr, outs: list[str], domain: itir.FunCall):
    ik_field_type = ts.FieldType(dims=[IDim, KDim], dtype=float_type)
    return program_factory(
        params=[
            im.sym("cond", ts.ScalarType(kind=ts.ScalarKind.BOOL)),
            im.sym("inp", ik_field_type),
            *(im.sym(out, ik_field_type) for out in outs),
        ],
        body=[
            itir.SetAt(
                target=im.make_tuple(*outs) if len(outs) > 1 else im.ref(outs[0]),
                expr=body_expr,
                domain=domain,
            )
        ],
    )


def test_if_branch_reused_at_shifted_position(uids: utils.IDGeneratorPool):
    # Corresponds to the field operator
    #   w = inp * 2.0
    #   w2 = w
    #   if cond:
    #       w2 = w
    #   return w2, w(KDim + 1)
    # `w` is needed on K [0, 36), the `if_` (and its temporary) only on K [0, 35). Both branches
    # must write K [0, 35) only.
    domain = im.domain("cartesian_domain", {IDim: (0, 4), KDim: (0, 35)})
    Koff = "Koff"
    testee = _if_branch_program(
        im.let("w", im.as_fieldop(im.lambda_("x")(im.multiplies_(im.deref("x"), 2.0)))("inp"))(
            im.make_tuple(
                im.if_("cond", "w", "w"),
                im.as_fieldop(im.lambda_("x")(im.deref(im.shift(Koff, 1)("x"))))("w"),
            )
        ),
        ["out0", "out1"],
        domain,
    )
    testee = type_inference.infer(testee, offset_provider_type={"Koff": KDim})
    testee = infer_domain.infer_program(testee, offset_provider={"Koff": KDim})

    actual = global_tmps.create_global_tmps(testee, {"Koff": KDim}, uids=uids)

    _assert_writes_within_targets(actual, {"out0": domain, "out1": domain})


def test_tuple_if_branch_reused_at_shifted_position(uids: utils.IDGeneratorPool):
    # Same as `test_if_branch_reused_at_shifted_position`, but with a tuple-valued `if_` whose
    # branches are a tuple temporary that is also read shifted.
    domain = im.domain("cartesian_domain", {IDim: (0, 4), KDim: (0, 35)})
    Koff = "Koff"
    testee = _if_branch_program(
        im.let(
            "t", im.as_fieldop(im.lambda_("x")(im.make_tuple(im.deref("x"), im.deref("x"))))("inp")
        )(
            im.let("t2", im.if_("cond", "t", "t"))(
                im.make_tuple(
                    im.tuple_get(0, "t2"),
                    im.tuple_get(1, "t2"),
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift(Koff, 1)("x"))))(
                        im.tuple_get(0, "t")
                    ),
                )
            )
        ),
        ["out0", "out1", "out2"],
        domain,
    )
    testee = type_inference.infer(testee, offset_provider_type={"Koff": KDim})
    testee = infer_domain.infer_program(testee, offset_provider={"Koff": KDim})

    actual = global_tmps.create_global_tmps(testee, {"Koff": KDim}, uids=uids)

    _assert_writes_within_targets(actual, {"out0": domain, "out1": domain, "out2": domain})
