# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import numbers
import os
import sys
import warnings
from typing import Optional, Protocol, cast

from gt4py import eve
from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir, pretty_printer
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import (
    cart_unroll,
    concat_where,
    dead_code_elimination,
    fuse_as_fieldop,
    global_tmps,
    infer_domain,
    infer_domain_ops,
    inline_dynamic_shifts,
    inline_fundefs,
    inline_lifts,
    prune_empty_concat_where,
    remove_broadcast,
    simplify_cart_shifts,
    symbol_ref_utils,
)
from gt4py.next.iterator.transforms.collapse_list_get import CollapseListGet
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.cse import CommonSubexpressionElimination
from gt4py.next.iterator.transforms.fuse_maps import FuseMaps
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.transforms.inline_scalar import InlineScalar
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.transforms.unroll_reduce import UnrollReduce
from gt4py.next.iterator.type_system.inference import infer


class GTIRTransform(Protocol):
    def __call__(
        self, _: itir.Program, *, offset_provider: common.OffsetProvider
    ) -> itir.Program: ...


class _FieldviewDebugStats(eve.NodeVisitor):
    def __init__(self) -> None:
        self.cartesian_reduce_nodes: list[itir.FunCall] = []
        self.nested_as_fieldop_nodes: list[itir.FunCall] = []
        self.named_range_nodes: list[itir.FunCall] = []
        self.named_range_arg_debug: list[tuple[str, str, str]] = []
        self.named_range_count = 0

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        if cpm.is_call_to(node, "cartesian_reduce"):
            self.cartesian_reduce_nodes.append(node)
        if cpm.is_call_to(node, "named_range"):
            self.named_range_count += 1
            self.named_range_nodes.append(node)
            axis, start, stop = node.args
            self.named_range_arg_debug.append(
                (
                    f"{type(axis).__name__}:{getattr(axis, 'value', None)}",
                    f"{type(start).__name__}:{getattr(start, 'value', getattr(start, 'name', None))}",
                    f"{type(stop).__name__}:{getattr(stop, 'value', getattr(stop, 'name', None))}",
                )
            )

        if cpm.is_applied_as_fieldop(node):
            if any(
                cpm.is_applied_as_fieldop(arg) for arg in node.args if isinstance(arg, itir.FunCall)
            ):
                self.nested_as_fieldop_nodes.append(node)

        self.generic_visit(node, **kwargs)


def _write_debug_output(text: str, *, stream=None) -> None:
    global _DEBUG_FILE_INITIALIZED

    if stream is not None:
        print(text, file=stream, end="")

    debug_file = os.environ.get("GT4PY_PRINT_IR_FILE", "ir_out.txt")
    if debug_file:
        mode = "a" if _DEBUG_FILE_INITIALIZED else "w"
        with open(debug_file, mode, encoding="utf-8") as output:
            output.write(text)
        _DEBUG_FILE_INITIALIZED = True


_DEBUG_FILE_INITIALIZED = False


def _print_ir_block(title: str, ir: itir.Program, *, enabled: bool) -> None:
    if not enabled:
        return

    text = "\n" + "=" * 60 + "\n" + title + "\n" + "=" * 60 + "\n" + f"{ir}\n" + "=" * 60 + "\n\n"
    _write_debug_output(text)


def _debug_dump_fieldview_ir(stage: str, ir: itir.Program) -> None:
    if not os.environ.get("GT4PY_DEBUG_FIELDVIEW_IR"):
        return

    stats = _FieldviewDebugStats()
    stats.visit(ir)

    _write_debug_output(
        f"[GT4PY_DEBUG_FIELDVIEW_IR] stage={stage} "
        f"cartesian_reduce_calls={len(stats.cartesian_reduce_nodes)} "
        f"nested_as_fieldop_calls={len(stats.nested_as_fieldop_nodes)} "
        f"named_range_calls={stats.named_range_count}\n",
        stream=sys.stderr,
    )

    if stats.cartesian_reduce_nodes:
        _write_debug_output(
            "[GT4PY_DEBUG_FIELDVIEW_IR] unresolved cartesian_reduce snippets:\n",
            stream=sys.stderr,
        )
        for node in stats.cartesian_reduce_nodes[:5]:
            _write_debug_output(pretty_printer.pformat(node) + "\n", stream=sys.stderr)

    if stats.nested_as_fieldop_nodes:
        _write_debug_output(
            "[GT4PY_DEBUG_FIELDVIEW_IR] nested as_fieldop snippets:\n",
            stream=sys.stderr,
        )
        for node in stats.nested_as_fieldop_nodes[:5]:
            _write_debug_output(pretty_printer.pformat(node) + "\n", stream=sys.stderr)

    if stats.named_range_nodes:
        _write_debug_output("[GT4PY_DEBUG_FIELDVIEW_IR] named_range snippets:\n", stream=sys.stderr)
        for node in stats.named_range_nodes[:5]:
            _write_debug_output(pretty_printer.pformat(node) + "\n", stream=sys.stderr)
        _write_debug_output(
            "[GT4PY_DEBUG_FIELDVIEW_IR] named_range arg types:\n", stream=sys.stderr
        )
        for axis_dbg, start_dbg, stop_dbg in stats.named_range_arg_debug[:5]:
            _write_debug_output(
                f"axis={axis_dbg} start={start_dbg} stop={stop_dbg}\n", stream=sys.stderr
            )

    if os.environ.get("GT4PY_DEBUG_FIELDVIEW_IR_FULL"):
        _write_debug_output(
            "[GT4PY_DEBUG_FIELDVIEW_IR] full pre-infer-domain IR:\n", stream=sys.stderr
        )
        _write_debug_output(pretty_printer.pformat(ir) + "\n", stream=sys.stderr)


def _apply_unroll_reduce_pipeline(
    ir: itir.Program,
    *,
    offset_provider_type: common.OffsetProviderType,
    uids: utils.IDGeneratorPool,
    use_offset_literal_index: bool = True,
) -> itir.Program:
    for _ in range(10):
        try:
            unrolled = cast(
                itir.Program,
                UnrollReduce.apply(
                    ir,
                    offset_provider_type=offset_provider_type,
                    uids=uids,
                    use_offset_literal_index=use_offset_literal_index,
                ),
            )
            unrolled = cast(itir.Program, CollapseListGet().visit(unrolled))
            unrolled = cast(itir.Program, NormalizeShifts().visit(unrolled))
            # this is required as nested neighbor reductions can contain lifts, e.g.,
            # `neighbors(V2Eₒ, ↑f(...))`
            unrolled = cast(itir.Program, inline_lifts.InlineLifts().visit(unrolled))
            unrolled = cast(itir.Program, NormalizeShifts().visit(unrolled))
        except Exception as e:
            raise RuntimeError("Failed inside _apply_unroll_reduce_pipeline") from e
        if unrolled == ir:
            break
        ir = unrolled
    else:
        raise RuntimeError("Reduction unrolling failed.")
    return ir


def _max_domain_range_sizes(offset_provider: common.OffsetProvider) -> dict[str, itir.Literal]:
    """
    Extract horizontal domain sizes from an `offset_provider`.

    Considers the shape of the neighbor table to get the size of each `source_dim` and the maximum
    value inside the neighbor table to get the size of each `codomain`.
    """
    sizes: dict[str, int] = {}
    for provider in offset_provider.values():
        if common.is_neighbor_connectivity(provider):
            src_dim = provider.__gt_type__().source_dim.value
            codomain_dim = provider.__gt_type__().codomain.value
            sizes[src_dim] = max(sizes.get(src_dim, 0), provider.ndarray.shape[0])
            sizes[codomain_dim] = max(
                sizes.get(codomain_dim, 0),
                int(provider.ndarray.max()) + 1,  # type: ignore[attr-defined] # TODO(havogt): improve typing for NDArrayObject
            )

    sizes_exprs = {k: im.literal_from_value(v) for k, v in sizes.items()}
    return sizes_exprs


def _has_dynamic_domains(ir: itir.Program) -> bool:
    # note: this function does not respect symbol collisions with builtins. As it is a temporary
    # workaround we don't care about this corner case.
    domains = set()
    domains |= ir.walk_values().if_isinstance(itir.SetAt).getattr("domain").to_set()
    for as_fop in (
        ir.walk_values()
        .if_isinstance(itir.FunCall)
        .filter(lambda node: cpm.is_call_to(node, "as_fieldop") and len(node.args) == 2)
    ):
        domains.add(as_fop.args[1])
    return len(symbol_ref_utils.collect_symbol_refs(domains)) > 0


def _process_symbolic_domains_option(
    ir: itir.Program,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, itir.Expr]],
    use_max_domain_range_on_unstructured_shift: Optional[bool],
) -> Optional[dict[str, itir.Expr]]:
    """
    Given a program, offset_provider and some configuration options determine how domains are
    inferred.

    The output of this function is used as `symbolic_domain_sizes` argument of domain inference, i.e.
    :func:`infer_domain.infer_program`.

    Right now domains of `as_fieldop` expressions can be inferred either a) using static information
    from the offset provider, or b) they are set to an expression controlled by
    the user and configured in the backend, or c) they are set to the maximum possible domain /
    everywhere (see :func:`_max_domain_range_sizes`)

    Option a) applies when the program is decorated with `static_domains = True` (unless option c)
    is explicitly requested). Then all dynamic domains were replaced with static ones
    which we recognize here. The domain inference then uses this static information which we
    communicate by returning `None`, i.e. no symbolic domain sizes.
    Option b) applies when the user explicitly configured `symbolic_domain_sizes` in the backend.
    In that case we just forward the value.
    Option c) applies when `static_domains = False` or when explicitly configured in the backend
    with `use_max_domain_range_on_unstructured_shift = True`. In that case we determine the
    maximum sizes using :func:`_max_domain_range_sizes` and return them.
    """
    if symbolic_domain_sizes:
        assert not use_max_domain_range_on_unstructured_shift, "Options are mutually exclusive."
        return symbolic_domain_sizes

    if use_max_domain_range_on_unstructured_shift is None:
        use_max_domain_range_on_unstructured_shift = _has_dynamic_domains(ir)
    elif use_max_domain_range_on_unstructured_shift:
        if not _has_dynamic_domains(ir):
            warnings.warn(
                "You are using static domains together with "
                "'use_max_domain_range_on_unstructured_shift'. This is "
                "likely not what you wanted.",
                stacklevel=2,
            )
    if use_max_domain_range_on_unstructured_shift:
        # TODO(havogt): ICON4Py uses this codepath as default for now. Once we use the minimal domain range, we should re-enable this warning.
        # if not _has_dynamic_domains(ir):
        #     warnings.warn(
        #         "You are using static domains together with "
        #         "'use_max_domain_range_on_unstructured_shift'. This is "
        #         "likely not what you wanted.",
        #         stacklevel=2,
        #     )  # noqa: ERA001, RUF100
        assert not symbolic_domain_sizes, "Options are mutually exclusive."
        symbolic_domain_sizes = _max_domain_range_sizes(offset_provider)  # type: ignore[assignment]
    return symbolic_domain_sizes


def pre_inline_scalar_params(
    program: itir.Program, sds: dict | None
) -> itir.Program:
    """Substitute known scalar program param values from sds into the ITIR body.

    Run this once, before any pass pipeline, when USE_STRUCTURED_BACKEND=1.  The result:
    - Boolean params (e.g. limited_area=True) become ir.Literal so dead_code_elimination
      can fold if_(True/False, ...) branches in the standard pipeline.
    - Integer params (e.g. vertical_start=0, vertical_end=5) become OffsetLiteral values,
      making domain bounds concrete before the structural passes run.

    Threshold params are intentionally skipped because ThresholdConditionRewriter matches
    them by SymRef name; substituting them first would suppress that rewriting.
    """
    if not sds or os.environ.get("USE_STRUCTURED_BACKEND", "0") != "1":
        return program

    # ThresholdConditionRewriter pattern-matches these by SymRef id — do not substitute.
    _unsafe: frozenset[str] = frozenset({
        "start_2nd_nudge_line_idx_e", "start_nudging_line_idx_e",
        "start_halo_level_2_idx_e", "start_edge_lateral_boundary",
        "start_edge_lateral_boundary_level_7", "start_edge_nudging_level_2",
        "end_edge_nudging", "end_edge_halo", "horizontal_start_distance",
        "horizontal_end_distance", "lateral_boundary_level_2",
    })
    # Every parameter the threshold rewriter translates from its ICON zone (e.g. interior_idx / halo_idx, which the
    # driver passes as compile-time constants): substituting them would leave an untranslated `index(Cell)` comparison.
    from gt4py.next.iterator.transforms.structured_backend_passes import ZONE_THRESHOLDS

    _unsafe = _unsafe | frozenset(ZONE_THRESHOLDS)

    from gt4py.next.type_system import type_specifications as _ts

    subst: dict[str, itir.Expr] = {}
    for param in program.params:
        pid = str(param.id)
        if pid in _unsafe:
            continue
        val = sds.get(pid)
        # bool must be checked before Integral because bool subclasses int in Python.
        if isinstance(val, bool):
            subst[pid] = itir.Literal(
                value=str(val), type=_ts.ScalarType(kind=_ts.ScalarKind.BOOL)
            )
        elif isinstance(val, numbers.Integral):
            subst[pid] = itir.OffsetLiteral(value=int(val))

    if not subst:
        return program

    class _Substitutor(eve.NodeTranslator):
        def visit_SymRef(self, node: itir.SymRef, **kw) -> itir.Expr:
            r = subst.get(node.id)
            return copy.deepcopy(r) if r is not None else node

    new_body = [_Substitutor().visit(stmt) for stmt in program.body]
    return itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=program.declarations,
        body=new_body,
    )


# TODO(tehrengruber): Revisit interface to configure temporary extraction. We currently forward
#  `extract_temporaries` and `temporary_extraction_heuristics` which is inconvenient.
def apply_common_transforms(
    ir: itir.Program,
    *,
    offset_provider: common.OffsetProvider | common.OffsetProviderType,
    extract_temporaries=False,
    unroll_reduce=False,
    common_subexpression_elimination=True,
    force_inline_lambda_args=False,
    transform_concat_where_to_as_fieldop=True,
    #: A dictionary mapping axes names to their length. See :func:`infer_domain.infer_expr` for
    #: more details.
    symbolic_domain_sizes: Optional[dict[str, itir.Expr]] = None,
    # TODO(tehrengruber): Remove this option again as soon as we have the necessary builtins
    #  to work with / translate domains.
    use_max_domain_range_on_unstructured_shift: Optional[bool] = None,
    cartesian_reduce_axis_ranges: Optional[dict[common.Dimension, tuple[int, int]]] = None,
) -> itir.Program:
    assert isinstance(ir, itir.Program)
    # TODO(tehrengruber): Allow `common.OffsetProviderType`, but domain inference currently
    #  relies on static information or `symbolic_domain_sizes`.
    assert common.is_offset_provider(offset_provider)

    offset_provider_type = common.offset_provider_to_type(offset_provider)
    print_ir = True
    _print_ir_block("=== FINAL GTIR HANDED TO GTFN BACKEND ===", ir, enabled=print_ir)

    symbolic_domain_sizes = _process_symbolic_domains_option(
        ir, offset_provider, symbolic_domain_sizes, use_max_domain_range_on_unstructured_shift
    )

    # Substitute compile-time scalar params (e.g. limited_area=False) so that the
    # dead_code_elimination call below can fold the inactive if/else branch.
    ir = pre_inline_scalar_params(ir, symbolic_domain_sizes)

    uids = utils.IDGeneratorPool()
    ir = MergeLet().visit(ir)
    ir = inline_fundefs.InlineFundefs().visit(ir)
    _print_ir_block("=== GTIR AFTER INLINING FUNDEFS ===", ir, enabled=print_ir)

    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = NormalizeShifts().visit(ir)
    _print_ir_block(
        "=== GTIR AFTER PRUNING UNREFERENCED FUNDEFS AND NORMALIZING SHIFTS ===",
        ir,
        enabled=print_ir,
    )

    # TODO(tehrengruber): Many iterator test contain lifts that need to be inlined, e.g.
    #  test_can_deref. We didn't notice previously as FieldOpFusion did this implicitly everywhere.
    ir = inline_lifts.InlineLifts().visit(ir)

    ir = concat_where.expand_tuple_args(ir, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program
    ir = dead_code_elimination.dead_code_elimination(
        ir, uids=uids, offset_provider_type=offset_provider_type
    )  # domain inference does not support dead-code
    _print_ir_block("=== GTIR AFTER DEAD CODE ELIMINATION ===", ir, enabled=print_ir)
    ir = inline_dynamic_shifts.InlineDynamicShifts.apply(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )  # domain inference does not support dynamic offsets yet

    # ir = cart_unroll.CartUnroll.apply(ir, symbolic_domain_sizes=symbolic_domain_sizes)
    if os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
        ir = cart_unroll.CartesianDomainAndTypeRemapper.apply(  # type: ignore[assignment]
            ir,
            symbolic_domain_sizes=cast(dict[str, str | int] | None, symbolic_domain_sizes),
            offset_provider=offset_provider,
        )
        _print_ir_block(
            "=== GTIR AFTER CARTESIAN DOMAIN AND TYPE REMAPPING ===", ir, enabled=print_ir
        )
        ir = cart_unroll.CartesianReductionUnroller.apply(ir)  # type: ignore[assignment]
        ir = NormalizeShifts().visit(ir)
        _print_ir_block("=== GTIR AFTER CARTESIAN UNROLLING ===", ir, enabled=print_ir)
        # ir = cart_unroll.KolorConstantPropagation.apply(ir)  # type: ignore[assignment]
        # _print_ir_block("=== GTIR AFTER KOLOR CONSTANT PROPAGATION ===", ir, enabled=print_ir)

    ir = infer_domain_ops.InferDomainOps.apply(ir)
    ir = concat_where.canonicalize_domain_argument(ir)
    _print_ir_block("=== GTIR AFTER CANONICALIZING DOMAIN ARGUMENTS ===", ir, enabled=print_ir)
    # if cartesian_reduce_axis_ranges is None:
    #     cartesian_reduce_axis_ranges = {common.Dimension("Kolor"): (0, 3)}
    # ir = UnrollCartesianReduce.apply(ir, axis_ranges=cartesian_reduce_axis_ranges)
    # _print_ir_block("=== GTIR AFTER UNROLLING CARTESIAN REDUCE ===", ir, enabled=print_ir)
    ir = infer_domain.infer_program(
        ir,
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
        allow_uninferred=True,
    )
    ir = prune_empty_concat_where.prune_empty_concat_where(ir)
    ir = remove_broadcast.RemoveBroadcast.apply(ir)
    ir = cast(itir.Program, ConstantFolding.apply(ir))
    # After ConstantFolding some K ranges may now be provably empty (e.g. when vertical_end
    # is small); prune those dead concat_where branches so they don't inflate the IR.
    ir = prune_empty_concat_where.prune_empty_concat_where(ir)
    _print_ir_block(
        "=== GTIR AFTER COMMON TRANSFORMS BEFORE INFER_DOMAIN ===", ir, enabled=print_ir
    )
    if transform_concat_where_to_as_fieldop:
        ir = concat_where.transform_to_as_fieldop(ir)
    elif (
        os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1"
        and os.environ.get("GT4PY_CONCAT_WHERE_AS_FIELDOP", "0") == "1"
    ):
        # OPT-IN (default OFF). Structured DaCe path: the caller passes
        # transform_concat_where_to_as_fieldop=False because DaCe cannot lower a tuple-output
        # as_fieldop (gtir_to_sdfg_primitives.py:255 raises NotImplementedError). A SINGLE-FIELD
        # concat_where can be rewritten into one `if_`-based as_fieldop, which the fusion loop
        # below collapses into a SINGLE DaCe map (apply_diffusion_to_vn: 12 -> 3 kernels).
        #
        # WARNING — this is OFF by default because full fusion evaluates BOTH branches at every
        # position: for asymmetric-cost branches (e.g. diffusion_vn, whose nudging branch runs an
        # expensive E2C2V nabla4) this runs the expensive branch over the whole domain incl. the
        # boundary frame + adds branch divergence, regressing runtime (~+22% on the 516 grid)
        # even though kernel count drops. Only enable it for stencils whose branches are cheap.
        # The default path keeps nudging/boundary as separate restricted maps and removes only
        # the copy kernels via MapSplitter + GT4PyMapBufferElimination (see translation.py).
        ir = concat_where.transform_to_as_fieldop(ir, only_single_field=True)
    _print_ir_block("=== GTIR AFTER TRANSFORM AS FIELDOP ===", ir, enabled=print_ir)
    for _ in range(10):
        inlined = ir

        inlined = InlineLambdas.apply(inlined, opcount_preserving=True)
        inlined = ConstantFolding.apply(inlined)  # type: ignore[assignment]  # always an itir.Program
        # This pass is required to be in the loop such that when an `if_` call with tuple arguments
        # is constant-folded the surrounding tuple_get calls can be removed.
        inlined = CollapseTuple.apply(
            inlined,
            enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
            uids=uids,
            offset_provider_type=offset_provider_type,
        )  # type: ignore[assignment]  # always an itir.Program
        inlined = InlineScalar.apply(inlined, offset_provider_type=offset_provider_type)
        if os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
            inlined = simplify_cart_shifts.SimplifyCartesianShifts.apply(inlined)  # type: ignore[assignment]

        # This pass is required to run after CollapseTuple as otherwise we can not inline
        # expressions like `tuple_get(make_tuple(as_fieldop(stencil)(...)))` where stencil returns
        # a list. Such expressions must be inlined however because no backend supports such
        # field operators right now.
        try:
            inlined = fuse_as_fieldop.FuseAsFieldOp.apply(
                inlined, uids=uids, offset_provider_type=offset_provider_type
            )
        except Exception:
            pass

        # FuseAsFieldOp intersects as_fieldop domains which introduces nested maximum/minimum
        # expressions in the K bounds (e.g. maximum(maximum(X, Y), Y)). Run ConstantFolding
        # immediately after to algebraically simplify them before the next fusion round, otherwise
        # the nesting grows unboundedly over iterations.
        inlined = ConstantFolding.apply(inlined)  # type: ignore[assignment]

        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError("Inlining 'lift' and 'lambdas' did not converge.")

    _print_ir_block("=== GTIR AFTER INLINING LIFTS AND LAMBDAS ===", ir, enabled=print_ir)
    if os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1" and (
        os.environ.get("GT4PY_ENABLE_CSI", "1") == "1"
    ):
        # ComposedShiftInliner needs the as_fieldop(λ(…, __iasfop_N, …) → …) structure that
        # FuseAsFieldOp (in the loop above) produces. Running it earlier (right after
        # CartesianReductionUnroller) is too soon — the intermediate field doesn't exist yet.
        #
        # OPT-IN (GT4PY_ENABLE_CSI=1), off by default: CSI fuses the V2E∘E2C2V chain into a
        # single kernel with no Kolor concat_where, but it forces the inner (E2C2V) stencil to
        # be recomputed at every outer neighbor slot. For rbf_nabla4 at 512×50 this measured
        # ~4× SLOWER than the 2-kernel materialized form (4.35 ms vs 1.02 ms) — the same penalty
        # the hand-written rbf_nabla4_direct pays. Keep it gated until a cost-aware policy or a
        # connectivity chain where recompute is cheap makes it a net win.
        ir = cart_unroll.ComposedShiftInliner.apply(ir)  # type: ignore[assignment]
    # breaks in test_zero_dim_tuple_arg as trivial tuple_get is not inlined
    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination.apply(
            ir, offset_provider_type=offset_provider_type, uids=uids
        )
        ir = MergeLet().visit(ir)
        ir = InlineLambdas.apply(ir, opcount_preserving=True)

    if extract_temporaries:
        ir = infer(ir, inplace=True, offset_provider_type=offset_provider_type)
        ir = global_tmps.create_global_tmps(
            ir,
            offset_provider=offset_provider,
            symbolic_domain_sizes=symbolic_domain_sizes,
            uids=uids,
        )

    _print_ir_block("=== GTIR AFTER COMMON TRANSFORMS ===", ir, enabled=print_ir)

    ir = NormalizeShifts().visit(ir)

    ir = FuseMaps(uids=uids).visit(ir)
    ir = CollapseListGet().visit(ir)

    if unroll_reduce:
        ir = _apply_unroll_reduce_pipeline(
            ir,
            offset_provider_type=offset_provider_type,
            uids=uids,
        )

    _print_ir_block("=== GTIR AFTER UNROLLING REDUCE ===", ir, enabled=print_ir)

    ir = InlineLambdas.apply(
        ir, opcount_preserving=False, force_inline_lambda_args=True
    )

    # NOTE: composed-reduction fusion (e.g. rbf_nabla4 v2e2c2v: E2C2V nabla4 -> edge field ->
    # V2E reduction) is deliberately NOT forced. The edge intermediate is materialised once as a
    # separate map; force-inlining it into the outer reduction recomputes the (itself E2C2V)
    # nabla4 at all 6 V2E slots x 2 outputs and measured ~9x SLOWER at 512x512 (21.0 ms vs
    # 2.25 ms). The 2-kernel materialised form is optimal. See Opt.md "Optimization 6".

    # Structured DaCe path: split a per-kolor edge-threshold `concat_where` SetAt into
    # separate restricted SetAts (interior nudging rect + boundary frame rects) that write the
    # target directly. This removes the native concat_where lowering's full-domain boundary
    # temp + per-kolor copy maps, while keeping the expensive interior (nabla4) stencil
    # restricted to the nudging zone (unlike the `if_` form). Runs before the final
    # infer_program so domain annexes are repopulated on the new restricted as_fieldops.
    # Set GT4PY_DISABLE_CONCAT_WHERE_SETAT_SPLIT=1 to fall back to the native lowering.
    if (
        os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1"
        and os.environ.get("GT4PY_DISABLE_CONCAT_WHERE_SETAT_SPLIT", "0") != "1"
    ):
        ir = cart_unroll.ConcatWhereSetAtSplitter.apply(ir)

    # Output fission (opt-in, DACE_FISSION_OUTPUTS=1): split a tuple-output SetAt
    # (e.g. rbf_nabla4 → {u_vert_out, v_vert_out} ← as_fieldop(λ → make_tuple(u, v))) into one
    # single-output SetAt per element, so each output lowers to its own DaCe map/kernel instead of
    # one register-heavy fused kernel that spills. Runs before the final infer_program so domain
    # annexes are repopulated on the split as_fieldops; re-fusion of the resulting maps is prevented
    # by the output-aware branch of `_kolor_aware_fusion_callback` in the DaCe translation step.
    if (
        os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1"
        and os.environ.get("DACE_FISSION_OUTPUTS", "0") == "1"
    ):
        ir = cart_unroll.TupleOutputSetAtSplitter.apply(ir)

    ir = infer_domain.infer_program(
        ir,
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
        allow_uninferred=True,
    )
    ir = prune_empty_concat_where.prune_empty_concat_where(ir)
    _print_ir_block("=== GTIR END ===", ir, enabled=print_ir)

    assert isinstance(ir, itir.Program)
    return ir


def apply_fieldview_transforms(
    ir: itir.Program,
    *,
    offset_provider: common.OffsetProvider,
    unroll_reduce: bool = False,
    cartesian_reduce_axis_ranges: Optional[dict[common.Dimension, tuple[int, int]]] = None,
    use_max_domain_range_on_unstructured_shift: Optional[bool] = None,
    symbolic_domain_sizes: Optional[dict[str, str | int]] = None,
) -> itir.Program:
    offset_provider_type = common.offset_provider_to_type(offset_provider)

    uids = utils.IDGeneratorPool()
    _print_ir_block("=== FIELDVIEW IR BEFORE TRANSFORMS ===", ir, enabled=True)
    symbolic_domain_sizes = _process_symbolic_domains_option(
        ir,
        offset_provider,
        cast(Optional[dict[str, itir.Expr]], symbolic_domain_sizes),
        use_max_domain_range_on_unstructured_shift,
    )
    _print_ir_block(
        "=== FIELDVIEW IR AFTER PROCESSING DOMAIN OPTIONS ===", ir, enabled=True
    )
    ir = inline_fundefs.InlineFundefs().visit(ir)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    _print_ir_block("=== FIELDVIEW IR AFTER INLINING FUNDEFS ===", ir, enabled=True)
    # required for dead-code-elimination and `prune_empty_concat_where` pass
    ir = concat_where.expand_tuple_args(ir, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program

    ir = dead_code_elimination.dead_code_elimination(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )
    _print_ir_block("=== FIELDVIEW IR AFTER DEAD CODE ELIMINATION ===", ir, enabled=True)
    ir = inline_dynamic_shifts.InlineDynamicShifts.apply(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )  # domain inference does not support dynamic offsets yet

    if os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
        ir = NormalizeShifts().visit(ir)
        ir = inline_lifts.InlineLifts().visit(ir)
        ir = cart_unroll.CartesianDomainAndTypeRemapper.apply(  # type: ignore[assignment]
            ir,
            symbolic_domain_sizes=cast(dict[str, str | int] | None, symbolic_domain_sizes),
            offset_provider=offset_provider,
        )
        ir = cart_unroll.CartesianReductionUnroller.apply(ir)  # type: ignore[assignment]
        ir = NormalizeShifts().visit(ir)
        ir = concat_where.expand_tuple_args(ir, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program
        ir = dead_code_elimination.dead_code_elimination(
            ir, offset_provider_type=offset_provider_type, uids=uids
        )
    _print_ir_block("=== FIELDVIEW IR AFTER CARTESIAN UNROLLING ===", ir, enabled=True)
    ir = infer_domain_ops.InferDomainOps.apply(ir)
    _print_ir_block("=== FIELDVIEW IR AFTER INFERRING DOMAIN OPS ===", ir, enabled=True)
    ir = concat_where.canonicalize_domain_argument(ir)

    ir = ConstantFolding.apply(ir)  # type: ignore[assignment]  # always an itir.Program

    # if cartesian_reduce_axis_ranges is None:
    #     cartesian_reduce_axis_ranges = {common.Dimension("Kolor"): (0, 3)}
    # ir = UnrollCartesianReduce.apply(ir, axis_ranges=cartesian_reduce_axis_ranges)

    # For the structured backend the first infer_program runs AFTER the fusion loop (below).
    # Running it here sets annex.domain annotations that cause FuseAsFieldOp to merge
    # as_fieldop nodes across kolor boundaries → CUDA_ERROR_ILLEGAL_ADDRESS on big grids.
    if os.environ.get("USE_STRUCTURED_BACKEND", "0") != "1":
        ir = infer_domain.infer_program(
            ir,
            symbolic_domain_sizes=symbolic_domain_sizes,
            offset_provider=offset_provider,
        )
        ir = ConstantFolding.apply(ir)  # type: ignore[assignment]  # always an itir.Program

        ir = prune_empty_concat_where.prune_empty_concat_where(ir)
        _print_ir_block("=== FIELDVIEW IR AFTER PRUNING EMPTY CONCAT WHERE ===", ir, enabled=True)

    # RemoveBroadcast reads node.annex.domain (set by infer_program). For the structured
    # backend, infer_program runs after the fusion loop, so RemoveBroadcast must also be deferred.
    if os.environ.get("USE_STRUCTURED_BACKEND", "0") != "1":
        ir = remove_broadcast.RemoveBroadcast.apply(ir)

    # Fuse the per-op as_fieldop nodes produced by the structured backend passes. Without this,
    # the IR keeps every `+`, `×`, `cast_`, `if`, `list_get` as a separate as_fieldop over the
    # full structured domain (513×513×3×K for a 512² grid). Each as_fieldop becomes its own
    # DaCe nested-SDFG/map writing to a per-op transient — works at tiny grids but exhausts and
    # corrupts the GPU transient pool at 512² (CUDA_ERROR_ILLEGAL_ADDRESS). See CLAUDE.md
    # § "Bug 6 — CUDA ILLEGAL_ADDRESS on big grids".
    #
    # We deliberately do NOT call `concat_where.transform_to_as_fieldop` here. The DaCe path
    # in grafrap3 restricts `translate_as_fieldop` to single-field outputs (see
    # gtir_to_sdfg_primitives.py:255 — raises NotImplementedError for tuple outputs). Calling
    # transform_to_as_fieldop on a tuple-returning concat_where collapses it into a single
    # tuple-output as_fieldop that DaCe cannot lower. concat_where is left intact and lowered
    # by gtir_to_sdfg_concat_where.py — this matches what grafrap_dace did with
    # `transform_concat_where_to_as_fieldop=False` in `apply_common_transforms`.
    if os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
        # Pre-fusion: annotate per-kolor domains so CSE in the fusion loop cannot merge
        # kolor-distinct outer branches into one shared node. Without this, CSE collapses
        # 3 identical stencil_as_fieldop branches (generated for non-split E2C2EO SetAts)
        # into one shared node → post-fusion infer_program uses the union context
        # Kolor:[0,3) → E2C back-propagation produces Kolor:[-2,3) for cell fields → GPU OOB.
        ir = infer_domain.infer_program(
            ir,
            symbolic_domain_sizes=symbolic_domain_sizes,
            offset_provider=offset_provider,
            allow_uninferred=True,
        )
        ir = prune_empty_concat_where.prune_empty_concat_where(ir)
        _prev_fuse_made_progress = True  # allow first iteration always
        for _iter in range(10):
            # If the previous iteration's FuseAsFieldOp made no progress, one more cleanup
            # pass was already run. No further iterations will reduce the IR, so stop.
            # Root cause of the 10x overrun: CSE creates new tlet_N names each iteration
            # (advancing the global uids), so `inlined == ir` never fires even when the
            # computation is structurally identical. Use fusion progress as the real signal.
            if not _prev_fuse_made_progress:
                break
            inlined = ir
            inlined = InlineLambdas.apply(inlined, opcount_preserving=True)
            inlined = ConstantFolding.apply(inlined)  # type: ignore[assignment]
            inlined = CollapseTuple.apply(
                inlined,
                enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
                uids=uids,
                offset_provider_type=offset_provider_type,
            )  # type: ignore[assignment]
            inlined = InlineScalar.apply(inlined, offset_provider_type=offset_provider_type)
            inlined = simplify_cart_shifts.SimplifyCartesianShifts.apply(inlined)  # type: ignore[assignment]
            # CSE before FuseAsFieldOp: normalises repeated accesses to the same field.
            inlined = CommonSubexpressionElimination.apply(
                inlined, offset_provider_type=offset_provider_type, uids=uids
            )
            inlined = MergeLet().visit(inlined)
            _fuse_made_progress = False
            if not os.environ.get("GT4PY_DISABLE_FUSE_AS_FIELDOP"):
                _n_before = str(inlined).count("as_fieldop")
                _pre_fuse = inlined
                try:
                    inlined = fuse_as_fieldop.FuseAsFieldOp.apply(
                        inlined, uids=uids, offset_provider_type=offset_provider_type
                    )
                except Exception:
                    inlined = _pre_fuse
                _n_after = str(inlined).count("as_fieldop")
                # Reject fusion that collapses too aggressively in one step.
                # nabla2_smag: 249→3 (83×) — FuseAsFieldOp merges as_fieldop nodes shared
                # across kolor-split SetAts, creating a dangling connector (tlet_42_minus)
                # in DaCe SDFG lowering. Threshold of 20 blocks this (83 > 20) while
                # allowing typical 2-10× reductions for simpler stencils.
                _ratio = int(os.environ.get("GT4PY_FUSE_RATIO_THRESHOLD", "20"))
                if _n_before > 0 and _n_after > 0 and (_n_before // _n_after) > _ratio:
                    inlined = _pre_fuse
                    _n_after = _n_before
                print(f"[fusion] FuseAsFieldOp: {_n_before} -> {_n_after} as_fieldop nodes")
                _fuse_made_progress = _n_after < _n_before
            inlined = ConstantFolding.apply(inlined)  # type: ignore[assignment]
            if inlined == ir:
                break
            ir = inlined
            _prev_fuse_made_progress = _fuse_made_progress
        ir = NormalizeShifts().visit(ir)
        ir = InlineLambdas.apply(ir, opcount_preserving=True, force_inline_lambda_args=True)
        # ComposedShiftInliner needs the as_fieldop(λ(…, __iasfop_N, …) → …) structure that
        # FuseAsFieldOp produces; run it now, after the fusion loop.
        # OPT-IN (GT4PY_ENABLE_CSI=1), off by default — see the apply_common_transforms call
        # site for the rationale (fusion recomputes the inner stencil per slot → slower at scale).
        if os.environ.get("GT4PY_ENABLE_CSI", "0") == "1":
            ir = cart_unroll.ComposedShiftInliner.apply(ir)  # type: ignore[assignment]
        # The fusion loop rebuilds the IR tree (InlineLambdas / FuseAsFieldOp create new nodes),
        # which strips the `node.annex.domain` that gtir_to_sdfg_concat_where.translate_concat_where
        # reads at lowering time. Re-run domain inference so the annex is repopulated on the
        # rebuilt concat_where nodes.
        ir = infer_domain.infer_program(
            ir,
            symbolic_domain_sizes=symbolic_domain_sizes,
            offset_provider=offset_provider,
            allow_uninferred=True,
        )
        ir = prune_empty_concat_where.prune_empty_concat_where(ir)
        ir = remove_broadcast.RemoveBroadcast.apply(ir)
    _print_ir_block("=== FINAL FIELDVIEW IR ===", ir, enabled=True)
    return ir
