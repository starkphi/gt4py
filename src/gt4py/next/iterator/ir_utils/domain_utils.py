# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import warnings
from typing import Callable, Iterable, Literal, Optional

import numpy as np

from gt4py.next import common
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import trace_shifts
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding


#: Threshold fraction of domain points which may be added to a domain on translation in order
#: to have a contiguous domain before a warning is raised.
_NON_CONTIGUOUS_DOMAIN_WARNING_THRESHOLD: float = 1 / 4

#: Offset tags for which a non-contiguous domain warning has already been printed
_NON_CONTIGUOUS_DOMAIN_WARNING_SKIPPED_OFFSET_TAGS: set[str] = set()


def _static_bound(bound: itir.Expr) -> int | None:
    """Integer value of a range bound: an integer `Literal`, or an integer `OffsetLiteral` (the structured backend
    writes Cartesian bounds as `OffsetLiteral`, e.g. `190ₒ`). None if not static."""
    if isinstance(bound, itir.Literal):
        try:
            return int(bound.value)
        except ValueError:
            return None
    if isinstance(bound, itir.OffsetLiteral) and isinstance(bound.value, int):
        return bound.value
    return None


@dataclasses.dataclass(frozen=True)
class SymbolicRange:
    start: itir.Expr
    stop: itir.Expr

    def __post_init__(self) -> None:
        # TODO(havogt): added this defensive checks as code seems to make this reasonable assumption
        assert self.start is not itir.InfinityLiteral.POSITIVE
        assert self.stop is not itir.InfinityLiteral.NEGATIVE

    def translate(self, distance: int) -> SymbolicRange:
        return SymbolicRange(im.plus(self.start, distance), im.plus(self.stop, distance))

    def empty(self) -> bool | None:
        start, stop = _static_bound(self.start), _static_bound(self.stop)
        if start is not None and stop is not None:
            return start >= stop
        elif self.start == self.stop:
            return True
        return None


_GRID_TYPE_MAPPING = {
    "unstructured_domain": common.GridType.UNSTRUCTURED,
    "cartesian_domain": common.GridType.CARTESIAN,
}


def _unstructured_translate_range_statically(
    range_: SymbolicRange,
    tag: str,
    val: itir.OffsetLiteral
    | Literal[trace_shifts.Sentinel.VALUE, trace_shifts.Sentinel.ALL_NEIGHBORS],
    offset_provider: common.OffsetProvider,
    expr: itir.Expr | None = None,
) -> SymbolicRange:
    """
    Translate `range_` using static connectivity information from `offset_provider`.
    """
    assert common.is_offset_provider(offset_provider)
    connectivity = offset_provider[tag]
    assert common.is_neighbor_connectivity(connectivity)
    skip_value = connectivity.skip_value

    # fold & convert expr into actual integers
    start_expr, stop_expr = range_.start, range_.stop
    # note: if you find tuple expressions on literals here, you likely forgot to collapse tuple
    # expressions beforehand
    assert isinstance(start_expr, itir.Literal) and isinstance(stop_expr, itir.Literal)
    start, stop = int(start_expr.value), int(stop_expr.value)

    nb_index: slice | int
    if val in [trace_shifts.Sentinel.ALL_NEIGHBORS, trace_shifts.Sentinel.VALUE]:
        nb_index = slice(None)
    else:
        nb_index = val.value  # type: ignore[assignment]  # assert above

    accessed = connectivity.ndarray[start:stop, nb_index]

    if isinstance(val, itir.OffsetLiteral) and np.any(accessed == skip_value):
        # TODO(tehrengruber): Turn this into a configurable error. This is currently
        #  not possible since some test cases starting from ITIR containing
        #  `can_deref` might lead here. The frontend never emits such IR and domain
        #  inference runs after we transform reductions into stmts containing
        #  `can_deref`.
        warnings.warn(
            UserWarning(f"Translating '{expr}' using '{tag}' has an out-of-bounds access."),
            stacklevel=2,
        )

    new_start, new_stop = accessed.min(), accessed.max() + 1  # type: ignore[attr-defined]  # TODO(havogt): improve typing for NDArrayObject

    fraction_accessed = np.unique(accessed).size / (new_stop - new_start)  # type: ignore[call-overload]  # TODO(havogt): improve typing for NDArrayObject

    if fraction_accessed < _NON_CONTIGUOUS_DOMAIN_WARNING_THRESHOLD and (
        tag not in _NON_CONTIGUOUS_DOMAIN_WARNING_SKIPPED_OFFSET_TAGS
    ):
        _NON_CONTIGUOUS_DOMAIN_WARNING_SKIPPED_OFFSET_TAGS.add(tag)
        warnings.warn(
            UserWarning(
                f"Translating '{expr}' using '{tag}' requires "
                f"computations on many additional points "
                f"({round((1 - fraction_accessed) * 100)}%) in order to get a contiguous "
                f"domain. Please consider reordering your mesh."
            ),
            stacklevel=2,
        )

    return SymbolicRange(
        im.literal(str(new_start), builtins.INTEGER_INDEX_BUILTIN),
        im.literal(str(new_stop), builtins.INTEGER_INDEX_BUILTIN),
    )


@dataclasses.dataclass(frozen=True)
class SymbolicDomain:
    grid_type: common.GridType
    ranges: dict[common.Dimension, SymbolicRange]

    def __hash__(self) -> int:
        return hash((self.grid_type, frozenset(self.ranges.items())))

    def empty(self) -> bool | None:
        if any(r.empty() for r in self.ranges.values()):
            return True
        if any(r.empty() is None for r in self.ranges.values()):
            return None
        return False

    @classmethod
    def from_expr(cls, node: itir.Node) -> SymbolicDomain:
        assert cpm.is_call_to(node, ("unstructured_domain", "cartesian_domain"))

        ranges: dict[common.Dimension, SymbolicRange] = {}
        for named_range in node.args:
            assert cpm.is_call_to(named_range, "named_range")
            axis_literal, lower_bound, upper_bound = named_range.args
            assert isinstance(axis_literal, itir.AxisLiteral)

            ranges[common.Dimension(value=axis_literal.value, kind=axis_literal.kind)] = (
                SymbolicRange(lower_bound, upper_bound)
            )
        return cls(_GRID_TYPE_MAPPING[node.fun.id], ranges)

    def as_expr(self) -> itir.FunCall:
        converted_ranges: dict[common.Dimension, tuple[itir.Expr, itir.Expr]] = {
            key: (value.start, value.stop) for key, value in self.ranges.items()
        }
        return im.domain(self.grid_type, converted_ranges)

    def translate(
        self: SymbolicDomain,
        shift: tuple[
            itir.OffsetLiteral
            | Literal[trace_shifts.Sentinel.VALUE, trace_shifts.Sentinel.ALL_NEIGHBORS],
            ...,
        ],
        offset_provider: common.OffsetProvider | common.OffsetProviderType,
        #: A dictionary mapping axes names to their length. See
        #: func:`gt4py.next.iterator.transforms.infer_domain.infer_expr` for more details.
        symbolic_domain_sizes: Optional[dict[str, itir.Expr]] = None,
    ) -> SymbolicDomain:
        offset_provider_type = common.offset_provider_to_type(offset_provider)

        dims = list(self.ranges.keys())
        new_ranges = {dim: self.ranges[dim] for dim in dims}
        if len(shift) == 0:
            return self
        if len(shift) == 2:
            off, val = shift
            assert isinstance(off, itir.OffsetLiteral) and isinstance(off.value, str)
            connectivity_type = common.get_offset_type(offset_provider_type, off.value)

            if isinstance(connectivity_type, common.Dimension):
                if val is trace_shifts.Sentinel.VALUE:
                    raise NotImplementedError("Dynamic offsets not supported.")
                assert isinstance(val, itir.OffsetLiteral) and isinstance(val.value, int)
                current_dim = connectivity_type
                # cartesian offset
                if current_dim not in new_ranges:
                    new_ranges[current_dim] = SymbolicRange(
                        itir.InfinityLiteral.NEGATIVE,
                        itir.InfinityLiteral.POSITIVE,
                    )
                new_ranges[current_dim] = SymbolicRange.translate(
                    new_ranges[current_dim], val.value
                )
            elif isinstance(connectivity_type, common.NeighborConnectivityType):
                # unstructured shift
                assert (
                    isinstance(val, itir.OffsetLiteral) and isinstance(val.value, int)
                ) or val in [
                    trace_shifts.Sentinel.ALL_NEIGHBORS,
                    trace_shifts.Sentinel.VALUE,
                ]
                old_dim = connectivity_type.source_dim
                new_dim = connectivity_type.codomain
                assert new_dim not in new_ranges or old_dim == new_dim

                # If the new_dim is not in symbolic_domain_sizes, try to compute it from the connectivity
                if symbolic_domain_sizes is None or new_dim.value not in symbolic_domain_sizes:
                    # Try to extract the size from the offset provider itself
                    offset_name = off.value if isinstance(off, itir.OffsetLiteral) else None
                    if offset_name and common.is_offset_provider(offset_provider):
                        provider = offset_provider.get(offset_name)
                        if provider is not None and common.is_neighbor_connectivity(provider):
                            # Extract codomain size from the connectivity array
                            max_neighbor = int(provider.ndarray.max())  # type: ignore[attr-defined]
                            # ensure the dict exists before assigning
                            if symbolic_domain_sizes is None:
                                symbolic_domain_sizes = {}
                            symbolic_domain_sizes[new_dim.value] = im.literal(
                                str(max_neighbor + 1), builtins.INTEGER_INDEX_BUILTIN
                            )

                    # If still not found, skip this translation - return unchanged domain
                    if symbolic_domain_sizes is None or new_dim.value not in symbolic_domain_sizes:
                        return self

                if symbolic_domain_sizes is not None and new_dim.value in symbolic_domain_sizes:
                    new_range = SymbolicRange(
                        im.literal(str(0), builtins.INTEGER_INDEX_BUILTIN),
                        im.ensure_expr(symbolic_domain_sizes[new_dim.value]),
                    )
                else:
                    assert common.is_offset_provider(offset_provider)
                    new_range = _unstructured_translate_range_statically(
                        new_ranges[old_dim], off.value, val, offset_provider, self.as_expr()
                    )

                new_ranges = dict(
                    (dim, range_) if dim != old_dim else (new_dim, new_range)
                    for dim, range_ in new_ranges.items()
                )
            else:
                raise AssertionError()
            return SymbolicDomain(self.grid_type, new_ranges)
        elif len(shift) > 2:
            return self.translate(shift[0:2], offset_provider, symbolic_domain_sizes).translate(
                shift[2:], offset_provider, symbolic_domain_sizes
            )
        else:
            raise AssertionError("Number of shifts must be a multiple of 2.")


def _reduce_ranges(
    *ranges: SymbolicRange,
    start_reduce_op: Callable[[itir.Expr, itir.Expr], itir.Expr],
    stop_reduce_op: Callable[[itir.Expr, itir.Expr], itir.Expr],
) -> SymbolicRange:
    """Uses start_op and stop_op to fold the start and stop of a list of ranges."""
    start = functools.reduce(
        lambda current_expr, el_expr: start_reduce_op(current_expr, el_expr),
        [range_.start for range_ in ranges],
    )
    stop = functools.reduce(
        lambda current_expr, el_expr: stop_reduce_op(current_expr, el_expr),
        [range_.stop for range_ in ranges],
    )
    # constant fold expression to keep the tree small
    start, stop = ConstantFolding.apply(start), ConstantFolding.apply(stop)  # type: ignore[assignment]  # always an itir.Expr
    return SymbolicRange(start, stop)


_range_union = functools.partial(
    _reduce_ranges, start_reduce_op=im.minimum, stop_reduce_op=im.maximum
)
_range_intersection = functools.partial(
    _reduce_ranges, start_reduce_op=im.maximum, stop_reduce_op=im.minimum
)


def _reduce_domains(
    *domains: SymbolicDomain,
    range_reduce_op: Callable[..., SymbolicRange],
) -> SymbolicDomain:
    """
    Applies range_op to the ranges of a list of domains with same dimensions and grid_type.
    """
    assert all(domain.grid_type == domains[0].grid_type for domain in domains)

    dims: list[common.Dimension] = [*domains[0].ranges.keys()]
    for domain in domains[1:]:
        for dim in domain.ranges.keys():
            if dim not in dims:
                dims.append(dim)

    promoted_domains = [promote_domain(domain, dims) for domain in domains]
    new_domain_ranges = {
        dim: range_reduce_op(*(domain.ranges[dim] for domain in promoted_domains)) for dim in dims
    }

    return SymbolicDomain(domains[0].grid_type, new_domain_ranges)


def _range_is_empty(range_: SymbolicRange) -> bool:
    """Return True if the range [start, stop) is empty or inverted after constant folding.

    Handles:
    - Empty ranges [k, k) where start == stop → zero width.
    - Inverted ranges [k', k) where start > stop → arise from intersections of disjoint
      domains (e.g. Kolor:[2,3) ∩ Kolor:[-inf,1) = Kolor:[2,1)) and are semantically empty.

    Handles both OffsetLiteral (int .value) and Literal (str .value) node types.
    InfinityLiteral.NEGATIVE/POSITIVE lack a numeric .value and are treated as non-empty.
    """
    start = ConstantFolding.apply(range_.start)
    stop = ConstantFolding.apply(range_.stop)
    try:
        return int(start.value) >= int(stop.value)  # empty (==) or inverted (>)
    except (AttributeError, ValueError, TypeError):
        return False  # InfinityLiteral or non-constant → assume non-empty


def domain_union(*domains: SymbolicDomain) -> SymbolicDomain:
    """Return the (set) union of a list of domains.

    Empty ranges [a, a) that arise from intersections with disjoint conditions (e.g.
    Kolor:[-inf,0) ∩ Kolor:[0,3) = Kolor:[0,0)) are excluded from the union. Without
    this, _domain_union([0,0), [1,3)) = [0,3) (wrong) instead of [1,3) (correct).
    The structured backend generates canonical concat_where conditions via
    canonicalize_domain_argument that produce such empty intersections for kolor branches
    that are out of range; including them widens E2C cell-field kolor domains to negative
    indices → CUDA_ERROR_ILLEGAL_ADDRESS on big grids.
    """
    assert all(domain.grid_type == domains[0].grid_type for domain in domains)

    dims: list[common.Dimension] = [*domains[0].ranges.keys()]
    for domain in domains[1:]:
        for dim in domain.ranges.keys():
            if dim not in dims:
                dims.append(dim)

    promoted_domains = [promote_domain(domain, dims) for domain in domains]
    new_domain_ranges: dict[common.Dimension, SymbolicRange] = {}
    for dim in dims:
        all_ranges = [domain.ranges[dim] for domain in promoted_domains]
        non_empty = [r for r in all_ranges if not _range_is_empty(r)]
        if non_empty:
            new_domain_ranges[dim] = _range_union(*non_empty)
        else:
            # All ranges are empty; keep first (caller should handle NEVER propagation)
            new_domain_ranges[dim] = all_ranges[0]

    return SymbolicDomain(domains[0].grid_type, new_domain_ranges)

domain_intersection = functools.partial(_reduce_domains, range_reduce_op=_range_intersection)
"""Return the intersection of a list of domains."""


def domain_complement(domain: SymbolicDomain) -> SymbolicDomain:
    """
    Return the (set) complement of a half-infinite domain.

    Note: after canonicalization of concat_where, the domain is always half-infinite,
    i.e. it has ranges of the form `]-inf, a[` or `[a, inf[`.
    """
    dims_dict = {}
    for dim in domain.ranges.keys():
        lb, ub = domain.ranges[dim].start, domain.ranges[dim].stop
        assert (lb == itir.InfinityLiteral.NEGATIVE) != (ub == itir.InfinityLiteral.POSITIVE)
        # `]-inf, a[` -> `[a, inf[`
        if lb == itir.InfinityLiteral.NEGATIVE:
            dims_dict[dim] = SymbolicRange(start=ub, stop=itir.InfinityLiteral.POSITIVE)
        # `[a, inf]` -> `]-inf, a]`
        else:  # ub == itir.InfinityLiteral.POSITIVE:
            dims_dict[dim] = SymbolicRange(start=itir.InfinityLiteral.NEGATIVE, stop=lb)
    return SymbolicDomain(domain.grid_type, dims_dict)


def promote_domain(
    domain: SymbolicDomain, target_dims: Iterable[common.Dimension]
) -> SymbolicDomain:
    """Return a domain that is extended with the dimensions of target_dims."""
    assert set(domain.ranges.keys()).issubset(target_dims)
    dims_dict = {
        dim: domain.ranges[dim]
        if dim in domain.ranges
        else SymbolicRange(itir.InfinityLiteral.NEGATIVE, itir.InfinityLiteral.POSITIVE)
        for dim in target_dims
    }
    return SymbolicDomain(domain.grid_type, dims_dict)


def is_finite(range_or_domain: SymbolicRange | SymbolicDomain) -> bool:
    """
    Return whether a range is unbounded in (at least) one direction.

    The expression is required to be constant folded before for the result to be reliable.
    """
    match range_or_domain:
        case SymbolicRange() as range_:
            infinity_literals = (itir.InfinityLiteral.POSITIVE, itir.InfinityLiteral.NEGATIVE)
            return not (range_.start in infinity_literals or range_.stop in infinity_literals)
        case SymbolicDomain() as domain:
            return all(is_finite(range_) for range_ in domain.ranges.values())
        case _:
            raise ValueError("Expected a 'SymbolicRange' or 'SymbolicDomain'.")
