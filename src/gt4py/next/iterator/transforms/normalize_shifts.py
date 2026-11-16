# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir


class NormalizeShifts(PreserveLocationVisitor, NodeTranslator):
    _IMPLICIT_OFF_AXIS_PREFIX = "_Off"

    @staticmethod
    def _is_zero_offset(expr: ir.Expr) -> bool:
        if isinstance(expr, ir.OffsetLiteral):
            return expr.value == 0
        if isinstance(expr, ir.Literal):
            return expr.value in {"0", "0.0"}
        return False

    @staticmethod
    def _is_implicit_off_axis(expr: ir.Expr) -> bool:
        if isinstance(expr, ir.CartesianOffset):
            # merge-test-main: Cartesian shifts are `CartesianOffset` nodes since upstream #2411
            return expr.domain == expr.codomain
        if isinstance(expr, ir.OffsetLiteral):
            return isinstance(expr.value, str) and expr.value.startswith(
                NormalizeShifts._IMPLICIT_OFF_AXIS_PREFIX
            )
        if isinstance(expr, ir.SymRef):
            return expr.id.startswith(NormalizeShifts._IMPLICIT_OFF_AXIS_PREFIX)
        return False

    @staticmethod
    def _strip_zero_shift_pairs(node: ir.FunCall) -> ir.FunCall | ir.Expr:
        if not (
            isinstance(node.fun, ir.FunCall)
            and isinstance(node.fun.fun, ir.SymRef)
            and node.fun.fun.id == "shift"
            and len(node.args) == 1
            and len(node.fun.args) % 2 == 0
        ):
            return node

        filtered_args: list[ir.Expr] = []
        shift_args = node.fun.args
        for idx in range(0, len(shift_args), 2):
            axis = shift_args[idx]
            offset = shift_args[idx + 1]
            if NormalizeShifts._is_implicit_off_axis(axis) and NormalizeShifts._is_zero_offset(
                offset
            ):
                continue
            filtered_args.extend([axis, offset])

        if not filtered_args:
            return node.args[0]
        if len(filtered_args) == len(shift_args):
            return node

        return ir.FunCall(
            fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=filtered_args),
            args=node.args,
        )

    def visit_FunCall(self, node: ir.FunCall):
        visited = self.generic_visit(node)

        if not isinstance(visited, ir.FunCall):
            return visited

        node_fc = visited

        stripped = self._strip_zero_shift_pairs(node_fc)
        if isinstance(stripped, ir.Expr) and not isinstance(stripped, ir.FunCall):
            return stripped
        if isinstance(stripped, ir.FunCall):
            node_fc = stripped

        if (
            isinstance(node_fc.fun, ir.FunCall)
            and isinstance(node_fc.fun.fun, ir.SymRef)
            and node_fc.fun.fun.id == "shift"
            and node_fc.args
            and isinstance(node_fc.args[0], ir.FunCall)
            and isinstance(node_fc.args[0].fun, ir.FunCall)
            and isinstance(node_fc.args[0].fun.fun, ir.SymRef)
            and node_fc.args[0].fun.fun.id == "shift"
        ):
            # shift(args1...)(shift(args2...)(it)) -> shift(args2..., args1...)(it)
            assert len(node_fc.args) == 1
            merged = ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="shift"), args=node_fc.args[0].fun.args + node_fc.fun.args
                ),
                args=node_fc.args[0].args,
            )
            stripped_merged = self._strip_zero_shift_pairs(merged)
            if isinstance(stripped_merged, ir.FunCall):
                merged = stripped_merged
            else:
                return stripped_merged
            return merged
        return node_fc
