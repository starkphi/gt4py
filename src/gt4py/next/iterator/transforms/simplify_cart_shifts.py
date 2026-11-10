# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from gt4py.eve import NodeTranslator
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


@dataclasses.dataclass
class SimplifyCartesianShifts(NodeTranslator):
    # Keep the inferred domains: FuseAsFieldOp (run right after this pass in the fusion loop) needs
    # `annex.domain` on make_tuple field elements. Without it fusion raised and was skipped entirely.
    PRESERVED_ANNEX_ATTRS = ("type", "domain")

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        # 1. Bottom-up traversal: visit children first
        visited_node = self.generic_visit(node, **kwargs)  # already a new node, no deepcopy needed

        if cpm.is_applied_shift(visited_node):
            shift_args = list(visited_node.fun.args)
            target_arg = visited_node.args[0]

            # 2. Collapse nested shifts: shift(B)(shift(A)(arg)) -> shift(A, B)(arg)
            while cpm.is_applied_shift(target_arg):
                shift_args.extend(target_arg.fun.args)
                target_arg = target_arg.args[0]

            accumulated_shifts: dict[str, int] = {}
            can_simplify = True

            # 3. Accumulate shifts per dimension
            for i in range(0, len(shift_args), 2):
                dim_node = shift_args[i]
                off_node = shift_args[i + 1]

                if (
                    isinstance(dim_node, ir.OffsetLiteral)
                    and isinstance(dim_node.value, str)
                    and isinstance(off_node, ir.OffsetLiteral)
                    and isinstance(off_node.value, int)
                ):
                    dim_name = dim_node.value
                    accumulated_shifts[dim_name] = (
                        accumulated_shifts.get(dim_name, 0) + off_node.value
                    )
                else:
                    can_simplify = False
                    break

            # 4. Reconstruct the simplified shift
            if can_simplify:
                new_shift_args = []
                # Sort to maintain a predictable order in the IR (e.g., IDim, JDim, Kolor)
                for dim_name in sorted(accumulated_shifts.keys()):
                    offset = accumulated_shifts[dim_name]
                    if offset != 0:
                        new_shift_args.append(ir.OffsetLiteral(value=dim_name))
                        new_shift_args.append(ir.OffsetLiteral(value=offset))

                if not new_shift_args:
                    # Shifts perfectly cancelled out! Just return the underlying field.
                    return target_arg

                return im.call(im.call("shift")(*new_shift_args))(target_arg)

        return visited_node
