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
import os
from pathlib import Path
from typing import Any, Final, Optional

import factory
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve import codegen
from gt4py.next import common
from gt4py.next.ffront import fbuiltins
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import pass_manager
from gt4py.next.otf import arguments, code_specs, definitions, stages, workflow
from gt4py.next.otf.binding import cpp_interface, interface
from gt4py.next.program_processors.codegens.gtfn.codegen import GTFNCodegen, GTFNIMCodegen
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir_to_gtfn_im_ir import GTFN_IM_lowering
from gt4py.next.program_processors.codegens.gtfn.itir_to_gtfn_ir import GTFN_lowering
from gt4py.next.type_system import type_specifications as ts, type_translation


GENERATED_CONNECTIVITY_PARAM_PREFIX = "gt_conn_"


def get_param_description(name: str, type_: Any) -> interface.Parameter:
    return interface.Parameter(name, type_)


@dataclasses.dataclass(frozen=True)
class GTFNTranslationStep(
    workflow.ReplaceEnabledWorkflowMixin[
        definitions.CompilableProgramDef,
        stages.ProgramSource[code_specs.HeaderAndSourceCodeSpec],
    ],
    workflow.ChainableWorkflowMixin[
        definitions.CompilableProgramDef,
        stages.ProgramSource[code_specs.HeaderAndSourceCodeSpec],
    ],
):
    code_spec: Optional[code_specs.HeaderAndSourceCodeSpec] = None
    # TODO replace by more general mechanism, see https://github.com/GridTools/gt4py/issues/1135
    enable_itir_transforms: bool = True
    use_imperative_backend: bool = False
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
    symbolic_domain_sizes: dict[str, Any] | None = None
    use_max_domain_range_on_unstructured_shift: bool | None = None

    def _default_code_spec(self) -> code_specs.HeaderAndSourceCodeSpec:
        match self.device_type:
            case core_defs.DeviceType.CUDA:
                return code_specs.CUDACodeSpec()
            case core_defs.DeviceType.ROCM:
                return code_specs.HIPCodeSpec()
            case core_defs.DeviceType.CPU:
                return code_specs.CPPCodeSpec()
            case _:
                raise self._not_implemented_for_device_type()

    def _process_regular_arguments(
        self,
        program: itir.Program,
        arg_types: tuple[ts.TypeSpec, ...],
        offset_provider_type: common.OffsetProviderType,
    ) -> tuple[list[interface.Parameter], list[str]]:
        parameters: list[interface.Parameter] = []
        arg_exprs: list[str] = []

        for arg_type, program_param in zip(arg_types, program.params, strict=True):
            # parameter
            parameter = get_param_description(program_param.id, arg_type)
            parameters.append(parameter)

            arg = f"std::forward<decltype({parameter.name})>({parameter.name})"

            if isinstance(parameter.type_, ts.FieldType):
                for dim in parameter.type_.dims:
                    if (
                        isinstance(
                            dim, fbuiltins.FieldOffset
                        )  # TODO(havogt): remove support for FieldOffset as Dimension
                        or dim.kind is common.DimensionKind.LOCAL
                    ):
                        # translate sparse dimensions to tuple dtype
                        dim_name = dim.value
                        connectivity = common.get_offset_type(offset_provider_type, dim_name)
                        assert isinstance(connectivity, common.NeighborConnectivityType)
                        size = connectivity.max_neighbors
                        arg = f"gridtools::sid::dimension_to_tuple_like<generated::{dim_name}_t, {size}>({arg})"
            arg_exprs.append(arg)
        return parameters, arg_exprs

    def _process_connectivity_args(
        self,
        offset_provider_type: common.OffsetProviderType,
        *,
        used_offset_tags: Optional[set[str]] = None,
    ) -> tuple[list[interface.Parameter], list[str]]:
        parameters: list[interface.Parameter] = []
        arg_exprs: list[str] = []

        for name, connectivity_type in offset_provider_type.items():
            if used_offset_tags is not None and name not in used_offset_tags:
                continue
            if isinstance(connectivity_type, common.NeighborConnectivityType):
                if connectivity_type.dtype.scalar_type not in [np.int32, np.int64]:
                    raise ValueError(
                        "Neighbor table indices must be of type 'np.int32' or 'np.int64'."
                    )

                # parameter
                parameters.append(
                    interface.Parameter(
                        name=GENERATED_CONNECTIVITY_PARAM_PREFIX + name.lower(),
                        type_=ts.FieldType(
                            dims=list(connectivity_type.domain),
                            dtype=type_translation.from_dtype(connectivity_type.dtype),
                        ),
                    )
                )

                # connectivity argument expression
                nbtbl = (
                    f"gridtools::fn::sid_neighbor_table::as_neighbor_table<"
                    f"generated::{connectivity_type.domain[0].value}_t, "
                    f"generated::{connectivity_type.domain[1].value}_t, "
                    f"{connectivity_type.max_neighbors}"
                    f">(std::forward<decltype({GENERATED_CONNECTIVITY_PARAM_PREFIX}{name.lower()})>({GENERATED_CONNECTIVITY_PARAM_PREFIX}{name.lower()}))"
                )
                arg_exprs.append(
                    f"gridtools::hymap::keys<generated::{name}_t>::make_values({nbtbl})"
                )
            elif isinstance(connectivity_type, common.Dimension):
                pass
            else:
                raise AssertionError(
                    f"Expected offset provider type '{name}' to be a 'NeighborConnectivityType' or 'Dimension', "
                    f"got '{type(connectivity_type).__name__}'."
                )

        return parameters, arg_exprs

    @staticmethod
    def _collect_used_offset_tags(program: itir.Program) -> set[str]:
        return (
            program.walk_values()
            .if_isinstance(itir.OffsetLiteral)
            .filter(lambda offset_literal: isinstance(offset_literal.value, str))
            .getattr("value")
            .to_set()
        )

    @staticmethod
    def _resolve_effective_arg_types(
        program: itir.Program, arg_types: tuple[ts.TypeSpec, ...]
    ) -> tuple[ts.TypeSpec, ...]:
        effective_arg_types: list[ts.TypeSpec] = []
        for arg_type, param in zip(arg_types, program.params, strict=True):
            effective_arg_types.append(param.type if param.type is not None else arg_type)
        return tuple(effective_arg_types)

    def _preprocess_program(
        self,
        program: itir.Program,
        offset_provider: common.OffsetProvider | common.OffsetProviderType,
        cartesian_reduce_axis_ranges: Optional[dict[common.Dimension, tuple[int, int]]] = None,
        symbolic_domain_sizes: Optional[dict[str, Any]] = None,
    ) -> itir.Program:
        if os.environ.get("USE_STRUCTURED_BACKEND", "0") != "1":
            # upstream path: the resolved static-arg dict must not reach domain inference as
            # `symbolic_domain_sizes`, a non-empty dict suppresses the dynamic-domain sizes
            # (`_process_symbolic_domains_option`) and static translate then hits symbolic ranges.
            effective_symbolic_domain_sizes = self.symbolic_domain_sizes
            cartesian_reduce_axis_ranges = None
        else:
            effective_symbolic_domain_sizes = (
                self.symbolic_domain_sizes if symbolic_domain_sizes is None else symbolic_domain_sizes
            )
            program = pass_manager.pre_inline_scalar_params(program, effective_symbolic_domain_sizes)
        apply_common_transforms = functools.partial(
            pass_manager.apply_common_transforms,
            extract_temporaries=True,
            offset_provider=offset_provider,
            symbolic_domain_sizes=effective_symbolic_domain_sizes,
            cartesian_reduce_axis_ranges=cartesian_reduce_axis_ranges,
            use_max_domain_range_on_unstructured_shift=self.use_max_domain_range_on_unstructured_shift,
        )

        new_program = apply_common_transforms(
            program, unroll_reduce=not self.use_imperative_backend
        )

        if self.use_imperative_backend and any(
            node.id == "neighbors"
            for node in new_program.pre_walk_values().if_isinstance(itir.SymRef)
        ):
            # if we don't unroll, there may be lifts left in the itir which can't be lowered to
            # gtfn. In this case, just retry with unrolled reductions.
            new_program = apply_common_transforms(program, unroll_reduce=True)

        return new_program

    def _resolve_symbolic_domain_sizes(
        self,
        program: itir.Program,
        argument_descriptor_contexts: arguments.ArgStaticDescriptorsContextsByType,
    ) -> Optional[dict[str, Any]]:
        resolved: dict[str, Any] = {}
        if self.symbolic_domain_sizes is not None:
            resolved.update(self.symbolic_domain_sizes)

        static_arg_descriptors = argument_descriptor_contexts.get(arguments.StaticArg)
        if static_arg_descriptors is not None:
            for name, descriptor in static_arg_descriptors.items():
                if not isinstance(descriptor, arguments.StaticArg):
                    continue
                if isinstance(descriptor.value, (int, np.integer)):
                    resolved[name] = int(descriptor.value)

            for name in (
                "max_i",
                "max_j",
                "domain_max_i",
                "domain_max_j",
                "nx",
                "ny",
                "edge_phase_size",
            ):
                descriptor = static_arg_descriptors.get(name)
                if not isinstance(descriptor, arguments.StaticArg):
                    continue
                if isinstance(descriptor.value, (int, np.integer)):
                    resolved[name] = int(descriptor.value)

        if not any(
            key in resolved
            for key in (
                "i_min",
                "i_max",
                "j_min",
                "j_max",
                "max_i",
                "max_j",
                "domain_max_i",
                "domain_max_j",
                "nx",
                "ny",
            )
        ):
            resolved.update(self._resolve_symbolic_domain_sizes_from_mesh_metadata())

        return resolved or None

    @staticmethod
    def _env_flag_enabled(*names: str) -> bool:
        for name in names:
            raw = os.environ.get(name)
            if raw is None:
                continue
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return False

    @staticmethod
    def _program_uses_edge_horizontal_unstructured_axis(program: itir.Program) -> bool:
        for fun_call in program.pre_walk_values().if_isinstance(itir.FunCall):
            if not (
                isinstance(fun_call.fun, itir.SymRef)
                and fun_call.fun.id == "named_range"
                and len(fun_call.args) == 3
            ):
                continue
            axis_expr = fun_call.args[0]
            axis_name = getattr(axis_expr, "value", None)
            if axis_name == "Edge":
                return True
        return False

    def _route_edge_lateral_symbolic_sizes(
        self,
        program: itir.Program,
        resolved: dict[str, Any],
    ) -> None:
        use_edge_lateral = self._env_flag_enabled(
            "GT4PY_TRANSLATOR_EDGE_LATERAL",
            "GT4PY_TRANSLATOR_USE_EDGE_LATERAL",
        )
        if not use_edge_lateral:
            return
        if not self._program_uses_edge_horizontal_unstructured_axis(program):
            return

        lateral_edge = resolved.get("lateral_edge")
        if lateral_edge is None:
            lateral_edge = resolved.get("lateral")
        if lateral_edge is None:
            return

        resolved["lateral_edge"] = lateral_edge

        if isinstance(lateral_edge, (int, np.integer)):
            # Preserve current domain-bound behavior: 10->interior, 9->nud2->4.5, 8->nud->4, 7->lat8->3.5, 6->lat7->3, 5->lat6->2.5, 4->lat5->2, 3->lat4->1.5, 2->lat3->1, 1->lat2->0.5, 0->lat->0
            reduced_lateral = (int(lateral_edge)) // 2
            resolved.setdefault("lateral_bounds", reduced_lateral)
            # Compatibility: some transforms still read `lateral` directly.
            resolved["lateral"] = int(resolved["lateral_bounds"])

    @staticmethod
    def _resolve_symbolic_domain_sizes_from_mesh_metadata() -> dict[str, Any]:
        # Only inject structured mapping metadata when the structured backend is active.
        # When USE_STRUCTURED_BACKEND=0 the cart_unroll pass is not applied and these
        # sizes are unused, but they change the cache hash — causing unstructured runs to
        # pick up cached structured (4-D) compilations.
        if os.environ.get("USE_STRUCTURED_BACKEND", "0") != "1":
            return {}
        mesh_path = os.environ.get("GT4PY_TRANSLATOR_MESH")
        lateral_env = os.environ.get("GT4PY_TRANSLATOR_LATERAL")
        if lateral_env is None:
            lateral_env = os.environ.get("LATERAL_BOUNDARY_LEVEL", "0")
        try:
            lateral = int(lateral_env)
        except ValueError:
            lateral = 0
        lateral = max(0, lateral)

        if not mesh_path:
            repo_root = Path(__file__).resolve().parents[6]
            candidate_paths = (
                repo_root / "../grid-generator/parallelogram_grid.nc",
                repo_root / "grid-generator/parallelogram_grid.nc",
                Path.cwd() / "grid-generator/parallelogram_grid.nc",
                Path.cwd() / "parallelogram_grid.nc",
            )
            for candidate in candidate_paths:
                resolved_candidate = candidate.resolve()
                if resolved_candidate.is_file():
                    mesh_path = str(resolved_candidate)
                    break

        if not mesh_path:
            return {}

        try:
            import gt4py.next.modules.translator as translator  # type: ignore[import-not-found]

            sizes = translator.load_structured_remap_sizes_from_netcdf(mesh_path, lateral=lateral)
        except Exception:
            return {}

        result: dict[str, Any] = {
            "i_min": 0,
            "j_min": 0,
            "i_max": int(sizes.max_i),
            "j_max": int(sizes.max_j),
            "max_i": int(sizes.max_i),
            "max_j": int(sizes.max_j),
            "domain_max_i": int(sizes.max_i),
            "domain_max_j": int(sizes.max_j),
            "nx": int(sizes.nx),
            "ny": int(sizes.ny),
            "mesh_path": str(Path(mesh_path).expanduser().resolve()),
        }

        # Also inject the edge_to_ijk mapping so all stencils use the horizontal_start
        # mapping path instead of the legacy lateral path. This ensures that even utility
        # stencils (geometry, math helpers) compiled via the standard GTFN backend go
        # through the structured pass correctly when USE_STRUCTURED_BACKEND=1.
        try:
            from gt4py.next.modules.cartesian_interceptor import get_global_grid_mapping
            index_map, _ = get_global_grid_mapping()
            if index_map is not None:
                import numpy as np
                edge_to_ijk = getattr(index_map, "edge_to_ijk", None)
                if edge_to_ijk is not None:
                    result["use_horizontal_start_mapping"] = True
                    result["edge_to_ijk"] = [
                        (int(i), int(j), int(k)) for i, j, k in np.asarray(edge_to_ijk)
                    ]
                vertex_to_ij = getattr(index_map, "vertex_to_ij", None)
                if vertex_to_ij is not None:
                    result["vertex_to_ij"] = [
                        (int(i), int(j)) for i, j in np.asarray(vertex_to_ij)
                    ]
        except Exception:
            pass

        return result

    @staticmethod
    def _infer_kolor_extent_from_program(program: itir.Program) -> Optional[int]:
        kolor_offset = common.dimension_to_implicit_offset("Kolor")
        kolor_shift_indices: list[int] = []

        for node in program.pre_walk_values().if_isinstance(itir.FunCall):
            if not (
                isinstance(node.fun, itir.SymRef) and node.fun.id == "shift" and len(node.args) == 2
            ):
                continue

            axis_offset, index_offset = node.args
            if not (
                isinstance(axis_offset, itir.OffsetLiteral)
                and axis_offset.value == kolor_offset
                and isinstance(index_offset, itir.OffsetLiteral)
                and isinstance(index_offset.value, int)
            ):
                continue

            kolor_shift_indices.append(index_offset.value)

        if not kolor_shift_indices:
            return None

        max_backward_offset = max((-idx for idx in kolor_shift_indices if idx < 0), default=0)
        inferred_extent = max_backward_offset + 1
        return inferred_extent if inferred_extent >= 2 else None

    @staticmethod
    def _resolve_cartesian_reduce_axis_ranges(
        program: itir.Program,
        argument_descriptor_contexts: arguments.ArgStaticDescriptorsContextsByType,
    ) -> Optional[dict[common.Dimension, tuple[int, int]]]:
        inferred_kolor_extent = GTFNTranslationStep._infer_kolor_extent_from_program(program)

        static_arg_descriptors = argument_descriptor_contexts.get(arguments.StaticArg)
        if static_arg_descriptors is None:
            if inferred_kolor_extent is not None:
                return {common.Dimension("Kolor"): (0, inferred_kolor_extent)}
            return None

        domain_max_kolor = static_arg_descriptors.get("domain_max_kolor")
        if not isinstance(domain_max_kolor, arguments.StaticArg):
            if inferred_kolor_extent is not None:
                return {common.Dimension("Kolor"): (0, inferred_kolor_extent)}
            return None
        if not isinstance(domain_max_kolor.value, (int, np.integer)):
            if inferred_kolor_extent is not None:
                return {common.Dimension("Kolor"): (0, inferred_kolor_extent)}
            return None

        domain_max_kolor_value = int(domain_max_kolor.value)
        if domain_max_kolor_value >= 2:
            return {common.Dimension("Kolor"): (0, domain_max_kolor_value)}

        if inferred_kolor_extent is not None:
            return {common.Dimension("Kolor"): (0, inferred_kolor_extent)}

        if domain_max_kolor_value == 1:
            # In many reduction kernels this static arg reflects the reduced output extent.
            # Fall back to a 2-color reduction when no stronger signal is available.
            return {common.Dimension("Kolor"): (0, 2)}

        if domain_max_kolor_value <= 0:
            return None
        return None

    def generate_stencil_source(
        self,
        program: itir.Program,
        offset_provider: common.OffsetProvider | common.OffsetProviderType,
        column_axis: Optional[common.Dimension],
        cartesian_reduce_axis_ranges: Optional[dict[common.Dimension, tuple[int, int]]] = None,
        *,
        already_preprocessed: bool = False,
    ) -> str:
        if already_preprocessed:
            new_program = program
        elif self.enable_itir_transforms:
            resolved_symbolic_domain_sizes = self._resolve_symbolic_domain_sizes(program, {})
            new_program = self._preprocess_program(
                program,
                offset_provider,
                cartesian_reduce_axis_ranges=cartesian_reduce_axis_ranges,
                symbolic_domain_sizes=resolved_symbolic_domain_sizes,
            )
        else:
            assert isinstance(program, itir.Program)
            new_program = program

        gtfn_ir = GTFN_lowering.apply(
            new_program,
            offset_provider_type=common.offset_provider_to_type(offset_provider),
            column_axis=column_axis,
        )

        if self.use_imperative_backend:
            gtfn_im_ir = GTFN_IM_lowering().visit(node=gtfn_ir)
            generated_code = GTFNIMCodegen.apply(gtfn_im_ir)
        else:
            generated_code = GTFNCodegen.apply(gtfn_ir)

        return codegen.format_source("cpp", generated_code, style="LLVM")

    def __call__(
        self, inp: definitions.CompilableProgramDef
    ) -> stages.ProgramSource[code_specs.HeaderAndSourceCodeSpec]:
        """Generate GTFN C++ code from the ITIR definition."""
        program: itir.Program = inp.data
        arg_types = inp.args.args
        cartesian_reduce_axis_ranges = self._resolve_cartesian_reduce_axis_ranges(
            program, inp.args.argument_descriptor_contexts
        )
        resolved_symbolic_domain_sizes = self._resolve_symbolic_domain_sizes(
            program, inp.args.argument_descriptor_contexts
        )

        if self.enable_itir_transforms:
            transformed_program = self._preprocess_program(
                program,
                inp.args.offset_provider,
                cartesian_reduce_axis_ranges=cartesian_reduce_axis_ranges,
                symbolic_domain_sizes=resolved_symbolic_domain_sizes,
            )
        else:
            transformed_program = program

        used_offset_tags = self._collect_used_offset_tags(transformed_program)
        effective_arg_types = self._resolve_effective_arg_types(transformed_program, arg_types)

        # handle regular parameters and arguments of the program (i.e. what the user defined in
        #  the program)
        regular_parameters, regular_args_expr = self._process_regular_arguments(
            transformed_program, effective_arg_types, inp.args.offset_provider_type
        )

        # handle connectivity parameters and arguments (i.e. what the user provided in the offset
        #  provider)
        connectivity_parameters, connectivity_args_expr = self._process_connectivity_args(
            inp.args.offset_provider_type,
            used_offset_tags=used_offset_tags,
        )

        # combine into a format that is aligned with what the backend expects
        parameters: list[interface.Parameter] = regular_parameters + connectivity_parameters
        backend_arg = self._backend_type()
        args_expr: list[str] = [backend_arg, *regular_args_expr]

        function = interface.Function(program.id, tuple(parameters))
        decl_body = (
            f"return generated::{function.name}("
            f"{', '.join(connectivity_args_expr)})({', '.join(args_expr)});"
        )
        decl_src = cpp_interface.render_function_declaration(function, body=decl_body)
        stencil_src = self.generate_stencil_source(
            transformed_program,
            inp.args.offset_provider,
            inp.args.column_axis,
            cartesian_reduce_axis_ranges=cartesian_reduce_axis_ranges,
            already_preprocessed=True,
        )
        source_code = interface.format_source(
            self._code_spec(),
            f"""
                    #include <{self._backend_header()}>
                    #include <gridtools/sid/dimension_to_tuple_like.hpp>
                    {stencil_src}
                    {decl_src}
                    """.strip(),
        )

        module: stages.ProgramSource[code_specs.HeaderAndSourceCodeSpec] = stages.ProgramSource(
            entry_point=function,
            library_deps=(interface.LibraryDependency(self._library_name(), "master"),),
            source_code=source_code,
            code_spec=self._code_spec(),
        )
        return module

    def _backend_header(self) -> str:
        match self.device_type:
            case core_defs.DeviceType.CUDA | core_defs.DeviceType.ROCM:
                return "gridtools/fn/backend/gpu.hpp"
            case core_defs.DeviceType.CPU:
                return "gridtools/fn/backend/naive.hpp"
            case _:
                raise self._not_implemented_for_device_type()

    def _backend_type(self) -> str:
        match self.device_type:
            case core_defs.DeviceType.CUDA | core_defs.DeviceType.ROCM:
                return "gridtools::fn::backend::gpu<generated::block_sizes_t>{}"
            case core_defs.DeviceType.CPU:
                return "gridtools::fn::backend::naive{}"
            case _:
                raise self._not_implemented_for_device_type()

    def _code_spec(self) -> code_specs.HeaderAndSourceCodeSpec:
        return self.code_spec if self.code_spec is not None else self._default_code_spec()

    def _library_name(self) -> str:
        match self.device_type:
            case core_defs.DeviceType.CUDA | core_defs.DeviceType.ROCM:
                return "gridtools_gpu"
            case core_defs.DeviceType.CPU:
                return "gridtools_cpu"
            case _:
                raise self._not_implemented_for_device_type()

    def _not_implemented_for_device_type(self) -> NotImplementedError:
        return NotImplementedError(
            f"{self.__class__.__name__} is not implemented for device type {self.device_type.name}"
        )


class GTFNTranslationStepFactory(factory.Factory[GTFNTranslationStep]):
    class Meta:
        model = GTFNTranslationStep


translate_program_cpu: Final[definitions.TranslationStep] = GTFNTranslationStepFactory()  # type: ignore[assignment] # factory-boy typing not precise enough

translate_program_gpu: Final[definitions.TranslationStep] = GTFNTranslationStepFactory(  # type: ignore[assignment] # factory-boy typing not precise enough
    device_type=core_defs.DeviceType.CUDA
)
