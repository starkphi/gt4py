# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import re
from typing import Any
import pathlib
from collections.abc import Callable
from typing import TypeAlias

import numpy as np

import gt4py._core.definitions as core_defs
from gt4py.next import backend, common, field_utils
from gt4py.next.embedded import nd_array_field
from gt4py.next.instrumentation import metrics
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import artifacts, recipes, stages, workflow
from gt4py.next.otf.binding import nanobind
from gt4py.next.otf.compilation import cache, compiler
from gt4py.next.otf.compilation.build_systems import compiledb
from gt4py.next.program_processors.codegens.gtfn import gtfn_module


def convert_arg(arg: Any) -> Any:
    # Note: this function is on the hot path and needs to have minimal overhead.
    if (origin := getattr(arg, "__gt_origin__", None)) is not None:
        # `Field` is the most likely case, we use `__gt_origin__` as the property is needed anyway
        # and (currently) uniquely identifies a `NDArrayField` (which is the only supported `Field`)
        assert isinstance(arg, nd_array_field.NdArrayField)
        return arg.ndarray, origin
    if isinstance(arg, tuple):
        return tuple(convert_arg(a) for a in arg)
    if isinstance(arg, np.integer):
        return int(arg)
    if isinstance(arg, np.bool_):
        # nanobind does not support implicit conversion of `np.bool` to `bool`
        return bool(arg)
    # TODO(havogt): if this function still appears in profiles,
    # we should avoid going through the previous isinstance checks for detecting a scalar.
    # E.g. functools.cache on the arg type, returning a function that does the conversion
    return arg


def convert_args(
    inp: artifacts.ExecutableProgram, device: core_defs.DeviceType = core_defs.DeviceType.CPU
) -> artifacts.ExecutableProgram:
    def decorated_program(
        *args: Any,
        offset_provider: dict[str, common.OffsetProviderElem],
        out: Any = None,
    ) -> None:
        # Note: this function is on the hot path and needs to have minimal overhead.
        if out is not None:
            args = (*args, out)
        converted_args = tuple(convert_arg(arg) for arg in args)
        conn_args = extract_connectivity_args(offset_provider, device, inp)

        opt_kwargs: dict[str, Any] = {}
        if collect_metrics := metrics.is_level_enabled(metrics.PERFORMANCE):
            # If we are collecting metrics, we need to add the `exec_info` argument
            # to the `inp` call, which will be used to collect performance metrics.
            exec_info: dict[str, float] = {}
            opt_kwargs["exec_info"] = exec_info

        # generate implicit domain size arguments only if necessary, using `iter_size_args()`
        try:
            inp(
                *converted_args,
                *conn_args,
                **opt_kwargs,
            )
        except TypeError as error:
            if conn_args and "incompatible function arguments" in str(error):
                inp(
                    *converted_args,
                    **opt_kwargs,
                )
            else:
                raise

        if collect_metrics:
            metrics.add_sample_to_current_source(
                metrics.COMPUTE_METRIC, exec_info["run_cpp_duration"]
            )

    return decorated_program


def extract_connectivity_args(
    offset_provider: dict[str, common.Connectivity | common.Dimension],
    device: core_defs.DeviceType,
    inp: stages.ExecutableProgram | None = None,
) -> list[tuple[core_defs.NDArrayObject, tuple[int, ...]]]:
    # Note: this function is on the hot path and needs to have minimal overhead.
    zero_origin = (0, 0)
    connectivities_by_name = {
        name.lower(): conn for name, conn in offset_provider.items() if hasattr(conn, "ndarray")
    }
    exact = _connectivity_names_from_binding_doc(inp) if inp is not None else None
    if exact is not None:
        # The binding states exactly which tables it takes (possibly none): no guessing.
        return [(connectivities_by_name[name].ndarray, zero_origin) for name in exact]
    required_connectivities: tuple[str, ...] = _extract_required_connectivity_names(inp)

    if required_connectivities:
        selected_connectivities = [
            connectivities_by_name[name]
            for name in required_connectivities
            if name in connectivities_by_name
        ]
    else:
        selected_connectivities = [
            conn for conn in offset_provider.values() if hasattr(conn, "ndarray")
        ]

    assert all(
        hasattr(conn, "ndarray") or isinstance(conn, common.Dimension)
        for conn in offset_provider.values()
    )
    assert all(
        common.is_neighbor_table(conn) and field_utils.verify_device_field_type(conn, device)
        for conn in selected_connectivities
    )
    args: list[tuple[core_defs.NDArrayObject, tuple[int, ...]]] = [
        (conn.ndarray, zero_origin) for conn in selected_connectivities
    ]

    return args


@functools.cache
def _extract_required_connectivity_names_from_doc(doc: str) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\bgt_conn_([a-zA-Z0-9_]+)\b", doc):
        name = match.group(1).lower()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


_CONNECTIVITY_DOC_MARKER = "gt4py_connectivities:"


def _connectivity_names_from_binding_doc(inp: stages.ExecutableProgram) -> tuple[str, ...] | None:
    """Exact neighbor-table list written into the binding docstring by `nanobind.create_bindings`
    (may be empty). None if the binding predates the marker."""
    for candidate in (inp, *(getattr(inp, a, None) for a in ("func", "__wrapped__"))):
        doc = getattr(candidate, "__doc__", None) or ""
        i = doc.find(_CONNECTIVITY_DOC_MARKER)
        if i >= 0:
            line = doc[i + len(_CONNECTIVITY_DOC_MARKER) :].split("\n", 1)[0]
            return _extract_required_connectivity_names_from_doc(line)
    return None


def _extract_required_connectivity_names(inp: stages.ExecutableProgram | None) -> tuple[str, ...]:
    if inp is None:
        return ()

    candidates: list[Any] = [inp]
    for accessor in ("func", "__wrapped__", "__func__"):
        candidate = getattr(inp, accessor, None)
        if candidate is not None:
            candidates.append(candidate)

    seen_candidate_ids: set[int] = set()
    for candidate in candidates:
        candidate_id = id(candidate)
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)

        for attr_name in ("__doc__", "__text_signature__"):
            text = getattr(candidate, attr_name, None)
            if text:
                names = _extract_required_connectivity_names_from_doc(text)
                if names:
                    return names

        repr_text = repr(candidate)
        names = _extract_required_connectivity_names_from_doc(repr_text)
        if names:
            return names

        str_text = str(candidate)
        names = _extract_required_connectivity_names_from_doc(str_text)
        if names:
            return names

    function_name = getattr(inp, "__name__", "")
    if function_name:
        return _extract_required_connectivity_names_from_cache(function_name)

    return ()


@functools.cache
def _extract_required_connectivity_names_from_cache(function_name: str) -> tuple[str, ...]:
    cache_dir = config.BUILD_CACHE_DIR / "gtfn_cache"
    if not cache_dir.exists():
        return ()

    function_name_bytes = function_name.encode()
    for cache_file in sorted(cache_dir.glob("*.pkl"), reverse=True):
        try:
            data = cache_file.read_bytes()
        except OSError:
            continue

        if function_name_bytes not in data:
            continue

        decoded = data.decode("latin1", errors="ignore")
        function_decl_match = re.search(
            rf"decltype\(auto\)\s+{re.escape(function_name)}\((.*?)\)\s*\{{",
            decoded,
            flags=re.DOTALL,
        )
        if function_decl_match is not None:
            names = _extract_required_connectivity_names_from_doc(function_decl_match.group(1))
            if names:
                return names

        names = _extract_required_connectivity_names_from_doc(decoded)
        if names:
            return names

    return ()


@dataclasses.dataclass(frozen=True)
class GTFNCompilationArtifact(compiler.CPPCompilationArtifact):
    def load(self) -> artifacts.ExecutableProgram:
        return convert_args(super().load(), device=self.device_type)


@dataclasses.dataclass(frozen=True)
class GTFNCompiler(compiler.CPPCompiler):
    def _make_artifact(
        self, src_dir: pathlib.Path, module: pathlib.Path, entry_point_name: str
    ) -> GTFNCompilationArtifact:
        return GTFNCompilationArtifact(
            src_dir=src_dir,
            module=module,
            entry_point_name=entry_point_name,
            device_type=self.device_type,
        )


@dataclasses.dataclass(frozen=True)
class GTFNConfig(backend.ToolchainConfig):
    """Settings shared by the steps of a GTFN toolchain, see `ToolchainConfig`."""


#: Step builders: callables creating a step from the toolchain configuration.
GTFNTranslationBuilder: TypeAlias = Callable[[GTFNConfig], stages.TranslationStep]
GTFNBindingsBuilder: TypeAlias = Callable[
    [GTFNConfig], workflow.Workflow[artifacts.ProgramSource, artifacts.ExtensionSource]
]
GTFNCompilationBuilder: TypeAlias = Callable[
    [GTFNConfig], workflow.Workflow[artifacts.ExtensionSource, artifacts.CompilationArtifact]
]


def make_gtfn_translation(
    cfg: GTFNConfig,
    /,
    *,
    enable_itir_transforms: bool = True,
    symbolic_domain_sizes: dict[str, itir.Expr] | None = None,
    use_max_domain_range_on_unstructured_shift: bool | None = None,
) -> gtfn_module.GTFNTranslationStep:
    """
    Build the GTIR -> C++ translation step.

    Args:
        cfg: The toolchain configuration.
        enable_itir_transforms: Run the GTIR transformation passes before code
            generation.
        symbolic_domain_sizes: Symbolic sizes of the domains, by dimension name.
        use_max_domain_range_on_unstructured_shift: See `GTFNTranslationStep`.

    Returns:
        The translation step, targeting `cfg.device_type`.
    """
    return gtfn_module.GTFNTranslationStep(
        device_type=cfg.device_type,
        enable_itir_transforms=enable_itir_transforms,
        symbolic_domain_sizes=symbolic_domain_sizes,
        use_max_domain_range_on_unstructured_shift=use_max_domain_range_on_unstructured_shift,
    )


def make_gtfn_bindings(
    cfg: GTFNConfig, /
) -> workflow.Workflow[artifacts.ProgramSource, artifacts.ExtensionSource]:
    """Build the step generating the nanobind bindings of the translated program."""
    # `OTFCompileWorkflow` is not parameterized over the code spec, so its
    # `bindings` field is typed for `ProgramSource[Any]` while
    # `ExtensionGenerator` accepts only C++-like specs. Parameterizing the
    # pipeline is the real fix and belongs with the pipeline rework.
    return nanobind.ExtensionGenerator(  # type: ignore[return-value] # see comment above
        unstructured_horizontal_has_unit_stride=cfg.unstructured_horizontal_has_unit_stride
    )


def make_gtfn_build_system(
    cfg: GTFNConfig,
    /,
    *,
    cmake_extra_flags: list[str] | None = None,
    renew_compiledb: bool = False,
) -> compiledb.CompiledbFactory:
    """
    Build the build-system project generator used by the GTFN compiler.

    Args:
        cfg: The toolchain configuration.
        cmake_extra_flags: Extra flags passed to CMake.
        renew_compiledb: Regenerate the compilation database even if one exists.

    Returns:
        A `CompiledbFactory` using `cfg.cmake_build_type`.
    """
    return compiledb.CompiledbFactory(
        cmake_build_type=cfg.cmake_build_type,
        cmake_extra_flags=cmake_extra_flags or [],
        renew_compiledb=renew_compiledb,
    )


def make_gtfn_compiler(
    cfg: GTFNConfig,
    /,
    *,
    build_system: Callable[[GTFNConfig], compiler.BuildSystemProjectGenerator] = (
        make_gtfn_build_system
    ),
    force_recompile: bool = False,
) -> GTFNCompiler:
    """
    Build the compilation step.

    Args:
        cfg: The toolchain configuration.
        build_system: Builder of the build-system project generator.
        force_recompile: Recompile even if a cached build exists.

    Returns:
        The compilation step, targeting `cfg.device_type`.
    """
    return GTFNCompiler(
        cache_lifetime=cfg.cache_lifetime,
        builder_factory=build_system(cfg),
        device_type=cfg.device_type,
        force_recompile=force_recompile,
    )


def make_gtfn_compile_workflow(
    cfg: GTFNConfig | None = None,
    /,
    *,
    translation: GTFNTranslationBuilder = make_gtfn_translation,
    bindings: GTFNBindingsBuilder = make_gtfn_bindings,
    compilation: GTFNCompilationBuilder = make_gtfn_compiler,
) -> recipes.OTFCompileWorkflow:
    """
    Build the GTFN translation -> bindings -> compilation workflow.

    Every step is created by a step builder that receives `cfg`, so all steps
    agree on the settings in it. To customize a step, pass a different
    builder: a `functools.partial` of the default one to change a
    step-local setting, e.g.
    `translation=functools.partial(make_gtfn_translation, enable_itir_transforms=False)`,
    or any callable taking the config to replace the step. The translation
    step is wrapped in the cache here, after its builder ran, so a custom
    translation step is cached like the default one. A custom step builder is
    responsible for configuring its step from `cfg`.

    Args:
        cfg: The toolchain configuration. Defaults to `GTFNConfig()`.
        translation: Builder of the translation step.
        bindings: Builder of the bindings step.
        compilation: Builder of the compilation step.

    Returns:
        The composed compile workflow.
    """
    if cfg is None:
        cfg = GTFNConfig()

    translation_step = translation(cfg)
    compilation_step = compilation(cfg)

    if cfg.cached_translation:
        translation_step = cache.persistent_translation_cache(
            translation_step, "gtfn", cfg.cache_lifetime
        )

    return recipes.OTFCompileWorkflow(
        translation=translation_step, bindings=bindings(cfg), compilation=compilation_step
    )


def make_gtfn_toolchain(
    cfg: GTFNConfig | None = None,
    /,
    *,
    name_postfix: str = "",
    translation: GTFNTranslationBuilder = make_gtfn_translation,
    bindings: GTFNBindingsBuilder = make_gtfn_bindings,
    compilation: GTFNCompilationBuilder = make_gtfn_compiler,
) -> backend.Backend:
    """
    Build a GTFN toolchain.

    Args:
        cfg: The toolchain configuration. Defaults to `GTFNConfig()`.
        name_postfix: Appended to the toolchain name, which must stay unique.
        translation: Builder of the translation step, see
            `make_gtfn_compile_workflow`.
        bindings: Builder of the bindings step.
        compilation: Builder of the compilation step.

    Returns:
        The configured toolchain.
    """
    if cfg is None:
        cfg = GTFNConfig()

    return backend.Backend(
        name=f"run_gtfn_{cfg.device_name}{name_postfix}",
        executor=make_gtfn_compile_workflow(
            cfg, translation=translation, bindings=bindings, compilation=compilation
        ),
        allocator=cfg.make_allocator(),
        transforms=backend.DEFAULT_TRANSFORMS,
    )


run_gtfn = make_gtfn_toolchain()

run_gtfn_gpu = make_gtfn_toolchain(GTFNConfig(gpu=True))

run_gtfn_no_transforms = make_gtfn_toolchain(
    name_postfix="_no_transforms",
    translation=functools.partial(make_gtfn_translation, enable_itir_transforms=False),
)
