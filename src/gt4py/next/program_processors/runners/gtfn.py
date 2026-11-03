# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import re
from typing import Any

import factory
import numpy as np

import gt4py._core.definitions as core_defs
import gt4py.next.custom_layout_allocators as next_allocators
from gt4py._core import filecache
from gt4py.next import backend, common, config, field_utils
from gt4py.next.embedded import nd_array_field
from gt4py.next.instrumentation import metrics
from gt4py.next.otf import recipes, stages, workflow
from gt4py.next.otf.binding import nanobind
from gt4py.next.otf.compilation import compiler
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
    inp: stages.ExecutableProgram, device: core_defs.DeviceType = core_defs.DeviceType.CPU
) -> stages.ExecutableProgram:
    def decorated_program(
        *args: Any,
        offset_provider: dict[str, common.Connectivity | common.Dimension],
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


class GTFNCompileWorkflowFactory(factory.Factory):
    class Meta:
        model = recipes.OTFCompileWorkflow

    class Params:
        device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
        cmake_build_type: config.CMakeBuildType = factory.LazyFunction(  # type: ignore[assignment] # factory-boy typing not precise enough
            lambda: config.CMAKE_BUILD_TYPE
        )
        builder_factory: compiler.BuildSystemProjectGenerator = factory.LazyAttribute(  # type: ignore[assignment] # factory-boy typing not precise enough
            lambda o: compiledb.CompiledbFactory(cmake_build_type=o.cmake_build_type)
        )

        cached_translation = factory.Trait(
            translation=factory.LazyAttribute(
                lambda o: workflow.CachedStep(
                    o.bare_translation,
                    hash_function=stages.fingerprint_compilable_program,
                    cache=filecache.FileCache(str(config.BUILD_CACHE_DIR / "gtfn_cache")),
                )
            ),
        )

        bare_translation = factory.SubFactory(
            gtfn_module.GTFNTranslationStepFactory,
            device_type=factory.SelfAttribute("..device_type"),
        )

    translation = factory.LazyAttribute(lambda o: o.bare_translation)

    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableProject] = (
        nanobind.bind_source
    )
    compilation = factory.SubFactory(
        compiler.CompilerFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        builder_factory=factory.SelfAttribute("..builder_factory"),
    )
    decoration = factory.LazyAttribute(
        lambda o: functools.partial(convert_args, device=o.device_type)
    )


class GTFNBackendFactory(factory.Factory):
    class Meta:
        model = backend.Backend

    class Params:
        name_device = "cpu"
        name_cached = ""
        name_temps = ""
        name_postfix = ""
        gpu = factory.Trait(
            allocator=next_allocators.StandardGPUFieldBufferAllocator(),
            device_type=core_defs.CUPY_DEVICE_TYPE or core_defs.DeviceType.CUDA,
            name_device="gpu",
        )
        cached = factory.Trait(
            executor=factory.LazyAttribute(
                lambda o: workflow.CachedStep(o.otf_workflow, hash_function=o.hash_function)
            ),
            name_cached="_cached",
        )
        device_type = core_defs.DeviceType.CPU
        hash_function = stages.compilation_hash
        otf_workflow = factory.SubFactory(
            GTFNCompileWorkflowFactory, device_type=factory.SelfAttribute("..device_type")
        )

    name = factory.LazyAttribute(
        lambda o: f"run_gtfn_{o.name_device}{o.name_temps}{o.name_cached}{o.name_postfix}"
    )

    executor = factory.LazyAttribute(lambda o: o.otf_workflow)
    allocator = next_allocators.StandardCPUFieldBufferAllocator()
    transforms = backend.DEFAULT_TRANSFORMS


run_gtfn = GTFNBackendFactory()

run_gtfn_imperative = GTFNBackendFactory(
    name_postfix="_imperative", otf_workflow__translation__use_imperative_backend=True
)

run_gtfn_cached = GTFNBackendFactory(cached=True, otf_workflow__cached_translation=True)

run_gtfn_gpu = GTFNBackendFactory(gpu=True)

run_gtfn_gpu_cached = GTFNBackendFactory(
    gpu=True, cached=True, otf_workflow__cached_translation=True
)

run_gtfn_no_transforms = GTFNBackendFactory(
    otf_workflow__bare_translation__enable_itir_transforms=False
)
