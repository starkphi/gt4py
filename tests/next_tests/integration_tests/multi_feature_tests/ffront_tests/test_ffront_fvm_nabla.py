# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple
import os

import numpy as np
import pytest
from packaging import version

from gt4py import next as gtx
from gt4py.next import neighbor_sum
from gt4py.next.program_processors.program_setup_utils import setup_program
from gt4py.next.program_processors.runners import gtfn as gtfn_runner


from next_tests import definitions as test_definitions
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
)
from next_tests.integration_tests.multi_feature_tests.fvm_nabla_setup import (
    E2V,
    V2E,
    E2VDim,
    Edge,
    V2EDim,
    Vertex,
    assert_close,
    nabla_setup,
)
from gt4py.next.modules.translator import load_structured_remap_sizes_from_netcdf
from gt4py.next.modules.translator import (
    IDim,
    JDim,
    Kolor,
    pack_sparse_local_field_to_structured,
    build_index_map_from_lonlat_e2v,
    pack_edge_field_to_structured,
    pack_vertex_field,
    pack_vertex_field_to_structured,
    unpack_edge_field,
    unpack_vertex_field_to_unstructured,
    pack_cell_field,
    unpack_cell_field,
)


def _first_present(ds, names, required=True):
    for name in names:
        if name in ds:
            return ds[name].values
    if required:
        raise KeyError(f"None of the variables {names} found in dataset.")
    return None


def _read_e2v(ds):
    raw = _first_present(ds, ["E2V", "edge_vertices", "edges2nodes", "edge_node_connectivity"])
    arr = np.asarray(raw, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError("e2v dataset must be 2-D")
    if arr.shape[1] != 2:
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError(f"e2v must have shape (n_edge, 2), got {arr.shape}")
    return np.where(arr > 0, arr - 1, -1)


def _read_v2e(ds):
    raw = _first_present(
        ds,
        ["V2E", "vertex_edges", "nodes2edges", "node_edge_connectivity", "edges_of_vertex"],
        required=False,
    )
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError("v2e dataset must be 2-D")
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return np.where(arr > 0, arr - 1, -1)


def _read_lonlat(ds):
    if "longitude_vertices" in ds and "latitude_vertices" in ds:
        lon = ds["longitude_vertices"].values.astype(np.float64)
        lat = ds["latitude_vertices"].values.astype(np.float64)
        return np.stack([lon, lat], axis=1)
    return _first_present(ds, ["lonlat", "vertex_lonlat", "node_lonlat"], required=False)


def _interior_ij_bounds(remap_sizes):
    lateral = int(getattr(remap_sizes, "lateral", 0))
    return lateral, int(remap_sizes.max_i) - lateral, lateral, int(remap_sizes.max_j) - lateral


def _vertex_interior_mask(index_map, remap_sizes):
    i_lo, i_hi, j_lo, j_hi = _interior_ij_bounds(remap_sizes)
    ij = index_map.vertex_to_ij
    i = ij[:, 0]
    j = ij[:, 1]
    return (i >= i_lo) & (i < i_hi) & (j >= j_lo) & (j < j_hi)


def _edge_interior_mask(index_map, remap_sizes):
    i_lo, i_hi, j_lo, j_hi = _interior_ij_bounds(remap_sizes)
    ijk = index_map.edge_to_ijk
    i = ijk[:, 0]
    j = ijk[:, 1]
    valid = (i >= 0) & (j >= 0)
    return valid & (i >= i_lo) & (i < i_hi) & (j >= j_lo) & (j < j_hi)


def _edge_structured_active_mask(index_map, remap_sizes):
    """Kolor-aware edge mask matching structured edge write extents."""
    i_lo, i_hi, j_lo, j_hi = _interior_ij_bounds(remap_sizes)
    ijk = index_map.edge_to_ijk
    i = ijk[:, 0]
    j = ijk[:, 1]
    k = ijk[:, 2]

    valid = (i >= 0) & (j >= 0) & (k >= 0)
    k0 = (k == 0) & (i >= i_lo) & (i < i_hi) & (j >= j_lo) & (j < (j_hi - 1))
    k1 = (k == 1) & (i >= i_lo) & (i < (i_hi - 1)) & (j >= j_lo) & (j < j_hi)
    k2 = (k == 2) & (i >= i_lo) & (i < (i_hi - 1)) & (j >= j_lo) & (j < (j_hi - 1))
    return valid & (k0 | k1 | k2)


def _skip_if_cupy_14_0_1_hip(exec_alloc_descriptor) -> None:
    if exec_alloc_descriptor is test_definitions.cupy_execution:
        cp = test_definitions.cupy_execution.allocator
        if cp.cuda.runtime.is_hip:
            cp_version = version.parse(cp.__version__)
            if cp_version.major == 14 and cp_version.minor == 0 and cp_version.micro < 2:
                # see https://github.com/cupy/cupy/issues/9742 and https://github.com/cupy/cupy/issues/9829
                pytest.skip(
                    "CuPy 14.0.0 and 14.0.1 have a bug that causes this test to fail when jit compiling for hip."
                )


@gtx.field_operator
def compute_zavgS(
    pp: gtx.Field[[Vertex], float], S_M: gtx.Field[[Edge], float]
) -> gtx.Field[[Edge], float]:
    zavg = 0.5 * (pp(E2V[0]) + pp(E2V[1]))
    return S_M * zavg

EdgeDim = gtx.Dimension("Edge")


@gtx.program
def compute_zavgS_program_(
    pp: gtx.Field[[Vertex], float],
    S_M: gtx.Field[[Edge], float],
    out: gtx.Field[[Edge], float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    compute_zavgS(pp, S_M, out=out, domain={EdgeDim: (horizontal_start, horizontal_end)})


@gtx.field_operator
def compute_pnabla(
    pp: gtx.Field[[Vertex], float],
    S_M: gtx.Field[[Edge], float],
    sign: gtx.Field[[Vertex, V2EDim], float],
    vol: gtx.Field[[Vertex], float],
) -> gtx.Field[[Vertex], float]:
    zavgS = compute_zavgS(pp, S_M)
    pnabla_M = neighbor_sum(zavgS(V2E) * sign, axis=V2EDim)
    return pnabla_M / vol


@gtx.field_operator
def compute_neighbor_sum_unweighted(
    zavgS: gtx.Field[[Edge], float],
) -> gtx.Field[[Vertex], float]:
    return neighbor_sum(zavgS(V2E), axis=V2EDim)


@gtx.field_operator
def compute_neighbor_sum_weighted(
    zavgS: gtx.Field[[Edge], float],
    sign: gtx.Field[[Vertex, V2EDim], float],
) -> gtx.Field[[Vertex], float]:
    return neighbor_sum(zavgS(V2E) * sign, axis=V2EDim)


@gtx.field_operator
def compute_divide_volume(
    pnabla_M: gtx.Field[[Vertex], float],
    vol: gtx.Field[[Vertex], float],
) -> gtx.Field[[Vertex], float]:
    return pnabla_M / vol


@gtx.field_operator
def pnabla(
    pp: gtx.Field[[Vertex], float],
    S_M: Tuple[gtx.Field[[Edge], float], gtx.Field[[Edge], float]],
    sign: gtx.Field[[Vertex, V2EDim], float],
    vol: gtx.Field[[Vertex], float],
) -> Tuple[gtx.Field[[Vertex], float], gtx.Field[[Vertex], float]]:
    return compute_pnabla(pp, S_M[0], sign, vol), compute_pnabla(pp, S_M[1], sign, vol)


@pytest.mark.uses_unstructured_shift
@pytest.mark.requires_atlas
def test_ffront_compute_zavgS(exec_alloc_descriptor):
    _skip_if_cupy_14_0_1_hip(exec_alloc_descriptor)

    setup = nabla_setup(allocator=exec_alloc_descriptor.allocator)

    zavgS = gtx.zeros({Edge: setup.edges_size}, allocator=exec_alloc_descriptor.allocator)

    compute_zavgS.with_backend(
        None if exec_alloc_descriptor.executor is None else exec_alloc_descriptor
    )(
        setup.input_field,
        setup.S_fields[0],
        out=zavgS,
        offset_provider={"E2V": setup.edges2node_connectivity},
    )

    assert_close(-199755464.25741270, np.min(zavgS.asnumpy()))
    assert_close(388241977.58389181, np.max(zavgS.asnumpy()))


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_tuple_returns
@pytest.mark.requires_atlas
def test_ffront_compute_zavgS_parallelogram_grid(exec_alloc_descriptor):
    mesh_nc = os.environ.get(
        "GT4PY_TRANSLATOR_MESH",
        "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc",
    )
    xr = pytest.importorskip("xarray")

    lateral = 1

    with xr.open_dataset(mesh_nc) as ds:
        e2v = _read_e2v(ds)
        v2e = _read_v2e(ds)
        lonlat = _read_lonlat(ds)
        remap_sizes = load_structured_remap_sizes_from_netcdf(mesh_nc, lateral=lateral)

        print(f"remap sies: ", remap_sizes)

        setup = nabla_setup.from_connectivity(
            allocator=exec_alloc_descriptor.allocator,
            e2v=e2v,
            v2e=v2e,
            lonlat_deg=lonlat,
        )

        assert setup.nodes_size == remap_sizes.vertex_size
        assert setup.edges_size <= remap_sizes.edge_size_padded
        assert int(ds.sizes["cell"]) == remap_sizes.cell_size

    index_map = build_index_map_from_lonlat_e2v(
        lonlat,
        e2v,
        nodes_size=setup.nodes_size,
        edges_size=setup.edges_size,
    )

    if "edge_translation_index" in ds:
        edge_translator = ds["edge_translation_index"].values - 1
    else:
        raise KeyError("edge_translation_index variable not found in dataset")

    if "vertex_translation_index" in ds:
        vertex_translator = ds["vertex_translation_index"].values - 1
    else:
        raise KeyError("vertex_translation_index variable not found in dataset")

    if "cell_translation_index" in ds:
        cell_translator = ds["cell_translation_index"].values - 1
    else:
        raise KeyError("cell_translation_index variable not found in dataset")

    print(f"vertex_translator: ", vertex_translator)
    pp_struct_np = pack_vertex_field_to_structured(setup.input_field.asnumpy(), index_map)
    s_m_struct_np = pack_edge_field_to_structured(setup.S_fields[0].asnumpy(), index_map)
    zavgS_struct_np = np.zeros_like(s_m_struct_np)
    # pp_struct_np = np.zeros(setup.nodes_size, dtype=setup.input_field.dtype)
    # print(f"input field: ",setup.input_field.asnumpy())
    # pp_struct_np[vertex_translator] = setup.input_field.asnumpy()
    # print(f"pp_struct_np after translation: ", pp_struct_np)
    # s_m_struct_np = np.zeros((remap_sizes.max_i, remap_sizes.max_j, 3), dtype=setup.S_fields[0].dtype)
    # s_m_struct_np[edge_translator] = setup.S_fields[0].asnumpy()
    # zavgS_struct_np = np.zeros_like(s_m_struct_np)

    assert pp_struct_np.shape[0] == remap_sizes.max_i
    assert pp_struct_np.shape[1] == remap_sizes.max_j
    assert s_m_struct_np.shape[2] == 3

    pp_struct = gtx.as_field(
        [IDim, JDim, Kolor], pp_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    s_m_struct = gtx.as_field(
        [IDim, JDim, Kolor], s_m_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    zavgS_struct = gtx.as_field(
        [IDim, JDim, Kolor], zavgS_struct_np, allocator=exec_alloc_descriptor.allocator
    )

    selected_backend = gtfn_runner.GTFNBackendFactory(
        cached=True,
        otf_workflow__cached_translation=True,
        otf_workflow__bare_translation__symbolic_domain_sizes={
            "use_horizontal_start_mapping": True,
            "edge_to_ijk": index_map.edge_to_ijk,
            "horizontal_start": int(844),
            "horizontal_end": setup.edges_size,
        },
    )
    print("the selected_backend is:", selected_backend)
    compute_zavgS_program = setup_program(
        compute_zavgS,
        backend=selected_backend,
        # horizontal_sizes={"horizontal_start": int(844), "horizontal_end": setup.edges_size},
        offset_provider={"E2V": setup.edges2node_connectivity},
    )

    compute_zavgS_program(
        pp=pp_struct,
        S_M=s_m_struct,
        out=zavgS_struct,
    )

    zavgS_np = unpack_edge_field(zavgS_struct.asnumpy(), index_map, setup.edges_size)

    e2v_conn = setup.edges2node_connectivity.asnumpy()
    valid = np.all(e2v_conn >= 0, axis=1)
    active = valid & _edge_structured_active_mask(index_map, remap_sizes)
    pp = setup.input_field.asnumpy()
    s_m = setup.S_fields[0].asnumpy()
    ref = np.zeros_like(s_m)
    ref[active] = s_m[active] * 0.5 * (pp[e2v_conn[active, 0]] + pp[e2v_conn[active, 1]])

    print(f"numpy solution: ", ref[active])
    print(f"GT4Py solution: ", zavgS_np)

    assert np.isfinite(zavgS_np).all()
    np.testing.assert_allclose(zavgS_np, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.requires_atlas
def test_ffront_nabla(exec_alloc_descriptor):
    _skip_if_cupy_14_0_1_hip(exec_alloc_descriptor)

    setup = nabla_setup(allocator=exec_alloc_descriptor.allocator)

    pnabla_MXX = gtx.zeros({Vertex: setup.nodes_size}, allocator=exec_alloc_descriptor.allocator)
    pnabla_MYY = gtx.zeros({Vertex: setup.nodes_size}, allocator=exec_alloc_descriptor.allocator)

    pnabla.with_backend(None if exec_alloc_descriptor.executor is None else exec_alloc_descriptor)(
        setup.input_field,
        setup.S_fields,
        setup.sign_field,
        setup.vol_field,
        out=(pnabla_MXX, pnabla_MYY),
        offset_provider={
            "E2V": setup.edges2node_connectivity,
            "V2E": setup.nodes2edge_connectivity,
        },
    )

    # TODO this check is not sensitive enough, need to implement a proper numpy reference!
    assert_close(-3.5455427772566003e-003, np.min(pnabla_MXX.asnumpy()))
    assert_close(3.5455427772565435e-003, np.max(pnabla_MXX.asnumpy()))
    assert_close(-3.3540113705465301e-003, np.min(pnabla_MYY.asnumpy()))
    assert_close(3.3540113705465301e-003, np.max(pnabla_MYY.asnumpy()))


def test_ffront_nabla_parallelogram_grid(exec_alloc_descriptor):
    mesh_nc = os.environ.get(
        "GT4PY_TRANSLATOR_MESH",
        "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc",
    )
    xr = pytest.importorskip("xarray")

    lateral = 1
    with xr.open_dataset(mesh_nc) as ds:
        e2v = _read_e2v(ds)
        v2e = _read_v2e(ds)
        lonlat = _read_lonlat(ds)
        remap_sizes = load_structured_remap_sizes_from_netcdf(mesh_nc, lateral=lateral)
        dual_volumes = _first_present(ds, ["dual_area"])

        print(f"remap sies: ", remap_sizes)

        setup = nabla_setup.from_connectivity(
            allocator=exec_alloc_descriptor.allocator,
            e2v=e2v,
            v2e=v2e,
            lonlat_deg=lonlat,
            dual_volumes=dual_volumes,
        )

        assert setup.nodes_size == remap_sizes.vertex_size
        assert setup.edges_size <= remap_sizes.edge_size_padded
        assert int(ds.sizes["cell"]) == remap_sizes.cell_size

    index_map = build_index_map_from_lonlat_e2v(
        lonlat,
        e2v,
        nodes_size=setup.nodes_size,
        edges_size=setup.edges_size,
    )

    # # 1. Extract the raw numpy array (size: max_i x max_j x 1)
    pp_struct_np = pack_vertex_field_to_structured(setup.input_field.asnumpy(), index_map)

    # # 2. Create a padded array (+1 in IDim and JDim) initialized to zero
    # pp_struct_np_padded = np.zeros(
    #     (remap_sizes.max_i + 1, remap_sizes.max_j + 1, 1),
    #     dtype=pp_raw.dtype
    # )

    # # 3. Insert the raw data into the valid interior of the padded array
    # pp_struct_np_padded[:remap_sizes.max_i, :remap_sizes.max_j, :] = pp_raw

    # # 4. Wrap the padded array into a GT4Py field
    # pp_struct = gtx.as_field(
    #     [IDim, JDim, Kolor],
    #     pp_struct_np_padded,
    #     allocator=exec_alloc_descriptor.allocator
    # )
    s_m_struct_np = pack_edge_field_to_structured(setup.S_fields[0].asnumpy(), index_map)
    sign_struct_np = pack_sparse_local_field_to_structured(
        coeff=setup.sign_field.asnumpy(),
        connectivity=setup.nodes2edge_connectivity.asnumpy(),
        index_map=index_map,
        local_dim_name="V2E",
    )

    assert sign_struct_np.ndim == 4
    assert sign_struct_np.shape[2] == 3
    assert sign_struct_np.shape[3] == 6
    vol_struct_np = pack_vertex_field_to_structured(setup.vol_field.asnumpy(), index_map)
    pnabla_mxx_struct_np = np.zeros_like(vol_struct_np)
    pnabla_myy_struct_np = np.zeros_like(vol_struct_np)

    pp_struct = gtx.as_field(
        [IDim, JDim, Kolor], pp_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    s_m_struct = gtx.as_field(
        [IDim, JDim, Kolor], s_m_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    sign_struct = gtx.as_field(
        [IDim, JDim, Kolor, V2EDim], sign_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    vol_struct = gtx.as_field(
        [IDim, JDim, Kolor], vol_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    pnabla_mxx_struct = gtx.as_field(
        [IDim, JDim, Kolor], pnabla_mxx_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    pnabla_myy_struct = gtx.as_field(
        [IDim, JDim, Kolor], pnabla_myy_struct_np, allocator=exec_alloc_descriptor.allocator
    )

    selected_backend = gtfn_runner.GTFNBackendFactory(
        cached=True,
        otf_workflow__cached_translation=True,
        otf_workflow__bare_translation__symbolic_domain_sizes={
            "max_i": int(remap_sizes.max_i),
            "max_j": int(remap_sizes.max_j),
            "lateral": int(remap_sizes.lateral),
        },
    )
    print("the selected_backend is:", selected_backend)
    pnabla_mxx_program = setup_program(
        compute_pnabla,
        backend=selected_backend,
        offset_provider={
            "E2V": setup.edges2node_connectivity,
            "V2E": setup.nodes2edge_connectivity,
        },
    )
    pnabla_myy_program = setup_program(
        compute_pnabla,
        backend=selected_backend,
        offset_provider={
            "E2V": setup.edges2node_connectivity,
            "V2E": setup.nodes2edge_connectivity,
        },
    )

    pnabla_mxx_program(
        pp=pp_struct,
        S_M=s_m_struct,
        sign=sign_struct,
        vol=vol_struct,
        out=pnabla_mxx_struct,
    )
    pnabla_myy_program(
        pp=pp_struct,
        S_M=s_m_struct,
        sign=sign_struct,
        vol=vol_struct,
        out=pnabla_myy_struct,
    )

    pnabla_mxx_np = unpack_vertex_field_to_unstructured(pnabla_mxx_struct.asnumpy(), index_map)
    pnabla_myy_np = unpack_vertex_field_to_unstructured(pnabla_myy_struct.asnumpy(), index_map)
    print(pnabla_mxx_np)
    # print(pnabla_myy_np)

    # Numpy reference implementation (use unstructured connectivity from setup)
    e2v_un = setup.edges2node_connectivity.asnumpy()  # (n_edge, 2)
    valid_e = np.all(e2v_un >= 0, axis=1)
    pp_un = setup.input_field.asnumpy()  # (n_vertex,)
    s_m_un = setup.S_fields[0].asnumpy()  # (n_edge,)
    vol_un = setup.vol_field.asnumpy()  # (n_vertex,)
    sign_un = setup.sign_field.asnumpy()  # (n_vertex, max_deg)
    v2e_un = setup.nodes2edge_connectivity.asnumpy()  # (n_vertex, max_deg)

    # zavg on unstructured edges
    zavg_un = np.zeros((s_m_un.shape[0],), dtype=s_m_un.dtype)
    zavg_un[valid_e] = (
        s_m_un[valid_e] * 0.5 * (pp_un[e2v_un[valid_e, 0]] + pp_un[e2v_un[valid_e, 1]])
    )

    # accumulate per-vertex neighbor sum
    n_vertex = v2e_un.shape[0]
    pnabla_mxx_numpy = np.zeros((n_vertex,), dtype=float)
    pnabla_myy_numpy = np.zeros((n_vertex,), dtype=float)

    interior_vertices = _vertex_interior_mask(index_map, remap_sizes)

    for v in range(n_vertex):
        if not interior_vertices[v]:
            continue
        edges = v2e_un[v]  # list of neighbor edge indices (pad -1)
        mask = edges >= 0
        if not np.any(mask):
            continue
        e_idx = edges[mask]
        svals = zavg_un[e_idx]
        sgns = sign_un[v, mask]
        pnabla_mxx_numpy[v] = float((svals * sgns).sum())
        pnabla_myy_numpy[v] = pnabla_mxx_numpy[
            v
        ]  # if identical here, else compute from S_MYY similarly

    # divide by vertex volume
    pnabla_mxx_numpy /= vol_un
    pnabla_myy_numpy /= vol_un

    # print(f"numpy pnabla_mxx: ", pnabla_mxx_numpy)
    print(f"pp: ", pp_un)

    print(f"difference: ", pnabla_mxx_np - pnabla_mxx_numpy)
    print(f"pnabla_mxx_np: ", pnabla_mxx_np)

    # compare to GT4Py result (unstructured)
    assert np.allclose(pnabla_mxx_np, pnabla_mxx_numpy, rtol=1e-9, atol=0)
    assert np.allclose(pnabla_myy_np, pnabla_myy_numpy, rtol=1e-9, atol=0)

    # # TODO this check is not sensitive enough, need to implement a proper numpy reference!
    # assert_close(-3.5455427772566003e-003, np.min(pnabla_mxx_np))
    # assert_close(3.5455427772565435e-003, np.max(pnabla_mxx_np))
    # assert_close(-3.3540113705465301e-003, np.min(pnabla_myy_np))
    # assert_close(3.3540113705465301e-003, np.max(pnabla_myy_np))


def _prepare_parallelogram_structured_case(exec_alloc_descriptor):
    mesh_nc = os.environ.get(
        "GT4PY_TRANSLATOR_MESH",
        "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc",
    )
    xr = pytest.importorskip("xarray")
    lateral = 1

    with xr.open_dataset(mesh_nc) as ds:
        e2v = _read_e2v(ds)
        v2e = _read_v2e(ds)
        lonlat = _read_lonlat(ds)
        remap_sizes = load_structured_remap_sizes_from_netcdf(mesh_nc, lateral=lateral)
        setup = nabla_setup.from_connectivity(
            allocator=exec_alloc_descriptor.allocator,
            e2v=e2v,
            v2e=v2e,
            lonlat_deg=lonlat,
        )

    index_map = build_index_map_from_lonlat_e2v(
        lonlat,
        e2v,
        nodes_size=setup.nodes_size,
        edges_size=setup.edges_size,
    )

    pp_struct_np = pack_vertex_field_to_structured(setup.input_field.asnumpy(), index_map)
    # # 2. Create a padded array (+1 in IDim and JDim) initialized to zero
    # pp_struct_np_padded = np.zeros(
    #     (remap_sizes.max_i + 1, remap_sizes.max_j + 1, 1),
    #     dtype=pp_struct_np.dtype
    # )

    # # 3. Insert the raw data into the valid interior of the padded array
    # pp_struct_np_padded[:remap_sizes.max_i, :remap_sizes.max_j, :] = pp_struct_np

    # # 4. Wrap the padded array into a GT4Py field
    # pp_struct = gtx.as_field(
    #     [IDim, JDim, Kolor],
    #     pp_struct_np_padded,
    #     allocator=exec_alloc_descriptor.allocator
    # )
    s_m_struct_np = pack_edge_field_to_structured(setup.S_fields[0].asnumpy(), index_map)
    sign_struct_np = pack_sparse_local_field_to_structured(
        coeff=setup.sign_field.asnumpy(),
        connectivity=setup.nodes2edge_connectivity.asnumpy(),
        index_map=index_map,
        local_dim_name="V2E",
    )
    vol_struct_np = pack_vertex_field_to_structured(setup.vol_field.asnumpy(), index_map)

    pp_struct = gtx.as_field(
        [IDim, JDim, Kolor], pp_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    s_m_struct = gtx.as_field(
        [IDim, JDim, Kolor], s_m_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    sign_struct = gtx.as_field(
        [IDim, JDim, Kolor, V2EDim], sign_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    vol_struct = gtx.as_field(
        [IDim, JDim, Kolor], vol_struct_np, allocator=exec_alloc_descriptor.allocator
    )

    selected_backend = gtfn_runner.GTFNBackendFactory(
        cached=True,
        otf_workflow__cached_translation=True,
        otf_workflow__bare_translation__symbolic_domain_sizes={
            "max_i": int(remap_sizes.max_i),
            "max_j": int(remap_sizes.max_j),
            "lateral": int(remap_sizes.lateral),
        },
    )

    return {
        "setup": setup,
        "index_map": index_map,
        "backend": selected_backend,
        "pp_struct": pp_struct,
        "s_m_struct": s_m_struct,
        "sign_struct": sign_struct,
        "vol_struct": vol_struct,
        "remap_sizes": remap_sizes,
    }


def _require_gtfn_run_gtfn_backend(request):
    param_id = str(getattr(getattr(request.node, "callspec", None), "id", ""))
    if not param_id.startswith("gtfn.run_gtfn"):
        pytest.skip("decomposition debug test is restricted to gtfn.run_gtfn")


@pytest.mark.requires_atlas
def test_ffront_nabla_parallelogram_part_zavgS(exec_alloc_descriptor, request):
    _require_gtfn_run_gtfn_backend(request)
    case = _prepare_parallelogram_structured_case(exec_alloc_descriptor)
    zavgS_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 3,
        },
        allocator=exec_alloc_descriptor.allocator,
    )

    program = setup_program(
        compute_zavgS,
        backend=case["backend"],
        offset_provider={"E2V": case["setup"].edges2node_connectivity},
    )

    program(pp=case["pp_struct"], S_M=case["s_m_struct"], out=zavgS_struct)

    # print(f"pp: ", case["pp_struct"][:,:,0])
    print(
        f"zavgS: ",
        zavgS_struct.asnumpy()[:, :, 0],
        "\n",
        zavgS_struct.asnumpy()[:, :, 1],
        "\n",
        zavgS_struct.asnumpy()[:, :, 2],
    )

    assert np.isfinite(zavgS_struct.asnumpy()).all()


@pytest.mark.requires_atlas
def test_ffront_nabla_parallelogram_part_neighbor_sum_unweighted(exec_alloc_descriptor, request):
    _require_gtfn_run_gtfn_backend(request)
    case = _prepare_parallelogram_structured_case(exec_alloc_descriptor)
    max_i = case["remap_sizes"].max_i
    max_j = case["remap_sizes"].max_j
    zavgS_struct = gtx.zeros(
        {
            IDim: int(max_i + 1),
            JDim: int(max_j + 1),
            Kolor: 3,
        },
        allocator=exec_alloc_descriptor.allocator,
    )
    pnabla_m_struct = gtx.zeros(
        {
            IDim: int(max_i + 1),
            JDim: int(max_j + 1),
            Kolor: 1,
        },
        allocator=exec_alloc_descriptor.allocator,
    )

    zavg_program = setup_program(
        compute_zavgS,
        backend=case["backend"],
        offset_provider={"E2V": case["setup"].edges2node_connectivity},
        # horizontal_size
    )
    unweighted_program = setup_program(
        compute_neighbor_sum_unweighted,
        backend=case["backend"],
        offset_provider={"V2E": case["setup"].nodes2edge_connectivity},
        # horizontal_sizes={
        #     "domain_max_i": gtx.int32(max_i),
        #     "domain_max_j": gtx.int32(max_j),
        #     "domain_max_kolor": gtx.int32(1),
        # },
    )

    zavg_program(
        pp=case["pp_struct"],
        S_M=case["s_m_struct"],
        out=zavgS_struct,
        domain={IDim: (0, max_i), JDim: (0, max_j), Kolor: (0, 3)},
    )
    unweighted_program(
        zavgS=zavgS_struct,
        out=pnabla_m_struct,
        domain={IDim: (0, max_i), JDim: (0, max_j), Kolor: (0, 1)},
    )

    # numpy reference implementation of the unweighted neighbor sum
    # with zero-padded boundaries (no negative-index wraparound).
    zavg_np = zavgS_struct.asnumpy()
    pnabla_numpy = np.zeros_like(pnabla_m_struct.asnumpy())

    def _zavg_or_zero(i: int, j: int, k: int) -> float:
        if i < 0 or j < 0 or i >= max_i or j >= max_j:
            return 0.0
        return float(zavg_np[i, j, k])

    i_lo, i_hi, j_lo, j_hi = _interior_ij_bounds(case["remap_sizes"])
    for i in range(i_lo, i_hi):
        for j in range(j_lo, j_hi):
            pnabla_numpy[i, j, 0] = (
                _zavg_or_zero(i, j, 0)
                + _zavg_or_zero(i, j - 1, 0)
                + _zavg_or_zero(i, j, 1)
                + _zavg_or_zero(i - 1, j, 1)
                + _zavg_or_zero(i, j - 1, 2)
                + _zavg_or_zero(i - 1, j, 2)
            )

    # print(zavgS_struct.asnumpy()[:,:,0])
    # print(zavg_np[:,:,0])
    # print(zavgS_struct.asnumpy()[:,:,1])
    # print(zavg_np[:,:,1])
    # print(zavgS_struct.asnumpy()[:,:,2])
    # print(zavg_np[:,:,2])
    print(f"pp: ", case["pp_struct"][:, :, 0])
    print(
        f"zavgS difference: \n",
        (zavgS_struct.asnumpy() - zavg_np)[:, :, 0],
        "\n",
        (zavgS_struct.asnumpy() - zavg_np)[:, :, 1],
        "\n",
        (zavgS_struct.asnumpy() - zavg_np)[:, :, 2],
    )
    # print(f"numpy pnabla: ", pnabla_numpy[:,:,0])
    # print(f"GT4py pnabla: ", pnabla_m_struct.asnumpy()[:,:,0])
    print(f"Pnabla difference: \n", pnabla_m_struct.asnumpy()[:, :, 0] - pnabla_numpy[:, :, 0])
    assert np.isfinite(pnabla_m_struct.asnumpy()).all()
    assert np.allclose(pnabla_m_struct.asnumpy(), pnabla_numpy, rtol=1e-10, atol=0)


@pytest.mark.requires_atlas
def test_ffront_nabla_parallelogram_part_neighbor_sum_weighted(exec_alloc_descriptor, request):
    _require_gtfn_run_gtfn_backend(request)
    case = _prepare_parallelogram_structured_case(exec_alloc_descriptor)

    zavgS_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 3,
        },
        allocator=exec_alloc_descriptor.allocator,
    )
    pnabla_m_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 1,
        },
        allocator=exec_alloc_descriptor.allocator,
    )

    zavg_program = setup_program(
        compute_zavgS,
        backend=case["backend"],
        offset_provider={"E2V": case["setup"].edges2node_connectivity},
    )
    weighted_program = setup_program(
        compute_neighbor_sum_weighted,
        backend=case["backend"],
        offset_provider={"V2E": case["setup"].nodes2edge_connectivity},
        # horizontal_sizes={
        #     "domain_max_i": gtx.int32(max_i),
        #     "domain_max_j": gtx.int32(max_j),
        #     "domain_max_kolor": gtx.int32(1),
        # },
    )
    #         domain={IDim: (0, domain_max_i), JDim: (0, domain_max_j), Kolor: (0, domain_max_kolor)},
    # horizontal_sizes={
    # #         "domain_max_i": gtx.int32(max_i),
    # #         "domain_max_j": gtx.int32(max_j),
    # #         "domain_max_kolor": gtx.int32(3),
    # #     },

    zavg_program(
        pp=case["pp_struct"],
        S_M=case["s_m_struct"],
        out=zavgS_struct,
        domain={
            IDim: (0, case["remap_sizes"].max_i),
            JDim: (0, case["remap_sizes"].max_j),
            Kolor: (0, 3),
        },
    )
    weighted_program(
        zavgS=zavgS_struct,
        sign=case["sign_struct"],
        out=pnabla_m_struct,
        domain={
            IDim: (0, case["remap_sizes"].max_i),
            JDim: (0, case["remap_sizes"].max_j),
            Kolor: (0, 1),
        },
    )

    # Numpy reference implementation:
    zavg_np = zavgS_struct.asnumpy()
    sign_np = case["sign_struct"].asnumpy()
    pnabla_numpy = np.zeros_like(pnabla_m_struct.asnumpy())

    def _zavg_or_zero(i: int, j: int, k: int) -> float:
        if i < 0 or j < 0 or i >= zavg_np.shape[0] or j >= zavg_np.shape[1]:
            return 0.0
        return float(zavg_np[i, j, k])

    i_lo, i_hi, j_lo, j_hi = _interior_ij_bounds(case["remap_sizes"])
    for i in range(i_lo, i_hi):
        for j in range(j_lo, j_hi):
            pnabla_numpy[i, j, 0] = (
                _zavg_or_zero(i, j, 0) * sign_np[i, j, 0, 0]
                + _zavg_or_zero(i, j - 1, 0) * sign_np[i, j, 0, 3]
                + _zavg_or_zero(i, j, 1) * sign_np[i, j, 0, 1]
                + _zavg_or_zero(i - 1, j, 1) * sign_np[i, j, 0, 4]
                + _zavg_or_zero(i, j - 1, 2) * sign_np[i, j, 0, 2]
                + _zavg_or_zero(i - 1, j, 2) * sign_np[i, j, 0, 5]
            )

    print(f"pp: ", case["pp_struct"][:, :, 0])
    print(
        f"zavgS difference: \n",
        (zavgS_struct.asnumpy() - zavg_np)[:, :, 0],
        "\n",
        (zavgS_struct.asnumpy() - zavg_np)[:, :, 1],
        "\n",
        (zavgS_struct.asnumpy() - zavg_np)[:, :, 2],
    )
    print(f"pnabla difference: \n", pnabla_m_struct.asnumpy() - pnabla_numpy)
    assert np.isfinite(pnabla_m_struct.asnumpy()).all()
    assert np.allclose(pnabla_m_struct.asnumpy(), pnabla_numpy, rtol=1e-10, atol=0)


@pytest.mark.requires_atlas
def test_ffront_nabla_parallelogram_part_divide(exec_alloc_descriptor, request):
    _require_gtfn_run_gtfn_backend(request)
    case = _prepare_parallelogram_structured_case(exec_alloc_descriptor)

    zavgS_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 3,
        },
        allocator=exec_alloc_descriptor.allocator,
    )
    pnabla_m_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 1,
        },
        allocator=exec_alloc_descriptor.allocator,
    )
    out_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 1,
        },
        allocator=exec_alloc_descriptor.allocator,
    )

    zavg_program = setup_program(
        compute_zavgS,
        backend=case["backend"],
        offset_provider={"E2V": case["setup"].edges2node_connectivity},
    )
    weighted_program = setup_program(
        compute_neighbor_sum_weighted,
        backend=case["backend"],
        offset_provider={"V2E": case["setup"].nodes2edge_connectivity},
    )
    divide_program = setup_program(
        compute_divide_volume,
        backend=case["backend"],
    )

    zavg_program(pp=case["pp_struct"], S_M=case["s_m_struct"], out=zavgS_struct)
    weighted_program(zavgS=zavgS_struct, sign=case["sign_struct"], out=pnabla_m_struct)
    divide_program(pnabla_M=pnabla_m_struct, vol=case["vol_struct"], out=out_struct)

    # Numpy reference implementation:
    zavg_np = zavgS_struct.asnumpy()
    sign_np = case["sign_struct"].asnumpy()
    vol_np = case["vol_struct"].asnumpy()
    pnabla_numpy = np.zeros_like(pnabla_m_struct.asnumpy())
    out_numpy = np.zeros_like(out_struct.asnumpy())

    def _zavg_or_zero(i: int, j: int, k: int) -> float:
        if i < 0 or j < 0 or i >= zavg_np.shape[0] or j >= zavg_np.shape[1]:
            return 0.0
        return float(zavg_np[i, j, k])

    i_lo, i_hi, j_lo, j_hi = _interior_ij_bounds(case["remap_sizes"])
    for i in range(i_lo, i_hi):
        for j in range(j_lo, j_hi):
            pnabla_numpy[i, j, 0] = (
                _zavg_or_zero(i, j, 0) * sign_np[i, j, 0, 0]
                + _zavg_or_zero(i, j - 1, 0) * sign_np[i, j, 0, 3]
                + _zavg_or_zero(i, j, 1) * sign_np[i, j, 0, 1]
                + _zavg_or_zero(i - 1, j, 1) * sign_np[i, j, 0, 4]
                + _zavg_or_zero(i, j - 1, 2) * sign_np[i, j, 0, 2]
                + _zavg_or_zero(i - 1, j, 2) * sign_np[i, j, 0, 5]
            )
            out_numpy[i, j, 0] = pnabla_numpy[i, j, 0] / vol_np[i, j, 0]

    print(f"pp: ", case["pp_struct"][:, :, 0])
    print(
        f"zavgS difference: \n",
        (zavgS_struct.asnumpy() - zavg_np)[:, :, 0],
        "\n",
        (zavgS_struct.asnumpy() - zavg_np)[:, :, 1],
        "\n",
        (zavgS_struct.asnumpy() - zavg_np)[:, :, 2],
    )
    print(f"pnabla difference: \n", pnabla_m_struct.asnumpy() - pnabla_numpy)
    print(f"divide difference: \n", out_struct.asnumpy() - out_numpy)

    assert np.isfinite(out_struct.asnumpy()).all()
    assert np.allclose(out_struct.asnumpy(), out_numpy, rtol=1e-10, atol=0)


from gt4py.next.modules.structured_wrapper import setup_smart_program


@pytest.mark.requires_atlas
def test_ffront_pnabla_clean(exec_alloc_descriptor):
    case = _prepare_parallelogram_structured_case(exec_alloc_descriptor)
    setup = case["setup"]

    # 1. Standard Unstructured Allocation!
    out_unstruct = gtx.zeros({Vertex: setup.nodes_size}, allocator=exec_alloc_descriptor.allocator)

    # 2. Use the smart setup
    pnabla_program = setup_smart_program(
        compute_pnabla,
        setup=setup,
        index_map=case["index_map"],
        remap_sizes=case["remap_sizes"],
        allocator=exec_alloc_descriptor.allocator,
        offset_provider={
            "E2V": setup.edges2node_connectivity,
            "V2E": setup.nodes2edge_connectivity,
        },
    )

    # 3. Call it! The wrapper intercepts, packs, runs Cartesian, and unpacks automatically.
    pnabla_program(
        pp=setup.input_field,
        S_M=setup.S_fields[0],
        sign=setup.sign_field,
        vol=setup.vol_field,
        out=out_unstruct,
    )

    # 4. Assert directly on the unstructured field
    assert np.isfinite(out_unstruct.asnumpy()).all()

    # 5. Implement a proper numpy reference and compare
    # Numpy reference implementation (use unstructured connectivity from setup)
    e2v_un = setup.edges2node_connectivity.asnumpy()  # (n_edge, 2)
    valid_e = np.all(e2v_un >= 0, axis=1)
    pp_un = setup.input_field.asnumpy()  # (n_vertex,)
    s_m_un = setup.S_fields[0].asnumpy()  # (n_edge,)
    vol_un = setup.vol_field.asnumpy()  # (n_vertex,)
    sign_un = setup.sign_field.asnumpy()  # (n_vertex, max_deg)
    v2e_un = setup.nodes2edge_connectivity.asnumpy()  # (n_vertex, max_deg)

    # zavg on unstructured edges
    zavg_un = np.zeros((s_m_un.shape[0],), dtype=s_m_un.dtype)
    zavg_un[valid_e] = (
        s_m_un[valid_e] * 0.5 * (pp_un[e2v_un[valid_e, 0]] + pp_un[e2v_un[valid_e, 1]])
    )

    # accumulate per-vertex neighbor sum
    n_vertex = v2e_un.shape[0]
    pnabla_mxx_numpy = np.zeros((n_vertex,), dtype=float)
    pnabla_myy_numpy = np.zeros((n_vertex,), dtype=float)

    interior_vertices = _vertex_interior_mask(case["index_map"], case["remap_sizes"])

    for v in range(n_vertex):
        if not interior_vertices[v]:
            continue
        edges = v2e_un[v]  # list of neighbor edge indices (pad -1)
        mask = edges >= 0
        if not np.any(mask):
            continue
        e_idx = edges[mask]
        svals = zavg_un[e_idx]
        sgns = sign_un[v, mask]
        pnabla_mxx_numpy[v] = float((svals * sgns).sum())

    # divide by vertex volume
    pnabla_mxx_numpy /= vol_un

    assert np.allclose(out_unstruct.asnumpy(), pnabla_mxx_numpy, rtol=1e-10, atol=0)
