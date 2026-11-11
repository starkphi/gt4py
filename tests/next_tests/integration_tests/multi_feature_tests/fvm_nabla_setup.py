# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np

try:
    from atlas4py import (
        Config,
        StructuredGrid,
        StructuredMeshGenerator,
        Topology,
        build_edges,
        build_median_dual_mesh,
        build_node_to_edge_connectivity,
        functionspace,
    )
except ImportError as error:
    # Stay importable so collection succeeds and `requires_atlas` can skip the tests.
    # `nabla_setup` re-raises the error, since the marker is probed with `find_spec`
    # and would not skip an `atlas4py` that is installed but not loadable.
    _ATLAS_IMPORT_ERROR: ImportError | None = error
    Config = StructuredGrid = StructuredMeshGenerator = Topology = None
    build_edges = build_median_dual_mesh = build_node_to_edge_connectivity = None
    functionspace = None
else:
    _ATLAS_IMPORT_ERROR = None

from gt4py import next as gtx
from gt4py.next.iterator import atlas_utils
from typing import Optional


Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)

V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))


def assert_close(expected, actual):
    assert np.allclose(expected, actual), "expected={}, actual={}".format(expected, actual)


class nabla_setup:
    @staticmethod
    def _default_config():
        config = Config()
        config["triangulate"] = True
        config["angle"] = 20.0
        return config

    def __init__(self, *, allocator, grid=None, config=None):
        if _ATLAS_IMPORT_ERROR is not None:
            # Fresh exception: re-raising one object grows its traceback per call.
            raise ImportError(str(_ATLAS_IMPORT_ERROR)) from _ATLAS_IMPORT_ERROR
        if grid is None:
            # Not a default argument: those are evaluated at import time.
            grid = StructuredGrid("O32")
        if config is None:
            config = self._default_config()
        self.allocator = allocator
        mesh = StructuredMeshGenerator(config).generate(grid)

        fs_edges = functionspace.EdgeColumns(mesh, halo=1)
        fs_nodes = functionspace.NodeColumns(mesh, halo=1)

        build_edges(mesh)
        build_node_to_edge_connectivity(mesh)
        build_median_dual_mesh(mesh)

        edges_per_node = max(
            [mesh.nodes.edge_connectivity.cols(node) for node in range(0, fs_nodes.size)]
        )

        self.mesh = mesh
        self.fs_edges = fs_edges
        self.fs_nodes = fs_nodes
        self.edges_per_node = edges_per_node

        # Optional injected data (used by from_connectivity)
        self._edges2node_connectivity = None
        self._nodes2edge_connectivity = None
        self._sign_field = None
        self._S_fields = None
        self._vol_field = None
        self._input_field = None
        self._nodes_size = None
        self._edges_size = None

    @staticmethod
    def _build_v2e_from_e2v(e2v: np.ndarray, nodes_size: int, edges_per_node: Optional[int] = None):
        deg = np.zeros((nodes_size,), dtype=np.int32)
        for e in range(e2v.shape[0]):
            v0, v1 = int(e2v[e, 0]), int(e2v[e, 1])
            deg[v0] += 1
            deg[v1] += 1
        max_deg = int(deg.max()) if edges_per_node is None else int(edges_per_node)

        v2e = np.full((nodes_size, max_deg), -1, dtype=np.int32)
        fill = np.zeros((nodes_size,), dtype=np.int32)
        for e in range(e2v.shape[0]):
            for v in (int(e2v[e, 0]), int(e2v[e, 1])):
                k = fill[v]
                if k >= max_deg:
                    raise ValueError("edges_per_node too small for provided connectivity.")
                v2e[v, k] = e
                fill[v] += 1
        return v2e, max_deg

    @classmethod
    def from_connectivity(
        cls,
        *,
        allocator,
        e2v: np.ndarray,
        nodes_size: Optional[int] = None,
        v2e: Optional[np.ndarray] = None,
        edges_per_node: Optional[int] = None,
        lonlat_deg: Optional[np.ndarray] = None,  # (n_vertex,2), optional
        dual_normals: Optional[np.ndarray] = None,  # (n_edge,2), optional
        dual_volumes: Optional[np.ndarray] = None,  # (n_vertex,), optional
        input_values: Optional[np.ndarray] = None,  # (n_vertex,), optional
    ):
        """
        Build a setup object directly from connectivity (+ optional geometry/fields).

        If dual_normals / dual_volumes / lonlat are missing, synthetic fields are used.
        """
        obj = cls.__new__(cls)
        obj.allocator = allocator
        obj.mesh = None
        obj.fs_edges = None
        obj.fs_nodes = None

        e2v = np.asarray(e2v, dtype=np.int32)
        if e2v.ndim != 2 or e2v.shape[1] != 2:
            raise ValueError("e2v must have shape (n_edge, 2).")

        n_edge = int(e2v.shape[0])
        n_vertex = int(nodes_size if nodes_size is not None else (e2v.max() + 1))

        if v2e is None:
            v2e, max_deg = cls._build_v2e_from_e2v(e2v, n_vertex, edges_per_node)
        else:
            v2e = np.asarray(v2e, dtype=np.int32)
            if v2e.ndim != 2:
                raise ValueError("v2e must have shape (n_vertex, max_deg).")
            max_deg = int(v2e.shape[1])

        obj.edges_per_node = max_deg  # should be 6
        obj._nodes_size = n_vertex
        obj._edges_size = n_edge

        obj._edges2node_connectivity = gtx.as_connectivity(
            domain={Edge: n_edge, E2VDim: 2},
            codomain=Vertex,
            data=e2v,
            allocator=allocator,
        )
        obj._nodes2edge_connectivity = gtx.as_connectivity(
            domain={Vertex: n_vertex, V2EDim: max_deg},
            codomain=Edge,
            data=v2e,
            allocator=allocator,
        )

        # sign from e2v/v2e: +1 at first endpoint, -1 at second endpoint
        sign = np.zeros((n_vertex, max_deg), dtype=np.float64)
        for v in range(n_vertex):
            for j in range(max_deg):
                e = int(v2e[v, j])
                if e < 0:
                    # print(f"Warning: vertex {v} has no edge at position {j} (v2e[v,j]={e}).")
                    continue
                sign[v, j] = 1.0 if int(e2v[e, 0]) == v else -1.0
                # print(f"v={v}, j={j}, e={e}, sign={sign[v,j]}")
        # print("raw sign: ", sign)
        obj._sign_field = gtx.as_field([Vertex, V2EDim], sign, allocator=allocator)
        # print("sign: ", obj._sign_field[:, :].asnumpy())

        # S fields
        if dual_normals is not None:
            S = np.asarray(dual_normals, dtype=np.float64)
            if S.shape != (n_edge, 2):
                raise ValueError("dual_normals must have shape (n_edge, 2).")
            rpi = 2.0 * math.asin(1.0)
            radius = 6371.22e03
            deg2rad = 2.0 * rpi / 360.0
            S_MXX = S[:, 0] * radius * deg2rad
            S_MYY = S[:, 1] * radius * deg2rad
        else:
            # synthetic fallback (for plumbing tests only)
            S_MXX = np.ones((n_edge,), dtype=np.float64)
            S_MYY = np.ones((n_edge,), dtype=np.float64)

        obj._S_fields = (
            gtx.as_field([Edge], S_MXX, allocator=allocator),
            gtx.as_field([Edge], S_MYY, allocator=allocator),
        )

        # vol field
        if dual_volumes is not None:
            vol_atlas = np.asarray(dual_volumes, dtype=np.float64)
            if vol_atlas.shape != (n_vertex,):
                raise ValueError("dual_volumes must have shape (n_vertex,).")
            rpi = 2.0 * math.asin(1.0)
            radius = 6371.22e03
            deg2rad = 2.0 * rpi / 360.0
            vol = vol_atlas * (deg2rad**2) * (radius**2)
        else:
            vol = np.ones((n_vertex,), dtype=np.float64)

        obj._vol_field = gtx.as_field([Vertex], vol, allocator=allocator)

        # input field
        if input_values is not None:
            inp = np.asarray(input_values, dtype=np.float64)
            if inp.shape != (n_vertex,):
                raise ValueError("input_values must have shape (n_vertex,).")
        elif lonlat_deg is not None:
            # same analytic formula as original setup, but without Atlas functionspace fields
            lonlat = np.asarray(lonlat_deg, dtype=np.float64)
            if lonlat.shape != (n_vertex, 2):
                raise ValueError("lonlat_deg must have shape (n_vertex, 2).")

            rpi = 2.0 * math.asin(1.0)
            radius = 6371.22e03
            deg2rad = 2.0 * rpi / 360.0
            zh0 = 2000.0
            zrad = 3.0 * rpi / 4.0 * radius
            zeta = rpi / 16.0 * radius
            zlatc = 0.0
            zlonc = 3.0 * rpi / 2.0

            lon = lonlat[:, 0] * deg2rad
            lat = lonlat[:, 1] * deg2rad
            rcosa = np.cos(lat)
            rsina = np.sin(lat)

            inp = np.zeros((n_vertex,), dtype=np.float64)
            for v in range(n_vertex):
                zdist = math.sin(zlatc) * rsina[v] + math.cos(zlatc) * rcosa[v] * math.cos(
                    lon[v] - zlonc
                )
                zdist = radius * math.acos(zdist)
                if zdist < zrad:
                    inp[v] = (
                        0.5
                        * zh0
                        * (1.0 + math.cos(rpi * zdist / zrad))
                        * (math.cos(rpi * zdist / zeta) ** 2)
                    )
        else:
            inp = np.zeros((n_vertex,), dtype=np.float64)

        obj._input_field = gtx.as_field([Vertex], inp, allocator=allocator)
        return obj

    @property
    def edges2node_connectivity(self) -> gtx.Connectivity:
        if self._edges2node_connectivity is not None:
            return self._edges2node_connectivity
        return gtx.as_connectivity(
            domain={Edge: self.edges_size, E2VDim: 2},
            codomain=Vertex,
            data=atlas_utils.AtlasTable(self.mesh.edges.node_connectivity).asnumpy(),
            allocator=self.allocator,
        )

    @property
    def nodes2edge_connectivity(self) -> gtx.Connectivity:
        if self._nodes2edge_connectivity is not None:
            return self._nodes2edge_connectivity
        return gtx.as_connectivity(
            domain={Vertex: self.nodes_size, V2EDim: self.edges_per_node},
            codomain=Edge,
            data=atlas_utils.AtlasTable(self.mesh.nodes.edge_connectivity).asnumpy(),
            allocator=self.allocator,
        )

    @property
    def nodes_size(self):
        return self._nodes_size if self._nodes_size is not None else self.fs_nodes.size

    @property
    def edges_size(self):
        return self._edges_size if self._edges_size is not None else self.fs_edges.size

    @staticmethod
    def _is_pole_edge(e, edge_flags):
        return Topology.check(edge_flags[e], Topology.POLE)

    @property
    def is_pole_edge_field(self) -> gtx.Field:
        edge_flags = np.array(self.mesh.edges.flags(), copy=False)

        pole_edge_field = np.zeros((self.edges_size,), dtype=bool)
        for e in range(self.edges_size):
            pole_edge_field[e] = self._is_pole_edge(e, edge_flags)
        return gtx.as_field([Edge], pole_edge_field, allocator=self.allocator)

    @property
    def sign_field(self) -> gtx.Field:
        if self._sign_field is not None:
            return self._sign_field
        node2edge_sign = np.zeros((self.nodes_size, self.edges_per_node))
        edge_flags = np.array(self.mesh.edges.flags(), copy=False)

        for jnode in range(0, self.nodes_size):
            node_edge_con = self.mesh.nodes.edge_connectivity
            edge_node_con = self.mesh.edges.node_connectivity
            for jedge in range(0, node_edge_con.cols(jnode)):
                iedge = node_edge_con[jnode, jedge]
                ip1 = edge_node_con[iedge, 0]
                if jnode == ip1:
                    node2edge_sign[jnode, jedge] = 1.0
                else:
                    node2edge_sign[jnode, jedge] = -1.0
                    if self._is_pole_edge(iedge, edge_flags):
                        node2edge_sign[jnode, jedge] = 1.0
        return gtx.as_field([Vertex, V2EDim], node2edge_sign, allocator=self.allocator)

    @property
    def S_fields(self) -> tuple[gtx.Field, gtx.Field]:
        if self._S_fields is not None:
            return self._S_fields
        S = np.array(self.mesh.edges.field("dual_normals"), copy=False)
        S_MXX = np.zeros((self.edges_size))
        S_MYY = np.zeros((self.edges_size))

        MXX = 0
        MYY = 1

        rpi = 2.0 * math.asin(1.0)
        radius = 6371.22e03
        deg2rad = 2.0 * rpi / 360.0

        for i in range(0, self.edges_size):
            S_MXX[i] = S[i, MXX] * radius * deg2rad
            S_MYY[i] = S[i, MYY] * radius * deg2rad

        assert math.isclose(min(S_MXX), -103437.60479272791)
        assert math.isclose(max(S_MXX), 340115.33913622628)
        assert math.isclose(min(S_MYY), -2001577.7946404363)
        assert math.isclose(max(S_MYY), 2001577.7946404363)

        return gtx.as_field([Edge], S_MXX, allocator=self.allocator), gtx.as_field(
            [Edge], S_MYY, allocator=self.allocator
        )

    @property
    def vol_field(self) -> gtx.Field:
        if self._vol_field is not None:
            return self._vol_field
        rpi = 2.0 * math.asin(1.0)
        radius = 6371.22e03
        deg2rad = 2.0 * rpi / 360.0
        vol_atlas = np.array(self.mesh.nodes.field("dual_volumes"), copy=False)
        # dual_volumes 4.6510228700066421    68.891611253882218    12.347560975609632
        assert_close(4.6510228700066421, min(vol_atlas))
        assert_close(68.891611253882218, max(vol_atlas))

        vol = np.zeros((vol_atlas.size))
        for i in range(0, vol_atlas.size):
            vol[i] = vol_atlas[i] * pow(deg2rad, 2) * pow(radius, 2)
        # VOL(min/max):  57510668192.214096    851856184496.32886
        assert_close(57510668192.214096, min(vol))
        assert_close(851856184496.32886, max(vol))
        return gtx.as_field([Vertex], vol, allocator=self.allocator)

    @property
    def input_field(self) -> gtx.Field:
        if self._input_field is not None:
            return self._input_field
        klevel = 0
        MXX = 0
        MYY = 1
        rpi = 2.0 * math.asin(1.0)
        radius = 6371.22e03
        deg2rad = 2.0 * rpi / 360.0

        zh0 = 2000.0
        zrad = 3.0 * rpi / 4.0 * radius
        zeta = rpi / 16.0 * radius
        zlatc = 0.0
        zlonc = 3.0 * rpi / 2.0

        m_rlonlatcr = self.fs_nodes.create_field(
            name="m_rlonlatcr", levels=1, dtype=np.float64, variables=self.edges_per_node
        )
        rlonlatcr = np.array(m_rlonlatcr, copy=False)

        m_rcoords = self.fs_nodes.create_field(
            name="m_rcoords", levels=1, dtype=np.float64, variables=self.edges_per_node
        )
        rcoords = np.array(m_rcoords, copy=False)

        m_rcosa = self.fs_nodes.create_field(name="m_rcosa", levels=1, dtype=np.float64)
        rcosa = np.array(m_rcosa, copy=False)

        m_rsina = self.fs_nodes.create_field(name="m_rsina", levels=1, dtype=np.float64)
        rsina = np.array(m_rsina, copy=False)

        m_pp = self.fs_nodes.create_field(name="m_pp", levels=1, dtype=np.float64)
        rzs = np.array(m_pp, copy=False)

        rcoords_deg = np.array(self.mesh.nodes.field("lonlat"), copy=False)

        for jnode in range(0, self.nodes_size):
            for i in range(0, 2):
                rcoords[jnode, klevel, i] = rcoords_deg[jnode, i] * deg2rad
                rlonlatcr[jnode, klevel, i] = rcoords[jnode, klevel, i]  # This is not my pattern!
            rcosa[jnode, klevel] = math.cos(rlonlatcr[jnode, klevel, MYY])
            rsina[jnode, klevel] = math.sin(rlonlatcr[jnode, klevel, MYY])
        for jnode in range(0, self.nodes_size):
            zlon = rlonlatcr[jnode, klevel, MXX]
            zdist = math.sin(zlatc) * rsina[jnode, klevel] + math.cos(zlatc) * rcosa[
                jnode, klevel
            ] * math.cos(zlon - zlonc)
            zdist = radius * math.acos(zdist)
            rzs[jnode, klevel] = 0.0
            if zdist < zrad:
                rzs[jnode, klevel] = rzs[jnode, klevel] + 0.5 * zh0 * (
                    1.0 + math.cos(rpi * zdist / zrad)
                ) * math.pow(math.cos(rpi * zdist / zeta), 2)

        assert_close(0.0000000000000000, min(rzs))
        assert_close(1965.4980340735883, max(rzs))

        return gtx.as_field([Vertex], rzs[:, klevel], allocator=self.allocator)
