"""
Icosahedral Spherical Mesh Module for MAUSAM.
Eliminates pole singularities and planar distortion by mapping atmospheric fields onto a spherical geodesic graph.
"""
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False

from scipy.spatial import KDTree
from typing import Tuple, List, Dict, Any, Optional

class IcosahedralMesh:
    def __init__(self, subdivision_level: int = 3, radius: float = 6371.0):
        """
        Creates an icosahedral geodesic mesh for the spherical Earth.
        radius: Earth radius in km (~6371 km).
        """
        self.subdivision_level = subdivision_level
        self.radius = radius
        self.vertices, self.faces = self._generate_icosphere(subdivision_level)
        self.num_nodes = len(self.vertices)
        
        # Convert vertices to spherical lat, lon
        self.lats, self.lons = self._cartesian_to_latlon(self.vertices)
        
        # Build spatial KDTree for fast nearest-neighbor lookups
        self.kdtree = KDTree(self.vertices)
        
        # Build edge index for GNN message passing [2, num_edges]
        self.edge_index = self._build_edge_index()
        self.graph = self._build_networkx_graph()

    def _generate_icosphere(self, level: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate icosahedral mesh subdivided to specified level."""
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        
        # 12 initial vertices of an icosahedron
        v = np.array([
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
        ], dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        
        vertices = list(v)
        midpoint_cache = {}

        def get_midpoint(i1, i2):
            key = tuple(sorted([i1, i2]))
            if key in midpoint_cache:
                return midpoint_cache[key]
            p1 = vertices[i1]
            p2 = vertices[i2]
            mid = (p1 + p2) / 2.0
            mid /= np.linalg.norm(mid)
            idx = len(vertices)
            vertices.append(mid)
            midpoint_cache[key] = idx
            return idx

        for _ in range(level):
            new_faces = []
            for tri in faces:
                a = get_midpoint(tri[0], tri[1])
                b = get_midpoint(tri[1], tri[2])
                c = get_midpoint(tri[2], tri[0])
                new_faces.extend([
                    [tri[0], a, c],
                    [tri[1], b, a],
                    [tri[2], c, b],
                    [a, b, c]
                ])
            faces = new_faces

        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int64)

    def _cartesian_to_latlon(self, verts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 3D unit coordinates to latitude (-90 to +90) and longitude (-180 to 180)."""
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        lats = np.arcsin(np.clip(z, -1.0, 1.0)) * 180.0 / np.pi
        lons = np.arctan2(y, x) * 180.0 / np.pi
        return lats, lons

    def latlon_to_cartesian(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Convert lat, lon (in degrees) to 3D unit sphere cartesian vectors."""
        phi = np.radians(lats)
        theta = np.radians(lons)
        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)
        return np.column_stack([x, y, z])

    def _build_edge_index(self):
        """Build bidirectional edge connections for GNN message passing."""
        edges = set()
        for face in self.faces:
            for i in range(3):
                u, v = face[i], face[(i + 1) % 3]
                edges.add((u, v))
                edges.add((v, u))
        edge_list = list(edges)
        u_nodes = [e[0] for e in edge_list]
        v_nodes = [e[1] for e in edge_list]
        if TORCH_AVAILABLE and torch is not None:
            return torch.tensor([u_nodes, v_nodes], dtype=torch.long)
        return np.array([u_nodes, v_nodes], dtype=np.int64)

    def _build_networkx_graph(self):
        """Construct NetworkX graph representing the spherical mesh connectivity."""
        if not NETWORKX_AVAILABLE or nx is None:
            adj = {i: [] for i in range(self.num_nodes)}
            for face in self.faces:
                for i in range(3):
                    u, v = int(face[i]), int(face[(i + 1) % 3])
                    adj[u].append(v)
            return adj
        G = nx.Graph()
        for i in range(self.num_nodes):
            G.add_node(i, lat=float(self.lats[i]), lon=float(self.lons[i]))
        for face in self.faces:
            for i in range(3):
                u, v = int(face[i]), int(face[(i + 1) % 3])
                G.add_edge(u, v)
        return G

    def project_grid_to_mesh(self, grid_lats: np.ndarray, grid_lons: np.ndarray, field: np.ndarray) -> np.ndarray:
        """
        Projects 2D regular atmospheric field (H, W) onto the icosahedral spherical mesh nodes.
        Uses fast KDTree spherical vector matching.
        """
        grid_pts = self.latlon_to_cartesian(grid_lats.ravel(), grid_lons.ravel())
        grid_vals = field.ravel()
        
        # Query nearest grid point for each mesh node
        grid_tree = KDTree(grid_pts)
        _, indices = grid_tree.query(self.vertices, k=1)
        mesh_values = grid_vals[indices]
        return mesh_values.astype(np.float32)

    def project_mesh_to_grid(self, mesh_data: np.ndarray, target_lats: np.ndarray, target_lons: np.ndarray) -> np.ndarray:
        """
        Interpolates spherical mesh data back onto target regular lat-lon coordinates.
        """
        target_pts = self.latlon_to_cartesian(target_lats.ravel(), target_lons.ravel())
        _, indices = self.kdtree.query(target_pts, k=1)
        grid_values = mesh_data[indices]
        return grid_values.reshape(target_lats.shape)
