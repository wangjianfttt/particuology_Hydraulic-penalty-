"""Auditable D3Q19 partially saturated-cell IMB kernel.

This module is intentionally compact. It mirrors the two-dimensional PSC branch
in `prototype2d.py` and provides the missing three-dimensional node-summed
force/torque path needed before local LIGGGHTS coupling can honestly be called
LBM-IMB-DEM rather than voxelized LBM plus a drag proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp

import numpy as np


C = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        [0, 1, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
    ],
    dtype=int,
)
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=float)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
CS2 = 1.0 / 3.0


@dataclass
class Particle3D:
    x: float
    y: float
    z: float
    r: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0


def _sparse_particle_alpha_task(args: tuple[Particle3D, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Build sparse PSC solid fractions for one particle.

    This helper intentionally mirrors ``SparsePscD3Q19.rebuild_geometry``. It is
    kept at module level so it can be used by ``multiprocessing`` on platforms
    that require pickleable worker functions.
    """

    p, n, subsamples = args
    ids: list[int] = []
    vals: list[float] = []
    samples = max(1, int(subsamples))
    offsets = (np.arange(samples, dtype=float) + 0.5) / samples
    ix0 = max(0, int(np.floor(p.x - p.r - 1.0)))
    ix1 = min(n, int(np.ceil(p.x + p.r + 1.0)))
    iy0 = max(0, int(np.floor(p.y - p.r - 1.0)))
    iy1 = min(n, int(np.ceil(p.y + p.r + 1.0)))
    iz0 = max(0, int(np.floor(p.z - p.r - 1.0)))
    iz1 = min(n, int(np.ceil(p.z + p.r + 1.0)))
    for ix in range(ix0, ix1):
        for iy in range(iy0, iy1):
            for iz in range(iz0, iz1):
                inside = 0
                for ox in offsets:
                    for oy in offsets:
                        for oz in offsets:
                            dx = ix + ox - p.x
                            dy = iy + oy - p.y
                            dz = iz + oz - p.z
                            inside += dx * dx + dy * dy + dz * dz <= p.r * p.r
                alpha = inside / float(samples**3)
                if alpha > 0.0:
                    ids.append((ix * n + iy) * n + iz)
                    vals.append(alpha)
    return np.asarray(ids, dtype=np.int64), np.asarray(vals, dtype=float)


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> np.ndarray:
    uu = ux * ux + uy * uy + uz * uz
    feq = np.empty((19, *rho.shape), dtype=float)
    for i, (cx, cy, cz) in enumerate(C):
        cu = cx * ux + cy * uy + cz * uz
        feq[i] = W[i] * rho * (1.0 + cu / CS2 + 0.5 * cu * cu / (CS2 * CS2) - 0.5 * uu / CS2)
    return feq


def guo_force(fx: float, fy: float, fz: float, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, tau: float) -> np.ndarray:
    term = np.empty((19, *ux.shape), dtype=float)
    prefactor = 1.0 - 0.5 / tau
    uf = ux * fx + uy * fy + uz * fz
    for i, (cx, cy, cz) in enumerate(C):
        cu = cx * ux + cy * uy + cz * uz
        cf = cx * fx + cy * fy + cz * fz
        term[i] = prefactor * W[i] * ((cf - uf) / CS2 + cu * cf / (CS2 * CS2))
    return term


class PscD3Q19:
    """Small D3Q19 PSC-IMB solver with node-summed particle loads."""

    def __init__(
        self,
        n: int,
        particles: list[Particle3D],
        tau: float = 0.72,
        subsamples: int = 4,
        body_force_z: float = 0.0,
        static_geometry: bool = False,
    ):
        self.n = int(n)
        self.particles = particles
        self.tau = float(tau)
        self.subsamples = int(subsamples)
        self.body_force_z = float(body_force_z)
        self.static_geometry = bool(static_geometry)
        self._geometry_built = False
        self.rho = np.ones((n, n, n), dtype=float)
        self.ux = np.zeros_like(self.rho)
        self.uy = np.zeros_like(self.rho)
        self.uz = np.zeros_like(self.rho)
        self.f = equilibrium(self.rho, self.ux, self.uy, self.uz)
        self.alpha_p = np.zeros((len(particles), n, n, n), dtype=float)
        self.owner_weights = np.zeros_like(self.alpha_p)
        self.solid = np.zeros_like(self.rho)
        self.imb_mx = np.zeros_like(self.rho)
        self.imb_my = np.zeros_like(self.rho)
        self.imb_mz = np.zeros_like(self.rho)
        self.hydro_forces = np.zeros((len(particles), 3), dtype=float)
        self.hydro_torques = np.zeros((len(particles), 3), dtype=float)
        self.momentum_residual = 0.0

    def rebuild_geometry(self) -> None:
        self.alpha_p.fill(0.0)
        samples = max(1, self.subsamples)
        offsets = (np.arange(samples, dtype=float) + 0.5) / samples
        for ip, p in enumerate(self.particles):
            ix0 = max(0, int(np.floor(p.x - p.r - 1.0)))
            ix1 = min(self.n, int(np.ceil(p.x + p.r + 1.0)))
            iy0 = max(0, int(np.floor(p.y - p.r - 1.0)))
            iy1 = min(self.n, int(np.ceil(p.y + p.r + 1.0)))
            iz0 = max(0, int(np.floor(p.z - p.r - 1.0)))
            iz1 = min(self.n, int(np.ceil(p.z + p.r + 1.0)))
            for ix in range(ix0, ix1):
                for iy in range(iy0, iy1):
                    for iz in range(iz0, iz1):
                        inside = 0
                        for ox in offsets:
                            for oy in offsets:
                                for oz in offsets:
                                    dx = ix + ox - p.x
                                    dy = iy + oy - p.y
                                    dz = iz + oz - p.z
                                    inside += dx * dx + dy * dy + dz * dz <= p.r * p.r
                        self.alpha_p[ip, ix, iy, iz] = inside / float(samples**3)
        alpha_sum = self.alpha_p.sum(axis=0)
        self.solid = np.clip(alpha_sum, 0.0, 1.0)
        self.owner_weights.fill(0.0)
        mask = alpha_sum > 1.0e-14
        self.owner_weights[:, mask] = self.alpha_p[:, mask] / alpha_sum[mask]
        self._geometry_built = True

    def solid_velocity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        usx = np.zeros_like(self.rho)
        usy = np.zeros_like(self.rho)
        usz = np.zeros_like(self.rho)
        alpha_sum = self.alpha_p.sum(axis=0)
        xx, yy, zz = np.indices(self.rho.shape)
        xnode = xx + 0.5
        ynode = yy + 0.5
        znode = zz + 0.5
        for ip, p in enumerate(self.particles):
            alpha = self.alpha_p[ip]
            rx = xnode - p.x
            ry = ynode - p.y
            rz = znode - p.z
            # u_s = U + omega x r
            vx = p.vx + p.wy * rz - p.wz * ry
            vy = p.vy + p.wz * rx - p.wx * rz
            vz = p.vz + p.wx * ry - p.wy * rx
            usx += alpha * vx
            usy += alpha * vy
            usz += alpha * vz
        mask = alpha_sum > 1.0e-14
        usx[mask] /= alpha_sum[mask]
        usy[mask] /= alpha_sum[mask]
        usz[mask] /= alpha_sum[mask]
        return usx, usy, usz

    def collide_stream_psc(self) -> dict[str, float]:
        if (not self.static_geometry) or (not self._geometry_built):
            self.rebuild_geometry()
        rho = np.sum(self.f, axis=0)
        ux = np.sum(self.f * C[:, 0, None, None, None], axis=0) / rho
        uy = np.sum(self.f * C[:, 1, None, None, None], axis=0) / rho
        uz = (np.sum(self.f * C[:, 2, None, None, None], axis=0) + 0.5 * self.body_force_z) / rho
        usx, usy, usz = self.solid_velocity()

        ux_eff = ux
        uy_eff = uy
        uz_eff = uz + 0.5 * self.body_force_z / rho
        feq_f = equilibrium(rho, ux_eff, uy_eff, uz_eff)
        omega_bgk = -(self.f - feq_f) / self.tau

        f_opp = self.f[OPP]
        feq_s = equilibrium(rho, usx, usy, usz)
        feq_f_opp = feq_f[OPP]
        omega_s = f_opp - self.f + feq_s - feq_f_opp

        alpha = np.clip(self.solid, 0.0, 1.0)
        b_weight = alpha / (alpha + self.tau * (1.0 - alpha) + 1.0e-14)
        collision = (1.0 - b_weight) * omega_bgk + b_weight * omega_s
        forcing = guo_force(0.0, 0.0, self.body_force_z, ux_eff, uy_eff, uz_eff, self.tau)
        self.f += collision + forcing

        self.imb_mx = np.sum((b_weight * omega_s) * C[:, 0, None, None, None], axis=0)
        self.imb_my = np.sum((b_weight * omega_s) * C[:, 1, None, None, None], axis=0)
        self.imb_mz = np.sum((b_weight * omega_s) * C[:, 2, None, None, None], axis=0)
        self.accumulate_particle_loads()

        streamed = np.empty_like(self.f)
        for i, (cx, cy, cz) in enumerate(C):
            streamed[i] = np.roll(self.f[i], shift=(cx, cy, cz), axis=(0, 1, 2))
        self.f = streamed
        self.rho = np.sum(self.f, axis=0)
        self.ux = np.sum(self.f * C[:, 0, None, None, None], axis=0) / self.rho
        self.uy = np.sum(self.f * C[:, 1, None, None, None], axis=0) / self.rho
        self.uz = (np.sum(self.f * C[:, 2, None, None, None], axis=0) + 0.5 * self.body_force_z) / self.rho
        return {
            "solid_volume_lu": float(np.sum(self.solid)),
            "mean_uz": float(np.mean(self.uz)),
            "momentum_residual": float(self.momentum_residual),
            "max_force": float(np.max(np.linalg.norm(self.hydro_forces, axis=1))) if len(self.hydro_forces) else 0.0,
            "max_torque": float(np.max(np.linalg.norm(self.hydro_torques, axis=1))) if len(self.hydro_torques) else 0.0,
        }

    def accumulate_particle_loads(self) -> None:
        xx, yy, zz = np.indices(self.rho.shape)
        xnode = xx + 0.5
        ynode = yy + 0.5
        znode = zz + 0.5
        total = np.zeros(3, dtype=float)
        self.hydro_forces.fill(0.0)
        self.hydro_torques.fill(0.0)
        for ip, p in enumerate(self.particles):
            w = self.owner_weights[ip]
            node_fx = -w * self.imb_mx
            node_fy = -w * self.imb_my
            node_fz = -w * self.imb_mz
            fx = float(node_fx.sum())
            fy = float(node_fy.sum())
            fz = float(node_fz.sum())
            rx = xnode - p.x
            ry = ynode - p.y
            rz = znode - p.z
            tx = float((ry * node_fz - rz * node_fy).sum())
            ty = float((rz * node_fx - rx * node_fz).sum())
            tz = float((rx * node_fy - ry * node_fx).sum())
            self.hydro_forces[ip] = [fx, fy, fz]
            self.hydro_torques[ip] = [tx, ty, tz]
            total += self.hydro_forces[ip]
        fluid = np.array([self.imb_mx.sum(), self.imb_my.sum(), self.imb_mz.sum()])
        self.momentum_residual = float(np.linalg.norm(total + fluid))


class SparsePscD3Q19(PscD3Q19):
    """Sparse PSC-IMB variant for local packed-bed windows.

    The dense solver above stores one ``alpha`` field per particle and is useful
    for transparent single-particle tests. Real local windows contain O(10^3)
    spheres, so here each particle owns only the lattice nodes touched by its
    bounding box. The collision is unchanged; only geometry and particle-load
    accumulation are sparse.
    """

    def __init__(
        self,
        n: int,
        particles: list[Particle3D],
        tau: float = 0.72,
        subsamples: int = 3,
        body_force_z: float = 0.0,
        static_geometry: bool = False,
        geometry_workers: int = 1,
    ):
        self.n = int(n)
        self.particles = particles
        self.tau = float(tau)
        self.subsamples = int(subsamples)
        self.body_force_z = float(body_force_z)
        self.static_geometry = bool(static_geometry)
        self.geometry_workers = max(1, int(geometry_workers))
        self._geometry_built = False
        self.rho = np.ones((n, n, n), dtype=float)
        self.ux = np.zeros_like(self.rho)
        self.uy = np.zeros_like(self.rho)
        self.uz = np.zeros_like(self.rho)
        self.f = equilibrium(self.rho, self.ux, self.uy, self.uz)
        self.solid = np.zeros_like(self.rho)
        self.alpha_sum_flat = np.zeros(n * n * n, dtype=float)
        self.node_ids: list[np.ndarray] = []
        self.alpha_vals: list[np.ndarray] = []
        self.imb_mx = np.zeros_like(self.rho)
        self.imb_my = np.zeros_like(self.rho)
        self.imb_mz = np.zeros_like(self.rho)
        self.hydro_forces = np.zeros((len(particles), 3), dtype=float)
        self.hydro_torques = np.zeros((len(particles), 3), dtype=float)
        self.momentum_residual = 0.0

    def rebuild_geometry(self) -> None:
        n = self.n
        self.alpha_sum_flat.fill(0.0)
        self.node_ids = []
        self.alpha_vals = []
        tasks = [(p, n, self.subsamples) for p in self.particles]
        if self.geometry_workers > 1 and len(tasks) > 1:
            with mp.Pool(processes=self.geometry_workers) as pool:
                results = pool.map(_sparse_particle_alpha_task, tasks)
        else:
            results = [_sparse_particle_alpha_task(task) for task in tasks]
        for ids, vals in results:
            if len(ids):
                np.add.at(self.alpha_sum_flat, ids, vals)
            self.node_ids.append(ids)
            self.alpha_vals.append(vals)
        self.solid = np.clip(self.alpha_sum_flat.reshape((n, n, n)), 0.0, 1.0)
        self._geometry_built = True

    def solid_velocity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n
        usx_flat = np.zeros(n * n * n, dtype=float)
        usy_flat = np.zeros_like(usx_flat)
        usz_flat = np.zeros_like(usx_flat)
        for p, ids, alpha in zip(self.particles, self.node_ids, self.alpha_vals):
            if len(ids) == 0:
                continue
            ix = ids // (n * n)
            rem = ids - ix * n * n
            iy = rem // n
            iz = rem - iy * n
            rx = ix + 0.5 - p.x
            ry = iy + 0.5 - p.y
            rz = iz + 0.5 - p.z
            vx = p.vx + p.wy * rz - p.wz * ry
            vy = p.vy + p.wz * rx - p.wx * rz
            vz = p.vz + p.wx * ry - p.wy * rx
            usx_flat[ids] += alpha * vx
            usy_flat[ids] += alpha * vy
            usz_flat[ids] += alpha * vz
        mask = self.alpha_sum_flat > 1.0e-14
        usx_flat[mask] /= self.alpha_sum_flat[mask]
        usy_flat[mask] /= self.alpha_sum_flat[mask]
        usz_flat[mask] /= self.alpha_sum_flat[mask]
        shape = (n, n, n)
        return usx_flat.reshape(shape), usy_flat.reshape(shape), usz_flat.reshape(shape)

    def accumulate_particle_loads(self) -> None:
        n = self.n
        imb_x = self.imb_mx.ravel()
        imb_y = self.imb_my.ravel()
        imb_z = self.imb_mz.ravel()
        total = np.zeros(3, dtype=float)
        self.hydro_forces.fill(0.0)
        self.hydro_torques.fill(0.0)
        for ip, (p, ids, alpha) in enumerate(zip(self.particles, self.node_ids, self.alpha_vals)):
            if len(ids) == 0:
                continue
            owner = alpha / np.maximum(self.alpha_sum_flat[ids], 1.0e-14)
            node_fx = -owner * imb_x[ids]
            node_fy = -owner * imb_y[ids]
            node_fz = -owner * imb_z[ids]
            fx = float(node_fx.sum())
            fy = float(node_fy.sum())
            fz = float(node_fz.sum())
            ix = ids // (n * n)
            rem = ids - ix * n * n
            iy = rem // n
            iz = rem - iy * n
            rx = ix + 0.5 - p.x
            ry = iy + 0.5 - p.y
            rz = iz + 0.5 - p.z
            tx = float((ry * node_fz - rz * node_fy).sum())
            ty = float((rz * node_fx - rx * node_fz).sum())
            tz = float((rx * node_fy - ry * node_fx).sum())
            self.hydro_forces[ip] = [fx, fy, fz]
            self.hydro_torques[ip] = [tx, ty, tz]
            total += self.hydro_forces[ip]
        fluid = np.array([self.imb_mx.sum(), self.imb_my.sum(), self.imb_mz.sum()])
        self.momentum_residual = float(np.linalg.norm(total + fluid))
