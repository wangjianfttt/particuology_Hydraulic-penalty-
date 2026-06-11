"""Small helper layer for LIGGGHTS force-coupling experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .liggghts_ctypes import LiggghtsLibrary


@dataclass
class ParticleState:
    ids: np.ndarray
    x: np.ndarray
    v: np.ndarray
    radius: np.ndarray
    omega: np.ndarray


class LiggghtsForceCoupler:
    """Library-mode DEM driver using `couple/lb/onetoone` dragforce."""

    def __init__(self, input_script: str | Path, quiet: bool = True):
        self.input_script = Path(input_script)
        self.lmp = LiggghtsLibrary(quiet=quiet)
        self.lmp.commands(self.input_script.read_text().splitlines())

    def state(self) -> ParticleState:
        n = self.lmp.natoms()
        ids = np.array(self.lmp.gather_int("id", 1, n), dtype=int)
        x = np.array(self.lmp.gather_double("x", 3, n), dtype=float).reshape(n, 3)
        v = np.array(self.lmp.gather_double("v", 3, n), dtype=float).reshape(n, 3)
        omega = np.array(self.lmp.gather_double("omega", 3, n), dtype=float).reshape(n, 3)
        radius = np.array(self.lmp.gather_double("radius", 1, n), dtype=float)
        return ParticleState(ids=ids, x=x, v=v, radius=radius, omega=omega)

    def set_dragforces(self, ids: np.ndarray, forces: np.ndarray) -> None:
        for atom_id, force in zip(ids, forces):
            fx, fy, fz = force
            self.lmp.command(
                f"set atom {int(atom_id)} property/atom dragforce {fx:.16e} {fy:.16e} {fz:.16e}"
            )

    def set_hdtorques(self, ids: np.ndarray, torques: np.ndarray) -> None:
        for atom_id, torque in zip(ids, torques):
            tx, ty, tz = torque
            self.lmp.command(
                f"set atom {int(atom_id)} property/atom hdtorque {tx:.16e} {ty:.16e} {tz:.16e}"
            )

    def run(self, steps: int) -> None:
        self.lmp.command(f"run {int(steps)}")

    def write_data(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lmp.command(f"write_data {path}")

    def close(self) -> None:
        self.lmp.close()

    def __enter__(self) -> "LiggghtsForceCoupler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
