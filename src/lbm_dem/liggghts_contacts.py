"""Utilities for LIGGGHTS pair/gran/local contact-force dumps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ContactTable:
    """Column view of a `pair/gran/local` dump used for force-chain analysis."""

    ids: np.ndarray
    total_force: np.ndarray
    normal_force: np.ndarray
    tangential_force: np.ndarray
    torque_over_radius: np.ndarray
    overlap: np.ndarray
    contact_point: np.ndarray

    @property
    def force_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.total_force, axis=1)

    @property
    def normal_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.normal_force, axis=1)

    @property
    def tangential_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.tangential_force, axis=1)


def load_liggghts_local_frames(path: str | Path) -> list[np.ndarray]:
    """Read numeric frames from a LIGGGHTS `dump local` text file."""

    frames: list[np.ndarray] = []
    rows: list[list[float]] = []
    in_entries = False
    with Path(path).open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("ITEM: ENTRIES"):
                in_entries = True
                rows = []
                continue
            if stripped.startswith("ITEM:"):
                if in_entries:
                    frames.append(np.asarray(rows, dtype=float) if rows else np.empty((0, 0), dtype=float))
                in_entries = False
                continue
            if in_entries:
                rows.append([float(item) for item in stripped.split()])
    if in_entries:
        frames.append(np.asarray(rows, dtype=float) if rows else np.empty((0, 0), dtype=float))
    return frames


def load_liggghts_local_dump(path: str | Path, frame: int = -1) -> np.ndarray:
    """Read one numeric frame from a LIGGGHTS `dump local` text file.

    By default the last frame is returned. This avoids double-counting the same
    contact when a smoke test dumps both timestep 0 and timestep 1.
    """

    frames = load_liggghts_local_frames(path)
    if not frames:
        return np.empty((0, 0), dtype=float)
    try:
        return frames[frame]
    except IndexError as exc:
        raise IndexError(f"Frame {frame} is out of range for {len(frames)} frames in {path}") from exc


def parse_pair_gran_local(table: np.ndarray, has_index: bool = False) -> ContactTable:
    """Parse the column order used by `library_contact_local.in`.

    Expected compute order is
    `id force force_normal force_tangential torque delta contactPoint`.
    If the dump includes an initial `index` column, pass `has_index=True`.
    """

    if table.size == 0:
        empty_i = np.empty((0, 2), dtype=int)
        empty_v = np.empty((0, 3), dtype=float)
        return ContactTable(empty_i, empty_v, empty_v, empty_v, empty_v, np.empty(0), empty_v)
    offset = 1 if has_index else 0
    expected = offset + 19
    if table.ndim != 2 or table.shape[1] < expected:
        raise ValueError(f"Expected at least {expected} columns, got {table.shape}")
    core = table[:, offset : offset + 19]
    return ContactTable(
        ids=core[:, :2].astype(int),
        total_force=core[:, 3:6],
        normal_force=core[:, 6:9],
        tangential_force=core[:, 9:12],
        torque_over_radius=core[:, 12:15],
        overlap=core[:, 15],
        contact_point=core[:, 16:19],
    )


def contact_components(ids: np.ndarray) -> list[list[int]]:
    """Return connected contact components by atom ID."""

    atom_ids = sorted({int(i) for row in ids for i in row[:2]})
    if not atom_ids:
        return []
    index = {atom_id: n for n, atom_id in enumerate(atom_ids)}
    parent = list(range(len(atom_ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(index[int(a)]), find(index[int(b)])
        if ra != rb:
            parent[rb] = ra

    for a, b in ids[:, :2]:
        union(int(a), int(b))

    groups: dict[int, list[int]] = {}
    for atom_id in atom_ids:
        groups.setdefault(find(index[atom_id]), []).append(atom_id)
    return list(groups.values())


def contact_graph_metrics(contacts: ContactTable) -> dict[str, float]:
    """Compact force-chain metrics for smoke tests and DEM blockage outputs."""

    components = contact_components(contacts.ids)
    largest = max((len(comp) for comp in components), default=0)
    return {
        "n_contacts": float(len(contacts.overlap)),
        "n_components": float(len(components)),
        "largest_component": float(largest),
        "max_force": float(np.max(contacts.force_magnitude)) if len(contacts.overlap) else 0.0,
        "sum_force": float(np.sum(contacts.force_magnitude)) if len(contacts.overlap) else 0.0,
        "max_overlap": float(np.max(contacts.overlap)) if len(contacts.overlap) else 0.0,
    }
