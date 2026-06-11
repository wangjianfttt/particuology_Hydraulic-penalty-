"""Minimal Python 3 ctypes bindings for a LIGGGHTS/LAMMPS-style library."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable


DEFAULT_LIB = Path(os.environ.get("LIGGGHTS_LIBRARY", "libliggghts.so"))


class LiggghtsLibrary:
    """Tiny wrapper around the exported C API.

    It intentionally covers only the smoke-test path first: open, command,
    gather_atoms and close. Force-transfer helpers should be added only after
    the chosen LIGGGHTS fix (`external` or `couple/lb/onetoone`) is verified.
    """

    def __init__(self, lib_path: str | Path = DEFAULT_LIB, args: Iterable[str] | None = None, quiet: bool = True):
        self.lib_path = Path(lib_path)
        self.lib = ctypes.CDLL(str(self.lib_path))
        self.handle = ctypes.c_void_p()
        self._configure_signatures()

        argv_items = ["liggghts"]
        if quiet:
            argv_items.extend(["-screen", "none", "-log", "none"])
        if args:
            argv_items.extend(args)
        self._argv_keepalive = [ctypes.create_string_buffer(a.encode("utf-8")) for a in argv_items]
        argv = (ctypes.c_char_p * len(self._argv_keepalive))()
        argv[:] = [ctypes.cast(item, ctypes.c_char_p) for item in self._argv_keepalive]
        self.lib.lammps_open_no_mpi(len(argv_items), argv, ctypes.byref(self.handle))
        if not self.handle:
            raise RuntimeError("lammps_open_no_mpi returned a null handle")

    def _configure_signatures(self) -> None:
        self.lib.lammps_open_no_mpi.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_void_p)]
        self.lib.lammps_open_no_mpi.restype = None
        self.lib.lammps_close.argtypes = [ctypes.c_void_p]
        self.lib.lammps_close.restype = None
        self.lib.lammps_command.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.lammps_command.restype = ctypes.c_char_p
        self.lib.lammps_get_natoms.argtypes = [ctypes.c_void_p]
        self.lib.lammps_get_natoms.restype = ctypes.c_int
        self.lib.lammps_gather_atoms.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.lammps_gather_atoms.restype = None
        self.lib.lammps_extract_atom.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.lammps_extract_atom.restype = ctypes.c_void_p

    def command(self, text: str) -> None:
        self.lib.lammps_command(self.handle, text.encode("utf-8"))

    def commands(self, lines: Iterable[str]) -> None:
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                self.command(stripped)

    def gather_double(self, name: str, count: int, nlocal: int) -> list[float]:
        array_type = ctypes.c_double * (nlocal * count)
        buffer = array_type()
        self.lib.lammps_gather_atoms(self.handle, name.encode("utf-8"), 1, count, ctypes.byref(buffer))
        return list(buffer)

    def gather_int(self, name: str, count: int, nlocal: int) -> list[int]:
        array_type = ctypes.c_int * (nlocal * count)
        buffer = array_type()
        self.lib.lammps_gather_atoms(self.handle, name.encode("utf-8"), 0, count, ctypes.byref(buffer))
        return list(buffer)

    def natoms(self) -> int:
        return int(self.lib.lammps_get_natoms(self.handle))

    def close(self) -> None:
        if self.handle:
            self.lib.lammps_close(self.handle)
            self.handle = ctypes.c_void_p()

    def __enter__(self) -> "LiggghtsLibrary":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
