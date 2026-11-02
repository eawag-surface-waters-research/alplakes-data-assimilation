"""Simstrat's binary snapshot (simulation-snapshot.dat): read and write it from Python.

The snapshot is a Fortran sequential unformatted file written by save_snapshot in Simstrat's
src/simstrat.f90. This mirrors its write order exactly, so a read-then-write round trip is
byte-identical. read_snapshot returns a Snapshot whose sections are OrderedDicts named as the
Fortran fields.
"""
from __future__ import annotations

import os
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# --- Fortran unformatted record I/O --------------------------------------
# Minimal drop-in for scipy.io.FortranFile, supporting exactly the calls the snapshot helpers make
# (read_ints / read_reals / write_record of a single array). It exists so snapshot I/O does NOT
# import scipy: `from scipy.io import FortranFile` costs ~2 s per interpreter, which the OpenDA
# wrapper paid on every member every step (a fresh process each time) — the bulk of the per-step
# "snapshot inject" storm. The native engine paid it once. numpy + struct only here.
#
# A Fortran sequential unformatted record is  [int32 nbytes][data][int32 nbytes]  with the marker in
# native byte order and 4-byte width (gfortran default) — matching scipy's defaults, so files stay
# byte-identical. read_* return writable copies (scipy semantics: callers mutate model['T'] in place).
class FortranFile:
    _MARK = np.dtype(np.uint32)   # record-length marker: uint32, native byte order (scipy default)

    def __init__(self, path, mode="r"):
        self._f = open(path, "rb" if mode == "r" else "wb")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()

    def close(self):
        self._f.close()

    def _read_record(self) -> bytes:
        head = self._f.read(4)
        if len(head) < 4:
            raise EOFError("end of Fortran file")
        n = int(np.frombuffer(head, dtype=self._MARK)[0])
        data = self._f.read(n)
        tail = self._f.read(4)
        if len(data) != n or len(tail) < 4 or int(np.frombuffer(tail, dtype=self._MARK)[0]) != n:
            raise ValueError("corrupt Fortran record (length-marker mismatch)")
        return data

    def read_ints(self, dtype=np.int32) -> np.ndarray:
        return np.frombuffer(self._read_record(), dtype=dtype).copy()

    def read_reals(self, dtype=np.float64) -> np.ndarray:
        return np.frombuffer(self._read_record(), dtype=dtype).copy()

    def write_record(self, arr: np.ndarray) -> None:
        b = np.ascontiguousarray(arr).tobytes()
        marker = np.array(len(b), dtype=self._MARK).tobytes()
        self._f.write(marker)
        self._f.write(b)
        self._f.write(marker)


# --- low-level record helpers --------------------------------------------

def _read_array(f: FortranFile) -> Tuple[np.ndarray, Tuple[int, int]]:
    lb, ub = f.read_ints(np.int32)
    n = ub - lb + 1
    data = f.read_reals(np.float64)
    if data.size != n:
        raise ValueError(
            f"array length mismatch: bounds [{lb},{ub}] imply {n}, got {data.size}"
        )
    return data, (int(lb), int(ub))


def _write_array(f: FortranFile, data: np.ndarray, bounds: Tuple[int, int]) -> None:
    lb, ub = bounds
    if data.size != ub - lb + 1:
        raise ValueError("array length does not match bounds")
    f.write_record(np.array([lb, ub], dtype=np.int32))
    f.write_record(np.ascontiguousarray(data, dtype=np.float64))


def _read_matrix(f: FortranFile) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    lb1, ub1, lb2, ub2 = f.read_ints(np.int32)
    rows, cols = ub1 - lb1 + 1, ub2 - lb2 + 1
    flat = f.read_reals(np.float64)
    if flat.size != rows * cols:
        raise ValueError(
            f"matrix size mismatch: bounds imply {rows}x{cols}={rows*cols}, "
            f"got {flat.size}"
        )
    mat = flat.reshape((rows, cols), order="F")
    return mat, (int(lb1), int(ub1), int(lb2), int(ub2))


def _write_matrix(
    f: FortranFile, mat: np.ndarray, bounds: Tuple[int, int, int, int]
) -> None:
    lb1, ub1, lb2, ub2 = bounds
    rows, cols = ub1 - lb1 + 1, ub2 - lb2 + 1
    if mat.shape != (rows, cols):
        raise ValueError(
            f"matrix shape {mat.shape} does not match bounds {bounds}"
        )
    f.write_record(np.array([lb1, ub1, lb2, ub2], dtype=np.int32))
    f.write_record(np.asfortranarray(mat, dtype=np.float64).ravel(order="F"))


def _read_any_array(f: FortranFile):
    """Read a 1-D array (2-integer bounds) or 2-D matrix (4-integer bounds)."""
    bounds = f.read_ints(np.int32)
    data = f.read_reals(np.float64)
    if len(bounds) == 2:
        lb, ub = int(bounds[0]), int(bounds[1])
        n = ub - lb + 1
        if data.size != n:
            raise ValueError(
                f"array length mismatch: bounds [{lb},{ub}] imply {n}, got {data.size}"
            )
        return data, (lb, ub)
    elif len(bounds) == 4:
        lb1, ub1, lb2, ub2 = (int(b) for b in bounds)
        rows, cols = ub1 - lb1 + 1, ub2 - lb2 + 1
        if data.size != rows * cols:
            raise ValueError(
                f"matrix size mismatch: bounds imply {rows}x{cols}={rows*cols}, "
                f"got {data.size}"
            )
        return data.reshape((rows, cols), order="F"), (lb1, ub1, lb2, ub2)
    else:
        raise ValueError(f"Unexpected bounds record with {len(bounds)} integers")


def _write_any_array(f: FortranFile, data: np.ndarray, bounds) -> None:
    """Write a 1-D array or 2-D matrix depending on bounds length."""
    if len(bounds) == 2:
        _write_array(f, data.ravel(), bounds)
    else:
        _write_matrix(f, data, bounds)


def _read_int_array(f: FortranFile) -> Tuple[np.ndarray, Tuple[int, int]]:
    lb, ub = f.read_ints(np.int32)
    data = f.read_ints(np.int32)
    return data, (int(lb), int(ub))


def _write_int_array(f: FortranFile, data: np.ndarray, bounds: Tuple[int, int]) -> None:
    lb, ub = bounds
    f.write_record(np.array([lb, ub], dtype=np.int32))
    f.write_record(np.ascontiguousarray(data, dtype=np.int32))


def _read_logical_array(f: FortranFile) -> Tuple[np.ndarray, Tuple[int, int]]:
    # gfortran default logical is 4 bytes
    lb, ub = f.read_ints(np.int32)
    data = f.read_ints(np.int32)
    return data.astype(bool), (int(lb), int(ub))


def _write_logical_array(f: FortranFile, data: np.ndarray, bounds: Tuple[int, int]) -> None:
    lb, ub = bounds
    f.write_record(np.array([lb, ub], dtype=np.int32))
    f.write_record(np.asarray(data, dtype=bool).astype(np.int32))


# --- snapshot container --------------------------------------------------

@dataclass
class Snapshot:
    couple_aed2: bool = False
    inflow_mode: int = 0
    has_lateral_state: bool = False
    model: "OrderedDict[str, object]" = field(default_factory=OrderedDict)
    grid: "OrderedDict[str, object]" = field(default_factory=OrderedDict)
    absorption: "OrderedDict[str, object]" = field(default_factory=OrderedDict)
    lateral: "OrderedDict[str, object]" = field(default_factory=OrderedDict)
    logger: "OrderedDict[str, object]" = field(default_factory=OrderedDict)


# --- model state ---------------------------------------------------------

# Mirror src/strat_simdata.f90:370-433. Three groups of fields:
#   - 1D arrays written via save_array / save_array_pointer (bounds + data)
#   - scalar groups written as single mixed records
#   - special / conditional fields handled inline below.

_MODEL_ARRAYS_BEFORE_E_SEICHE = [
    "U", "V", "T", "S", "dS", "rho",
    "k", "ko", "avh", "eps", "num", "nuh",
    "P", "B", "NN", "cmue1", "cmue2", "P_Seiche",
]

_MODEL_SCALAR_GROUPS = [
    ("u10_v10_uv10_Wf", ["u10", "v10", "uv10", "Wf"]),
    ("u_taub_drag_u_taus_rain", ["u_taub", "drag", "u_taus", "rain"]),
    ("tx_ty", ["tx", "ty"]),
    ("C10", ["C10"]),
    ("SST_heat_group", ["SST", "heat", "heat_snow", "heat_ice", "heat_snowice"]),
    ("T_atm", ["T_atm"]),
]

_MODEL_SCALAR_SOLO_AFTER_LAT = [
    "snow_h", "total_ice_h", "black_ice_h", "white_ice_h",
    "snow_dens", "ice_temp", "precip",
    "ha", "hw", "hk", "hv", "rad0",
]


def _read_model_state(f: FortranFile, snap: Snapshot) -> None:
    m = snap.model

    for name in _MODEL_ARRAYS_BEFORE_E_SEICHE:
        m[name], m[f"{name}_bounds"] = _read_array(f)

    e_seiche, gamma = f.read_reals(np.float64)
    m["E_Seiche"] = float(e_seiche)
    m["gamma"] = float(gamma)

    m["absorb"], m["absorb_bounds"] = _read_array(f)
    m["absorb_vol"], m["absorb_vol_bounds"] = _read_array(f)

    for group_name, fields in _MODEL_SCALAR_GROUPS:
        vals = f.read_reals(np.float64)
        if vals.size != len(fields):
            raise ValueError(
                f"{group_name}: expected {len(fields)} reals, got {vals.size}"
            )
        for name, v in zip(fields, vals):
            m[name] = float(v)

    m["rad"], m["rad_bounds"] = _read_array(f)

    # albedo_data is a fixed-shape 9x12 matrix written without explicit bounds
    flat = f.read_reals(np.float64)
    if flat.size != 9 * 12:
        raise ValueError(f"albedo_data: expected 108 reals, got {flat.size}")
    m["albedo_data"] = flat.reshape((9, 12), order="F")

    m["albedo_water"] = float(f.read_reals(np.float64)[0])
    m["lat_number"] = int(f.read_ints(np.int32)[0])

    for name in _MODEL_SCALAR_SOLO_AFTER_LAT:
        m[name] = float(f.read_reals(np.float64)[0])

    cde, cm0 = f.read_reals(np.float64)
    m["cde"] = float(cde)
    m["cm0"] = float(cm0)
    m["fsed"] = float(f.read_reals(np.float64)[0])

    m["fgeo_add"], m["fgeo_add_bounds"] = _read_array(f)

    if snap.couple_aed2:
        m["AED2_state"], m["AED2_state_bounds"] = _read_any_array(f)
        m["AED2_diagnostic"], m["AED2_diagnostic_bounds"] = _read_any_array(f)
        m["AED2_diagnostic_sheet"], m["AED2_diagnostic_sheet_bounds"] = _read_any_array(f)

    if snap.inflow_mode > 0:
        m["Q_inp"], m["Q_inp_bounds"] = _read_matrix(f)
        m["Q_vert"], m["Q_vert_bounds"] = _read_array(f)


def _write_model_state(f: FortranFile, snap: Snapshot) -> None:
    m = snap.model

    for name in _MODEL_ARRAYS_BEFORE_E_SEICHE:
        _write_array(f, m[name], m[f"{name}_bounds"])

    f.write_record(np.array([m["E_Seiche"], m["gamma"]], dtype=np.float64))

    _write_array(f, m["absorb"], m["absorb_bounds"])
    _write_array(f, m["absorb_vol"], m["absorb_vol_bounds"])

    for _, fields in _MODEL_SCALAR_GROUPS:
        f.write_record(np.array([m[k] for k in fields], dtype=np.float64))

    _write_array(f, m["rad"], m["rad_bounds"])

    f.write_record(
        np.asfortranarray(m["albedo_data"], dtype=np.float64).ravel(order="F")
    )
    f.write_record(np.array([m["albedo_water"]], dtype=np.float64))
    f.write_record(np.array([m["lat_number"]], dtype=np.int32))

    for name in _MODEL_SCALAR_SOLO_AFTER_LAT:
        f.write_record(np.array([m[name]], dtype=np.float64))

    f.write_record(np.array([m["cde"], m["cm0"]], dtype=np.float64))
    f.write_record(np.array([m["fsed"]], dtype=np.float64))

    _write_array(f, m["fgeo_add"], m["fgeo_add_bounds"])

    if snap.couple_aed2:
        _write_any_array(f, m["AED2_state"], m["AED2_state_bounds"])
        _write_any_array(f, m["AED2_diagnostic"], m["AED2_diagnostic_bounds"])
        _write_any_array(f, m["AED2_diagnostic_sheet"], m["AED2_diagnostic_sheet_bounds"])

    if snap.inflow_mode > 0:
        _write_matrix(f, m["Q_inp"], m["Q_inp_bounds"])
        _write_array(f, m["Q_vert"], m["Q_vert_bounds"])


# --- grid ----------------------------------------------------------------

# Mirror src/strat_grid.f90:110-129
_GRID_ARRAYS_BEFORE_VOLUME = ["h", "z_face", "z_volume", "Az", "dAz", "meanint"]
_GRID_ARRAYS_AFTER_VOLUME = [
    "AreaFactor_1", "AreaFactor_2",
    "AreaFactor_k1", "AreaFactor_k2", "AreaFactor_eps",
]


def _read_grid(f: FortranFile, snap: Snapshot) -> None:
    g = snap.grid
    for name in _GRID_ARRAYS_BEFORE_VOLUME:
        g[name], g[f"{name}_bounds"] = _read_any_array(f)
    volume, h_old = f.read_reals(np.float64)
    g["volume"] = float(volume)
    g["h_old"] = float(h_old)
    for name in _GRID_ARRAYS_AFTER_VOLUME:
        g[name], g[f"{name}_bounds"] = _read_any_array(f)
    nz_grid, nz_occupied, max_input = f.read_ints(np.int32)
    g["nz_grid"] = int(nz_grid)
    g["nz_occupied"] = int(nz_occupied)
    g["max_length_input_data"] = int(max_input)
    ubnd_vol, ubnd_fce, length_vol, length_fce = f.read_ints(np.int32)
    g["ubnd_vol"] = int(ubnd_vol)
    g["ubnd_fce"] = int(ubnd_fce)
    g["length_vol"] = int(length_vol)
    g["length_fce"] = int(length_fce)
    z_zero, lake_level, lake_level_old, max_depth = f.read_reals(np.float64)
    g["z_zero"] = float(z_zero)
    g["lake_level"] = float(lake_level)
    g["lake_level_old"] = float(lake_level_old)
    g["max_depth"] = float(max_depth)


def _write_grid(f: FortranFile, snap: Snapshot) -> None:
    g = snap.grid
    for name in _GRID_ARRAYS_BEFORE_VOLUME:
        _write_any_array(f, g[name], g[f"{name}_bounds"])
    f.write_record(np.array([g["volume"], g["h_old"]], dtype=np.float64))
    for name in _GRID_ARRAYS_AFTER_VOLUME:
        _write_any_array(f, g[name], g[f"{name}_bounds"])
    f.write_record(np.array(
        [g["nz_grid"], g["nz_occupied"], g["max_length_input_data"]],
        dtype=np.int32,
    ))
    f.write_record(np.array(
        [g["ubnd_vol"], g["ubnd_fce"], g["length_vol"], g["length_fce"]],
        dtype=np.int32,
    ))
    f.write_record(np.array(
        [g["z_zero"], g["lake_level"], g["lake_level_old"], g["max_depth"]],
        dtype=np.float64,
    ))


# --- absorption ----------------------------------------------------------

# Mirror src/strat_absorption.f90:81-91
def _read_absorption(f: FortranFile, snap: Snapshot) -> None:
    a = snap.absorption
    a["number_of_lines_read"] = int(f.read_ints(np.int32)[0])
    tb_start, tb_end = f.read_reals(np.float64)
    a["tb_start"] = float(tb_start)
    a["tb_end"] = float(tb_end)
    eof, nval = f.read_ints(np.int32)
    a["eof"] = int(eof)
    a["nval"] = int(nval)
    a["z_absorb"], a["z_absorb_bounds"] = _read_array(f)
    a["absorb_start"], a["absorb_start_bounds"] = _read_array(f)
    a["absorb_end"], a["absorb_end_bounds"] = _read_array(f)


def _write_absorption(f: FortranFile, snap: Snapshot) -> None:
    a = snap.absorption
    f.write_record(np.array([a["number_of_lines_read"]], dtype=np.int32))
    f.write_record(np.array([a["tb_start"], a["tb_end"]], dtype=np.float64))
    f.write_record(np.array([a["eof"], a["nval"]], dtype=np.int32))
    _write_array(f, a["z_absorb"], a["z_absorb_bounds"])
    _write_array(f, a["absorb_start"], a["absorb_start_bounds"])
    _write_array(f, a["absorb_end"], a["absorb_end_bounds"])


# --- lateral (optional) --------------------------------------------------

# Mirror src/strat_lateral.f90:154-185
_LATERAL_INT_ARRAYS = [
    "number_of_lines_read", "eof", "nval", "nval_deep", "nval_surface", "fnum",
]
_LATERAL_LOGICAL_ARRAYS = ["has_surface_input", "has_deep_input"]
_LATERAL_REAL_ARRAYS = ["tb_start", "tb_end"]
_LATERAL_MATRICES = [
    "z_Inp", "Q_start", "Qs_start", "Q_end", "Qs_end",
    "Q_read_start", "Q_read_end",
    "Inp_read_start", "Inp_read_end",
    "Qs_read_start", "Qs_read_end",
]


def _read_lateral(f: FortranFile, snap: Snapshot) -> None:
    lat = snap.lateral
    has = f.read_ints(np.int32)
    lat["has_allocated"] = bool(has[0])
    for name in _LATERAL_INT_ARRAYS:
        lat[name], lat[f"{name}_bounds"] = _read_int_array(f)
    for name in _LATERAL_LOGICAL_ARRAYS:
        lat[name], lat[f"{name}_bounds"] = _read_logical_array(f)
    for name in _LATERAL_REAL_ARRAYS:
        lat[name], lat[f"{name}_bounds"] = _read_array(f)
    for name in _LATERAL_MATRICES:
        lat[name], lat[f"{name}_bounds"] = _read_matrix(f)


def _write_lateral(f: FortranFile, snap: Snapshot) -> None:
    lat = snap.lateral
    f.write_record(np.array(
        [1 if lat.get("has_allocated", True) else 0], dtype=np.int32,
    ))
    for name in _LATERAL_INT_ARRAYS:
        _write_int_array(f, lat[name], lat[f"{name}_bounds"])
    for name in _LATERAL_LOGICAL_ARRAYS:
        _write_logical_array(f, lat[name], lat[f"{name}_bounds"])
    for name in _LATERAL_REAL_ARRAYS:
        _write_array(f, lat[name], lat[f"{name}_bounds"])
    for name in _LATERAL_MATRICES:
        _write_matrix(f, lat[name], lat[f"{name}_bounds"])


# --- logger --------------------------------------------------------------

def _read_logger(f: FortranFile, snap: Snapshot) -> None:
    snap.logger["last_iteration_data"], snap.logger["last_iteration_data_bounds"] = (
        _read_matrix(f)
    )


def _write_logger(f: FortranFile, snap: Snapshot) -> None:
    _write_matrix(
        f,
        snap.logger["last_iteration_data"],
        snap.logger["last_iteration_data_bounds"],
    )


# --- top-level read/write ------------------------------------------------

def flags_from_par(par_path: str) -> dict:
    """Read CoupleAED2 and InflowMode from a Simstrat .par (JSON) config.

    Returns a dict with `couple_aed2`, `inflow_mode`, and `has_lateral_state`
    suitable for splatting into `read_snapshot(...)`. `has_lateral_state` is
    set to `inflow_mode > 0` since the lateral block is normally allocated
    whenever inflow is enabled; override explicitly if your run differs.
    """
    with open(par_path) as fh:
        cfg = json.load(fh)
    mc = cfg.get("ModelConfig", {})
    couple_aed2 = bool(mc.get("CoupleAED2", False))
    inflow_mode = int(mc.get("InflowMode", 0))
    return {
        "couple_aed2": couple_aed2,
        "inflow_mode": inflow_mode,
        "has_lateral_state": inflow_mode > 0,
    }


def read_snapshot(
    path: str,
    couple_aed2: bool = False,
    inflow_mode: int = 0,
    has_lateral_state: bool = False,
    par_path: Optional[str] = None,
) -> Snapshot:
    """Read a Simstrat snapshot file into a Snapshot dataclass.

    If `par_path` is given, all three flags are taken from the .par file and
    the explicit flag args are ignored. To override par-derived values, call
    `flags_from_par()`, edit the dict, and pass without `par_path`.
    """
    if par_path is not None:
        flags = flags_from_par(par_path)
        couple_aed2 = flags["couple_aed2"]
        inflow_mode = flags["inflow_mode"]
        has_lateral_state = flags["has_lateral_state"]
    snap = Snapshot(
        couple_aed2=couple_aed2,
        inflow_mode=inflow_mode,
        has_lateral_state=has_lateral_state,
    )
    with FortranFile(path, "r") as f:
        _read_model_state(f, snap)
        _read_grid(f, snap)
        _read_absorption(f, snap)
        if inflow_mode > 0 and has_lateral_state:
            _read_lateral(f, snap)
        _read_logger(f, snap)
    return snap


def write_snapshot(path: str, snap: Snapshot) -> None:
    """Write a Snapshot back to a Fortran unformatted file."""
    with FortranFile(path, "w") as f:
        _write_model_state(f, snap)
        _write_grid(f, snap)
        _write_absorption(f, snap)
        if snap.inflow_mode > 0 and snap.has_lateral_state:
            _write_lateral(f, snap)
        _write_logger(f, snap)


# --- snapshot T-column round-trip: single source of truth for BOTH engines ------------------------
# The native engine (read_snapshot_T/write_snapshot_T, keyed by member_id) and the OpenDA wrapper
# (keyed by an explicit instance path) both read/inject the analysed temperature column. These two
# path-based helpers are that shared core; the member-id wrappers below just build the paths.

def read_snapshot_T_at(snap_path, par_path):
    """Read a snapshot's temperature column + grid geometry -> (T, z_volume, lake_level)."""
    snap  = read_snapshot(snap_path, par_path=par_path)
    T     = snap.model["T"]
    # VERIFIED: there is a vertical-alignment assumption. This takes the TOP len(T) cells of
    # z_volume, i.e. it assumes T is surface-aligned (z_volume[-1] == surface). If T is
    # bottom-aligned in the Fortran grid this silently maps every obs to the wrong depth.
    # Verified on inputs/upperlugano
    z_vol = snap.grid["z_volume"][-len(T):]
    return T.copy(), z_vol.copy(), float(snap.grid["lake_level"])


def write_snapshot_T_at(snap_path, par_path, T_new):
    """Inject an analysed temperature column into a snapshot (atomic via tmp + os.replace).
    Raises if T_new's length doesn't match the snapshot grid (caller guards/handles)."""
    snap = read_snapshot(snap_path, par_path=par_path)
    if len(T_new) != len(snap.model["T"]):
        raise ValueError(f"T length {len(T_new)} != snapshot grid {len(snap.model['T'])}")
    snap.model["T"][:] = T_new
    tmp = snap_path + ".tmp"
    write_snapshot(tmp, snap)
    os.replace(tmp, snap_path)
