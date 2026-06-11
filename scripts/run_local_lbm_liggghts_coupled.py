"""Quasi-resolved local D3Q19-LIGGGHTS coupling for fines in a 3-D window.

The loop is intentionally compact:
DEM state -> voxelized D3Q19 flow or sparse PSC-IMB load reconstruction ->
write dragforce/hdtorque -> DEM subcycles. It is a local time-coupled bridge;
long calibrated PSC-IMB production campaigns still require force-scale and
resolution convergence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_local_fines_lbm_permeability import voxelize_variable
from scripts.run_3d_window_lbm import run_lbm
from src.lbm_dem.liggghts_contacts import load_liggghts_local_frames, parse_pair_gran_local
from src.lbm_dem.liggghts_coupler import LiggghtsForceCoupler
from src.lbm_dem.psc3d import Particle3D, SparsePscD3Q19


BASE = ROOT / "data" / "production" / "3d_pebble_bed" / "local_windows" / "strong_force_chain"
FIG = ROOT / "figures"
BOX_SIZE = 0.009
CS2 = 1.0 / 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=str(ROOT / "liggghts" / "local_window_fines_coupled.in"))
    parser.add_argument("--data-file", default="data/production/3d_pebble_bed/local_windows/strong_force_chain/window_with_fines_packed_slug_900.data")
    parser.add_argument("--dump-prefix", default="coupled")
    parser.add_argument("--resolution", type=int, default=40)
    parser.add_argument("--lbm-steps", type=int, default=180)
    parser.add_argument("--coupling-steps", type=int, default=18)
    parser.add_argument("--dem-substeps", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.72)
    parser.add_argument("--force-z", type=float, default=2.0e-7)
    parser.add_argument("--target-superficial-velocity", type=float, default=0.05)
    parser.add_argument("--drag-scale", type=float, default=1.0)
    parser.add_argument("--fine-radius-inflate-dx", type=float, default=0.0)
    parser.add_argument("--force-model", choices=("drag_proxy", "psc_imb"), default="drag_proxy")
    parser.add_argument("--psc-steps", type=int, default=8)
    parser.add_argument("--psc-subsamples", type=int, default=2)
    parser.add_argument("--psc-force-scale", type=float, default=1.0, help="Converts PSC-IMB lattice-unit particle loads to DEM force units.")
    parser.add_argument("--psc-torque-scale", type=float, default=1.0, help="Converts PSC-IMB lattice-unit particle torques to DEM torque units.")
    parser.add_argument("--apply-psc-torque", action="store_true")
    parser.add_argument("--velocity-lu-scale", type=float, default=0.0, help="Optional SI-to-lattice velocity scale for particle velocities in PSC-IMB.")
    parser.add_argument("--angular-lu-scale", type=float, default=0.0, help="Optional SI-to-lattice angular-velocity scale for particle rotations in PSC-IMB.")
    parser.add_argument("--velocity-feedback-factor", type=float, default=1.0, help="Relaxation factor applied to DEM translational velocity feedback in PSC-IMB.")
    parser.add_argument("--angular-feedback-factor", type=float, default=1.0, help="Relaxation factor applied to DEM angular velocity feedback in PSC-IMB.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--use-si-scaling", action="store_true", help="Compute PSC force/torque/velocity scaling from helium properties and lattice resolution.")
    parser.add_argument("--temperature-k", type=float, default=773.15)
    parser.add_argument("--pressure-pa", type=float, default=2.0e5)
    parser.add_argument("--mu-he", type=float, default=3.7e-5)
    return parser.parse_args()


def apply_si_scaling(args: argparse.Namespace) -> dict[str, float]:
    r_he = 2077.1
    rho_he = args.pressure_pa / (r_he * args.temperature_k)
    nu_phys = args.mu_he / rho_he
    nu_lu = CS2 * (args.tau - 0.5)
    dx = BOX_SIZE / args.resolution
    dt = nu_lu * dx * dx / nu_phys
    u_scale = dx / dt
    force_scale = rho_he * dx**4 / (dt * dt)
    torque_scale = force_scale * dx
    if args.use_si_scaling:
        args.psc_force_scale = force_scale
        args.psc_torque_scale = torque_scale
        args.velocity_lu_scale = 1.0 / u_scale
        args.angular_lu_scale = dt
    return {
        "rho_he_kg_m3": rho_he,
        "nu_phys_m2_s": nu_phys,
        "nu_lu": nu_lu,
        "dx_m": dx,
        "dt_s": dt,
        "u_scale_m_s": u_scale,
        "force_scale_N_per_lu": force_scale,
        "torque_scale_Nm_per_lu": torque_scale,
        "velocity_lu_per_m_s": args.velocity_lu_scale,
        "angular_lu_per_rad_s": args.angular_lu_scale,
        "velocity_feedback_factor": args.velocity_feedback_factor,
        "angular_feedback_factor": args.angular_feedback_factor,
        "applied_force_scale": args.psc_force_scale,
        "applied_torque_scale": args.psc_torque_scale,
    }


def write_case_input(template: Path, data_file: str, dump_prefix: str) -> Path:
    dump_base = BASE.relative_to(ROOT)
    lines = []
    for line in template.read_text().splitlines():
        if line.startswith("read_data "):
            lines.append(f"read_data {data_file}")
        elif line.startswith("dump traj all custom"):
            lines.append(
                "dump traj all custom 100 "
                f"{dump_base / (dump_prefix + '_traj_*.dump')} "
                "id type x y z vx vy vz fx fy fz radius c_contacts"
            )
        elif line.startswith("dump dlocal all local"):
            lines.append(
                "dump dlocal all local 100 "
                f"{dump_base / (dump_prefix + '_contact_*.local')} "
                "c_cpgl[1] c_cpgl[2] c_cpgl[3] c_cpgl[4] c_cpgl[5] c_cpgl[6] "
                "c_cpgl[7] c_cpgl[8] c_cpgl[9] c_cpgl[10] c_cpgl[11] c_cpgl[12] "
                "c_cpgl[13] c_cpgl[14] c_cpgl[15] c_cpgl[16] c_cpgl[17] c_cpgl[18] c_cpgl[19]"
            )
        else:
            lines.append(line)
    path = BASE / f"{dump_prefix}_coupled.in"
    path.write_text("\n".join(lines) + "\n")
    return path


def particles_from_state(state) -> np.ndarray:
    types = np.where(state.radius < 0.0003, 2.0, 1.0)
    return np.column_stack([state.ids.astype(float), types, state.x[:, 0], state.x[:, 1], state.x[:, 2], state.radius])


def trilinear(field: np.ndarray, xyz: np.ndarray, box_size: float) -> np.ndarray:
    n = field.shape[0]
    coords = np.clip(xyz / box_size * n - 0.5, 0.0, n - 1.001)
    i0 = np.floor(coords).astype(int)
    t = coords - i0
    i1 = np.minimum(i0 + 1, n - 1)
    vals = np.empty(len(xyz), dtype=float)
    for m, ((x0, y0, z0), (x1, y1, z1), (tx, ty, tz)) in enumerate(zip(i0, i1, t)):
        c000 = field[x0, y0, z0]
        c100 = field[x1, y0, z0]
        c010 = field[x0, y1, z0]
        c110 = field[x1, y1, z0]
        c001 = field[x0, y0, z1]
        c101 = field[x1, y0, z1]
        c011 = field[x0, y1, z1]
        c111 = field[x1, y1, z1]
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        vals[m] = c0 * (1 - tz) + c1 * tz
    return vals


def psc_particles_from_state(state, resolution: int, velocity_lu_scale: float, angular_lu_scale: float) -> list[Particle3D]:
    scale = resolution / BOX_SIZE
    vscale = float(velocity_lu_scale)
    wscale = float(angular_lu_scale)
    return [
        Particle3D(
            x=float(pos[0] * scale),
            y=float(pos[1] * scale),
            z=float(pos[2] * scale),
            r=float(radius * scale),
            vx=float(vel[0] * vscale),
            vy=float(vel[1] * vscale),
            vz=float(vel[2] * vscale),
            wx=float(omega[0] * wscale),
            wy=float(omega[1] * wscale),
            wz=float(omega[2] * wscale),
        )
        for pos, vel, omega, radius in zip(state.x, state.v, state.omega, state.radius)
    ]


def psc_imb_loads(state, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    particles = psc_particles_from_state(
        state,
        args.resolution,
        args.velocity_lu_scale * args.velocity_feedback_factor,
        args.angular_lu_scale * args.angular_feedback_factor,
    )
    solver = SparsePscD3Q19(
        n=args.resolution,
        particles=particles,
        tau=args.tau,
        subsamples=args.psc_subsamples,
        body_force_z=args.force_z,
    )
    diag = {}
    for _ in range(args.psc_steps):
        diag = solver.collide_stream_psc()
    forces = args.psc_force_scale * solver.hydro_forces
    torques = args.psc_torque_scale * solver.hydro_torques
    return forces, torques, diag


def contact_snapshot(prefix: str, n_skeleton: int) -> tuple[float, float, float]:
    files = sorted(BASE.glob(f"{prefix}_contact_*.local"))
    if not files:
        return 0.0, 0.0, 0.0
    frames = load_liggghts_local_frames(files[-1])
    if not frames:
        return 0.0, 0.0, 0.0
    contacts = parse_pair_gran_local(frames[-1])
    if not len(contacts.overlap):
        return 0.0, 0.0, 0.0
    ids = contacts.ids
    fine_i = ids[:, 0] > n_skeleton
    fine_j = ids[:, 1] > n_skeleton
    fine_contact = fine_i | fine_j
    fine_fine = fine_i & fine_j
    return float(np.sum(fine_contact)), float(np.sum(fine_fine)), float(np.max(contacts.force_magnitude[fine_contact]) if np.any(fine_contact) else 0.0)


def main() -> None:
    args = parse_args()
    scale_info = apply_si_scaling(args)
    for pattern in (f"{args.dump_prefix}_traj_*.dump", f"{args.dump_prefix}_contact_*.local"):
        for old in BASE.glob(pattern):
            old.unlink()
    case_input = write_case_input(Path(args.template), args.data_file, args.dump_prefix)
    rows = []
    failed = ""
    out = BASE / f"{args.dump_prefix}_coupled_summary.csv"
    scale_out = BASE / f"{args.dump_prefix}_scales.csv"
    header = (
        "coupling_step,voxel_porosity,permeability_lu,superficial_uz_lu,"
        "retained_fraction,mean_applied_force_N,max_applied_force_N,"
        "n_fine_contacts,n_fine_fine_contacts,max_fine_contact_force_N,"
        "imb_momentum_residual,psc_force_scale,psc_torque_scale,"
        "velocity_lu_scale,angular_lu_scale"
    )
    out.write_text(header + "\n")
    with scale_out.open("w") as fh:
        fh.write("quantity,value\n")
        for key, value in scale_info.items():
            fh.write(f"{key},{value}\n")
    try:
        with LiggghtsForceCoupler(case_input, quiet=True) as dem:
            for step in range(args.coupling_steps):
                state = dem.state()
                particles = particles_from_state(state)
                n_skeleton = int(np.sum(particles[:, 1] == 1))
                if args.force_model == "drag_proxy":
                    solid = voxelize_variable(particles, BOX_SIZE, args.resolution, args.fine_radius_inflate_dx)
                    flow = run_lbm(solid, tau=args.tau, force_z=args.force_z, steps=args.lbm_steps, return_fields=True)
                    scale = args.target_superficial_velocity / max(float(flow["superficial_uz"]), 1.0e-30)
                    fluid_u = np.column_stack(
                        [
                            trilinear(flow["ux"], state.x, BOX_SIZE),
                            trilinear(flow["uy"], state.x, BOX_SIZE),
                            trilinear(flow["uz"], state.x, BOX_SIZE),
                        ]
                    ) * scale
                    rel = fluid_u - state.v
                    radii = state.radius
                    mu_he = 3.7e-5
                    applied = args.drag_scale * (6.0 * np.pi * mu_he * radii)[:, None] * rel
                    applied[particles[:, 1] == 1] = 0.0
                    flow_porosity = float(flow["porosity_voxel"])
                    permeability = float(flow["permeability_lu"])
                    superficial = float(flow["superficial_uz"])
                    imb_residual = 0.0
                else:
                    applied, torques, diag = psc_imb_loads(state, args)
                    applied[particles[:, 1] == 1] = 0.0
                    torques[particles[:, 1] == 1] = 0.0
                    flow_porosity = 1.0 - float(diag.get("solid_volume_lu", 0.0)) / float(args.resolution**3)
                    permeability = float("nan")
                    superficial = float(diag.get("mean_uz", 0.0))
                    imb_residual = float(diag.get("momentum_residual", 0.0))
                dem.set_dragforces(state.ids, applied)
                if args.force_model == "psc_imb" and args.apply_psc_torque:
                    dem.set_hdtorques(state.ids, torques)
                dem.run(args.dem_substeps)
                fine = particles[:, 1] == 2
                retained = float(np.mean((state.x[fine, 2] > 0.0) & (state.x[fine, 2] < BOX_SIZE))) if np.any(fine) else 0.0
                n_fine_contacts, n_fine_fine, max_fine_force = contact_snapshot(args.dump_prefix, n_skeleton)
                rows.append(
                    [
                        step,
                        flow_porosity,
                        permeability,
                        superficial,
                        retained,
                        float(np.mean(np.linalg.norm(applied[fine], axis=1))) if np.any(fine) else 0.0,
                        float(np.max(np.linalg.norm(applied[fine], axis=1))) if np.any(fine) else 0.0,
                        n_fine_contacts,
                        n_fine_fine,
                        max_fine_force,
                        imb_residual,
                        args.psc_force_scale if args.force_model == "psc_imb" else float("nan"),
                        args.psc_torque_scale if args.force_model == "psc_imb" else float("nan"),
                    args.velocity_lu_scale if args.force_model == "psc_imb" else float("nan"),
                        args.angular_lu_scale if args.force_model == "psc_imb" else float("nan"),
                    ]
                )
                with out.open("a") as fh:
                    fh.write(",".join(str(v) for v in rows[-1]) + "\n")
                if step % max(args.log_every, 1) == 0 or step == args.coupling_steps - 1:
                    print(
                        f"step={step:03d} model={args.force_model} K={permeability:.4e} porosity={flow_porosity:.4f} "
                        f"retained={retained:.3f} fine_contacts={n_fine_contacts:.0f} imb_residual={imb_residual:.2e}"
                    )
    except Exception as exc:
        failed = f"{type(exc).__name__}: {exc}"
        print(f"coupling failed after {len(rows)} recorded steps: {failed}", file=sys.stderr)
    np.savetxt(
        out,
        np.asarray(rows, dtype=float).reshape((-1, 15)),
        delimiter=",",
        header=header,
        comments="",
    )
    with scale_out.open("a") as fh:
        fh.write(f"failed,{failed}\n")
    data = np.asarray(rows, dtype=float)
    FIG.mkdir(exist_ok=True)
    if len(rows):
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), constrained_layout=True)
        axes[0].plot(data[:, 0], np.maximum(np.abs(data[:, 10]), 1.0e-18), "o-", color="#3f6db5")
        axes[0].set_ylabel(r"$|\mathrm{IMB\ residual}|$")
        axes[0].set_yscale("log")
        axes[1].plot(data[:, 0], data[:, 7], "o-", color="#7e62a3", label="fine-involving")
        axes[1].plot(data[:, 0], data[:, 8], "s--", color="#5a9a9a", label="fine-fine")
        axes[1].set_ylabel("contact count")
        axes[1].legend(frameon=False, fontsize=8)
        axes[2].plot(data[:, 0], 1.0e9 * data[:, 5], "o-", color="#c7665a")
        axes[2].set_ylabel("mean applied force (nN)")
        for ax in axes:
            ax.set_xlabel("coupling step")
            ax.grid(True, alpha=0.25)
        fig.savefig(FIG / f"{args.dump_prefix}_coupled_summary.pdf")
        fig.savefig(FIG / f"{args.dump_prefix}_coupled_summary.png", dpi=220)
    print(f"wrote {out}")
    print(f"wrote {scale_out}")
    print(f"wrote figures/{args.dump_prefix}_coupled_summary.pdf")
    if failed:
        raise RuntimeError(failed)


if __name__ == "__main__":
    main()
