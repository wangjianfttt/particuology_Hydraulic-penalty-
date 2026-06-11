import numpy as np

from src.lbm_dem.psc3d import Particle3D, PscD3Q19


def test_stationary_particle_geometry_and_momentum_closure():
    particle = Particle3D(x=6.0, y=6.0, z=6.0, r=2.0)
    solver = PscD3Q19(
        n=12,
        particles=[particle],
        tau=0.72,
        subsamples=2,
        body_force_z=1.0e-7,
        static_geometry=True,
    )

    diagnostics = solver.collide_stream_psc()

    assert 0.0 < solver.solid.mean() < 1.0
    assert np.isfinite(solver.f).all()
    assert np.isfinite(solver.hydro_forces).all()
    assert np.isfinite(diagnostics["momentum_residual"])
