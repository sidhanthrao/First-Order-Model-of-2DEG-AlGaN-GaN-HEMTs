"""
First-order model of the 2D electron gas (2DEG) in AlGaN/GaN heterostructures.

Part A: 2DEG sheet density n_s vs Al fraction x and barrier thickness d
Part B: Sheet resistance vs number of parallel channels N (multi-channel idea)
Part C: (optional) Ron,sp vs breakdown voltage limit for Si, 4H-SiC, GaN

MODEL (Ambacher et al., J. Appl. Phys. 87, 334 (2000)):
    sigma(x) = P_SP(AlGaN) + P_PZ(AlGaN) - P_SP(GaN)        [C/m^2]
    n_s(x,d) = |sigma(x)|/e - (eps0*eps_r(x) / (d*e)) * (phi_b(x) + E_F - dEc(x))
"""


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- constants ----------------------------------
Q = 1.602e-19        # C
EPS0 = 8.854e-12     # F/m

# ----------------------------- PARAMETERS ------------------------
# Lattice constants (angstrom), piezo constants (C/m^2), elastic constants (GPa)
GaN = dict(a=3.189, e31=-0.35, e33=1.27, C13=106.0, C33=398.0)
AlN = dict(a=3.112, e31=-0.50, e33=1.79, C13=108.0, C33=373.0)

def lerp(prop, x):
    """Linear (Vegard) interpolation between GaN (x=0) and AlN (x=1)."""
    return GaN[prop] * (1 - x) + AlN[prop] * x

def p_sp(x):
    """Spontaneous polarization of AlxGa1-xN (C/m^2), with bowing term."""
    return -0.090 * x - 0.034 * (1 - x) + 0.021 * x * (1 - x)

def p_pz(x):
    """Piezoelectric polarization of AlGaN strained onto relaxed GaN (C/m^2)."""
    a_x = lerp("a", x)
    strain = (GaN["a"] - a_x) / a_x
    return 2 * strain * (lerp("e31", x) - lerp("e33", x) * lerp("C13", x) / lerp("C33", x))

def eps_r(x):
    return 10.31 * x + 8.9 * (1 - x)

def phi_b(x):
    """Schottky barrier height at the surface (V)."""
    return 1.3 * x + 0.84

def delta_ec(x):
    """Conduction band offset (eV) = 0.7 * bandgap difference."""
    eg_gan = 3.42
    eg_x = 6.13 * x + eg_gan * (1 - x) - 1.0 * x * (1 - x)
    return 0.7 * (eg_x - eg_gan)

E_F = 0.1            # Fermi level above GaN conduction band (eV). 
MU = 1500.0          # 2DEG mobility (cm^2/Vs). 

# ----------------------------- Part A -------------------------------------
def ns_cm2(x, d_nm):
    """2DEG sheet density in cm^-2 (returns 0 if the barrier is too thin)."""
    sigma = abs(p_sp(x) + p_pz(x) - p_sp(0.0))            # C/m^2
    d = d_nm * 1e-9
    drop = EPS0 * eps_r(x) / (d * Q) * (phi_b(x) + E_F - delta_ec(x))  # m^-2
    n = sigma / Q - drop                                   # m^-2
    return np.maximum(n, 0.0) / 1e4                        # cm^-2

def plot_part_a():
    d_vals = np.linspace(2, 40, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    for x in (0.15, 0.20, 0.25, 0.30, 0.35):
        ax.plot(d_vals, ns_cm2(x, d_vals) / 1e13, label=f"x = {x:.2f}")
    ax.set_xlabel("AlGaN barrier thickness d (nm)")
    ax.set_ylabel("2DEG sheet density n_s (1e13 cm$^{-2}$)")
    ax.set_title("2DEG density in AlGaN/GaN (first-order model)")
    ax.grid(alpha=0.3)
    ax.legend(title="Al fraction")
    fig.tight_layout()
    fig.savefig("fig_a_ns_vs_thickness.png", dpi=200)
    plt.close(fig)

# ----------------------------- Part B -------------------------------------
def rsheet_ohm_sq(n_s_cm2, mu_cm2, n_channels=1):
    """Ideal parallel channels: R = 1 / (q * n_s * mu * N). Inputs in cm units."""
    return 1.0 / (Q * n_s_cm2 * mu_cm2 * n_channels)

def plot_part_b():
    n_s = float(ns_cm2(0.25, 25.0))
    channels = np.arange(1, 11)
    r = rsheet_ohm_sq(n_s, MU, channels)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(channels, r, "o-")
    ax.set_xlabel("Number of parallel 2DEG channels N")
    ax.set_ylabel("Sheet resistance (ohm/sq)")
    ax.set_title(f"Ideal multi-channel stack (n_s = {n_s:.2e} cm$^{{-2}}$, mu = {MU:.0f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_b_rsheet_vs_channels.png", dpi=200)
    plt.close(fig)
    return n_s, r

# ----------------------------- Part C --------------------------
# Ron,sp = 4 V_B^2 / (eps_s * mu * Ec^3)   (Baliga limit, 1D drift region)
MATERIALS = {
    "Si":     dict(eps=11.7, mu=1350.0, Ec=0.3e6),   # Ec in V/cm
    "4H-SiC": dict(eps=9.7,  mu=950.0,  Ec=2.5e6),
    "GaN":    dict(eps=9.0,  mu=1000.0, Ec=3.3e6),
}

def ron_sp_limit(vb, eps, mu, ec):
    """Ron,sp in ohm*cm^2 for breakdown voltage vb (V)."""
    return 4 * vb**2 / (eps * (EPS0 / 100.0) * mu * ec**3)   # eps0 in F/cm

def plot_part_c():
    vb = np.logspace(2, 4, 100)          # 100 V to 10 kV
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, m in MATERIALS.items():
        ax.loglog(vb, ron_sp_limit(vb, m["eps"], m["mu"], m["Ec"]) * 1e3, label=name)
    ax.set_xlabel("Breakdown voltage (V)")
    ax.set_ylabel("Ron,sp limit (m$\\Omega\\cdot$cm$^2$)")
    ax.set_title("Theoretical unipolar limit")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_c_baliga_limit.png", dpi=200)
    plt.close(fig)

# ----------------------------- run ----------------------------------------
if __name__ == "__main__":
    plot_part_a()
    n_s, r = plot_part_b()
    plot_part_c()
    print(f"Sanity check, x=0.25, d=25 nm: n_s = {n_s:.2e} cm^-2 "
          f"(expect roughly 1e13), R_sheet(N=1) = {r[0]:.0f} ohm/sq")
    print(f"N=5: {r[4]:.0f} ohm/sq,  N=10: {r[9]:.0f} ohm/sq")
