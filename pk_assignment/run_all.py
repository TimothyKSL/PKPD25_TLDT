import os, csv, sys
import numpy as np
import matplotlib.pyplot as plt

# Add local package to path
HERE = os.path.dirname(__file__)
PKPKG = os.path.join(HERE, "pkpkg")
sys.path.insert(0, PKPKG)
from pk_models import (
    iv_conc_analytical, oral_conc_analytical,
    euler_iv, rk45_iv, rmse, find_crossing_time
)

# Parameters
Vd = 70.0          # L
ke = 0.175         # 1/hr
D  = 700.0         # mg (so IV C0 = 10 mg/L)
t_end = 48.0       # hr
dt_fine = 0.01     # hr for smooth analytical curve
kas = [0.35, 0.5]  # 1/hr
Fs  = [1.0, 0.7, 0.4, 0.1]

FIG_DIR = os.path.join(HERE, "figures")
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

t_fine = np.arange(0, t_end + dt_fine, dt_fine)

# IV: Analytical vs Euler vs RK45 
C_iv_anal = iv_conc_analytical(t_fine, D, Vd, ke)

euler_dts = [1.0, 0.5, 0.25, 0.1, 0.05]
rows = []

plt.figure(figsize=(8,5))
plt.plot(t_fine, C_iv_anal, label="IV Analytical")

for dt in euler_dts:
    t_eu, C_eu = euler_iv(t_end, dt, D, Vd, ke)
    C_true = np.interp(t_eu, t_fine, C_iv_anal)
    rows.append(["Euler", dt, rmse(C_true, C_eu), "vs analytical at same points"])
    plt.plot(t_eu, C_eu, label=f"Euler dt={dt} h")

# RK45 with a fixed output grid (t_fine) so we can overlay easily
t_rk, C_rk = rk45_iv(t_end, D, Vd, ke, t_eval=t_fine)
rows.append(["RK45", "adaptive", rmse(C_iv_anal, C_rk), "vs analytical on t_fine"])
plt.plot(t_rk, C_rk, label="RK45 (adaptive)", linestyle="--")

plt.xlabel("Time (h)"); plt.ylabel("Concentration C (mg/L)")
plt.title("IV Bolus: Analytical vs Euler vs RK45"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "iv_methods_comparison.png"), dpi=200); plt.close()

# ln(C) vs time (IV analytical)
plt.figure(figsize=(8,5))
plt.plot(t_fine, np.log(np.clip(C_iv_anal, 1e-12, None)))
plt.xlabel("Time (h)"); plt.ylabel("ln C (ln mg/L)")
plt.title("IV Bolus: ln(C) vs Time (Analytical)"); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "iv_lnC_vs_time.png"), dpi=200); plt.close()

with open(os.path.join(DATA_DIR, "iv_method_errors.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "dt_or_mode", "rmse_mg_per_L", "note"]); w.writerows(rows)

# Oral scenarios: analytical C vs t and ln C vs t
for ka in kas:
    # C vs time
    plt.figure(figsize=(8,5))
    for F in Fs:
        C = oral_conc_analytical(t_fine, D, Vd, ka, ke, F)
        plt.plot(t_fine, C, label=f"F={F}")
    plt.xlabel("Time (h)"); plt.ylabel("Concentration C (mg/L)")
    plt.title(f"Oral (ka={ka} hr^-1): C vs Time"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"oral_C_vs_time_ka_{str(ka).replace('.','_')}.png"), dpi=200); plt.close()

    # ln(C) vs time
    plt.figure(figsize=(8,5))
    for F in Fs:
        C = oral_conc_analytical(t_fine, D, Vd, ka, ke, F)
        plt.plot(t_fine, np.log(np.clip(C, 1e-12, None)), label=f"F={F}")
    plt.xlabel("Time (h)"); plt.ylabel("ln C (ln mg/L)")
    plt.title(f"Oral (ka={ka} hr^-1): ln(C) vs Time"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"oral_lnC_vs_time_ka_{str(ka).replace('.','_')}.png"), dpi=200); plt.close()

# IV vs Oral crossings
cross_rows = []
for ka in kas:
    for F in Fs:
        t_cross = find_crossing_time(D, Vd, ka, ke, F, t_max=t_end, step=0.01)
        cross_rows.append([ka, F, t_cross])

with open(os.path.join(DATA_DIR, "iv_oral_crossings.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ka_1_per_hr", "F", "t_cross_hr (None if no crossing)"]); w.writerows(cross_rows)

# Summary
print("\n=== Euler step size sensitivity (RMSE vs analytical) ===")
for r in rows:
    if r[0] == "Euler":
        print(f"  Euler dt={r[1]:.2f} h  ->  RMSE = {r[2]:.6f} mg/L")
print(f"  RK45 (adaptive) ->  RMSE = {[r[2] for r in rows if r[0]=='RK45'][0]:.6e} mg/L")

print("\n=== IV vs Oral crossing times (hours) ===")
for ka in kas:
    for F in Fs:
        match = [r[2] for r in cross_rows if r[0]==ka and r[1]==F][0]
        print(f"  ka={ka:.2f}, F={F:.1f}  ->  t_cross = {match}")
print("\nOutputs saved under pk_assignment/figures and pk_assignment/data\n")
