import numpy as np
from typing import Tuple

# Analytical IV 
def iv_conc_analytical(t, D, Vd, ke):
    """
    IV bolus analytical solution:
      C(t) = (D/Vd) * exp(-ke * t)
    """
    t = np.asarray(t)
    return (D / Vd) * np.exp(-ke * t)

# Analytical Oral 
def oral_conc_analytical(t, D, Vd, ka, ke, F):
    """
    One-compartment oral, first-order absorption & elimination.
      If ka != ke:
        C(t) = [F*D*ka / (Vd*(ka - ke))] * (exp(-ke*t) - exp(-ka*t))
      If ka == ke (limiting form):
        C(t) = F*D/Vd * (ka*t) * exp(-ka*t)
    """
    t = np.asarray(t)
    if abs(ka - ke) < 1e-12:
        return F * D / Vd * (ka * t) * np.exp(-ka * t)
    return (F * D * ka / (Vd * (ka - ke))) * (np.exp(-ke * t) - np.exp(-ka * t))

# Euler for IV
def euler_iv(t_end: float, dt: float, D: float, Vd: float, ke: float) -> Tuple[np.ndarray, np.ndarray]:
   
    n = int(np.ceil(t_end / dt)) + 1
    t = np.linspace(0, dt*(n-1), n)
    C = np.zeros_like(t)
    C[0] = D / Vd
    for i in range(1, n):
        C[i] = C[i-1] + dt * (-ke * C[i-1])
    return t, C

# ODE solver (RK45) for IV 
def iv_rhs(t, C, ke):
    return -ke * C

def rk45_iv(t_end: float, D: float, Vd: float, ke: float, t_eval: np.ndarray):

    from scipy.integrate import solve_ivp
    y0 = [D / Vd]
    sol = solve_ivp(iv_rhs, [0.0, t_end], y0, method="RK45", args=(ke,), t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[0]

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def find_crossing_time(D, Vd, ka, ke, F, t_max=48.0, step=0.01):

    t = np.arange(0, t_max + step, step)
    f = oral_conc_analytical(t, D, Vd, ka, ke, F) - iv_conc_analytical(t, D, Vd, ke)
    for i in range(1, len(t)):
        if f[i-1] == 0.0:
            return float(t[i-1])
        if f[i-1] * f[i] < 0:
            a, b = t[i-1], t[i]
            fa, fb = f[i-1], f[i]
            for _ in range(40):
                m = 0.5*(a+b)
                fm = oral_conc_analytical(m, D, Vd, ka, ke, F) - iv_conc_analytical(m, D, Vd, ke)
                if fa * fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            return float(0.5*(a+b))
    return None
