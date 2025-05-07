import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, minimize
from scipy.linalg import solve

# === (a) Shooting Method ===
def ode_system(x, Y):
    y, dy = Y
    d2y = -(x + 1) * dy + 2 * y + (1 - x**2) * np.exp(-x)
    return [dy, d2y]

def solve_shooting(alpha, x_eval):
    sol = solve_ivp(ode_system, [0, 1], [1, alpha], t_eval=x_eval, rtol=1e-8, atol=1e-8)
    return sol

def shooting_method(h=0.1):
    x_eval = np.arange(0, 1 + h, h)

    def boundary_residual(alpha):
        sol = solve_shooting(alpha, x_eval)
        return sol.y[0, -1] - 2  # target y(1) = 2

    sol = root_scalar(boundary_residual, bracket=[-10, 10], method='brentq')
    best_alpha = sol.root
    final_sol = solve_shooting(best_alpha, x_eval)
    return x_eval, final_sol.y[0]

# === (b) Finite-Difference Method ===
def finite_difference_method(h=0.1):
    x = np.arange(0, 1 + h, h)
    n = len(x)
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Boundary conditions
    A[0, 0] = 1
    b[0] = 1  # y(0) = 1
    A[-1, -1] = 1
    b[-1] = 2  # y(1) = 2

    for i in range(1, n - 1):
        xi = x[i]
        h2 = h**2
        A[i, i - 1] = 1/h2 + (xi + 1)/(2*h)
        A[i, i] = -2/h2 - 2
        A[i, i + 1] = 1/h2 - (xi + 1)/(2*h)
        b[i] = (1 - xi**2) * np.exp(-xi)

    y = solve(A, b)
    return x, y

# === (c) Variational Method ===
def variational_method(h=0.1):
    x = np.arange(0, 1 + h, h)

    def trial(x, a, b, c):
        return 1 + a*x + b*x**2 + c*x**3  # Trial function

    def residual_squared(params):
        a, b, c = params
        y = trial(x, a, b, c)
        dy = a + 2*b*x + 3*c*x**2
        d2y = 2*b + 6*c*x
        r = d2y + (x + 1)*dy - 2*y - (1 - x**2)*np.exp(-x)
        return np.trapz(r**2, x)

    # Boundary condition at x=1: y(1) = 2
    cons = ({'type': 'eq', 'fun': lambda p: trial(1, *p) - 2})  # y(1)=2

    res = minimize(residual_squared, x0=[0, 0, 0], constraints=cons)
    y = trial(x, *res.x)
    return x, y

# === Running All Methods ===
h = 0.1

# Running Shooting Method
x_shoot, y_shoot = shooting_method(h)

# Running Finite-Difference Method
x_fd, y_fd = finite_difference_method(h)

# Running Variational Method
x_var, y_var = variational_method(h)

# === Print Results ===
print("\n{:^10} {:^20} {:^20} {:^20}".format("x", "Shooting", "Finite-Diff", "Variational"))
print("-" * 70)
for i in range(len(x_fd)):
    print(f"{x_fd[i]:>5.2f} {y_shoot[i]:>20.8f} {y_fd[i]:>20.8f} {y_var[i]:>20.8f}")

# === Plotting Results ===
plt.figure(figsize=(10, 6))
plt.plot(x_shoot, y_shoot, 'b.-', label='Shooting Method')
plt.plot(x_fd, y_fd, 'r.--', label='Finite Difference')
plt.plot(x_var, y_var, 'g-', label='Variational Method')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Comparison of Numerical Methods')
plt.legend()
plt.grid(True)
plt.show()
