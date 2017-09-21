import numpy as np
from numpy import sqrt

tau_e = 15.0
tau_a = 12775.0
tau_d = 79935.0
# try 25
tau_T = 220.0 # days
sigma_y = 0.15
sigma_x = 0.005
# sigma_v = 1.0 / 10000.0
sigma_v = 10.
L_y = 8250.0

eps_T = tau_T / tau_d
eps = tau_e / tau_d
P_a = tau_d / tau_a
P_e = sigma_v * tau_d / L_y
P = sqrt(eps) * P_e

def dx_f(y, t):
    term1 = (-1 / eps_T) * (y[0] - 1)
    term2a = -(1 + P_a * (y[0] - y[1])**2)*y[0]
    term2b = 4 * y[2] * y[3]
    return term1 + term2a + term2b

def dy_f(y, t):
    term2 = (1 + P_a * (y[0] - y[1])**2)*y[1]
    term3 = 4 * y[2] * y[4]
    return 1 - term2 + term3

def dv_f(y, t):
    return -y[2] / eps

def dT_f(y, t):
    return (-1 / eps) * (y[3] + (2 * P**2 * y[2] * y[0]))


def dS_f(y, t):
    return (-1 / eps) * (y[4] + 2 * P**2 * y[2] * y[1])

def f(y, t):
    return np.array([dx_f(y, t),
                     dy_f(y, t),
                     dv_f(y, t),
                     dT_f(y, t),
                     dS_f(y, t)])

def G(y, t):
    arr = np.zeros((5, 5))
    arr[0,0] = sqrt(1 / eps_T) * sigma_x
    arr[1,1] = sigma_y
    arr[2,2] = sqrt(2. / eps)
    return arr

def jac(y, t):
    return np.array([[-1 - (2 * y[0] * (y[0] - y[1]) * P_a) - ((y[0] - y[1])**2 * P_a) - (1 / eps_T), 2 * y[0] * (y[0] - y[1]) * P_a, 4 * y[3], 4 * y[2], 0.0],
    [-2 * (y[0] - y[1]) * y[1] * P_a, -1 - (P_a * (y[0]-y[1])**2) + (2 * (y[0] - y[1]) * y[1] * P_a), 4 * y[4], 0.0, 4 * y[2]],
    [0.0, 0.0, -1.0 / eps, 0.0, 0.0],
    [-2 * P**2 * y[2] / eps, 0.0, -2 * P**2 * y[0] / eps, -1.0 / eps, 0.0],
    [0.0, -2 * P**2 * y[2] / eps, -2 * P**2 * y[1] / eps, 0.0, -1.0 / eps]])


def dx_x(y, t):
    return -1 - (2 * y[0] * (y[0] - y[1]) * P_a) - ((y[0] - y[1])**2 * P_a) - (1 / eps_T)


def dx_y(y, t):
    return 2 * y[0] * (y[0] - y[1]) * P_a


def dx_v(y, t):
    return 4 * y[3]


def dx_T(y, t):
    return 4 * y[2]


def dx_S(y, t):
    return 0.0


def dy_x(y, t):
    return -2 * (y[0] - y[1]) * y[1] * P_a


def dy_y(y, t):
    return -1 - (P_a * (y[0]-y[1])**2) + (2 * (y[0] - y[1]) * y[1] * P_a)


def dy_v(y, t):
    return 4 * y[4]


def dy_T(y, t):
    return 0.0


def dy_S(y, t):
    return 4 * y[2]


def dv_x(y, t):
    return 0.0


def dv_y(y, t):
    return 0.0


def dv_v(y, t):
    return -1.0 / eps


def dv_T(y, t):
    return 0.0


def dv_S(y, t):
    return 0.0


def dT_x(y, t):
    return -2 * P**2 * y[2] / eps


def dT_y(y, t):
    return 0.0


def dT_v(y, t):
    return -2 * P**2 * y[0] / eps


def dT_T(y, t):
    return -1.0 / eps


def dT_S(y, t):
    return 0.0


def dS_x(y, t):
    return 0.0


def dS_y(y, t):
    return -2 * P**2 * y[2] / eps


def dS_v(y, t):
    return -2 * P**2 * y[1] / eps


def dS_T(y, t):
    return 0.0


def dS_S(y, t):
    return -1.0 / eps
