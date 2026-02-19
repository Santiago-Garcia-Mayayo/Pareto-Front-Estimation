import numpy as np


# ZDT1
def zdt1(x):
    x = np.array(x)
    n = len(x)

    f1 = x[0]

    g = 1.0 + (9.0 / (n - 1)) * np.sum(x[1:])

    h = 1.0 - np.sqrt(f1 / g)

    f2 = g * h

    return [f1, f2]


def zdt1_f1(x):
    return zdt1(x)[0]


def zdt1_f2(x):
    return zdt1(x)[1]


# ZDT2
def zdt2(x):
    x = np.array(x)
    n = len(x)
    f1 = x[0]

    g = 1.0 + (9.0 / (n - 1)) * np.sum(x[1:])

    h = 1.0 - (f1 / g) ** 2

    f2 = g * h

    return [f1, f2]


def zdt2_f1(x):
    return zdt2(x)[0]


def zdt2_f2(x):
    return zdt2(x)[1]


# ZDT3
def zdt3(x):
    x = np.array(x)
    n = len(x)

    f1 = x[0]
    g = 1.0 + (9.0 / (n - 1)) * np.sum(x[1:])

    h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)

    f2 = g * h

    return [f1, f2]


def zdt3_f1(x):
    return zdt3(x)[0]


def zdt3_f2(x):
    return zdt3(x)[1]


# ZDT4
def zdt4(x):
    
    x = np.array(x)

    f1 = x[0]

    xm = x[1:]
    n_m = len(xm)

    g = 1.0 + 10.0 * n_m + np.sum(xm**2 - 10.0 * np.cos(4.0 * np.pi * xm))

    h = 1.0 - np.sqrt(f1 / g)

    f2 = g * h

    return [f1, f2]


def zdt4_f1(x):
    return zdt4(x)[0]


def zdt4_f2(x):
    return zdt4(x)[1]


# ZDT6
def zdt6(x):
    x = np.array(x)
    n = len(x)

    x1 = x[0]

    f1 = 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)

    xm = x[1:]
    g = 1.0 + 9.0 * ((np.sum(xm) / (n - 1)) ** 0.25)

    h = 1.0 - (f1 / g) ** 2

    f2 = g * h

    return [f1, f2]


def zdt6_f1(x):
    return zdt6(x)[0]


def zdt6_f2(x):
    return zdt6(x)[1]



# DTLZ1
def dtlz1(x, n_objs=3):
    """
    Multimodal, and planar Pareto Front
    """
    x = np.array(x)
    n_vars = len(x)
    k = n_vars - n_objs + 1

    xm = x[n_objs - 1 :]

    g = 100 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20 * np.pi * (xm - 0.5))))

    val = 0.5 * (1.0 + g)
    f = []

    for i in range(n_objs):
        f_val = val
        for j in range(n_objs - 1 - i):
            f_val *= x[j]
        if i > 0:
            f_val *= 1 - x[n_objs - 1 - i]

        f.append(f_val)

    return f


def dtlz1_f1(x):
    return dtlz1(x, 3)[0]


def dtlz1_f2(x):
    return dtlz1(x, 3)[1]


def dtlz1_f3(x):
    return dtlz1(x, 3)[2]


# DTLZ2
def dtlz2(x, n_objs=3):
    """
    Convex Pareto Front
    """
    x = np.array(x)
    k = len(x) - n_objs + 1
    g = np.sum((x[n_objs - 1 :] - 0.5) ** 2)
    val = 1.0 + g
    f = []
    for i in range(n_objs):
        f_val = val
        for j in range(n_objs - 1 - i):
            f_val *= np.cos(x[j] * np.pi / 2.0)
        if i > 0:
            f_val *= np.sin(x[n_objs - 1 - i] * np.pi / 2.0)
        f.append(f_val)
    return f


def dtlz2_f1(x):
    return dtlz2(x, 3)[0]


def dtlz2_f2(x):
    return dtlz2(x, 3)[1]


def dtlz2_f3(x):
    return dtlz2(x, 3)[2]


def dtlz3(x, n_objs=3):
    """
    Convex and multimodal Pareto Front
    """
    x = np.array(x)
    n_vars = len(x)
    k = n_vars - n_objs + 1

    xm = x[n_objs - 1 :]

    g = 100 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20 * np.pi * (xm - 0.5))))

    val = 1.0 + g

    f = []
    for i in range(n_objs):
        f_val = val
        for j in range(n_objs - 1 - i):
            f_val *= np.cos(x[j] * np.pi / 2.0)
        if i > 0:
            f_val *= np.sin(x[n_objs - 1 - i] * np.pi / 2.0)
        f.append(f_val)

    return f


def dtlz3_f1(x):
    return dtlz3(x, 3)[0]


def dtlz3_f2(x):
    return dtlz3(x, 3)[1]


def dtlz3_f3(x):
    return dtlz3(x, 3)[2]


# DTLZ4
def dtlz4(x, n_objs=3, alpha=100):
    """
    Biased problem and convex Pareto Front
    """
    x = np.array(x)
    n_vars = len(x)
    k = n_vars - n_objs + 1

    xm = x[n_objs - 1 :]
    g = np.sum((xm - 0.5) ** 2)
    val = 1.0 + g

    f = []

    x_p = np.power(x[: n_objs - 1], alpha)

    for i in range(n_objs):
        f_val = val
        for j in range(n_objs - 1 - i):
            f_val *= np.cos(x_p[j] * np.pi / 2.0)
        if i > 0:
            f_val *= np.sin(x_p[n_objs - 1 - i] * np.pi / 2.0)
        f.append(f_val)

    return f


def dtlz4_f1(x):
    return dtlz4(x, 3)[0]


def dtlz4_f2(x):
    return dtlz4(x, 3)[1]


def dtlz4_f3(x):
    return dtlz4(x, 3)[2]


# DTLZ5
def dtlz5(x, n_objs=3):
    """
    Degenerate front
    """
    x = np.array(x)
    xm = x[n_objs - 1 :]
    g = np.sum((xm - 0.5) ** 2)
    val = 1.0 + g

    theta1 = x[0] * np.pi / 2.0

    theta2 = (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[1])

    f1 = val * np.cos(theta1) * np.cos(theta2)
    f2 = val * np.cos(theta1) * np.sin(theta2)
    f3 = val * np.sin(theta1)

    return [f1, f2, f3]


def dtlz5_f1(x):
    return dtlz5(x)[0]


def dtlz5_f2(x):
    return dtlz5(x)[1]


def dtlz5_f3(x):
    return dtlz5(x)[2]


# DTLZ6
def dtlz6(x, n_objs=3):
    """
    Gradient and degenerate front
    """
    x = np.array(x)

    xm = x[n_objs - 1 :]
    g = np.sum(np.power(xm, 0.1))

    val = 1.0 + g

    theta1 = x[0] * np.pi / 2.0
    theta2 = (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[1])

    f1 = val * np.cos(theta1) * np.cos(theta2)
    f2 = val * np.cos(theta1) * np.sin(theta2)
    f3 = val * np.sin(theta1)

    return [f1, f2, f3]


def dtlz6_f1(x):
    return dtlz6(x)[0]


def dtlz6_f2(x):
    return dtlz6(x)[1]


def dtlz6_f3(x):
    return dtlz6(x)[2]


# DTLZ7
def dtlz7(x, n_objs=3):
    """
    Disconnected pareto front
    """
    x = np.array(x)
    n_vars = len(x)
    k = n_vars - n_objs + 1

    f = []
    for i in range(n_objs - 1):
        f.append(x[i])

    xm = x[n_objs - 1 :]
    g = 1.0 + (9.0 / k) * np.sum(xm)

    h = n_objs
    for i in range(n_objs - 1):
        fi = f[i]
        h -= (fi / (1.0 + g)) * (1.0 + np.sin(3.0 * np.pi * fi))

    f_last = (1.0 + g) * h
    f.append(f_last)

    return f


def dtlz7_f1(x):
    return dtlz7(x)[0]


def dtlz7_f2(x):
    return dtlz7(x)[1]


def dtlz7_f3(x):
    return dtlz7(x)[2]

