import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as npr

plt.close('all')

theta = np.pi / 6
phi = np.pi / 4
L1 = 1
L2 = 2
L3 = 3

R1 = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]])
R2 = np.array([[np.cos(phi), 0, np.sin(phi)],
               [0, 1, 0],
               [-np.sin(phi), 0, np.cos(phi)]])
R = R1.dot(R2)

print(f"Repère: {R[:, 0]}, {R[:, 1]}, {R[:, 2]}")

Matr = R.T @ np.diag([1 / L1 ** 2, 1 / L2 ** 2, 1 / L3 ** 2]) @ R
print(Matr)

a = Matr[0, 0]
b = Matr[1, 1]
c = Matr[2, 2]
d = 2 * Matr[0, 1]
e = 2 * Matr[0, 2]
f = 2 * Matr[1, 2]
# a = np.cos(phi)**2*np.cos(theta)**2/L1**2 + np.cos(phi)**2*np.sin(theta)*2/L2**2 + np.sin(phi)**2/L3**2
# b = np.cos(theta)**2/L2**2 + np.sin(theta)**2/L1**2
# c = np.cos(theta)**2*np.sin(phi)**2/L1**2 + np.sin(phi)**2*np.sin(theta)**2/L2**2 + np.cos(phi)**2/L3**2
# d =  - 2*np.cos(phi)*np.cos(theta)*np.sin(theta)/L1**2 + 2*np.cos(phi)*np.cos(theta)*np.sin(theta)/L2**2
# e = 2*np.cos(phi)*np.cos(theta)**2*np.sin(phi)/L1**2 + 2*np.cos(phi)*np.sin(phi)*np.sin(theta)**2/L2**2  - 2*np.cos(phi)*np.sin(phi)/L3**2
# f = - 2*np.cos(theta)*np.sin(phi)*np.sin(theta)/L1**2 + 2*np.cos(theta)*np.sin(phi)*np.sin(theta)/L2**2
g = -1
beta = b - d ** 2 / (4 * a)
gamma = c - e ** 2 / (4 * a)
delta = f - d * e / (2 * a)


# print(f"a={a}, b={b}, c={c}, d={d}, e={e}, f={f}, g={g}, beta={beta}, gamma={gamma}, delta={delta}")

def eta(z):
    return (delta ** 2 / (4 * beta) - gamma) * z ** 2 - g


def points(psi, z, bruit):
    perturb = bruit * npr.rand(2, len(psi))
    y = np.sqrt(eta(z) / beta) * np.sin(psi) - delta / (2 * beta) * z + perturb[0, :]
    x = np.sqrt(eta(z) / a) * np.cos(psi) - d / (2 * a) * y - e / (2 * a) * z + perturb[1, :]
    return x, y, z * np.ones(len(psi))


def moindre_carres(x, y, z):
    n = len(x)
    A = np.zeros((n, 9))
    A[:, 0] = x ** 2
    A[:, 1] = y ** 2
    A[:, 2] = z ** 2
    A[:, 3] = x * y
    A[:, 4] = x * z
    A[:, 5] = y * z
    A[:, 6] = x
    A[:, 7] = y
    A[:, 8] = z
    B = np.ones(n)
    coeff = np.linalg.solve(A.T @ A, A.T @ B)
    return coeff


def ellipsoide(coeff):
    A = np.array([[coeff[0], coeff[3] / 2, coeff[4] / 2],
                  [coeff[3] / 2, coeff[1], coeff[5] / 2],
                  [coeff[4] / 2, coeff[5] / 2, coeff[2]]])
    vp, vect_p = np.linalg.eigh(A)
    U = np.array(coeff[6:])
    centre = -0.5 * np.linalg.solve(A, U)
    longueurs = 1 / np.sqrt(vp / (1 - centre.dot(A @ centre)))
    return centre, vect_p, longueurs


def calc_pts_ellipsoide(centre, vect_p, longueurs):
    # Création d'une grille en coordonnées sphériques
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    U, V = np.meshgrid(u, v)
    # Conversion en coordonnées cartésiennes
    X = longueurs[0] * np.cos(U) * np.sin(V)
    Y = longueurs[1] * np.sin(U) * np.sin(V)
    Z = longueurs[2] * np.cos(V)
    T = np.zeros((3, 10000))
    T[0, :] = X.reshape(10000)[:]
    T[1, :] = Y.reshape(10000)[:]
    T[2, :] = Z.reshape(10000)[:]
    T = vect_p.T @ T
    # print(np.max(a*T[0,:]**2+b*T[1,:]**2+c*T[2,:]**2+d*T[0,:]*T[1,:]+e*T[0,:]*T[2,:]+f*T[1,:]*T[2,:]+g))
    X = T[0, :].reshape((100, 100))
    Y = T[1, :].reshape((100, 100))
    Z = T[2, :].reshape((100, 100))
    return X + centre[0], Y + centre[1], Z + centre[2]


if __name__ == "__main__":
    X, Y, Z = calc_pts_ellipsoide(np.zeros(3), R, np.array([L1, L2, L3]))

    # Tracé de l'ellipsoïde
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, color='b', alpha=0.3)

    # Configuration de l'affichage
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ellipsoïde 3D')

    psi = np.linspace(-np.pi / 2, np.pi / 2, 16)
    bruit = 0.3
    lpx = []
    lpy = []
    lpz = []
    # verif = []
    for z in np.linspace(0, 2, 5):
        px, py, pz = points(psi, z, bruit)
        # verif.append(np.max(a*px**2+b*py**2+c*pz**2+d*px*py+e*px*pz+f*py*pz+g))
        ax.scatter(px, py, pz)
        lpx.append(px)
        lpy.append(py)
        lpz.append(pz)
    # print(max(verif))
    PX = np.concatenate(lpx)
    PY = np.concatenate(lpy)
    PZ = np.concatenate(lpz)

    coeff = moindre_carres(PX, PY, PZ)
    centre, vect_p, longueurs = ellipsoide(coeff)
    Xcalc, Ycalc, Zcalc = calc_pts_ellipsoide(centre, vect_p.T, longueurs)
    ax.plot_surface(Xcalc, Ycalc, Zcalc, rstride=4, cstride=4, color='c', alpha=0.3)

    X0 = np.zeros(3)
    Y0 = np.zeros(3)
    Z0 = np.zeros(3)
    ax.quiver(X0, Y0, Z0, R[:, 0], R[:, 1], R[:, 2])
    ax.quiver(X0, Y0, Z0, vect_p[0, :], vect_p[1, :], vect_p[2, :])

    ax.set_aspect('equal')

    plt.show()