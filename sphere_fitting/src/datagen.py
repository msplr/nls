import numpy as np

r = 1.0
p = [1, 2, -3]
xs, ys, zs = [], [], []
for phi in np.linspace(-1.0, 1.0, 100):
    for psi in np.linspace(-0.7, 0.7, 100):
        n = 1e-2 * np.random.randn(3)
        x = p[0] + r * np.cos(psi) * np.cos(phi) + n[0]
        y = p[1] + r * np.cos(psi) * np.sin(phi) + n[1]
        z = p[2] + r * np.sin(psi) + n[2]
        print(f"{x},{y},{z},")
        xs += [x]
        ys += [y]
        zs += [z]


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(xs, ys, zs)

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = p[0] + r * np.outer(np.cos(u), np.sin(v))
y = p[1] + r * np.outer(np.sin(u), np.sin(v))
z = p[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color="grey")

ax.set_box_aspect([1, 1, 1])
plt.show()
