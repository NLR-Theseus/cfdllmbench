import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
v_inf = 1.0
d1 = 1.0
din = d1 / 2.0  # Inner boundary in the physical domain
d2 = 20.0 * d1
dout = d2 / 2.0  # outer boundary in the physical domain

# Dicretization and solver parameters
num_elem_theta = 40
num_elem_radial = 60
beta = 1.93  # over-relaxation factor
nu = 0.005  # Fluid kinematic viscosity
theta = np.linspace(0, 2 * np.pi, num_elem_theta)
r = np.linspace(din, dout, num_elem_radial)
dr = r[1] - r[0]
dtheta = theta[1] - theta[0]

t = np.linspace(0, 10, num=201, endpoint=True)
dt = t[1] - t[0]
print(dt)
max_time_steps = np.asarray(np.shape(t)[0])
# print(max_time_steps)

# Initialize arrays
omega0 = np.zeros([num_elem_radial, num_elem_theta])
psi0 = np.zeros([num_elem_radial, num_elem_theta])

## Getting x and y coordinates and boundary conditions for streamfunction (psi)
# getting x and y coordinates
x = np.zeros([num_elem_radial, num_elem_theta])
y = np.zeros([num_elem_radial, num_elem_theta])

for i in range(num_elem_radial):
    for j in range(num_elem_theta):
        x[i, j] = r[i] * np.cos(theta[j])
        y[i, j] = r[i] * np.sin(theta[j])

# set streamfunction = constant (=20) on cylinder wall
psi0[0, :] = 20.0

# set streamfunction = v_inf * y + constant (=20) on outer boundary of physical domain
psi0[-1, :] = v_inf * y[-1, :] + 20.0

# defining few arrays for holding coefficient in discretized equations
a = np.zeros([num_elem_radial, num_elem_theta])
b = np.zeros([num_elem_radial, num_elem_theta])
c = np.zeros([num_elem_radial, num_elem_theta])
d = np.zeros([num_elem_radial, num_elem_theta])
e = np.zeros([num_elem_radial, num_elem_theta])
a1 = np.zeros([num_elem_radial, num_elem_theta])
b1 = np.zeros([num_elem_radial, num_elem_theta])
c1 = np.zeros([num_elem_radial, num_elem_theta])
d1 = np.zeros([num_elem_radial, num_elem_theta])
e1 = np.zeros([num_elem_radial, num_elem_theta])

# defining arrays for velocity components and absolute velocity
u_r = np.zeros([num_elem_radial, num_elem_theta])
u_t = np.zeros([num_elem_radial, num_elem_theta])
u = np.zeros([num_elem_radial, num_elem_theta])
u_x = np.zeros([num_elem_radial, num_elem_theta])
u_y = np.zeros([num_elem_radial, num_elem_theta])

# defining arrays for extracting wall velocity data
probe_u_r = np.zeros([max_time_steps])
probe_u_t = np.zeros([max_time_steps])

## Main program
omega = omega0.copy()
psi = psi0.copy()

for time_step in range(max_time_steps):
    residual = 10.0
    # Solving Poisson's equation
    while (np.abs(residual) > 0.01):
        # Solve for interior points excluding periodic boundaries
        for i in range(1, num_elem_radial - 1):
            for j in range(1, num_elem_theta - 1):
                a[i, j] = (r[i + 1] + r[i]) / (2.0 * dr * dr)
                b[i, j] = (r[i - 1] + r[i]) / (2.0 * dr * dr)
                c[i, j] = 1.0 / (r[i] * dtheta * dtheta)
                d[i, j] = 1.0 / (r[i] * dtheta * dtheta)
                e[i, j] = (2.0 * r[i] + r[i + 1] + r[i - 1]) / (2.0 * dr * dr) + 2.0 / (r[i] * dtheta * dtheta)
                psi[i, j] = (1 - beta) * psi0[i, j] + (beta / e[i, j]) * (
                            a[i, j] * psi0[i + 1, j] + b[i, j] * psi[i - 1, j] \
                            + c[i, j] * psi0[i, j + 1] + d[i, j] * psi[i, j - 1] + r[i] * omega0[i, j])

        # apply periodic boundary conditions
        for i in range(1, num_elem_radial - 1):
            a[i, 0] = (r[i + 1] + r[i]) / (2.0 * dr * dr)
            b[i, 0] = (r[i - 1] + r[i]) / (2.0 * dr * dr)
            c[i, 0] = 1.0 / (r[i] * dtheta * dtheta)
            d[i, 0] = 1.0 / (r[i] * dtheta * dtheta)
            e[i, 0] = (2.0 * r[i] + r[i + 1] + r[i - 1]) / (2.0 * dr * dr) + 2.0 / (r[i] * dtheta * dtheta)
            psi[i, 0] = (1 - beta) * psi0[i, 0] + (beta / e[i, 0]) * (a[i, 0] * psi0[i + 1, 0] + b[i, 0] * psi[i - 1, 0] \
                                                                      + c[i, 0] * psi0[i, 1] + d[i, 0] * psi[i, -2] + r[
                                                                          i] * omega0[i, 0])

        psi[:, -1] = psi[:, 0]

        # calculating residue
        residual = np.sqrt(np.sum((psi - psi0) ** 2.0))
        psi0 = psi.copy()

    # Extracting velocity from psi data
    for i in range(num_elem_radial):
        for j in range(1, num_elem_theta - 1):
            u_r[i, j] = (1.0 / r[i]) * ((psi[i, j + 1] - psi[i, j - 1]) / (2 * dtheta))

    for i in range(num_elem_radial):
        u_r[i, 0] = (1.0 / r[i]) * ((psi[i, 1] - psi[i, 0]) / (dtheta))
        u_r[i, -1] = (1.0 / r[i]) * ((psi[i, -1] - psi[i, -2]) / (dtheta))

    for i in range(1, num_elem_radial - 1):
        for j in range(num_elem_theta):
            u_t[i, j] = -(psi[i + 1, j] - psi[i - 1, j]) / (2.0 * dr)

    for j in range(num_elem_theta):
        u_t[0, j] = -(psi[1, j] - psi[0, j]) / (dr)
        u_t[-1, j] = -(psi[-1, j] - psi[-2, j]) / (dr)

    for i in range(num_elem_radial):
        for j in range(num_elem_theta):
            u_x[i, j] = -np.sin(theta[j]) * u_t[i, j] + np.cos(theta[j]) * u_r[i, j]
            u_y[i, j] = np.sin(theta[j]) * u_r[i, j] + np.cos(theta[j]) * u_t[i, j]
            u[i, j] = np.sqrt(u_x[i, j] ** 2.0 + u_y[i, j] ** 2.0)

    # Vorticity calculations
    # Solving for all interior nodes excluding periodic boundaries
    for i in range(1, num_elem_radial - 1):
        for j in range(1, num_elem_theta - 1):
            a1[i, j] = ((-1.0 / dt) + (np.abs(u_r[i, j]) / dr) + np.abs(u_t[i, j] / r[i]) / dtheta + \
                        nu / dr ** 2.0 + nu / ((r[i] ** 2.0) * dtheta ** 2.0))
            b1[i, j] = (-u_r[i, j] / (2.0 * dr) - np.abs(u_r[i, j]) / (2.0 * dr) - nu / dr ** 2.0 + \
                        nu / (2.0 * r[i] * dr))
            c1[i, j] = (-u_t[i, j] / (2.0 * r[i] * dtheta) - np.abs(u_t[i, j] / r[i]) / (2 * dtheta) - \
                        nu / ((r[i] ** 2.0) * dtheta ** 2.0))
            d1[i, j] = (u_r[i, j] / (2.0 * dr) - abs(u_r[i, j]) / (2.0 * dr) - nu / dr ** 2.0 - \
                        nu / (2.0 * r[i] * dr))
            e1[i, j] = (u_t[i, j] / (r[i] * 2.0 * dtheta) - np.abs(u_t[i, j] / r[i]) / (2.0 * dtheta) - \
                        (nu / (r[i] ** 2.0 * dtheta ** 2.0)))
            omega[i, j] = -dt * (a1[i, j] * omega0[i, j] + b1[i, j] * omega0[i - 1, j] + c1[i, j] * omega0[i, j - 1] + \
                                 d1[i, j] * omega0[i + 1, j] + e1[i, j] * omega0[i, j + 1])

    # apply periodic boundary conditions
    for i in range(1, num_elem_radial - 1):
        a1[i, 0] = ((-1.0 / dt) + (np.abs(u_r[i, 0]) / dr) + np.abs(u_t[i, 0] / r[i]) / dtheta + \
                    nu / dr ** 2.0 + nu / ((r[i] ** 2.0) * dtheta ** 2.0))
        b1[i, 0] = (-u_r[i, 0] / (2.0 * dr) - np.abs(u_r[i, 0]) / (2.0 * dr) - nu / dr ** 2.0 + \
                    nu / (2.0 * r[i] * dr))
        c1[i, 0] = (-u_t[i, 0] / (2.0 * r[i] * dtheta) - np.abs(u_t[i, 0] / r[i]) / (2 * dtheta) - \
                    nu / ((r[i] ** 2.0) * dtheta ** 2.0))
        d1[i, 0] = (u_r[i, 0] / (2.0 * dr) - abs(u_r[i, 0]) / (2.0 * dr) - nu / dr ** 2.0 - \
                    nu / (2.0 * r[i] * dr))
        e1[i, 0] = (u_t[i, 0] / (r[i] * 2.0 * dtheta) - np.abs(u_t[i, 0] / r[i]) / (2.0 * dtheta) - \
                    (nu / (r[i] ** 2.0 * dtheta ** 2.0)))
        omega[i, 0] = -dt * (a1[i, 0] * omega0[i, 0] + b1[i, 0] * omega0[i - 1, 0] + c1[i, 0] * omega0[i, -2] + \
                             d1[i, 0] * omega0[i + 1, 0] + e1[i, 0] * omega0[i, 1])

    omega[:, -1] = omega[:, 0]
    omega[0, :] = 2.0 * (((psi[0, :] - psi[1, :]))) / dr ** 2.0
    omega[-1, :] = 0.0
    omega0 = omega.copy()

    probe_u_r[time_step] = u_r[19, 0]
    probe_u_t[time_step] = u_t[19, 0]
    # print(time_step, probe_u_r[time_step], probe_u_t[time_step])
    print(time_step, time_step * dt)

# Extracting velocity from psi data
for i in range(num_elem_radial):
    for j in range(1, num_elem_theta - 1):
        u_r[i, j] = (1.0 / r[i]) * ((psi[i, j + 1] - psi[i, j - 1]) / (2 * dtheta))

for i in range(num_elem_radial):
    u_r[i, 0] = (1.0 / r[i]) * ((psi[i, 1] - psi[i, 0]) / (dtheta))
    u_r[i, -1] = (1.0 / r[i]) * ((psi[i, -1] - psi[i, -2]) / (dtheta))

for i in range(1, num_elem_radial - 1):
    for j in range(num_elem_theta):
        u_t[i, j] = -(psi[i + 1, j] - psi[i - 1, j]) / (2.0 * dr)

for j in range(num_elem_theta):
    u_t[0, j] = -(psi[1, j] - psi[0, j]) / (dr)
    u_t[-1, j] = -(psi[-1, j] - psi[-2, j]) / (dr)

for i in range(num_elem_radial):
    for j in range(num_elem_theta):
        u_x[i, j] = -np.sin(theta[j]) * u_t[i, j] + np.cos(theta[j]) * u_r[i, j]
        u_y[i, j] = np.sin(theta[j]) * u_r[i, j] + np.cos(theta[j]) * u_t[i, j]
        u[i, j] = np.sqrt(u_x[i, j] ** 2.0 + u_y[i, j] ** 2.0)

plt.figure()
plt.plot(t, probe_u_r, '.r', t, probe_u_t, '.b')
plt.tight_layout()
plt.grid()
plt.legend(['radial velocity', 'tangential velocity'])
plt.xlabel('time')
plt.ylabel('velocity')
plt.title('radial and tangential velocity vs time')
plt.show()

plt.figure()
plt.plot(theta, u[0, :], '-o', markerfacecolor='r')
plt.tight_layout()
plt.grid()
plt.xlabel(r'$\Theta$ in radians')
plt.ylabel('velocity')
plt.title('velocity magnitude over the cylinder wall')
plt.show()

plt.figure()
plt.pcolormesh(x, y, omega, cmap=plt.cm.Spectral, shading='gouraud')
plt.fill_between(din * np.cos(np.linspace(0, 2 * np.pi)), din * np.sin(np.linspace(0, 2 * np.pi)), color='black')
plt.axis('equal')
plt.grid()
plt.title(r'Vorticity, $\Omega$')
plt.colorbar()
plt.show()

plt.figure()
plt.quiver(x, y, u_x, u_y, color='blue')
plt.fill_between(din * np.cos(np.linspace(0, 2 * np.pi)), din * np.sin(np.linspace(0, 2 * np.pi)), color='black')
plt.axis('equal')
plt.grid()
plt.title(r'Velocity vector')
plt.show()

plt.figure()
plt.pcolormesh(x, y, psi, cmap=plt.cm.Spectral, shading='gouraud')
plt.fill_between(din * np.cos(np.linspace(0, 2 * np.pi)), din * np.sin(np.linspace(0, 2 * np.pi)), color='black')
plt.axis('equal')
plt.grid()
plt.title(r'streamline')
plt.colorbar()
plt.show()

plt.figure()
plt.pcolormesh(x, y, u, cmap=plt.cm.Spectral, shading='gouraud')
plt.fill_between(din * np.cos(np.linspace(0, 2 * np.pi)), din * np.sin(np.linspace(0, 2 * np.pi)), color='black')
plt.axis('equal')
plt.grid()
plt.title(r'velocity magnitude')
plt.colorbar()
plt.show()

plt.subplot(3, 2, 1)
plt.plot(t, probe_u_r, '.r', t, probe_u_t, '.b')
plt.tight_layout()
plt.grid()
plt.legend(['radial velocity', 'tangential velocity'])
plt.xlabel('time')
plt.ylabel('velocity')
plt.title('radial and tangential velocity vs time')

plt.subplot(3, 2, 2)
plt.plot(theta, u[0, :], '-o', markerfacecolor='r')
plt.tight_layout()
plt.grid()
plt.xlabel(r'$\Theta$ in radians')
plt.ylabel('velocity')
plt.title('velocity magnitude over the cylinder wall')

plt.subplot(3, 2, 3)
plt.pcolormesh(x, y, omega, cmap=plt.cm.Spectral, shading='gouraud')
plt.fill_between(din * np.cos(np.linspace(0, 2 * np.pi)), din * np.sin(np.linspace(0, 2 * np.pi)), color='black')
plt.axis('equal')
plt.grid()
plt.title(r'Vorticity, $\Omega$')
plt.colorbar()

plt.subplot(3, 2, 4)
plt.quiver(x, y, u_x, u_y, color='cyan')
plt.fill_between(din * np.cos(np.linspace(0, 2 * np.pi)), din * np.sin(np.linspace(0, 2 * np.pi)), color='black')
plt.axis('equal')
plt.grid()
plt.title(r'Velocity vector')

plt.subplot(3, 2, 5)
plt.pcolormesh(x, y, psi, cmap=plt.cm.Spectral, shading='gouraud')
plt.fill_between(din * np.cos(np.linspace(0, 2 * np.pi)), din * np.sin(np.linspace(0, 2 * np.pi)), color='black')
plt.axis('equal')
plt.grid()
plt.title(r'streamline')
plt.colorbar()

plt.subplot(3, 2, 6)
plt.pcolormesh(x, y, u, cmap=plt.cm.Spectral, shading='gouraud')
plt.fill_between(din * np.cos(np.linspace(0, 2 * np.pi)), din * np.sin(np.linspace(0, 2 * np.pi)), color='black')
plt.axis('equal')
plt.grid()
plt.title(r'velocity magnitude')
plt.colorbar()
plt.show()

np.save('../../PDE_Benchmark/results/solution/psi_Flow_Past_Circular_Cylinder.npy', psi)
np.save('../../PDE_Benchmark/results/solution/omega_Flow_Past_Circular_Cylinder.npy', omega)








