# %% [markdown]
# # Double Pendulum
# 
# A double pendulum simulation using Lagrangian mechanics.

# %%
import numpy as np
from scipy.integrate import odeint
import sympy as smp
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation


# We will represent the time in seconds using the symbol t. The constants g, m1, m2, l1, l2 will represent the gravitational force, masses of the pendulums, and the length of the pendulums, respectively.
 
# %%
t = smp.symbols('t')

g, m1, m2, l1, l2 = smp.symbols("g m1:3 l1:3", positive = True)

the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)

the1_func = the1(t)
the2_func = the2(t)

x1 = smp.sin(the1_func) * l1
y1 = -smp.cos(the1_func) * l1
x2 = x1 + smp.sin(the2_func) * l2
y2 = y1 - smp.cos(the2_func) * l2

the1_d = smp.diff(the1_func, t)
the2_d = smp.diff(the2_func, t)
x1_dd = smp.diff(smp.diff(x1, t), t)
y1_dd = smp.diff(smp.diff(y1, t), t)
x2_dd = smp.diff(smp.diff(x2, t), t)
y2_dd = smp.diff(smp.diff(y2, t), t)

# Equations for The Second Derivative of Angles

eq1 = smp.Eq(smp.sin(the1_func) * (m1*y1_dd + m2*y2_dd + (m2 + m1)*g), -smp.cos(the1_func) * (m1 * x1_dd + m2 * x2_dd))
eq2 = smp.Eq(smp.sin(the2_func) * (m2 * (y2_dd + g)), -smp.cos(the2_func) * m2 * x2_dd)


the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)
solutions = smp.solve([eq1, eq2], the1_dd, the2_dd)
solutions[the2_dd].simplify()

dw1dt_f = smp.lambdify((t,g,m1,m2,l1,l2,the1_func,the2_func,the1_d,the2_d), solutions[the1_dd])
dw2dt_f = smp.lambdify((t,g,m1,m2,l1,l2,the1_func,the2_func,the1_d,the2_d), solutions[the2_dd])
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)

def dfdt(f, t, g, m1, m2, l1, l2):
    the_1, w1, the_2, w2 = f
    return [
        dthe1dt_f(w1),
        dw1dt_f(t, g, m1, m2, l1, l2, the_1, the_2, w1, w2),
        dthe2dt_f(w2),
        dw2dt_f(t, g, m1, m2, l1, l2, the_1, the_2, w1, w2),
    ]
print('preodeint done')

t =  np.linspace(0, 50, 1000)
g = 9.81
m1 = 10
m2 = 7
l1 = 1
l2 = 1
ans = odeint(dfdt, y0=[1.5, 2, 1, 1], t=t, args=(g, m1, m2, l1, l2))
ans1 = odeint(dfdt, y0=[1.50001, 2, 1, 1], t=t, args=(g, m1, m2, l1, l2))
pend_infos = [((10, 1), (7, 1)) for _ in range(5)]
theta_pairs = [
    # using list comprehension to get ode solutions for penduli.
    # each pendulum will be offset by 1e-4 radians of the previous pendulum.
    odeint(dfdt, y0=[3 + i*1e-4, 2, 1, 1], t=t, args=(g, 
                                                         pend_infos[i][0][0], 
                                                         pend_infos[i][1][0], 
                                                         pend_infos[i][0][1], 
                                                         pend_infos[i][1][1])).T
    for i in range(len(pend_infos))
]
print('odeint done')

the_1 = ans.T[0]
the_2 = ans.T[2]
the_3 = ans1.T[0]
the_4 = ans1.T[2]

def get_x1y1x2y2(t, the1, the2, l1, l2):
    return (l1*np.sin(the1),
            -l1*np.cos(the1),
            l1*np.sin(the1) + l2*np.sin(the2),
            -l1*np.cos(the1) - l2*np.cos(the2))
x1, y1, x2, y2 = get_x1y1x2y2(t, the_1, the_2, l1, l2)
x3, y3, x4, y4 = get_x1y1x2y2(t, the_3, the_4, l1, l2)

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_axis_off()
ax.set_aspect('equal', adjustable='box')
ax.set_ylim(-l1-l2 - 0.3, l1+l2 + 0.3)
ax.set_xlim(-l1-l2 - 0.3, l1+l2 + 0.3)
plt.grid(False)

plot_paths1, = ax.plot([], [])
plot_paths2, = ax.plot([], [])
artists = []
artist_objects = []
coordinate_pairs = []

for i in range(len(theta_pairs)):
    (m1, l1), (m2, l2) = pend_infos[i] # unboxes pendulum informations
    x1, y1, x2, y2 = get_x1y1x2y2(0, theta_pairs[i][0], theta_pairs[i][2], l1, l2)

    pend1, = ax.plot([0, x1[0]], [0, y1[0]], lw=1)
    mass1, = ax.plot([x1[0]], [y1[0]], 'o', markersize=2*m1, color='green')

    pend2, = ax.plot([x1[0], x2[0]], [y1[0], y2[0]], lw=1)
    mass2, = ax.plot([x2[0]], [y2[0]], 'o', markersize=2*m2, color='green')

    plot_path, = ax.plot([], [], zorder=0)
    artists.extend((pend1, mass1, pend2, mass2, plot_path))
    artist_objects.append((pend1, mass1, pend2, mass2, plot_path))
    coordinate_pairs.append((x1, y1, x2, y2))


def update_data(frame):

    for i in range(len(artist_objects)):
        pend1, mass1, pend2, mass2, plot_path = artist_objects[i]
        x1, y1, x2, y2 = coordinate_pairs[i]

        pend1.set_data([0, x1[frame]], [0, y1[frame]])
        mass1.set_data([x1[frame]], [y1[frame]])

        pend2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
        mass2.set_data([x2[frame]], [y2[frame]])

        plot_path.set_data(x2[:frame], y2[:frame])

    return artists

ani = animation.FuncAnimation(fig, update_data, frames=len(t), interval=70, blit=True)
ani.save("five-double-pendulums.gif")
