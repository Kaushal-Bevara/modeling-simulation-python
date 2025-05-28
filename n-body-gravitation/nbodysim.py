# N Body Gravitation Simulation

from matplotlib import pyplot as plt
from matplotlib import animation as animation
import numpy as np
import sympy as smp
# import vpython as vp


# sets positon of each planet
def set_positions(dt):
    # pairwise calculation of forces
    for i in range(len(bodies) - 1):
        for j in range(i + 1, len(bodies)):
            bodies[i].set_forces(bodies[j])

    for body in bodies:
        pos, vel, m, force = body.get_info()

        acc = force / m
        body.pos += vel * dt

        body.vel += acc * dt
        # makes sure that the speed is not too great
        if (np.linalg.norm(vel) > 10):
            body.vel = vel * 10.0 / np.linalg.norm(vel)
        # set body force equal to 0 to be recalculated for next frame
        body.force = np.array((0.0, 0.0))

# updates each frame
def update_func(frame):
    set_positions(.1)
    for i in range(len(body_anims)):
        pos, vel, m, force = bodies[i].get_info()
        print(bodies[i].get_info())
        body_anims[i].set_data([pos[0]], [pos[1]])
    return body_anims

class Body:
    def __init__(self, pos, vel, m):
        self.pos = pos
        self.vel = vel
        self.m = m
        self.force = np.array((0.0, 0.0))
    def set_forces(self, otherBody):
        # we will assume gravitational constant is 1, as it will not affect simulation

        r = otherBody.pos-self.pos # vector from self to otherbody
        r_mag = np.sqrt(r.dot(r))

        # Newton's universal law of gravitation used to find accelerations
        # added softening parameter of .01 to make sure that bodies aren't ejected
        f = 1000*(self.m * otherBody.m * r) / ((r_mag + 10) ** 3)

        # sums forces
        
        self.force += f
        otherBody.force += -f 

        return f
    def get_info(self):
        return (self.pos, self.vel, self.m, self.force)


bodies = []
body_anims = []

def main():
    for i in range(5):
        rand_pos = np.random.rand(2)*500.0 - 250
        bodies.append(Body(rand_pos, np.array((0.0, 0.0)), np.random.randint(3) + 1))

    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylim(-260, 260)
    ax.set_xlim(-260, 260)
    plt.grid(False)

    # make animation objects

    for body in bodies:
        body_anim, = ax.plot([body.pos[0]], [body.pos[1]], 'o', markersize=body.m, color='green')
        body_anims.append(body_anim)

    ani = animation.FuncAnimation(fig, update_func, frames=10000, interval=1, blit=True)
    plt.show()

if __name__ == "__main__":
    main()