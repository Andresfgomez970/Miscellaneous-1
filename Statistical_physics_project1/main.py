from random_walks import *  # Import all routines for th simulation


def SecondPoint1():
    N_try = 4
    N_steps = 100

    positions = TwoDimArrayLattice(N_try, N_steps)

    xpos = positions[0][:, 0:1].transpose()[0]
    ypos = positions[0][:, 1:].transpose()[0]

    xpos1 = positions[1][:, 0:1].transpose()[0]
    ypos1 = positions[1][:, 1:].transpose()[0]

    xpos2 = positions[2][:, 0:1].transpose()[0]
    ypos2 = positions[2][:, 1:].transpose()[0]

    xpos3 = positions[3][:, 0:1].transpose()[0]
    ypos3 = positions[3][:, 1:].transpose()[0]

    shift = 0.1
    plt.figure(figsize=(10, 6))
    plt.title("Some Random Paths in Lattice, Nsteps=100", size=14)
    plt.plot(xpos, ypos, 'k.-', label='path1')
    plt.plot(xpos1 + shift, ypos1, 'b.-', label='path2')
    plt.plot(xpos2 - shift, ypos2, 'g.-', label='path3')
    plt.plot(xpos3, ypos3 + shift, 'c.-', label='path4')
    N_limit = np.abs(positions).max() + 1
    plt.xlim(-N_limit, N_limit)
    plt.ylim(-N_limit, N_limit)
    plt.plot(0, 0, 'ro', label="initital position")
    plt.xlabel("x position", size=12)
    plt.ylabel("y position", size=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("alleatory_paths_lattice.png")
    plt.show()


def SecondPoint2():
    N_try = 4
    N_steps = 100

    positions = TwoDimArrayOffLattice(N_try, N_steps)

    xpos = positions[0][:, 0:1].transpose()[0]
    ypos = positions[0][:, 1:].transpose()[0]

    xpos1 = positions[1][:, 0:1].transpose()[0]
    ypos1 = positions[1][:, 1:].transpose()[0]

    xpos2 = positions[2][:, 0:1].transpose()[0]
    ypos2 = positions[2][:, 1:].transpose()[0]

    xpos3 = positions[3][:, 0:1].transpose()[0]
    ypos3 = positions[3][:, 1:].transpose()[0]

    shift = 0.1
    plt.figure(figsize=(10, 6))
    plt.title("Some Random Paths off lattice, Nsteps=100", size=14)
    plt.plot(xpos, ypos, 'k.-', label='path1')
    plt.plot(xpos1 + shift, ypos1, 'b.-', label='path2')
    plt.plot(xpos2 - shift, ypos2, 'g.-', label='path3')
    plt.plot(xpos3, ypos3 + shift, 'c.-', label='path4')
    N_limit = np.abs(positions).max() + 1
    plt.xlim(-N_limit, N_limit)
    plt.ylim(-N_limit, N_limit)
    plt.plot(0, 0, 'ro', label="initital position")
    plt.xlabel("x position", size=12)
    plt.ylabel("y position", size=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("alleatory_paths_off_lattice.png")
    plt.show()


def SecondPoint3():
    N_try = 100
    N_steps_list = np.arange(100, 1001, 50)
    d_mean_lattice = np.zeros(len(N_steps_list))
    d_mean_off_lattice = np.zeros(len(N_steps_list))
    for i in range(len(N_steps_list)):
        positions = TwoDimArrayLattice(N_try, N_steps_list[i])
        d = positions[:, -1, 0:1]**2 + positions[:, -1, 1:2]**2
        d_mean_lattice[i] = d.mean()
        positions = TwoDimArrayOffLattice(N_try, N_steps_list[i])
        d = positions[:, -1, 0:1]**2 + positions[:, -1, 1:2]**2
        d_mean_off_lattice[i] = d.mean()

    plt.figure(figsize=(10, 6))
    plt.title("Number of Steps vs Mean Square Distance", size=14)

    popt, pcov = optimize.curve_fit(lineal, N_steps_list, d_mean_lattice)
    popt2, pcov2 = optimize.curve_fit(lineal, N_steps_list, d_mean_off_lattice)

    plt.plot(N_steps_list, d_mean_lattice, "bo", label="In lattice data")
    plt.plot(N_steps_list, d_mean_off_lattice, "ko", label="Off latice data")

    plt.plot(N_steps_list, lineal(N_steps_list, *popt), "b-",
             label="In lattice adjust")
    plt.plot(N_steps_list, lineal(N_steps_list, *popt2), "k-",
             label="Off lattice adjust")

    plt.xlabel("Number of Steps", size=12)
    plt.ylabel("Mean distance", size=12)
    plt.legend(loc="best")
    plt.savefig("adjustements_relationship.png")
    plt.show()


# PositionMappingAndGraph(N_try=40000, N_steps=50, p=0.5)
# PositionMappingAndGraph(N_try=40000, N_steps=50, p=0.75)
# RightStepMappingAndGraph(N_try=40000, N_steps=50, p=1 / 32.)

# SecondPoint1()
# SecondPoint2()
# SecondPoint3()

Distinguishable_dice(2)
Distinguishable_dice(3)
Distinguishable_dice(4)
Indistinguishable_dice(2)
Indistinguishable_dice(3)
Indistinguishable_dice(4)
# End
