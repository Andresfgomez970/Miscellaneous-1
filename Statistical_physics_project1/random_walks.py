import numpy as np  # Library to generate random numbers
import random  # To generate random mubers
import matplotlib.pyplot as plt  # To plot results in general
from scipy.special import factorial
from scipy import optimize  # Import in order to do adjustements


def OneDimArray(N_try, N_steps, p):
    """
    input
      N: number of steps to be simulated.
      p: probabiliti of step to the right.

    output
      x_steps: array that saves in its i index the the position given that i
        steps have been made.
    """
    # Generate N uniform random number
    positions = np.zeros(N_try)
    for i in range(N_try):
        for j in range(N_steps):
            r = random.random()
            # Right step in x
            if r <= p:
                positions[i] += 1
            # Left step in x
            else:
                positions[i] -= 1
    return positions


def OneDimArrayPoisson(N_try, N_steps, p):
    """
    input
      N: number of steps to be simulated.
      p: probabiliti of step to the right.

    output
      x_steps: array that saves in its i index the the position given that i
        steps have been made.
    """
    # Generate N uniform random number
    positions = np.zeros(N_try)
    for i in range(N_try):
        for j in range(1, N_steps):
            r = random.random()
            # Right step in x
            if r <= p:
                positions[i] += 1
    return positions


def OneDimArrayShort(N_try, N_steps, p):
    """
    input
      N: number of steps to be simulated.
      p: probabiliti of step to the right.

    output
      x_steps: array that saves in its i index the the position given that i
        steps have been made.
    """
    # Generate N uniform random number
    x_steps = np.random.uniform(0, 1, (N_try, N_steps))
    # (x_steps < p) gives 1 in the steps that are going to be to the right in
    #   the simulation and 0 in the rest. ~ operator interchanges 0 and 1's,
    #   therefore ~(x_steps < p) * -1 gives -1 in the steps that are going to
    #   be to the left in the simulation and 0 to the rest, its sum is then the
    #   steps made in each step. Its commulative sum is consequently is the
    #   actual position in the sumulation.
    right = (x_steps < p)
    left = ~right * -1
    x_steps = right + left
    x_steps = np.cumsum(x_steps, axis=1)
    return x_steps


def PositionMappingAndGraph(N_try, N_steps, p):
    walk = OneDimArray(N_try, N_steps, p)

    # Organize from least to greatest
    walk = np.sort(walk)

    # Doing histogram
    label_message = ("\nParameters:\n  Nsamples = %d\n  Nsteps = %d" %
                     (N_try, N_steps))
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(walk, bins=N_steps + 1, rwidth=0.9,
                                range=(-N_steps - 0.5, N_steps + 0.5),
                                density=True, label=label_message)

    # Calculate correct x's to graph
    x_sample = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(n))])
    x_fit = np.linspace(x_sample[0], x_sample[-1], 100)

    # Doing graph of predicted tendency
    mu = (2 * p - 1) * N_steps
    sigma = 2 * (p * (1 - p) * N_steps) ** 0.5
    a = (1. / (2 * np.pi)) ** 0.5 / sigma
    print(a, mu, sigma)
    gaussian = Gaussian(x_fit, a, mu, sigma)
    label_msg = "Expected tendency:\n  a = %.3f\n  mu = %.3f\n  sigma= %.3f"
    plt.plot(x_fit, gaussian, label=label_msg % (a, mu, sigma))

    # Empirical, but, fair ranges for adjustement
    min_values = (a - a / 2., mu - sigma, sigma - sigma / 2.)
    max_values = (a + a / 2., mu + sigma, sigma + sigma / 2.)
    # Doing adjustement
    popt, pcov = optimize.curve_fit(Gaussian, x_sample, n,
                                    bounds=(min_values, max_values))

    plt.title("Final Position Histogram for Random 1D Walk, p = %.2f" % p,
              size=14)
    label_message = ("\nGaussian fit\n  a = %.3f\n  mu = %.3f\n  sigma = %.3f"
                     % (popt[0], popt[1], popt[2]))
    plt.plot(x_fit, Gaussian(x_fit, popt[0], popt[1], popt[2]),
             label=label_message)

    plt.xlabel("Position [l = 1]", size=12)
    plt.ylabel("Frequency", size=12)
    plt.legend()

    plt.savefig('p%.2fgaussian.png' % p)
    plt.show()


def RightStepMappingAndGraph(N_try, N_steps, p):
    walk = OneDimArrayPoisson(N_try, N_steps, p)

    # Organize from least to greatest
    walk = np.sort(walk)

    # Doing histogram
    plt.figure(figsize=(10, 6))
    label_message = ("\nParameters:\n  Nsamples = %d\n  Nsteps = %d" %
                     (N_try, N_steps))
    n, bins, patches = plt.hist(walk, bins=N_steps + 1,
                                range=(-0.5, N_steps + 0.5),
                                density=True, label=label_message)

    x_fit = np.arange(0, N_steps + 1, 1)

    popt, pcov = optimize.curve_fit(Poission, x_fit, n)

    # Doing graph of predicted tendency
    mu = p * N_steps
    a = 1
    poisson = Poission(x_fit, a, mu)
    plt.title("Right Steps Made for 1D walk, p = 1/32", size=14)
    plt.plot(x_fit, poisson, "k-o", label="Expected tendency:\n" +
             "  a=%.3f\n  mu=%.3f" % (a, mu))
    plt.plot(x_fit, Poission(x_fit, *popt), "b--o", markersize=4,
             label="Poisson fit:\n  a=%.3f\n  mu=%.3f" % (popt[0], popt[1]))

    plt.xlim(-0.5, 10)
    plt.xlabel("Position to right [l = 1]", size=12)
    plt.ylabel("Frequency", size=12)
    plt.legend()

    plt.savefig('p1_32poisson.png')
    plt.show()


def TwoDimArrayLattice(N_try, N_steps):
    """
    input
      N: number of steps to be simulated.
      p: probabiliti of step to the right.

    output
      x_steps: array that saves in its i index the the position given that i
        steps have been made.
    """
    # Generate N steps in 2D
    positions = np.zeros((N_try, N_steps, 2))
    X, Y = 0, 1
    for i in range(N_try):
        for j in range(1, N_steps):
            p = random.random()
            # Right step in x
            if p < 0.25:
                positions[i][j][X] = positions[i][j - 1][X] + 1
                positions[i][j][Y] = positions[i][j - 1][Y]
            # Left step in x
            elif 0.25 < p < 0.5:
                positions[i][j][X] = positions[i][j - 1][X] - 1
                positions[i][j][Y] = positions[i][j - 1][Y]
            # Up step in y
            elif 0.5 < p < 0.75:
                positions[i][j][X] = positions[i][j - 1][X]
                positions[i][j][Y] = positions[i][j - 1][Y] + 1
            # Down step in y
            else:
                positions[i][j][X] = positions[i][j - 1][X]
                positions[i][j][Y] = positions[i][j - 1][Y] - 1

    return positions


def TwoDimArrayOffLattice(N_try, N_steps):
    """
    input
      N: number of steps to be simulated.
      p: probabiliti of step to the right.

    output
      x_steps: array that saves in its i index the the position given that i
        steps have been made.
    """
    # Generate N steps in 2D
    positions = np.zeros((N_try, N_steps, 2))
    length_path = 1
    X, Y = 0, 1
    for i in range(N_try):
        for j in range(1, N_steps):
            alpha = random.random() * 2 * np.pi
            # Step in direction of unimorm alleatory angle
            positions[i][j][X] = (positions[i][j - 1][X] +
                                  length_path * np.cos(alpha))
            positions[i][j][Y] = (positions[i][j - 1][Y] +
                                  length_path * np.sin(alpha))
    return positions


def Gaussian(x, a, mu, std):
    """
    output
      value of a gaussian type in points x.
    """
    return a * np.exp(-(x - mu)**2.0 / (2 * std**2))


def Poission(x, a, mu):
    return a * mu ** x * np.exp(-mu) / factorial(x)


def lineal(x, a, b):
    return a + b * x


def Distinguishable_dice(N_dices):
    # Make all possible permutations, dices can be differentiated
    dice_faces = 6
    # Make an array of the possible sums, the array position represents the sum
    sumas = np.zeros(dice_faces * N_dices + 1)
    lista_good = np.zeros(N_dices, dtype=int)

    total_config = dice_faces ** N_dices
    for i in range(total_config):
        dummy = i
        for k in range(N_dices):
            steps = dummy % dice_faces
            lista_good[k] = steps + 1
            dummy = int(dummy / dice_faces)
        sumas[lista_good.sum()] += 1

    plt.figure(figsize=(10, 6))
    plt.title("Macrostates for distinguishable dices, N = %d" % (N_dices),
              size=14)
    plt.plot(np.arange(N_dices, dice_faces * N_dices + 1, 1), sumas[N_dices:],
             'o')
    plt.xlabel("Possible Macrostate", size=12)
    plt.ylabel("Possible Mircrostate", size=12)
    plt.savefig("Dices_distinguishableN%d" % (N_dices) + ".png")
    plt.show()


def Indistinguishable_dice(N_dices):
    # Make all possible permutations, dices can be differentiated
    dice_faces = 6
    # Make an array of the possible sums, the array position represents the sum
    sumas = np.zeros(dice_faces * N_dices + 1)
    lista_good = np.zeros(N_dices, dtype=int)
    results = []

    total_config = dice_faces ** N_dices
    for i in range(total_config):
        dummy = i
        for k in range(N_dices):
            steps = dummy % dice_faces
            lista_good[k] = steps + 1
            dummy = int(dummy / dice_faces)
        results.append(list(lista_good))

    # Organize each one inidividually
    results = np.array(results)
    results = np.sort(results, axis=1)
    N = len(results)
    results = [list(results[i]) for i in range(N)]

    new_results = []
    for elem in results:
        if elem not in new_results:
            new_results.append(elem)
    results = np.array(new_results)

    for i in range(len(results)):
        sumas[results[i].sum()] += 1

    plt.figure(figsize=(10, 6))
    plt.title("Macrostates for indistinguishable dices, N = %d" % (N_dices))
    plt.plot(np.arange(N_dices, dice_faces * N_dices + 1, 1), sumas[N_dices:],
             'o')
    plt.xlabel("Possible Macrostate")
    plt.ylabel("Possible Mircrostate")
    plt.savefig("Dices_indistinguishableN%d" % (N_dices) + ".png")
    plt.show()
