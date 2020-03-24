import numpy as np  # Library to generate random numbers
import random  # To generate random mubers
import matplotlib.pyplot as plt  # To plot results in general
from scipy.special import factorial
from scipy import optimize  # Import in order to do adjustements


def OneDimArray(N_try, N_steps, p):
    """
    input
      N_steps: number of steps to be simulated.
      N_try:   number of final position mapped.
      p: probability of step to the right.

    output
      x_steps: array that saves in its i index the final position of possible
        walk.
    """

    # Array to save the N_try final positions
    positions = np.zeros(N_try)
    # Cover all possible paths
    for i in range(N_try):
        # Do N_steps for each path
        for j in range(N_steps):
            # Use rejection method
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
      N_steps: number of steps to be simulated.
      N_try:   number of final position mapped.
      p: probability of step to the right.

    output
      x_steps: array that saves in its i index the final position of possible
        walk.
    """
    # Array to save the N_try final positions
    positions = np.zeros(N_try)
    # Cover all possible paths
    for i in range(N_try):
        # Do N_steps for each path
        for j in range(N_steps):
            # Use rejection method
            r = random.random()
            # Right step in x
            if r <= p:
                positions[i] += 1
    return positions


def PositionMappingAndGraph(N_try, N_steps, p):
    """
    input
      N_steps: number of steps to be simulated.
      N_try:   number of final position mapped.
      p: probability of step to the right.

    output
      Approximation to the distribution of the final psotion of the random walk
        as well as graph of its expected tendency (gaussian) and an adjustement
        made.
    """
    # Sample N_try final positions of random walk
    walk = OneDimArray(N_try, N_steps, p)

    # Organize from least to greatest for the histogram
    walk = np.sort(walk)

    # Do histogram, tak into acount that x = 2*n - N, so there are N_steps + 1
    #   values for the final position, also range is transalted to avoid
    #   rounding errros, an rwidth change for aesthetic reasons.
    label_message = ("\nParameters:\n  Nsamples = %d\n  Nsteps = %d" %
                     (N_try, N_steps))
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(walk, bins=N_steps + 1, rwidth=0.9,
                                range=(-N_steps - 0.5, N_steps + 0.5),
                                density=True, label=label_message)

    # Calculate correct x's to appriximate to a continuos
    x_sample = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(n))])
    # Do finner array to plo adjusment
    x_fit = np.linspace(x_sample[0], x_sample[-1], 100)

    ######################################################
    # Doing graph of predicted tendency
    ######################################################
    # Calculating expected parameters for expected tendency
    mu = (2 * p - 1) * N_steps
    sigma = 2 * (p * (1 - p) * N_steps) ** 0.5
    a = (1. / (2 * np.pi)) ** 0.5 / sigma
    gaussian = Gaussian(x_fit, a, mu, sigma)
    label_msg = "Expected tendency:\n  a = %.3f\n  mu = %.3f\n  sigma= %.3f"
    plt.plot(x_fit, gaussian, label=label_msg % (a, mu, sigma))

    ######################################################
    # Doing adjustment for histogram and graph
    ######################################################
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
    # Sample N_try number of right steps of random walk
    walk = OneDimArrayPoisson(N_try, N_steps, p)

    # Organize from least to greatest for histogram
    walk = np.sort(walk)

    # Do histogram, range is transalted to avoid rounding errros, and rwidth
    #   change for aesthetic reasons, result here are clearlly 50.
    plt.figure(figsize=(10, 6))
    label_message = ("\nParameters:\n  Nsamples = %d\n  Nsteps = %d" %
                     (N_try, N_steps))
    n, bins, patches = plt.hist(walk, bins=N_steps + 1,
                                range=(-0.5, N_steps + 0.5),
                                density=True, label=label_message)

    # Possible values of adjusment and tendency
    x_fit = np.arange(0, N_steps + 1, 1)

    # Doing the adjusment and graph
    popt, pcov = optimize.curve_fit(Poission, x_fit, n)

    plt.plot(x_fit, Poission(x_fit, *popt), "b--o", markersize=4,
             label="Poisson fit:\n  a=%.3f\n  mu=%.3f" % (popt[0], popt[1]))

    # Calulate predicted tendency and graph
    mu = p * N_steps
    a = 1
    poisson = Poission(x_fit, a, mu)
    plt.title("Right Steps Made for 1D walk, p = 1/32", size=14)
    plt.plot(x_fit, poisson, "k-o", label="Expected tendency:\n" +
             "  a=%.3f\n  mu=%.3f" % (a, mu))

    plt.xlim(-0.5, 10)
    plt.xlabel("Position to right [l = 1]", size=12)
    plt.ylabel("Frequency", size=12)
    plt.legend()

    plt.savefig('p1_32poisson.png')
    plt.show()


def TwoDimArrayLattice(N_try, N_steps):
    """
    input
      N_steps: number of steps to be simulated.
      N_try:   number of final position mapped.

    output
      positions: array that saves in its i index the final succesive positions
        of the random walk.
    """
    # Array to save N_try paths of N_steps done in 2 Dimensions
    positions = np.zeros((N_try, N_steps, 2))
    X, Y = 0, 1  # Variable to improve reaing of code.
    # Go over number of pats
    for i in range(N_try):
        # Go over number of steps to be made
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
      N_steps: number of steps to be simulated.
      N_try:   number of final position mapped.

    output
      positions: array that saves in its i index the final succesive positions
        of the random walk.
    """
    # Array to save N_try paths of N_steps done in 2 Dimensions
    positions = np.zeros((N_try, N_steps, 2))
    X, Y = 0, 1  # Variable to improve reaing of code.
    length_path = 1  # Define size of each step made in the walk
    # Go over number of pats
    for i in range(N_try):
        # Go over number of steps to be made
        for j in range(1, N_steps):
            # Map any direction with the same probability
            alpha = random.random() * 2 * np.pi
            # Step in direction of the uniform alleatory angle
            positions[i][j][X] = (positions[i][j - 1][X] +
                                  length_path * np.cos(alpha))
            positions[i][j][Y] = (positions[i][j - 1][Y] +
                                  length_path * np.sin(alpha))
    return positions


def Gaussian(x, a, mu, std):
    """
    Done to make adjustement
    output
      value of a gaussian type in points x.
    """
    return a * np.exp(-(x - mu)**2.0 / (2 * std**2))


def Poission(x, a, mu):
    """
    Done to make adjustement
    output
      value of a  possion type in points x.
    """
    return a * mu ** x * np.exp(-mu) / factorial(x)


def lineal(x, a, b):
    """
    Done to make adjustement
    output
      value of a lineal type in points x.
    """
    return a + b * x


def Distinguishable_dice(N_dices):
    """
    Make all possible permutations, dices can be differentiated

    input
      N_dices: number of dices to be thrown.
    """
    # Number of faces in each dice
    dice_faces = 6
    # Make an array of the possible sums, the array index position will
    #  represents the sum
    sumas = np.zeros(dice_faces * N_dices + 1)
    lista_good = np.zeros(N_dices, dtype=int)

    total_config = dice_faces ** N_dices  # Total configurations of the system
    for i in range(total_config):
        dummy = i  # dummy is defined to do not change i value.
        # This pass the number i to a dice_faces base with N_dice digits
        for k in range(N_dices):
            steps = dummy % dice_faces  # Get k digit
            lista_good[k] = steps + 1  # Move digit one up to go from 1-6
            # Define again by dice_faces to get the following digit in
            #   dummy % dice_faces
            dummy = int(dummy / dice_faces)
        sumas[lista_good.sum()] += 1  # Sum when the value desired is obtained

    # Plot of configurations (microstates) vs possible sums (macrostates)
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
    """
    Make all possible permutations, dices can be differentiated

    input
      N_dices: number of dices to be thrown.
    """
    # Number of faces in each dice
    dice_faces = 6
    # Make an array of the possible sums, the array index position will
    #  represents the sum
    sumas = np.zeros(dice_faces * N_dices + 1)
    lista_good = np.zeros(N_dices, dtype=int)
    results = []

    total_config = dice_faces ** N_dices  # Total configurations of the system
    for i in range(total_config):
        dummy = i  # dummy is defined to do not change i value.
        # This pass the number i to a dice_faces base with N_dice digits
        for k in range(N_dices):
            steps = dummy % dice_faces  # Get k digit
            lista_good[k] = steps + 1  # Move digit one up to go from 1-6
            # Define again by dice_faces to get the following digit in
            #   dummy % dice_faces
            dummy = int(dummy / dice_faces)
        results.append(list(lista_good))  # Save every config

    # Organize each one inidividually
    results = np.array(results)
    results = np.sort(results, axis=1)
    N = len(results)
    results = [list(results[i]) for i in range(N)]

    # Remove repeated configurations
    new_results = []
    for elem in results:
        if elem not in new_results:
            new_results.append(elem)
    results = np.array(new_results)

    # Finally count
    for i in range(len(results)):
        sumas[results[i].sum()] += 1

    # Plot of configurations (microstates) vs possible sums (macrostates)
    plt.figure(figsize=(10, 6))
    plt.title("Macrostates for indistinguishable dices, N = %d" % (N_dices))
    plt.plot(np.arange(N_dices, dice_faces * N_dices + 1, 1), sumas[N_dices:],
             'o')
    plt.xlabel("Possible Macrostate")
    plt.ylabel("Possible Mircrostate")
    plt.savefig("Dices_indistinguishableN%d" % (N_dices) + ".png")
    plt.show()


# Additional function to so map short of all positions that the particle had.
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
