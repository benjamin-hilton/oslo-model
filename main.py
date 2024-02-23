import oslo
import numpy as np
import matplotlib.pyplot as plt
import timeit as ti
import os
import pickle
import cycler
import scipy.optimize as spo
import scipy.interpolate as spi
import logbin2018 as logbin


# Change the font settings and use LaTeX as the text renderer.
# plt.rc('text', usetex=True)
# plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['cm']})
# plt.rc('text.latex', preamble=r'\usepackage{sfmath}')


def task_1(timing_sizes, tries):

    print("TASK 1")

    def run_trial(length, tries):

        naive = oslo.oslo_naive(length, 0.5, tries)
        efficient = oslo.oslo_efficient(length, 0.5, tries)

        naive_heights = naive['heights'][naive['crossover_step']:]
        naive_avlch_sizes = naive['avlch_sizes'][naive['crossover_step']:]

        efficient_heights = efficient['heights'][efficient['crossover_step']:]
        efficient_avlch_sizes =\
            efficient['avlch_sizes'][efficient['crossover_step']:]

        naive_mean_height = np.mean(naive_heights)
        naive_err_height = np.std(naive_heights)/np.sqrt(tries)

        efficient_mean_height = np.mean(efficient_heights)
        efficient_err_height = np.std(efficient_heights)/np.sqrt(tries)

        print("Naive mean height:", naive_mean_height, "+/-", naive_err_height)
        print("Efficient mean height:", efficient_mean_height,
              "+/-", efficient_err_height)

        naive_mean_avlch_size = np.mean(naive_avlch_sizes)
        naive_err_avlch_size = np.std(naive_avlch_sizes)/np.sqrt(tries)

        efficient_mean_avlch_size = np.mean(efficient_avlch_sizes)
        efficient_err_avlch_size = (np.std(efficient_avlch_sizes) /
                                    np.sqrt(tries))

        print("Naive mean avalanche size:", naive_mean_avlch_size,
              "+/-", naive_err_avlch_size)
        print("Efficient mean avalanche size:", efficient_mean_avlch_size,
              "+/-", efficient_err_avlch_size)

    print("System size:", 16)
    print("Theoretical height: 26.5")
    print("Theoretical avalanche size: 16")
    run_trial(16, tries)

    print("System size:", 32)
    print("Theoretical height: 53.9")
    print("Theoretical avalanche size: 32")
    run_trial(32, tries)

    print("Calculating algorithm runtimes.")

    naive_times = np.zeros(timing_sizes.size)
    efficient_times = np.zeros(timing_sizes.size)

    for i in range(timing_sizes.size):

        print("System size:", sizes[i])

        naive_string = ("oslo_naive(" + str(sizes[i]) + ", 0.5,"
                        + str(tries) + ")")
        efficient_string = ("oslo_efficient(" + str(sizes[i]) + ", 0.5,"
                            + str(tries) + ")")

        naive_times[i] = ti.timeit(naive_string,
                                   "from oslo import oslo_naive",
                                   number=5)

        print("Naive algorithm runtime:", naive_times[i])

        efficient_times[i] = ti.timeit(efficient_string,
                                       "from oslo import oslo_efficient",
                                       number=5)

        print("Efficient algorithm runtime:", efficient_times[i])

    naive_times /= tries
    efficient_times /= tries

    fig = plt.figure("Algorithm Speed")
    fig.gca().loglog(sizes, naive_times, 'r-')
    fig.gca().loglog(sizes, efficient_times, 'b-')
    fig.gca().set_xlabel(r"System size $L$")
    fig.gca().set_ylabel("Runtime per iteration (s)")
    fig.gca().legend(labels=["Naive Algorithm", "Efficient Algorithm"])

    plt.savefig("figures/1a_speeds.png")

    plt.show()


def generate_data(sizes, probability, tries):

    print("Generating data..")

    output_list = []

    for i in range(sizes.size):
        print("L = ", sizes[i])
        output_list.append(oslo.oslo_efficient(sizes[i], 0.5, int(tries[i])))

    output = dict(zip(sizes, output_list))

    return output


def task_2a(data):

    print("TASK 2A")

    height_time = plt.figure("Heights vs Time")
    labels = []

    height_time.gca().set_prop_cycle(cycler.cycler('color', ['b', 'b',
                                                             'g', 'g',
                                                             'r', 'r',
                                                             'c', 'c',
                                                             'm', 'm',
                                                             'y', 'y',
                                                             'k', 'k']))

    handles = []

    for key in data:

        step = np.arange(data[key]['heights'].size)
        handle, = height_time.gca().loglog(step, data[key]['heights'],
                                           ls='-')
        handles.append(handle)

        crossover_step = data[key]['crossover_step']
        height_time.gca().loglog(crossover_step,
                                 data[key]['heights'][crossover_step],
                                 ls='None', marker='x', markersize='10')

        height_time.gca().set_xlabel(r"Time $t$")
        height_time.gca().set_ylabel(r"Height $h$")

        labels.append("L = " + str(key))

    height_time.gca().legend(handles, labels)

    plt.savefig("figures/2a_heighttime.png")

    plt.show()


def task_2c(data, window_size):

    print("TASK 2c")

    def moving_average(ndarray, window_size):
        indices = np.arange(ndarray.size)
        start_average = indices - window_size
        start_average *= start_average > 0

        means = np.zeros(ndarray.size)

        for index in indices:
            start = start_average[index]
            means[index] = np.mean(ndarray[start:index])

        return means

    data_collapse = plt.figure("Data Collapse")
    labels = []

    for key in data:

        data[key]['norm_heights'] = moving_average(data[key]['heights'],
                                                   window_size)/key
        time = np.arange(data[key]['norm_heights'].size)/(key**2)
        data_collapse.gca().loglog(time,
                                   data[key]['norm_heights'],
                                   ls='-')

        data_collapse.gca().set_xlabel(r"Time$t$")
        data_collapse.gca().set_ylabel(r"Height$h$")

        labels.append("L = " + str(key))

    data_collapse.gca().legend(labels=labels)

    plt.savefig("figures/2c_datacollapse.png")

    plt.show()


def task_2d(sizes, iterations, tries):

    print("TASK 2d")

    def fit_func(L, a):
        return a * (L ** 2 + L)

    crossover_times = dict([(size, []) for size in sizes])

    for i in range(iterations):

        data = generate_data(sizes, 0.5, tries)

        for key in data:
            crossover_times[key].append(data[key]['crossover_step'])

    crossover_means = []
    crossover_err = []

    for size in sizes:

        print("System size:", size)

        mean = np.mean(crossover_times[size])
        err = np.std(crossover_times[size])/np.sqrt(iterations)
        crossover_means.append(mean)
        crossover_err.append(err)

        print("Crossover Time:", mean, "+/-", err)

    crossover_fig = plt.figure("Crossover Times")
    crossover_fig.gca().loglog(sizes, crossover_means, 'bx')
    crossover_fig.gca().errorbar(sizes, crossover_means, yerr=crossover_err,
                                 color='c', fmt=' ')

    po, po_cov = spo.curve_fit(fit_func, sizes, crossover_means,
                               sigma=crossover_err, absolute_sigma=True)

    print("From fitting by least squares, <z> =", 2 * po[0],
          "+/-", 2 * np.sqrt(po_cov[0][0]))

    x_fit = np.arange(sizes[0], sizes[-1] * 1.01)
    crossover_fig.gca().plot(x_fit, fit_func(x_fit, po[0]), 'b-')

    crossover_fig.gca().set_xlabel(r"System Size $L$")
    crossover_fig.gca().set_ylabel(r"Crossover Time $t_c$")

    plt.savefig("figures/2d_crossovertimes.png")

    plt.show()


def task_2e(data):

    print("TASK 2e")

    def fit_func(L, a, b, w):
        return a * L * (1 - b * L ** (-w))

    mean_heights = []
    sizes = data.keys()

    for key in data:

        print("System size:", key)

        heights = data[key]['heights'][data[key]['crossover_step']:]
        mean_heights.append(np.mean(heights))

        print("Mean Height:", mean_heights[-1])

    mean_height_fig = plt.figure("Mean Height vs System Size")
    mean_height_fig.gca().loglog(sizes, mean_heights, 'bx')

    po, po_cov = spo.curve_fit(fit_func, sizes, mean_heights)

    print("From fitting by least squares,\n",
          "a_0 =", po[0], "+/-", np.sqrt(po_cov[0][0]), ",\n",
          "a_1 =", po[1], "+/-", np.sqrt(po_cov[1][1]), ",\n",
          "omega_1 =", po[2], "+/-", np.sqrt(po_cov[2][2]), ".")

    x_fit = np.arange(sizes[0], sizes[-1] * 1.01)
    mean_height_fig.gca().plot(x_fit,
                               fit_func(x_fit, po[0], po[1], po[2]),
                               'b-')

    mean_height_fig.gca().set_xlabel(r"System Size $L$")
    mean_height_fig.gca().set_ylabel(r"Mean Height $\langle h \rangle$")

    plt.savefig("figures/2e_meanheight.png")

    plt.show()


def task_2f(data):

    print("TASK 2f")

    def fit_func(L, a, b):
        return a * L ** b

    std_heights = []

    for key in data:

        print("System size:", key)

        heights = data[key]['heights'][data[key]['crossover_step']:]
        std_heights.append(np.std(heights))

        print("Standard Deviation in Height:", std_heights[-1])

    std_heights_fig = plt.figure("Standard Deviation in Height vs System Size")
    std_heights_fig.gca().loglog(sizes, std_heights, 'bx')

    po, po_cov = spo.curve_fit(fit_func, sizes[6:], std_heights[6:])

    print("From fitting to y = Ax^b by least squares,\n",
          "A =", po[0], "+/-", np.sqrt(po_cov[0][0]), ",\n",
          "b =", po[1], "+/-", np.sqrt(po_cov[1][1]), ".")

    x_fit = np.arange(sizes[0], sizes[-1] * 1.01)
    std_heights_fig.gca().plot(x_fit,
                               fit_func(x_fit, po[0], po[1]),
                               'b-')

    std_heights_fig.gca().set_xlabel(r"System Size $L$")
    std_heights_fig.gca().\
        set_ylabel(r"Standard Deviation in Height $\sigma_h$")

    plt.savefig("figures/2f_stdheight.png")

    plt.show()


def task_2g(data):

    print("TASK 2g")

    height_dist = plt.figure("Height Distributions")
    data_collapse = plt.figure("Height Distribution Data Collapse")

    for key in data:

        heights = data[key]['heights'][data[key]['crossover_step']:]
        freqs = np.bincount(heights.astype('int64'))/heights.size
        height_dist.gca().semilogx(np.arange(freqs.size), freqs)

        std = np.std(heights)
        mean = np.mean(heights)

        collapsed_bins = (np.arange(freqs.size) - mean)/std
        collapsed_freqs = freqs * std
        data_collapse.gca().plot(collapsed_bins, collapsed_freqs)

    height_dist.gca().set_xlabel(r"Height $h$")
    height_dist.gca().set_ylabel(r"Probability $P(h)$")
    height_dist.savefig("figures/2g_distributions")

    data_collapse.gca().\
        set_xlabel(r"Collapsed Height $(h - \langle h \rangle ) / \sigma_h$")
    data_collapse.gca().set_ylabel(r"Probability $P(h) \sigma_h$")
    data_collapse.savefig("figures/2g_datacollapse")

    plt.show()


def task_3a(data):

    print("TASK 3a")

    avalanche_dist = plt.figure("Avalanche Size Distribution")

    labels = []

    for key in data:
        avlch_sizes = data[key]['avlch_sizes'][data[key]['crossover_step']:]

        bins, freqs = logbin.logbin(avlch_sizes.astype('int64'),
                                    scale=1.2)

        avalanche_dist.gca().loglog(bins, freqs/(avlch_sizes.size),
                                    '-', markersize=4)

        labels.append("L = " + str(key))

    avalanche_dist.gca().set_xlabel(r"Avalanche Size $s$")
    avalanche_dist.gca().set_ylabel(r"Probability Density $P(s)$")

    avalanche_dist.gca().legend(labels=labels)

    plt.savefig("figures/3a_avlchdist")

    plt.show()


def task_3b(data, tau, dimension):

    print("TASK 3b")

    data_collapse = plt.figure("Avalanche Size Distribution")

    labels = []
    new_data = {key: {} for key in data}

    for key in new_data:
        crossover = data[key]['crossover_step']
        avlch_sizes = data[key]['avlch_sizes'][crossover:]
        bins, freqs = logbin.logbin(avlch_sizes.astype('int64'),
                                    scale=1.2)
        new_data[key]['avlch_sizes'] = bins
        new_data[key]['prob_density'] = freqs/(avlch_sizes.size)

    def estimate_tau(tau, dimension, data):
        def peak_dist(tau, dimension=dimension, data=data):
            total = 0
            for size in data:
                bins = data[size]['avlch_sizes']
                x = bins/(key**dimension)
                y = (bins**tau)*data[size]['prob_density']
                max_index = np.max(np.where((y[1:] - y[:-1]) > 0))
                data[size]['peak'] = y[max_index]
            for size_1 in data:
                for size_2 in data:
                    diff = data[size_2]['peak'] - data[size_1]['peak']
                    total += np.abs(diff)
            return total

        return spo.fmin(peak_dist, tau, xtol=1e-20, ftol=1e-20, disp=0)

    def estimate_dimension(tau, dimension, data):
        def peak_dist(dimension, tau=tau, data=data):
            differences = 0
            total = 0
            for size in data:
                bins = data[size]['avlch_sizes']
                x = bins/(key**dimension)
                y = (bins**tau)*data[size]['prob_density']
                tck = spi.splrep(x, y)
                max_index = np.max(np.where((y[1:] - y[:-1]) > 0))
                data[size]['pos'] = x[max_index]
            for size_1 in data:
                for size_2 in data:
                        diff = data[size_2]['pos'] - data[size_1]['pos']
                        total += data[size_2]['pos'] + data[size_1]['pos']
                        differences += np.abs(diff)
            return differences/total

        return spo.fmin(peak_dist, dimension, xtol=1e-20, ftol=1e-20, disp=1)

    tau = estimate_tau(tau, dimension, new_data)
    dimension = estimate_dimension(tau, dimension, new_data)

    print("tau_s = ", tau)
    print("D =", dimension)

    for key in data:
        avlch_sizes = data[key]['avlch_sizes'][data[key]['crossover_step']:]
        bins, freqs = logbin.logbin(avlch_sizes.astype('int64'),
                                    scale=1.2)
        prob_density = freqs/(avlch_sizes.size)
        data_collapse.gca().loglog(bins/(key**dimension),
                                   (bins**tau)*prob_density,
                                   '-', marker='.')
        labels.append("L = " + str(key))

    data_collapse.gca().set_xlabel(r"Rescaled Avalanche Size $s/L^D$")
    data_collapse.gca().set_ylabel(r"$s^{\tau_s} P(s)$")

    data_collapse.gca().legend(labels=labels)

    plt.savefig("figures/3b_avlchcollapse")

    plt.show()


def task_3c(data, moments):

    print("TASK 3c")

    moment_fig = plt.figure("")
    moment_data = {i: [] for i in moments}

    labels = []
    handles = []
    moment_fig.gca().set_prop_cycle(cycler.cycler('color', ['b', 'b',
                                                            'g', 'g',
                                                            'r', 'r',
                                                            'c', 'c',
                                                            'm', 'm',
                                                            'y', 'y',
                                                            'k', 'k']))

    sizes = list(data.keys())

    gradients = []
    gradients_err = []

    def line_fit(x, gradient, c):
        return gradient * x + c

    for k in moments:

        print("k = ", k)

        for key in data:
            moment = np.mean(data[key]['avlch_sizes'].astype(np.float128) ** k)
            moment_data[k].append(moment)

        labels.append("$\langle s ^" + str(k) + "\\rangle$")

        moment_fig.gca().loglog(sizes, moment_data[k], 'x')

        log_data = np.log(moment_data[k], dtype=np.float128).astype(np.float64)
        po, po_cov = spo.curve_fit(line_fit, np.log(sizes)[6:], log_data[6:])

        print("Gradient =", po[0], "+/-", np.sqrt(po_cov[0][0]))
        print("Intecept =", po[1], "+/-", np.sqrt(po_cov[1][1]))

        x_fit = np.linspace(1, sizes[-1], 1000)
        y_fit = np.exp(line_fit(np.log(x_fit), po[0], po[1]))

        handle, = moment_fig.gca().loglog(x_fit, y_fit)
        handles.append(handle)

        gradients.append(po[0])
        gradients_err.append(np.sqrt(po_cov[0][0]))

    moment_fig.gca().legend(handles=handles, labels=labels)

    moment_fig.gca().set_ylabel(r"$k^{\rm{th}}$ Moment $\langle s^k \rangle$")
    moment_fig.gca().set_xlabel(r"System Size $L$")

    analysis_fig = plt.figure("Moment Analysis")

    analysis_fig.gca().errorbar(moments, gradients,
                                yerr=gradients_err, fmt='bx')

    po, po_cov = spo.curve_fit(line_fit, moments, gradients,
                               sigma=gradients_err, absolute_sigma=True)

    x_fit = np.linspace(moments[0] - 0.5, moments[-1] + 0.5, 1000)
    analysis_fig.gca().plot(x_fit, line_fit(x_fit, po[0], po[1]), 'b-')

    analysis_fig.gca().set_xlabel(r"Moment $k$")
    analysis_fig.gca().set_ylabel(r"$D(1+k-\tau_s)$")

    tau_s = - po[1]/po[0] + 1
    tau_s_err = (tau_s - 1) * np.sqrt(np.sum(np.diag(po_cov)/(po**2)))
    print("tau_s =", tau_s, "+/-", tau_s_err)
    print("D =", po[0], "+/-", np.sqrt(po_cov[0][0]))

    plt.savefig("figures/3c_momentsize")

    plt.show()


sizes = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
tries = sizes**2 + 4e6
# print(tries)

# task_1(sizes, tries)

if not os.path.isfile("data.pkl"):
    data = generate_data(sizes, 0.5, tries)
    with open("data.pkl", 'wb') as pickle_file:
        pickle.dump(data, pickle_file, pickle.HIGHEST_PROTOCOL)
else:
    with open("data.pkl", 'rb') as pickle_file:
        data = pickle.load(pickle_file)


# task_2a(data)

# task_2c(data, 100)

# task_2d([4, 8, 16, 32, 64, 128, 256], 2, tries)

# task_2e(data)

# task_2f(data)

# task_2g(data)

# task_3a(data)

# task_3b(data)

# task_3c(data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
