import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Make nice plots
# https://towardsdatascience.com/making-matplotlib-beautiful-by-default-d0d41e3534fd


TABLEAU = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

sns.set(
    font="Helvetica",
    rc={
        "axes.axisbelow": False,
        "axes.edgecolor": "black",
        "axes.facecolor": "None",
        "axes.grid": False,
        "axes.labelcolor": "black",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": "white",
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        "text.color": "black",
        "xtick.bottom": True,
        "xtick.color": "black",
        "xtick.direction": "out",
        "xtick.top": False,
        "ytick.color": "black",
        "ytick.direction": "out",
        "ytick.left": True,
        "ytick.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        # "text.usetex": True,
        "figure.figsize": (8, 5),
    },
)
sns.set_context(
    "notebook",
    rc={
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "figure.figsize": (8, 5),
    },
)


from scipy.interpolate import interp1d


def summarize_lines(xys, x0):
    "summarize a collection of lines by plotting their median and IQR"
    y0 = []
    for x, y in xys:
        f = interp1d(
            x, y, bounds_error=False
        )  # interpolate linearly to a common set of points
        y0.append(f(x0))
    return np.nanquantile(y0, [0.5, 0.25, 0.75], axis=0)  # median, q25, q75


def plot_summary(ax, lines, x, label=None, **kwargs):
    all_x = np.concatenate([l[0] for l in lines]).reshape(-1)
    m, q25, q75 = summarize_lines(lines, x)
    ax.plot(x, m, label=label, **kwargs)
    ax.fill_between(x, q25, q75, **kwargs, alpha=0.5)


def plot_de(de, Ne):
    "plot msprime demographic events"

    def ev_iter():
        last_g = 0.0
        last_a = Ne
        last_t = None
        for ev in de:
            t = ev.time
            a = ev.initial_size or last_a * np.exp(-last_g * (t - last_t))
            g = ev.growth_rate if ev.growth_rate is not None else last_g
            yield (t, a, g)
            last_a = a
            last_t = t
            last_g = g

    t, a, g = zip(*ev_iter())
    t = np.append(t, np.inf)

    def f(x):
        i = np.searchsorted(t, x, "right") - 1
        return a[i] * np.exp(-g[i] * (x - t[i]))

    return np.vectorize(f)
