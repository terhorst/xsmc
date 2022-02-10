from re import S
import xsmc
import xsmc.sampler
from xsmc import Segmentation
from xsmc.supporting.plotting import *
from xsmc.supporting.kde_ne import kde_ne
import matplotlib.pyplot as plt
import numpy as np
import msprime as msp
from scipy.interpolate import PPoly
import tskit
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import os
import subprocess
import time
import smcsmc
import stdpopsim

logging.getLogger("xsmc").setLevel(logging.INFO)

np.random.seed(1)

def seed():
    return np.random.randint(1, np.iinfo(np.int32).max)

L = int(5e7)  # length of simulated chromosome
mu = 1.4e-8  # mutation rate/bp/gen
M = 25  # number of replicates
rho_over_theta = [1.0] * (2*M) + [1e-9 / mu] * M

def sim_data(de, **kwargs):
    d = dict(
        sample_size=2,
        recombination_rate=1.4e-8,
        mutation_rate=mu,
        length=L,
        demographic_events=de,
    )

    d.update(kwargs)
    with ThreadPoolExecutor() as p:
        futs = [p.submit(msp.simulate, **d, random_seed=seed()) for i in range(M)]
        return [f.result() for f in futs]

os.environ[
    "PSMC_PATH"
] = "/nfs/turbo/lsa-jonth/calebki/psmc/psmc"  # update as needed if running locally
import mspsmc

def run_psmc(reps):
    def f(data, *args):
        return mspsmc.msPSMC([(data, (0, 1))]).estimate(*args)

    with ThreadPoolExecutor() as p:
        futs = [p.submit(f, data, "-r", 1.0 / rot) for data, rot in zip(reps, rho_over_theta)]
        res = [f.result() for f in futs]
    rescaled = []
    for r in res:
        # See Appendix I of https://github.com/lh3/psmc/blob/master/README
        N0 = r.theta / (4 * mu) / 100
        rescaled.append(r.Ne.rescale(2 * N0))
    return rescaled

def parallel_sample(reps, j=100, k=int(L / 50_000)):
    xs = [
        xsmc.XSMC(data, focal=0, panel=[1], rho_over_theta=rot)
        for data, rot in zip(reps, rho_over_theta)
    ]
    with ThreadPoolExecutor() as p:
        futs = [
            p.submit(x.sample_heights, j=j, k=k, seed=seed()) for i, x in enumerate(xs)
        ]
        return np.array(
            [f.result() * 2 * x.theta / (4 * mu) for f, x in zip(futs, xs)]
        )  # rescale each sampled path by 2N0 so that segment heights are in generations

def parallel_kde(sampled_heights, **kwargs):
    with ThreadPoolExecutor() as p:
        futs = [p.submit(kde_ne, h.reshape(-1), **kwargs) for h in sampled_heights]
        return [(f.result()[0], f.result()[1]) for f in futs]

def _scenario(i):
    if i // M == 0:
        scenario = "constant"
    elif i // M == 1:
        scenario = "growth"
    else:
        scenario = "zigzag"
    return scenario

def _rho(scenario):
    if scenario == "zigzag":
        return "1e-9"
    else:
        return "1.4e-8"

def run_smcpp(reps):
    def f(i):
        scenario = _scenario(i)
        rho = _rho(scenario)
        subprocess.run(
            [
                "singularity",
                "run",
                "docker://terhorst/smcpp",
                "estimate",
                str(mu),
                f"smcpp/input/{scenario}/{i%M}.smc.gz",
                "-r",
                rho,
                "--timepoints",
                "100",
                "100000",
                "--knots",
                "100", #change this for actual run 100
                "-o",
                f"smcpp/output/{scenario}",
                "--base",
                str(i%M),
            ]
        )

    for i, ts in enumerate(reps):
        scenario = _scenario(i)
        input_handle = f"smcpp/input/{scenario}/{i%M}.vcf"
        input_handle_gz = input_handle + ".gz"
        with open(input_handle, "w") as vcf_file:
            ts.write_vcf(vcf_file, ploidy=2)
        subprocess.run(["bgzip", input_handle])
        subprocess.run(["tabix", input_handle_gz])
        subprocess.run(
            [
                "singularity",
                "run",
                "docker://terhorst/smcpp",
                "vcf2smc",
                input_handle_gz,
                f"smcpp/input/{scenario}/{i%M}.smc.gz",
                "1",
                "Pop1:tsk_0",
            ]
        )

    with ThreadPoolExecutor() as p:
        for i in range(len(reps)):
            p.submit(f, i)


def run_smcsmc(reps):
    for i, ts in enumerate(reps):
        scenario = _scenario(i)

        ts.dump(f"smcsmc/input/{scenario}/sim{i%M}.tree")
        smcsmc.ts_to_seg(f"smcsmc/input/{scenario}/sim{i%M}.tree", n=[2])

    def f(i):
        scenario = _scenario(i)
        rho = _rho(scenario)

        args = {
            "seg": f"smcsmc/input/{scenario}/2.sim{i%M}.tree.seg",
            "nsam": "2",
            "Np": "10000", #Change this for actual run 10000 15
            "EM": "15",
            "mu": "1.4e-8",
            "rho": rho,
            "N0": "10000",
            "tmax": "100000",
            "P": "100 100000 99*1",
            "no_infer_recomb": "",
            "smcsmcpath": os.path.expandvars("${CONDA_PREFIX}/bin/smcsmc"),
            "o": f"smcsmc/output/{scenario}/output{i%M}",
        }

        smcsmc.run_smcsmc(args)

    with ThreadPoolExecutor() as p:
        for i in range(len(reps)):
            p.submit(f, i)

def save_lines(lines, label):
    lines1 = np.array([line[0] for line in lines])
    np.save(f"{label}1.npy", lines1)
    lines2 = np.array([line[1] for line in lines])
    np.save(f"{label}2.npy", lines2)


def main():
    # Constant

    de = [msp.PopulationParametersChange(time=0, initial_size=1e4)]
    data1 = sim_data(de)

    # Recent Growth

    de = [
        msp.PopulationParametersChange(time=0, initial_size=1e6),
        msp.PopulationParametersChange(time=1e3, initial_size=5e3),
        msp.PopulationParametersChange(time=2e3, initial_size=2e4),
    ]
    data2 = sim_data(de)

    # Zigzag

    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("Zigzag_1S14")
    de = [
        msp.PopulationParametersChange(time=0, initial_size=14312)
    ] + model.demographic_events
    data3 = sim_data(de, recombination_rate=1e-9)

    data = data1 + data2 + data3

    times = []

    start = time.time()
    run_smcsmc(data)
    times.append(time.time()-start)
    np.save("times_smcsmc.npy", np.array(times))

    start = time.time()
    run_smcpp(data)
    times.append(time.time()-start)
    np.save("times_smcpp.npy", np.array(times))

    start = time.time()
    sampled_heights = parallel_sample(data)
    lines_xsmc = parallel_kde(sampled_heights)
    times.append(time.time()-start)
    save_lines(lines_xsmc, "xsmc")

    start = time.time()
    x_psmc = np.geomspace(1e2, 1e5, 100)
    psmc_out = run_psmc(data)
    lines_psmc = [(x_psmc, r(x_psmc)) for r in psmc_out]
    times.append(time.time()-start)
    save_lines(lines_psmc, "psmc")

    np.save("times.npy", np.array(times))

if __name__ == '__main__':
    main()