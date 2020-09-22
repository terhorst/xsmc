import tskit
import xsmc

chrom1 = tskit.load("/scratch/1kg/1kg_chr22.trees").keep_intervals([[2e7, 2.5e7]]).trim()
paths = xsmc.sample_paths(chrom1, 0, [1], M=1, theta=1.4e-4, rho=1.4e-4 / 4)
aoeu
