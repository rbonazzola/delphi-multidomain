import sys
import os
import pickle
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

if len(sys.argv) < 2:
    sys.exit("Usage: python compute_mean_std.py <shap_pickle_file>")

# Input and output paths
infile = sys.argv[1]
outfile = os.path.splitext(infile)[0] + "_mean_and_std.pkl"

logging.info(f"Reading SHAP pickle: {infile}")
with open(infile, "rb") as f:
    shap_pkl = pickle.load(f)

tokens = shap_pkl["tokens"].astype(int)
values = shap_pkl["values"]

logging.info(f"Data loaded: {values.shape[0]} events × {values.shape[1]} features")

# Unique tokens and mapping
uniq, inv = np.unique(tokens, return_inverse=True)
n_tokens, n_features = len(uniq), values.shape[1]

logging.info(f"Found {n_tokens} unique tokens")

# Accumulators
sums = np.zeros((n_tokens, n_features), dtype=np.float64)
sumsq = np.zeros((n_tokens, n_features), dtype=np.float64)
counts = np.zeros(n_tokens, dtype=np.int64)

logging.info("Accumulating sums and squared sums...")
np.add.at(sums, inv, values)
np.add.at(sumsq, inv, values**2)
np.add.at(counts, inv, 1)

# Mean and std
logging.info("Computing mean and standard deviation...")
means = sums / counts[:, None]
vars_ = (sumsq - (sums**2) / counts[:, None]) / (counts[:, None] - 1)
stds = np.sqrt(vars_)

# Save results
logging.info(f"Saving results to {outfile}")
with open(outfile, "wb") as f:
    pickle.dump(
        {"tokens": uniq, "mean": means, "std": stds, "counts": counts},
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

logging.info("Done ✅")
