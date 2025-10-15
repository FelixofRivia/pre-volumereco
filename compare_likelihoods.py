import pickle
import matplotlib.pyplot as plt
import glob
import numpy as np

# File patterns
classic_files = sorted(glob.glob("/storage-hpc/ntosi/SAND-LAr-PROD/GRAIN/numu_CC_QE_Filippo/volumereco/3dreco_classic_*.pkl"))
pre_files = sorted(glob.glob("/storage-hpc/ntosi/SAND-LAr-PROD/GRAIN/numu_CC_QE_Filippo/volumereco/3dreco_pre_*_20cm.pkl"))

def load_logl(filename):
    with open(filename, 'rb') as f:
        outdata = pickle.load(f)  # not used here, but needed for pickle structure
        logl = pickle.load(f)
    return logl

def iterations_to_stable_delta_y(logl_event):
    y = np.array(logl_event)

    dy = np.abs(np.diff(y))
    threshold = 50
    for xi, dyi in enumerate(dy):
        if dyi <= threshold:
            return xi + 1
    return len(dy)

diff_iterations = []

# Loop over files
for classic_file, pre_file in zip(classic_files, pre_files):
    logl_classic = load_logl(classic_file)
    logl_pre = load_logl(pre_file)
    
    for u in logl_classic.keys():
        for evn in logl_classic[u]:
            iter_classic = iterations_to_stable_delta_y(logl_classic[u][evn])
            iter_pre = iterations_to_stable_delta_y(logl_pre[u][evn])
            diff_iterations.append(iter_classic - iter_pre)

# Compute statistics
mean_diff = np.mean(diff_iterations)
median_diff = np.median(diff_iterations)
std_diff = np.std(diff_iterations)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(diff_iterations, bins=50, range=(-50, 150), color='skyblue', edgecolor='black')
plt.title("$\\Delta$ number of iterations for convergence (Uniform - DNN)", fontsize=16)
plt.xlabel("Iteration difference", fontsize=16)
plt.ylabel("Number of events", fontsize=16)
plt.grid(True)

# Add text with averages
textstr = '\n'.join((
    f'Mean: {mean_diff:.2f}',
    f'Median: {median_diff:.2f}',
    f'Std: {std_diff:.2f}'
))
plt.text(0.98, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=16, va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("delta_y_convergence_diff_histogram_with_stats.png")
plt.show()
