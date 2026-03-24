import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import webbrowser
import math
import time

all_the_eigens = False

LATEX_REPORT = "final-project.tex"
LATEX_REPORT_PDF = "final-project.pdf"
MARCHENKO_PASTUR_IMAGE = "marchenko.png"
PROJECTED_DATA_IMAGE_A = "projected-A.png"
PROJECTED_DATA_IMAGE_B = "projected-B.png"
def brow_open(x): webbrowser.open("file://" + os.path.abspath(x))
def fstr(x): return format(float(x),".6f")

NUM_BINS = 25

def linreg(X, Y):
    xbar = np.mean(X)
    ybar = np.mean(Y)
    b = sum([(X[i] - xbar) * (Y[i] - ybar) for i in range(len(X))]) / sum([pow(xi - xbar, 2) for xi in X])
    a = ybar - b * xbar
    mse = sum([pow(a + b * X[i] - Y[i], 2) for i in range(len(X))]) / len(X)
    return (a, b, mse)

# ==================
# 1. Pick a dataset.
# ==================
SOURCE_URL    = "https://archive.ics.uci.edu/dataset/171/madelon"
SOURCE_NAME   = "MADELON"
SOURCE_SOURCE = "UC Irvine Machine Learning Repository"
SOURCE_CITE   = "Guyon, I. (2004). Madelon [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5602H."
df = pd.read_csv("dataset.csv")
#df = pd.read_excel("dataset.xlsx")
# remove first 2 rows
df.drop(0,inplace=True)
df.drop(1,inplace=True)
# remove rows with bad data
df = df.dropna()
df = df[~df.isin(['', '--']).any(axis=1)]
for col in df.columns.tolist():
    try:
        df[col].astype(float)
    except (ValueError, TypeError):
        df = df.drop(columns=[col])
# extract and normalize observations
O = df.to_numpy() # observations
O = O.astype(float)
if True: # to remove constant rows
    std = O.std(axis=0)
    O = O[:, std > 0]
O = (O - O.mean(axis=0)) / O.std(axis=0)
# parameters
NUM_ROWS = O.shape[0]
NUM_COLS = O.shape[1]
gamma = NUM_COLS / NUM_ROWS
gamma_is_within_bounds = (gamma > 1/20 and gamma < 20)

print("Gamma is " + str(gamma) + ", so it is " + ("within" if gamma_is_within_bounds else "out of") + " bounds.")
print("There are", len(df.columns), "columns.")
print("There are", len(df), "rows.")

# =======================
# 2. Compute eigenvalues.
# =======================
cov_matrix = np.cov(O.T)
complex_eigenvalues, complex_eigenvectors = np.linalg.eig(cov_matrix)
real_eigenvalues = np.real(complex_eigenvalues)
real_eigenvectors = np.real(complex_eigenvectors)
print("eigens!")
print(pd.DataFrame(data={"Eigenvalues":real_eigenvalues,"Eigenvectors":[str(i) for i in real_eigenvectors]}))
print(len(real_eigenvectors[0]))
# params
sigma_square = 1 # np.var(O)
lambda_m, lambda_p = [sigma_square * pow(1 + i * math.sqrt(gamma), 2) for i in (-1, 1)]
# this is the diagram (unscaled)
plt.hist(real_eigenvalues, density=True, bins=NUM_BINS)
x_mp = np.linspace(lambda_m, lambda_p, 500)
y_mp = (np.sqrt((x_mp - lambda_m) * (lambda_p - x_mp))) / (2 * np.pi * gamma * x_mp * sigma_square)
plt.axvline(x=lambda_m, color='red', linestyle='--', label='$\\lambda_-$')
plt.axvline(x=lambda_p, color='green', linestyle='--', label='$\\lambda_+$')
plt.legend()
plt.xlabel("Eigenvalue")
plt.ylabel("Density")
plt.title("Eigenvalue Distribution with Marchenko-Pastur Fit")
plt.plot(x_mp, y_mp)
plt.savefig(MARCHENKO_PASTUR_IMAGE, dpi=300, bbox_inches="tight", pad_inches=0.5)
plt.close()
brow_open(MARCHENKO_PASTUR_IMAGE)
# fit
hist_counts, bin_edges = np.histogram(real_eigenvalues, bins=NUM_BINS, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]
mp_expected = np.array([
    (np.sqrt((x - lambda_m) * (lambda_p - x)) / (2 * np.pi * gamma * x * sigma_square))
    if lambda_m <= x <= lambda_p else 0.0
    for x in bin_centers
])
mp_fit_pct = np.sum(np.minimum(hist_counts, mp_expected)) * bin_width * 100
mp_fit_str = fstr(mp_fit_pct)
mp_fits_well = mp_fit_pct > 50

# =====================
# 3. Identify outliers.
# =====================
print("lambda bounds", lambda_m, lambda_p)
outlier_indices = np.where((real_eigenvalues < lambda_m) | (real_eigenvalues > lambda_p))[0]
outliers = real_eigenvalues[outlier_indices]
print("# outliers", len(outliers))

# =======================
# 4. Principal components
# =======================
sorted_outlier_indices = sorted(outlier_indices, key=lambda i: real_eigenvalues[i], reverse=True)
principal_components = [real_eigenvectors[:,i] for i in sorted_outlier_indices]

# ===============
# 5. Project data
# ===============
selection = principal_components[:min(3, len(principal_components))]
projected = np.array([
    [
        np.dot(row, pc)
        for pc in selection
    ]
    for row in O
])

# ==========
# 6. Analyze
# ==========
xs = projected[:, 0]
ys = projected[:, 1]
zs = projected[:, 2]
for g in [(PROJECTED_DATA_IMAGE_A, 30, 20), (PROJECTED_DATA_IMAGE_B, 45, 45)]:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)
    plt.title("Datapoints projected onto new basis")
    ax.set_xlabel("Basis Vector 1")
    ax.set_ylabel("Basis Vector 2")
    ax.set_zlabel("Basis Vector 3")
    #plt.show()
    ax.view_init(elev=g[1], azim=g[2])
    plt.savefig(g[0], dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close()
    brow_open(g[0])
print(O[:,0])
print(O[:,1])
num_better = 0
_,_,mp_fit = linreg(xs, O[:,1])
for i in range(NUM_COLS - 1):
    _,_,naive_fit = linreg(O[:,0], O[:,i + 1])
    if naive_fit > mp_fit:
        num_better += 1
percent_better = num_better / (NUM_COLS - 1)


