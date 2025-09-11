import json
import matplotlib.pyplot as plt


with open('results/results-naive.json', 'r') as f:
    pairs_naive = json.load(f)

density_naive = [pair[0] for pair in pairs_naive]
time_naive = [pair[1] for pair in pairs_naive]

with open('results/results-csc.json', 'r') as f:
    pairs_csc = json.load(f)

density_csc = [pair[0] for pair in pairs_csc]
time_csc = [pair[1] for pair in pairs_csc]

with open('results/results-ssgblas.json', 'r') as f:
    pairs_ssgblas = json.load(f)

density_ssgblas = [pair[0] for pair in pairs_ssgblas]
time_ssgblas = [pair[1] for pair in pairs_ssgblas]

with open('results/results-custom.json', 'r') as f:
    pairs_custom = json.load(f)

density_custom = [pair[0] for pair in pairs_custom]
time_custom = [pair[1] for pair in pairs_custom]

# Plot
plt.scatter(density_naive, time_naive, s=10, label ='naive', color='blue')
plt.scatter(density_csc, time_csc, s=10, label ='csc', color='orange')
plt.scatter(density_ssgblas, time_ssgblas, s=10, label ='ssgblas', color='green')
plt.scatter(density_custom, time_custom, s=10, label ='custom', color='red')
plt.legend()
plt.xlabel('density')
plt.ylabel('time')
plt.title('Scatter plot of SpMM performance')
# plt.savefig("spmm-plot.png")

plt.savefig("results/spmm-plot-4.png")