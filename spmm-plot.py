import json
import matplotlib.pyplot as plt

# with open('spmm-anl_results.json', 'r') as f:
#     pairs = json.load(f)

with open('spmm-anl_results-pls.json', 'r') as f:
    pairs = json.load(f)

density = [pair[0] for pair in pairs]
time = [pair[1] for pair in pairs]

# Plot
plt.scatter(density, time, s=10)
plt.xlabel('density')
plt.ylabel('time')
plt.title('Scatter plot of SpMM performance')
# plt.savefig("spmm-plot.png")

plt.savefig("spmm-plot-pls.png")