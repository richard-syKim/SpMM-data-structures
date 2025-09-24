import json
import matplotlib.pyplot as plt


with open("res/refine/sparse-arrays-csc.json", "r") as f:
    pairs_sa_csc = json.load(f)

den_sa_csc = [pair[0] for pair in pairs_sa_csc]
time_sa_csc = [pair[1] for pair in pairs_sa_csc]


with open("res/refine/suite-sparse-graph-blas.json", "r") as f:
    pairs_ssgblas = json.load(f)

den_ssgblas = [pair[0] for pair in pairs_ssgblas]
time_ssgblas = [pair[1] for pair in pairs_ssgblas]


with open("res/refine/custom-coo.json", "r") as f:
    pairs_cus_coo = json.load(f)

den_cus_coo = [pair[0] for pair in pairs_cus_coo]
time_cus_coo = [pair[1] for pair in pairs_cus_coo]


with open("res/refine/finch-csc.json", "r") as f:
    pairs_fin_csc = json.load(f)

den_fin_csc = [pair[0] for pair in pairs_fin_csc]
time_fin_csc = [pair[1] for pair in pairs_fin_csc]


with open("res/refine/finch-csf.json", "r") as f:
    pairs_fin_csf = json.load(f)

den_fin_csf = [pair[0] for pair in pairs_fin_csf]
time_fin_csf = [pair[1] for pair in pairs_fin_csf]


with open("res/refine/finch-dcsc.json", "r") as f:
    pairs_fin_dcsc = json.load(f)

den_fin_dcsc = [pair[0] for pair in pairs_fin_dcsc]
time_fin_dcsc = [pair[1] for pair in pairs_fin_dcsc]


with open("res/refine/finch-dcsf.json", "r") as f:
    pairs_fin_dcsf = json.load(f)

den_fin_dcsf = [pair[0] for pair in pairs_fin_dcsf]
time_fin_dcsf = [pair[1] for pair in pairs_fin_dcsf]


with open("res/refine/finch-coo.json", "r") as f:
    pairs_fin_coo = json.load(f)

den_fin_coo = [pair[0] for pair in pairs_fin_coo]
time_fin_coo = [pair[1] for pair in pairs_fin_coo]


with open("res/refine/finch-hash.json", "r") as f:
    pairs_fin_hash = json.load(f)

den_fin_hash = [pair[0] for pair in pairs_fin_hash]
time_fin_hash = [pair[1] for pair in pairs_fin_hash]


with open("res/refine/finch-bm.json", "r") as f:
    pairs_fin_bm = json.load(f)

den_fin_bm = [pair[0] for pair in pairs_fin_bm]
time_fin_bm = [pair[1] for pair in pairs_fin_bm]



den_sa_csc = den_sa_csc[:len(den_fin_bm) // 5]
time_sa_csc = time_sa_csc[:len(time_fin_bm) // 5]
                          
den_ssgblas = den_ssgblas[:len(den_fin_bm) // 5]
time_ssgblas = time_ssgblas[:len(time_fin_bm) // 5]

den_cus_coo = den_cus_coo[:len(den_fin_bm) // 5]
time_cus_coo = time_cus_coo[:len(time_fin_bm) // 5]

den_fin_csc = den_fin_csc[:len(den_fin_bm) // 5]
time_fin_csc = time_fin_csc[:len(time_fin_bm) // 5]

den_fin_csf = den_fin_csf[:len(den_fin_bm) // 5]
time_fin_csf = time_fin_csf[:len(time_fin_bm) // 5]

den_fin_dcsc = den_fin_dcsc[:len(den_fin_bm) // 5]
time_fin_dcsc = time_fin_dcsc[:len(time_fin_bm) // 5]

den_fin_dcsf = den_fin_dcsf[:len(den_fin_bm) // 5]
time_fin_dcsf = time_fin_dcsf[:len(time_fin_bm) // 5]

den_fin_coo = den_fin_coo[:len(den_fin_bm) // 5]
time_fin_coo = time_fin_coo[:len(time_fin_bm) // 5]

den_fin_hash = den_fin_hash[:len(den_fin_bm) // 5]
time_fin_hash = time_fin_hash[:len(time_fin_bm) // 5]

den_fin_bm = den_fin_bm[:len(den_fin_bm) // 5]
time_fin_bm = time_fin_bm[:len(time_fin_bm) // 5]



# Plot
plt.figure(figsize=(10, 6))

plt.scatter(den_sa_csc, time_sa_csc, s=10, label='sparse-arrays-csc', color='blue')
# plt.scatter(den_ssgblas, time_ssgblas, s=10, label='suite-sparse-graph-blas', color='orange')
plt.scatter(den_cus_coo, time_cus_coo, s=10, label='custom-coo', color='green')

plt.scatter(den_fin_csc, time_fin_csc, s=10, label='finch-csc', color='red')
plt.scatter(den_fin_csf, time_fin_csf, s=10, label='finch-csf', color='purple')
plt.scatter(den_fin_dcsc, time_fin_dcsc, s=10, label='finch-dcsc', color='brown')
plt.scatter(den_fin_dcsf, time_fin_dcsf, s=10, label='finch-dcsf', color='pink')
plt.scatter(den_fin_coo, time_fin_coo, s=10, label='finch-coo', color='gray')
plt.scatter(den_fin_hash, time_fin_hash, s=10, label='finch-hash', color='cyan')
plt.scatter(den_fin_bm, time_fin_bm, s=10, label='finch-bm', color='magenta')

plt.legend()
plt.xlabel('density')
plt.ylabel('time (ns)')
plt.title('Scatter plot of SpMM performance')

plt.savefig("img/spmm-plot-refine-0.1.png")