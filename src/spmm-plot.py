import json
import matplotlib.pyplot as plt

with open("res/0929/dense.json", "r") as f:
    pairs_dense = json.load(f)

den_dense = [pair[0] for pair in pairs_dense]
time_dense = [pair[1] for pair in pairs_dense]


with open("res/0929/sparse-arrays-csc.json", "r") as f:
    pairs_sa_csc = json.load(f)

den_sa_csc = [pair[0] for pair in pairs_sa_csc]
time_sa_csc = [pair[1] for pair in pairs_sa_csc]


with open("res/0929/custom-coo.json", "r") as f:
    pairs_cus_coo = json.load(f)

den_cus_coo = [pair[0] for pair in pairs_cus_coo]
time_cus_coo = [pair[1] for pair in pairs_cus_coo]


with open("res/1001/finch-csc.json", "r") as f:
    pairs_fin_csc_mac = json.load(f)

den_fin_csc_mac = [pair[0] for pair in pairs_fin_csc_mac]
time_fin_csc_mac = [pair[1] for pair in pairs_fin_csc_mac]


with open("res/1001/finch-csf.json", "r") as f:
    pairs_fin_csf_mac = json.load(f)

den_fin_csf_mac = [pair[0] for pair in pairs_fin_csf_mac]
time_fin_csf_mac = [pair[1] for pair in pairs_fin_csf_mac]


with open("res/1001/finch-dcsc.json", "r") as f:
    pairs_fin_dcsc_mac = json.load(f)

den_fin_dcsc_mac = [pair[0] for pair in pairs_fin_dcsc_mac]
time_fin_dcsc_mac = [pair[1] for pair in pairs_fin_dcsc_mac]


with open("res/1001/finch-dcsf.json", "r") as f:
    pairs_fin_dcsf_mac = json.load(f)

den_fin_dcsf_mac = [pair[0] for pair in pairs_fin_dcsf_mac]
time_fin_dcsf_mac = [pair[1] for pair in pairs_fin_dcsf_mac]


with open("res/1001/finch-coo.json", "r") as f:
    pairs_fin_coo_mac = json.load(f)

den_fin_coo_mac = [pair[0] for pair in pairs_fin_coo_mac]
time_fin_coo_mac = [pair[1] for pair in pairs_fin_coo_mac]


with open("res/1001/finch-hash.json", "r") as f:
    pairs_fin_hash_mac = json.load(f)

den_fin_hash_mac = [pair[0] for pair in pairs_fin_hash_mac]
time_fin_hash_mac = [pair[1] for pair in pairs_fin_hash_mac]


with open("res/1001/finch-bm.json", "r") as f:
    pairs_fin_bm_mac = json.load(f)

den_fin_bm_mac = [pair[0] for pair in pairs_fin_bm_mac]
time_fin_bm_mac = [pair[1] for pair in pairs_fin_bm_mac]



# den_sa_csc = den_sa_csc[:len(den_fin_bm) // 3]
# time_sa_csc = time_sa_csc[:len(time_fin_bm) // 3]

# den_cus_coo = den_cus_coo[:len(den_fin_bm) // 3]
# time_cus_coo = time_cus_coo[:len(time_fin_bm) // 3]

# den_fin_csc = den_fin_csc[:len(den_fin_bm) // 3]
# time_fin_csc = time_fin_csc[:len(time_fin_bm) // 3]

# den_fin_csf = den_fin_csf[:len(den_fin_bm) // 3]
# time_fin_csf = time_fin_csf[:len(time_fin_bm) // 3]

# den_fin_dcsc = den_fin_dcsc[:len(den_fin_bm) // 3]
# time_fin_dcsc = time_fin_dcsc[:len(time_fin_bm) // 3]

# den_fin_dcsf = den_fin_dcsf[:len(den_fin_bm) // 3]
# time_fin_dcsf = time_fin_dcsf[:len(time_fin_bm) // 3]

# den_fin_coo = den_fin_coo[:len(den_fin_bm) // 3]
# time_fin_coo = time_fin_coo[:len(time_fin_bm) // 3]

# den_fin_hash = den_fin_hash[:len(den_fin_bm) // 3]
# time_fin_hash = time_fin_hash[:len(time_fin_bm) // 3]

# den_fin_bm = den_fin_bm[:len(den_fin_bm) // 3]
# time_fin_bm = time_fin_bm[:len(time_fin_bm) // 3]



# Plot
plt.figure(figsize=(10, 6))

plt.scatter(den_dense, time_dense, s=3, label='dense', color='orange')
plt.scatter(den_sa_csc, time_sa_csc, s=3, label='sparse-arrays-csc', color='blue')
plt.scatter(den_cus_coo, time_cus_coo, s=3, label='custom-coo', color='green')
plt.scatter(den_fin_csc_mac, time_fin_csc_mac, s=3, label='finch-csc', color='red')
plt.scatter(den_fin_csf_mac, time_fin_csf_mac, s=3, label='finch-csf', color='purple')
plt.scatter(den_fin_dcsc_mac, time_fin_dcsc_mac, s=3, label='finch-dcsc', color='brown')
plt.scatter(den_fin_dcsf_mac, time_fin_dcsf_mac, s=3, label='finch-dcsf', color='pink')
plt.scatter(den_fin_coo_mac, time_fin_coo_mac, s=3, label='finch-coo', color='gray')
plt.scatter(den_fin_hash_mac, time_fin_hash_mac, s=3, label='finch-hash', color='olive')
plt.scatter(den_fin_bm_mac, time_fin_bm_mac, s=3, label='finch-bm', color='cyan')

plt.legend()
plt.xlabel('density')
plt.ylabel('time (ns)')
plt.title('Scatter plot of SpMM performance with finch macros')

plt.xscale('log')
plt.yscale('log')
plt.savefig("img/1001/spmm-plot-log-mac.png")