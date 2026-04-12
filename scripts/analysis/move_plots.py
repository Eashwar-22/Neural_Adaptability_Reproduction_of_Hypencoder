
import os
import shutil

src_dir = "./msc_thesis/images/new_plots"
dst_dir = "./msc_thesis/images"

files = [
    "qnet_weight_vs_bias_rsd.png",
    "qnet_svd_spectrum_log.png",
    "qnet_entropy_by_category.png"
]

for f in files:
    src = os.path.join(src_dir, f)
    dst = os.path.join(dst_dir, f)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {f} to {dst_dir}")
    else:
        print(f"Not found: {src}")
