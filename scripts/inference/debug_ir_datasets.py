import ir_datasets

try:
    ds = ir_datasets.load("beir/nfcorpus")
    print(f"--- beir/nfcorpus ---")
    print(dir(ds))
except Exception as e:
    print(e)
