
import ir_datasets
try:
    ds = ir_datasets.load("beir/nfcorpus")
    print("Dataset loaded:", ds)
    print("Has qrels_iter:", hasattr(ds, "qrels_iter"))
    if hasattr(ds, "qrels_iter"):
        print("Iterating qrels...")
        count = 0
        for x in ds.qrels_iter():
            count += 1
            if count > 5: break
        print("Qrels check passed.")
    else:
        print("Attributes:", dir(ds))
except Exception as e:
    print("Error:", e)
