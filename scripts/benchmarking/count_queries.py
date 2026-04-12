import ir_datasets
datasets = [
    "msmarco-passage/dev/small",
    "msmarco-passage/trec-dl-2019/judged",
    "msmarco-passage/trec-dl-2020/judged",
    "beir/scifact/test",
    "beir/nfcorpus/test",
    "beir/fiqa/test",
    "beir/trec-covid",
    "beir/arguana",
    "beir/dbpedia-entity/test",
    "beir/webis-touche2020/v2"
]

print(f"{'Dataset':<40} | {'Queries':<10}")
print("-" * 55)

for ds_name in datasets:
    try:
        ds = ir_datasets.load(ds_name)
        count = sum(1 for _ in ds.queries_iter())
        print(f"{ds_name:<40} | {count:<10}")
    except Exception as e:
        print(f"{ds_name:<40} | ERROR: {e}")
