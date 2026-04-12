import json
from collections import defaultdict
from numbers import Number
from typing import Dict

import ir_datasets


def load_qrels_from_ir_datasets(
    dataset_name: str,
    binarize: bool = False,
    binarize_threshold: int = 1,
) -> Dict[str, Dict[str, Number]]:
    """
    Load the qrels from ir_datasets.

    Args:
        dataset_name (str): The dataset name to use.

    Returns:
        Dict[str, Dict[str, Number]]: The qrels.
    """
    try:
        dataset = ir_datasets.load(dataset_name)
    except KeyError:
         # Fallback for some naming conventions if needed, though usually load works
         pass

    qrels = defaultdict(dict)

    # CRITICAL FIX: Handle datasets where qrels are in a split (e.g. /test)
    iterator = None
    if hasattr(dataset, 'qrels_iter'):
        iterator = dataset.qrels_iter()
    else:
        # Try appending /test (Common for BEIR datasets in ir_datasets)
        print(f"[WARN] No qrels found for '{dataset_name}'. Trying '{dataset_name}/test'...")
        try:
            dataset_test = ir_datasets.load(dataset_name + "/test")
            if hasattr(dataset_test, 'qrels_iter'):
                iterator = dataset_test.qrels_iter()
        except Exception as e:
            print(f"[ERROR] Failed to load qrels from {dataset_name}/test: {e}")
    
    if iterator is None:
        raise ValueError(f"Could not find qrels (qrels_iter) for dataset {dataset_name}")

    for qrel in iterator:
        relevance = int(qrel.relevance)

        if binarize:
            relevance = relevance if relevance >= binarize_threshold else 0

        qrels[str(qrel.query_id)][str(qrel.doc_id)] = relevance

    return qrels


def load_qrels_from_json(
    input_json: str,
) -> Dict[str, Dict[str, Number]]:
    """
    Load the qrels from a json file.

    Args:
        input_json (str): The input json file.

    Returns:
        Dict[str, Dict[str, Number]]: The qrels.
    """

    with open(input_json, "r") as f:
        qrels = json.load(f)

    return qrels
