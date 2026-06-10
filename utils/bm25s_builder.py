# bm25s_builder.py

import argparse
import pickle
import datasets
import pandas as pd
import bm25s

def main():
    parser = argparse.ArgumentParser(description="Build BM25s cache")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., ariflaksito/exarank1)"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Top-k retrieval size"
    )

    args = parser.parse_args()

    ds_name = args.dataset
    top_k = args.k

    print(f"Loading dataset: {ds_name}")
    df = datasets.load_dataset(ds_name)

    _train = pd.DataFrame(df["train"])
    _test = pd.DataFrame(df["test"])

    _train["docs"] = _train["query"] + " [SEP] " + _train["doc"]
    _test["docs"] = _test["query"] + " [SEP] " + _test["doc"]

    print("Building BM25s index...")
    corpus = _train["docs"].tolist()
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    print("Retrieving for all test samples...")
    train = _train.to_dict(orient="records")
    test = _test.to_dict(orient="records")

    test_queries = [row["docs"] for row in test]
    query_tokens = bm25s.tokenize(test_queries, stopwords="en")

    results, scores = retriever.retrieve(query_tokens, k=top_k)

    all_indices = [results[i].tolist() for i in range(len(test))]

    # Clean dataset name for filename
    safe_name = ds_name.split("/")[-1]

    output_file = f"bm25s_topk_{safe_name}.pkl"

    print("Saving cache...")
    cache = {
        "ds_name": ds_name,
        "k": top_k,
        "train": train,
        "all_indices": all_indices,
    }

    with open(output_file, "wb") as f:
        pickle.dump(cache, f)

    print(f"Done! Saved {output_file} ({len(all_indices)} test samples)")


if __name__ == "__main__":
    main()