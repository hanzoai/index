import json
import statistics

def calculate_exact_rrf(k_values, dense, sparse):
    results = {}
    for k in k_values:
        fused = [1/(k+d) + 1/(k+s) for d, s in zip(dense, sparse)]
        results[f"k={k}"] = {"p99_score": round(statistics.quantiles(fused, n=100)[98], 5)}
    
    with open("ablation_study.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    import random
    dense_ranks = [random.randint(1, 100) for _ in range(500)]
    sparse_ranks = [random.randint(1, 100) for _ in range(500)]
    calculate_exact_rrf([10, 60, 100], dense_ranks, sparse_ranks)
