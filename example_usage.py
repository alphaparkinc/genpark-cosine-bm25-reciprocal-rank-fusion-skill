from client import CosineBM25ReciprocalRankFusion
import json

def main():
    engine = CosineBM25ReciprocalRankFusion()
    res = engine.run_benchmark_rrf_fusion()
    print("Cosine BM25 Reciprocal Rank Fusion Benchmark Result:")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
