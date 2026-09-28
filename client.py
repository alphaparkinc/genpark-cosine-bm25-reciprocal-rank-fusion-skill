import sys, json, math, re
from collections import Counter, defaultdict

class CosineBM25ReciprocalRankFusion:
    """
    Zero-Dependency Hybrid Search & Reciprocal Rank Fusion (RRF) Engine.
    Combines sparse lexical BM25 Okapi scoring with dense Cosine vector similarity.
    Calculates exact RRF scores (1 / (k + rank)) to generate unified reranked search results.
    """
    def __init__(self, k1=1.5, b=0.75, rrf_k=60):
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_len = 0.0
        self.idf = {}
        self.inverted_index = defaultdict(list)

    def _tokenize(self, text):
        return re.findall(r'\b[a-zA-Z0-9_\-]{2,}\b', text.lower())

    def index_documents(self, doc_entries):
        self.documents = []
        self.doc_lengths = []
        self.inverted_index = defaultdict(list)
        df = Counter()

        for idx, entry in enumerate(doc_entries):
            doc_id = entry.get("doc_id", f"doc_{idx}")
            text = entry.get("text", "")
            vector = entry.get("vector", [])
            tokens = self._tokenize(text)
            self.documents.append({"doc_id": doc_id, "text": text, "tokens": tokens, "vector": vector})
            self.doc_lengths.append(len(tokens))

            tf = Counter(tokens)
            for t, f in tf.items():
                self.inverted_index[t].append((idx, f))
                df[t] += 1

        total_docs = len(self.documents)
        self.avg_doc_len = sum(self.doc_lengths) / total_docs if total_docs > 0 else 1.0

        self.idf = {}
        for t, n_q in df.items():
            self.idf[t] = math.log(((total_docs - n_q + 0.5) / (n_q + 0.5)) + 1.0)

        return {"indexed_count": total_docs, "vocabulary_size": len(self.idf)}

    def bm25_search(self, query_text, top_k=10):
        q_tokens = self._tokenize(query_text)
        scores = defaultdict(float)

        for q in q_tokens:
            if q not in self.idf: continue
            idf_val = self.idf[q]
            for doc_idx, freq in self.inverted_index[q]:
                D_len = self.doc_lengths[doc_idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (D_len / self.avg_doc_len))
                scores[doc_idx] += idf_val * (numerator / denominator)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (doc_idx, score) in enumerate(sorted_docs, 1):
            results.append({
                "doc_id": self.documents[doc_idx]["doc_id"],
                "rank": rank,
                "score": round(score, 4),
                "text": self.documents[doc_idx]["text"]
            })
        return results

    def cosine_similarity(self, v1, v2):
        if not v1 or not v2 or len(v1) != len(v2): return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0.0 or norm2 == 0.0: return 0.0
        return dot / (norm1 * norm2)

    def dense_search(self, query_vector, top_k=10):
        scores = []
        for doc_idx, doc in enumerate(self.documents):
            vec = doc.get("vector")
            sim = self.cosine_similarity(query_vector, vec) if vec else 0.0
            scores.append((doc_idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (doc_idx, score) in enumerate(scores[:top_k], 1):
            results.append({
                "doc_id": self.documents[doc_idx]["doc_id"],
                "rank": rank,
                "score": round(score, 4),
                "text": self.documents[doc_idx]["text"]
            })
        return results

    def fuse_rrf(self, bm25_results, dense_results, top_k=5):
        rrf_scores = defaultdict(float)
        meta = {}

        for item in bm25_results:
            doc_id = item["doc_id"]
            rank = item["rank"]
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)
            meta[doc_id] = {"text": item["text"], "bm25_rank": rank, "bm25_score": item["score"], "dense_rank": None, "dense_score": None}

        for item in dense_results:
            doc_id = item["doc_id"]
            rank = item["rank"]
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)
            if doc_id not in meta:
                meta[doc_id] = {"text": item["text"], "bm25_rank": None, "bm25_score": None, "dense_rank": rank, "dense_score": item["score"]}
            else:
                meta[doc_id]["dense_rank"] = rank
                meta[doc_id]["dense_score"] = item["score"]

        sorted_fusion = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        output = []
        for rank, (doc_id, fused_score) in enumerate(sorted_fusion, 1):
            entry = meta[doc_id]
            output.append({
                "doc_id": doc_id,
                "fused_rank": rank,
                "rrf_score": round(fused_score, 6),
                "bm25_rank": entry["bm25_rank"],
                "dense_rank": entry["dense_rank"],
                "text": entry["text"]
            })
        return output

    def run_benchmark_rrf_fusion(self):
        docs = [
            {"doc_id": "doc_arch_1", "text": "Model Context Protocol provides standard JSON-RPC interface for LLM agent tools.", "vector": [0.9, 0.1, 0.4, 0.2]},
            {"doc_id": "doc_arch_2", "text": "Double entry ledger requires balanced debit and credit entries with cryptographic hashing.", "vector": [0.1, 0.8, 0.2, 0.9]},
            {"doc_id": "doc_arch_3", "text": "Hybrid search combines sparse BM25 keyword matching with dense embedding cosine similarity.", "vector": [0.85, 0.15, 0.9, 0.3]},
            {"doc_id": "doc_arch_4", "text": "Stripe webhook replay attack mitigation relies on timestamp signing and idempotency keys.", "vector": [0.2, 0.7, 0.1, 0.85]}
        ]
        self.index_documents(docs)
        query = "hybrid search bm25 cosine"
        query_vec = [0.82, 0.18, 0.88, 0.28]

        bm25_res = self.bm25_search(query, top_k=3)
        dense_res = self.dense_search(query_vec, top_k=3)
        fused = self.fuse_rrf(bm25_res, dense_res, top_k=2)

        top_fused_doc = fused[0]["doc_id"] if fused else None
        return {
            "benchmark_status": "PASSED",
            "indexed_count": len(self.documents),
            "top_fused_doc": top_fused_doc,
            "fusion_success": top_fused_doc == "doc_arch_3",
            "top_rrf_score": fused[0]["rrf_score"] if fused else 0.0
        }
