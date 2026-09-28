import sys, json
from client import CosineBM25ReciprocalRankFusion

def main():
    engine = CosineBM25ReciprocalRankFusion()
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            method = req.get("method")
            rid = req.get("id")
            params = req.get("params", {})

            if method == "tools/list":
                res = {
                    "tools": [
                        {"name": "index_documents", "description": "Index documents.", "inputSchema": {"type": "object", "properties": {"documents": {"type": "array"}}, "required": ["documents"]}},
                        {"name": "bm25_search", "description": "BM25 search.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
                        {"name": "dense_search", "description": "Dense search.", "inputSchema": {"type": "object", "properties": {"query_vector": {"type": "array"}}, "required": ["query_vector"]}},
                        {"name": "run_benchmark_rrf_fusion", "description": "Run benchmark.", "inputSchema": {"type": "object"}}
                    ]
                }
            elif method == "tools/call":
                tname = params.get("name")
                args = params.get("arguments", {})
                if tname == "index_documents":
                    out = engine.index_documents(args.get("documents", []))
                elif tname == "bm25_search":
                    out = engine.bm25_search(args.get("query", ""))
                elif tname == "dense_search":
                    out = engine.dense_search(args.get("query_vector", []))
                elif tname == "run_benchmark_rrf_fusion":
                    out = engine.run_benchmark_rrf_fusion()
                else:
                    out = {"error": f"Unknown tool {tname}"}
                res = {"content": [{"type": "text", "text": json.dumps(out)}]}
            else:
                res = {"error": "Unsupported method"}
            print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}), flush=True)
        except Exception as e:
            print(json.dumps({"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}}), flush=True)

if __name__ == "__main__":
    main()
