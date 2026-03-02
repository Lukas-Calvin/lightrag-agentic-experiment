import os, requests

HOST = os.getenv("LLM_BINDING_HOST", "https://api.zyai.online/v1")
KEY = os.getenv("LLM_BINDING_API_KEY", "")
MODEL = os.getenv("EMBEDDING_MODEL", "")

def check(texts):
    r = requests.post(
        f"{HOST}/embeddings",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "input": texts},
        timeout=60,
    )
    print("HTTP", r.status_code)
    r.raise_for_status()
    j = r.json()

    data = j.get("data", [])
    print("input_count =", len(texts), "data_count =", len(data))

    # 额外检查：每条 embedding 的维度是否一致
    dims = [len(item.get("embedding", [])) for item in data]
    print("dims =", dims[:5], ("..." if len(dims) > 5 else ""))
    if len(data) != len(texts):
        raise SystemExit(f"FAIL: expected {len(texts)} vectors but got {len(data)}")
    if len(set(dims)) != 1:
        raise SystemExit(f"FAIL: inconsistent embedding dims: {set(dims)}")
    print("PASS")

if __name__ == "__main__":
    if not KEY or not MODEL:
        raise SystemExit("Please set EMBED_KEY and EMBED_MODEL environment variables.")

    check(["test"])
    check(["a", "b"])
