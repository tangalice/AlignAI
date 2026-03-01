"""Quick test of Supermemory API - add a memory and search for it."""
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    from supermemory import Supermemory

    client = Supermemory()
    # API key loads from SUPERMEMORY_API_KEY env (set in .env)

    print("Adding memory...")
    client.add(
        content="User prefers dark mode",
        container_tags=["user-123"],
    )
    print("✓ Memory added")

    print("\nSearching memories...")
    results = client.search.documents(
        q="dark mode",
        container_tags=["user-123"],
    )
    print("✓ Search results:")
    print(f"  Total: {len(results.results)}")
    for r in results.results:
        text = getattr(r, "chunk", None) or getattr(r, "memory", None) or getattr(r, "content", None)
        print(f"  - {text}")
    if not results.results:
        print("  (Indexing may take a few seconds - run again in 10 sec)")

if __name__ == "__main__":
    main()
