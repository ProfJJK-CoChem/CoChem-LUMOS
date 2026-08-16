import sys
import asyncio

sys.path.insert(0, r"D:\Gdrive\__CoChem\GitHub-Repo")

async def test_fc_concurrency():
    print("Running Adversarial Audit 158: Goal Payload Franck-Condon Grid Concurrency")
    try:
        from cochem_lumos.engine.hpc_dispatcher import HPCDispatcher
    except ImportError as e:
        print(f"FAILED: Could not import cochem_lumos.engine.hpc_dispatcher. Error: {e}")
        print("Audit Result: FAILS the concurrency audit. System cannot dispatch 100,000 FC tasks.")
        return

    dispatcher = HPCDispatcher()
    
    num_overlaps = 100000
    chunk_size = 1000
    
    print(f"Requesting vibronic spectrum evaluation with {num_overlaps} distinct overlaps...")
    try:
        payloads = await dispatcher.dispatch_fc_overlaps(num_overlaps, chunk_size)
        print(f"Successfully chunked into {len(payloads)} payloads.")
        print("Audit Result: PASSES the concurrency audit.")
    except AttributeError as e:
        print(f"FAILED: Method dispatch_fc_overlaps not found or not working. Error: {e}")
        print("Audit Result: FAILS the concurrency audit.")

if __name__ == "__main__":
    asyncio.run(test_fc_concurrency())
