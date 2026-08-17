import asyncio
import time
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ECOSYSTEM_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(ECOSYSTEM_ROOT))

def test_namd_concurrency():
    print("Running Adversarial Audit 148: NAMD Trajectory Parallelism")
    
    try:
        from cochem_lumos.engine.hpc_dispatcher import HPCDispatcher
    except ImportError as e:
        print(f"FAILED: Could not import cochem_lumos.engine.hpc_dispatcher. Error: {e}")
        raise e
        
    dispatcher = HPCDispatcher()
    num_trajectories = 500
    duration_ps = 1.0
    
    print(f"Submitting NAMD run with {num_trajectories} independent Tully surface-hopping trajectories over {duration_ps} ps...")
    
    start_time = time.time()
    payloads = asyncio.run(dispatcher.dispatch_namd_trajectories(num_trajectories, duration_ps))
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"Dispatched 500 payloads in {elapsed:.4f} seconds.")
    
    # Assertions
    assert len(payloads) == 500, f"Expected 500 payloads, got {len(payloads)}"
    for payload in payloads:
        assert payload["endpoint"] == "/goal", "Endpoint must be /goal"
        assert payload["task"] == "namd_surface_hopping", "Task must be NAMD"
        assert payload["duration_ps"] == 1.0, "Duration must be 1.0 ps"

if __name__ == "__main__":
    test_namd_concurrency()
