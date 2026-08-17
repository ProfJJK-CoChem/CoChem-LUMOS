import json
import h5py
import sys
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ECOSYSTEM_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(ECOSYSTEM_ROOT))

def test_sspcm_provenance():
    print("Running Adversarial Audit 150: OpenAlex Excited State Solvation Models")
    try:
        from cochem_lumos.core.metadata import evaluate_excited_state_dipole
    except ImportError as e:
        print(f"FAILED: Could not import cochem_lumos.core.metadata. Error: {e}")
        raise e
    
    # Run the SS-PCM evaluation
    dipole = evaluate_excited_state_dipole(model="SS-PCM")
    print(f"Evaluated dipole: {dipole}")
    
    # Wait for the async OpenAlex fetch thread to write to the HDF5 file
    time.sleep(2)
    
    h5_path = "cochem_state.h5"
    assert os.path.exists(h5_path), "cochem_state.h5 file was not created!"
    
    with h5py.File(h5_path, "r") as f:
        existing_prov = f.attrs.get('provenance_algorithms', '{}')
        if isinstance(existing_prov, bytes):
            existing_prov = existing_prov.decode('utf-8')
        prov_dict = json.loads(existing_prov)
        
        found = False
        for k, v in prov_dict.items():
            if "10.1002/qua.560560850" in v:
                found = True
                break
                
        assert found, f"Failed to find Cammi 1995 DOI in provenance algorithms! Found: {prov_dict}"
        print("SUCCESS: The output metadata JSON contains the specific Cammi 1995 DOI string.")

if __name__ == "__main__":
    test_sspcm_provenance()
