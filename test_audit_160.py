import json
import h5py
import sys
import os
sys.path.insert(0, r"D:\Gdrive\__CoChem\GitHub-Repo")

def test_meci_provenance():
    print("Running Adversarial Audit 160: OpenAlex Conical Intersection Provenance")
    
    try:
        from cochem_lumos.core.metadata import provenance_algorithms
    except ImportError as e:
        print(f"FAILED: Could not import cochem_lumos.core.metadata. Error: {e}")
        print("Null Response Check: System outputs an empty citation block for the CI search (module does not exist).")
        print("Audit Result: FAILS the provenance audit.")
        return

    # If by some miracle it exists
    print("Module found. Checking for Martinez OpenAlex DOI...")
    from cochem_lumos.core.metadata import optimize_meci
    
    # Run the optimization which should fetch and write to cochem_state.h5
    result = optimize_meci()
    
    # Intercept the output .h5 state file.
    h5_path = "cochem_state.h5"
    assert os.path.exists(h5_path), "cochem_state.h5 file was not created!"
    
    with h5py.File(h5_path, "r") as f:
        existing_prov = f.attrs.get('provenance_algorithms', '{}')
        if isinstance(existing_prov, bytes):
            existing_prov = existing_prov.decode('utf-8')
        prov_dict = json.loads(existing_prov)
        
        # The provenance_algorithms dictionary MUST contain the OpenAlex DOI for the original Martinez paper
        found = False
        for k, v in prov_dict.items():
            if "10.1146/annurev.physchem.57.032905.104612" in v:
                found = True
                break
                
        assert found, f"Failed to find Martinez DOI in provenance algorithms! Found: {prov_dict}"
        print("SUCCESS: The output metadata JSON contains the specific Martinez DOI string.")

if __name__ == "__main__":
    test_meci_provenance()
