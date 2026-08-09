# CoChem-LUMOS: Software Engineering Specification
**Target Phase:** Python Implementation

This document serves as the exact coding blueprint for the next LLM agent to construct the `CoChem-LUMOS` repository.

## 1. Directory & File Architecture
```text
CoChem-LUMOS/
├── lumos_core/
│   ├── __init__.py
│   ├── dispatcher.py       # Entry point for BASE payload ingestion
│   ├── td_dft.py           # TDA integration and Casida equation setup
│   ├── steom_ccsd.py       # High-tier Time-Tier integration
│   ├── nto_generator.py    # Natural Transition Orbital parsing
│   └── crossing_guard.py   # Conical Intersection (CI) root tracker
├── tests/
│   ├── test_crossing.py
│   └── test_nto_parse.py
├── requirements.txt        # h5py, numpy, scipy
└── README.md
```

## 2. File-by-File Blueprint

### `lumos_core/crossing_guard.py`
- **Purpose:** Monitors geometrical relaxation of the $S_1$ state.
- **Functions:**
  - `def check_s1_s0_gap(s1_energy: float, s0_energy: float) -> bool:`
    - *Returns:* `True` if gap $> 0.2$ eV. If `< 0.2` eV, triggers `BaseException("CONICAL_INTERSECTION_DETECTED")`.

### `lumos_core/nto_generator.py`
- **Purpose:** Extracts hole/electron distributions.
- **Functions:**
  - `def parse_cube_to_array(cube_path: str) -> np.ndarray:`
    - *Returns:* A flattened numpy array representing the 3D density grid.
  - `def serialize_to_hdf5(h5_group, root_index: int, hole_array, elec_array):`
    - *Action:* Memory-maps the density arrays into `/lumos/nto_cubes/root_{N}`.

### `lumos_core/td_dft.py`
- **Purpose:** Writes the ORCA excitation block.
- **Functions:**
  - `def write_tda_block(n_roots: int, functional: str) -> str:`
    - *Returns:* The ORCA string explicitly enforcing `%cis tda true end`.

## 3. Execution Data Flow (The Payload Trace)
1. **Payload Ingest:** `dispatcher.py` triggers on UV-Vis/Excited State tasks from BASE.
2. **Method Selection:** If `Time-Tier <= 7`, calls `td_dft.py`. If `Time-Tier >= 8`, calls `steom_ccsd.py`.
3. **Execution & Parsing:** Executes via `CoChem-NODE`. Upon completion, parses the output to extract oscillator strengths ($f$) and vertical excitation energies.
4. **NTO Extraction:** Calls `nto_generator.py` to parse the ORCA-generated `.cube` files into NumPy arrays.
5. **Optimization Monitoring:** If an $S_1$ relaxation was requested, `crossing_guard.py` monitors every geometry step. If an avoided crossing collapses, it halts execution and alerts `BASE` to trigger `PULSE`.
6. **Serialization:** Writes transition dipole moments and NTO tensors into `/lumos/`.

## 4. PyTest Roadmap
- **Test 1 (`test_crossing.py`):** Provide a mock array of decreasing $S_1-S_0$ gaps (e.g., `[1.0, 0.5, 0.15]`). Assert that `check_s1_s0_gap` correctly throws the CI exception on the final step.
- **Test 2 (`test_nto_parse.py`):** Use a dummy `.cube` file. Assert that `parse_cube_to_array` correctly reshapes the header coordinates and extracts the scalar volumetric grid without memory leakage.
