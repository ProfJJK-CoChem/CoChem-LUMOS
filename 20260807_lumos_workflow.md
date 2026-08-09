# CoChem-LUMOS: Execution Workflow (2026-08-07)

## Phase 1: Pre-Screen & Method Selection
1. **sTDA Pre-Screen:** LUMOS executes a simplified TD-DFT (sTDA) calculation in milliseconds to locate the dominant absorbing energy spectrum and truncate the required virtual orbital space.
2. **Execution Selection:** Based on the Time-Tier, LUMOS fires standard TDA-DFT (using range-separated hybrids like CAM-B3LYP) or STEOM-CCSD via CoChem-NODE.

## Phase 2: Quantum Execution
1. **Vertical Excitation:** Davidson diagonalization solves the Casida equations iteratively, accelerated by RI-J approximations.
2. **$S_1$ Relaxation:** If Stokes shifts are requested, LUMOS analytically optimizes the $S_1$ geometry, monitoring the $S_1 - S_0$ gap to prevent falling into a Conical Intersection.

## Phase 3: Reporting & UX
1. **Interactive Visualization:** LUMOS streams the NTO `.cube` files directly to `cochem_state.h5`. The Jupyter UI allows the user to slide a bar to blend the "Hole" and "Electron" densities in Py3Dmol.
2. **Jablonski Generation:** The data generates a 2D Jablonski diagram explicitly separating Singlet/Triplet manifolds.
3. **Integrity Lock:** The use of TDA, exact functional parameters, and explicit solvent models are cryptographically stamped into the methodology output for SCRIBE.
