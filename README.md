# CoChem-LUMOS

**CoChem-LUMOS** is the UV-Vis and Excited State Static engine of the extended CoChem suite.

It is responsible for:
- Executing advanced TD-DFT analysis utilizing the Tamm-Dancoff Approximation (TDA) to circumvent triplet instabilities.
- Deploying STEOM-CCSD for high-accuracy vertical excitation energy benchmarks on complex molecular systems.
- Automatically generating state-specific Natural Transition Orbitals (NTOs) to enforce a rigorous hole-electron pairing visualization model.
- Implementing strict root-following during $S_1$ geometrical relaxations and automatically flagging Conical Intersections if the $S_1 - S_0$ gap collapses below critical thresholds.

## Usage
Please refer to the authoritative `CoChem_Master_User_Manual.md` located in the `CoChem-BASE` repository for full execution instructions across the entire pipeline.