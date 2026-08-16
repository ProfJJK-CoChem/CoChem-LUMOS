from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional

class LumosConfig(BaseModel):
    pump_nm: float = Field(default=266.0)
    temp_k: float = Field(default=298.15)
    tier: str = Field(default="T1-30min")
    samples: Optional[int] = Field(default=None)
    solvent: str = Field(default="water")
    input_xyz: Path

class LumosStatus(BaseModel):
    status: str
    pump_nm: float
    target_temp_k: float
    solvent: str
    tier_level: str
    gpu_crossover_mode: str
    basis_functions: int
    trajectories_tracked: int
    output_dir: str
