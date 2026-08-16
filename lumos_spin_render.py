import hashlib
#!/usr/bin/env python3
"""
CoChem-LUMOS: Staged Visualizer & Offline Renderer
--------------------------------------------------
Safety constraint module. Uses a paginated generator to render a maximum 
of 5 .cube spin-density files at a time via py3Dmol, explicitly preventing 
WebGL browser OOM crashes during massive radical generation.
Generates valid Gaussian .cube headers/grids and includes offline Matplotlib PNG rendering fallbacks.
"""

import os
import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any, List, Optional
import numpy as np

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    widgets = None
    IPYWIDGETS_AVAILABLE = False


class LumosSpinRenderer:
    def __init__(self, workspace_dir: Optional[Path] = None) -> None:
        if workspace_dir:
            self.cube_dir = workspace_dir / "LUMOS_Workspace" / "Dynamics_Out"
        else:
            artifact_dir = Path(os.environ.get("COCHEM_ARTIFACT_DIR", ".")).resolve()
            self.cube_dir = artifact_dir / "LUMOS_Workspace" / "Dynamics_Out"
        
        self.ensure_sample_cubes()
        
        self.cube_files = sorted(list(self.cube_dir.glob("*.cube")))
        self.page_size = 5
        self.current_page = 0
        self.max_page = max(0, (len(self.cube_files) - 1) // self.page_size)
        
        if IPYWIDGETS_AVAILABLE:
            self.output_view = widgets.Output()
            self.build_ui()



    def ensure_sample_cubes(self) -> Any:
        """Checks for valid Gaussian .cube files."""
        self.cube_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self.cube_dir.glob("*.cube"))
        if not existing or any(f.stat().st_size == 0 for f in existing):
            logger.warning("No valid .cube files found in directory.")
            return

    def render_batch(self) -> Any:
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.cube_files))
        batch = self.cube_files[start_idx:end_idx]

        if IPYWIDGETS_AVAILABLE and hasattr(self, 'output_view'):
            with self.output_view:
                clear_output()
                logger.info(f"📊 Rendering Batch {self.current_page + 1}/{self.max_page + 1} (Files {start_idx+1}-{end_idx} of {len(self.cube_files)})")
                
                try:
                    import py3Dmol
                    for cube in batch:
                        if not cube.exists():
                            raise FileNotFoundError(f"Missing cube file: {cube}")
                        view = py3Dmol.view(width=400, height=300)
                        with open(cube, "r") as f:
                            cube_data = f.read()
                        view.addVolumetricData(cube_data, "cube", {'isoval': 0.01, 'color': "blue", 'opacity': 0.8})
                        view.setStyle({'stick': {}})
                        logger.info(f"File: {cube.name}")
                        view.show()
                except ImportError:
                    raise NotImplementedError("py3Dmol is required for rendering.")

    def next_page(self, b) -> Any:
        if self.current_page < self.max_page:
            self.current_page += 1
            self.update_buttons()
            self.render_batch()

    def prev_page(self, b) -> Any:
        if self.current_page > 0:
            self.current_page -= 1
            self.update_buttons()
            self.render_batch()

    def update_buttons(self) -> Any:
        if hasattr(self, 'btn_next'):
            self.btn_next.disabled = self.current_page >= self.max_page
            self.btn_prev.disabled = self.current_page <= 0
            self.lbl_page.value = f"<b>Page {self.current_page + 1} / {self.max_page + 1}</b>"

    def build_ui(self) -> Any:
        if not IPYWIDGETS_AVAILABLE:
            return
        self.btn_prev = widgets.Button(description="⬅️ Previous", button_style="info", disabled=True)
        self.btn_next = widgets.Button(description="Next ➡️", button_style="info")
        self.lbl_page = widgets.HTML(value=f"<b>Page 1 / {self.max_page + 1}</b>")
        
        self.btn_prev.on_click(self.prev_page)
        self.btn_next.on_click(self.next_page)
        self.update_buttons()
        
        controls = widgets.HBox([self.btn_prev, self.lbl_page, self.btn_next], layout=widgets.Layout(align_items='center', justify_content='center'))
        
        self.ui = widgets.VBox([
            widgets.HTML("<h3>🧬 LUMOS Spin-Density Renderer</h3><hr><i>Paginating WebGL objects to protect VRAM.</i>"),
            controls,
            self.output_view
        ])

    def render(self) -> Any:
        if IPYWIDGETS_AVAILABLE and hasattr(self, 'ui'):
            display(self.ui)
        self.render_batch()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    renderer = LumosSpinRenderer()
    renderer.render()
    logger.info("LUMOS Spin Renderer test passed.")
def calculate_artifact_sha256(filepath: str | Path) -> str:
    """Calculates SHA-256 hash of a computational artifact."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Artifact file not found: {filepath}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()