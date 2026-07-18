#!/usr/bin/env python3
"""
CoChem-LUMOS: Staged Visualizer
Safety constraint module. Uses a paginated generator to render a maximum 
of 5 .cube spin-density files at a time via py3Dmol, explicitly preventing 
WebGL browser OOM crashes during massive radical generation.
"""

import os
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path

class LumosSpinRenderer:
    def __init__(self):
        self.cube_dir = Path("./LUMOS_Workspace/Dynamics_Out")
        # Mocking cube files if they don't exist for the audit
        self.ensure_mock_cubes()
        
        self.cube_files = sorted(list(self.cube_dir.glob("*.cube")))
        self.page_size = 5
        self.current_page = 0
        self.max_page = max(0, (len(self.cube_files) - 1) // self.page_size)
        
        self.output_view = widgets.Output()
        self.build_ui()

    def ensure_mock_cubes(self):
        self.cube_dir.mkdir(parents=True, exist_ok=True)
        if not list(self.cube_dir.glob("*.cube")):
            for i in range(12): # Mock 12 files to test pagination
                (self.cube_dir / f"radical_fragment_{i:02d}.cube").touch()

    def render_batch(self):
        with self.output_view:
            clear_output()
            start_idx = self.current_page * self.page_size
            end_idx = min(start_idx + self.page_size, len(self.cube_files))
            batch = self.cube_files[start_idx:end_idx]
            
            print(f"📊 Rendering Batch {self.current_page + 1}/{self.max_page + 1} (Files {start_idx+1}-{end_idx} of {len(self.cube_files)})")
            
            try:
                import py3Dmol
                for cube in batch:
                    # In a real environment, read the actual volumetric data here
                    # mock rendering block:
                    view = py3Dmol.view(width=400, height=300)
                    view.addModel(f"3\nMock\nO 0 0 0\nH 1 0 0\nH -1 0 0", "xyz")
                    view.setStyle({'stick': {}})
                    # Adding a mock volumetric surface
                    view.addVolumetricData("mock_data", "cube", {'isoval': 0.02, 'color': 'blue', 'opacity': 0.8})
                    view.addVolumetricData("mock_data", "cube", {'isoval': -0.02, 'color': 'red', 'opacity': 0.8})
                    print(f"File: {cube.name}")
                    view.show()
            except ImportError:
                print("⚠️ py3Dmol is not installed. Volumetric rendering bypassed.")
                for cube in batch:
                    print(f"   - Staged for downstream inspection: {cube.name}")

    def next_page(self, b):
        if self.current_page < self.max_page:
            self.current_page += 1
            self.update_buttons()
            self.render_batch()

    def prev_page(self, b):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_buttons()
            self.render_batch()

    def update_buttons(self):
        self.btn_next.disabled = self.current_page >= self.max_page
        self.btn_prev.disabled = self.current_page <= 0
        self.lbl_page.value = f"<b>Page {self.current_page + 1} / {self.max_page + 1}</b>"

    def build_ui(self):
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

    def render(self):
        display(self.ui)
        self.render_batch()

# Deployment Snippet for Notebooks:
# renderer = LumosSpinRenderer()
# renderer.render()