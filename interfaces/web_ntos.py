import plotly.graph_objects as go
from plotly.subplots import make_subplots

class NTORenderer:
    @staticmethod
    def render_isosurfaces(hole_grid, particle_grid, isovalue=0.02):
        """
        Renders Natural Transition Orbitals (NTOs) as dual WebGL isosurfaces.
        The javascript engine (plotly.js) intrinsically applies the marching cubes 
        algorithm on the client side to render the 3D volume grids.
        
        Args:
            hole_grid (dict): Dictionary with keys 'X', 'Y', 'Z', 'V' as flattened 1D arrays or 3D grids
            particle_grid (dict): Dictionary with keys 'X', 'Y', 'Z', 'V' as flattened 1D arrays or 3D grids
            isovalue (float): The exact isosurface boundary threshold
        """
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('Hole NTO', 'Particle NTO')
        )
        
        # We enforce exactly 2 surfaces (positive and negative phase)
        # using the standard RdBu (Red/Blue) phase color coding
        
        hole_surf = go.Isosurface(
            x=hole_grid['X'].flatten(),
            y=hole_grid['Y'].flatten(),
            z=hole_grid['Z'].flatten(),
            value=hole_grid['V'].flatten(),
            isomin=-isovalue,
            isomax=isovalue,
            surface_count=2,
            colorscale='RdBu',
            caps=dict(x_show=False, y_show=False, z_show=False)
        )
        
        particle_surf = go.Isosurface(
            x=particle_grid['X'].flatten(),
            y=particle_grid['Y'].flatten(),
            z=particle_grid['Z'].flatten(),
            value=particle_grid['V'].flatten(),
            isomin=-isovalue,
            isomax=isovalue,
            surface_count=2,
            colorscale='RdBu',
            caps=dict(x_show=False, y_show=False, z_show=False)
        )
        
        fig.add_trace(hole_surf, row=1, col=1)
        fig.add_trace(particle_surf, row=1, col=2)
        
        fig.update_layout(
            title_text=f"Natural Transition Orbitals (Isovalue = {isovalue})",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        )
        
        # Return complete HTML payload encapsulating the WebGL/Marching Cubes runtime
        return fig.to_html(include_plotlyjs='cdn', full_html=True)
