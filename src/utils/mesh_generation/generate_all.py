"""Generate and export meshes for all supported flow domains.

Run from the repo root with the fenics conda environment active:
    python src/utils/mesh_generation/generate_all.py
"""

from pathlib import Path

from utils.mesh_generation import cavity, cylinder, lidcavity, pinball

OUT = Path(__file__).parent / "data_output"
OUT.mkdir(exist_ok=True)

print("Cylinder...")
cylinder.generate_mesh(str(OUT / "cylinder"), verbose=True)

print("Cavity...")
cavity.generate_mesh(str(OUT / "cavity"), verbose=True)

print("Lid-driven cavity...")
lidcavity.generate_mesh(str(OUT / "lidcavity"), verbose=True)

print("Pinball...")
pinball.generate_mesh(str(OUT / "pinball"), verbose=True)

print(f"Done. Files written to {OUT}/")
