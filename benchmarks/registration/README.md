# Registration Benchmarks

Tests `dvfopt` as a **post-processing step** on displacement fields
produced by third-party registration algorithms. Each notebook runs a
registration, checks for negative Jacobian determinants, then applies
the SLSQP or barrier correction.

## Dependencies

Several of these notebooks require optional extras:

```bash
pip install -e ".[benchmarks]"  # itk-elastix, opencv, timm, torch, voxelmorph
```

ANTsPy is pulled in by `benchmark-registration-methods.ipynb` via
`antspyx` (install separately if not already present).

## Notebooks

### [benchmark-registration-methods.ipynb](benchmark-registration-methods.ipynb) · [benchmark-registration-methods-barrier.ipynb](benchmark-registration-methods-barrier.ipynb)
End-to-end sweep across multiple registration methods, with SLSQP
correction and a barrier-based companion:

| Method | Library | Notes |
|--------|---------|-------|
| Demons (diffeomorphic) | SimpleITK | Classic iterative, often folds. |
| SyN (symmetric normalization) | ANTsPy | State-of-the-art diffeomorphic. |
| B-spline (FFD) | SimpleITK | Parametric, can fold at large deformations. |
| VoxelMorph | voxelmorph | CNN-based learned registration. |

### [elastix-registration.ipynb](elastix-registration.ipynb)
Registers a 2D image pair with [ITKElastix](https://github.com/InsightSoftwareConsortium/ITKElastix)
using both a standard B-spline and an aggressive variant (reduced
regularisation, more folding), then corrects the resulting DVF.

### [voxelmorph-registration.ipynb](voxelmorph-registration.ipynb)
Trains a small [VoxelMorph](https://github.com/voxelmorph/voxelmorph)
network on synthetic 2D images (random circles/ellipses), extracts the
predicted displacement field from a held-out pair, and corrects folds
with `iterative_serial`.

### [transmorph-registration.ipynb](transmorph-registration.ipynb)
Trains a small [TransMorph](https://github.com/junyuchen245/TransMorph_Pytorch)
network (Swin-Transformer encoder + ConvNet decoder) on synthetic 2D
images. Self-attention captures longer-range spatial relationships but
typically produces more folding than VoxelMorph, giving the corrector a
heavier workload.

### [opencv-optflow-registration.ipynb](opencv-optflow-registration.ipynb)
Uses OpenCV dense optical flow (Farneback, TV-L1) to compute
displacement fields between image pairs. Optical flow isn't designed
for medical registration, which makes it a good stress test — the
fields routinely contain many negative Jacobians.

### [correct-real-ants-warp.ipynb](correct-real-ants-warp.ipynb)
Loads the ANTs SyN displacement field from registering
`B0039_brain_25.nii.gz` (moving) to `average_template_25.nii.gz`
(fixed). Because SyN is diffeomorphic by construction, this notebook
serves two roles: (1) **verification** that the warp is fold-free, and
(2) a reference for the Jdet distribution of a well-behaved real-world
field.

## Output

All notebooks write to `output/registration/<notebook_name>/` relative
to the repo root. See [`../README.md`](../README.md) for the shared
output convention.
