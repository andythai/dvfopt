"""
Low-level transform utilities.

Provides functions for applying affine transforms and deformation fields
to images and point clouds.
"""
import numpy as np
from scipy.ndimage import affine_transform, map_coordinates

from .io_utils import loadNiiImages


def applyAffineTransform(image , A, output_shape):
    """
    Applies Affine Transform defined  by A on the numpy image data represented by image

    Parameters:
    -----------
    image: Can be Numpy array or a path to nii Image
    A : Affine matrix of 3x4
    """
    data = loadNiiImages([image])
    transformedData = affine_transform(data,np.linalg.inv(A), output_shape = output_shape)
    return transformedData

def applyDeformationField(image, deformationField):
    """
    Morphs the numpy image data according to the deformation field.

    Uses scipy.ndimage.map_coordinates for fast C-level interpolation
    instead of a per-pixel Python callback (100-1000x faster).

    Parameters
    ----------
    image : np.ndarray or str
        Image data or path to NIfTI image.
    deformationField : array-like
        Deformation field with shape (3, X, Y, Z) where each component
        stores the displacement along that axis.
    """
    data = loadNiiImages([image])
    dx, dy, dz = deformationField[0], deformationField[1], deformationField[2]
    
    # Build coordinate grid + displacements
    coords = np.mgrid[:data.shape[0], :data.shape[1], :data.shape[2]].astype(np.float64)
    coords[0] += dx
    coords[1] += dy
    coords[2] += dz
    
    # Clamp coordinates to valid range (replicates original boundary behavior)
    for i in range(3):
        np.clip(coords[i], 0, data.shape[i] - 1, out=coords[i])
    
    transformedData = map_coordinates(data, coords, order=1, mode='constant', cval=0.0)
    return transformedData

def affineTransformPointCloud(points, A):
    points_ = points.copy()
    points_ = A@np.hstack([points, np.ones((points_.shape[0],1))]).T
    return points_[:3].T
