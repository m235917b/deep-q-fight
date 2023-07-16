"""
This file defines some utility math functions for the program.
"""

from enum import Enum

import numpy as np
from numba import njit, cuda


class Axis(Enum):
    """
    An enum class that defines the axis of the vector space the game is defined in (R^2 int his case).
    """
    X = 0,
    Y = 1


def rotate_axis(vector, angle, axis):
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    return rot_mat.dot(vector - axis) + axis


def rotate(vector, angle):
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    return rot_mat.dot(vector)


def flip(vector, axis):
    if axis == Axis.X:
        return np.array([[-vector[0][0]],
                         [vector[1][0]]])
    else:
        return np.array([[vector[0][0]],
                         [-vector[1][0]]])


def adjoint(A):
    """compute inverse without division by det; ...xv3xc3 input, or array of matrices assumed"""
    AI = np.empty_like(A)
    for i in range(2):
        AI[..., i, :] = np.cross(A[..., i - 2, :], A[..., i - 1, :])
    return AI


def inverse_transpose(A):
    """
    efficiently compute the inverse-transpose for stack of 3x3 matrices
    """
    I = adjoint(A)
    det = dot(I, A).mean(axis=-1)
    return I / det[..., None, None]


def inverse(A):
    """inverse of a stack of 3x3 matrices"""
    return np.swapaxes(inverse_transpose(A), -1, -2)


def dot(A, B):
    """dot arrays of vecs; contract over last indices"""
    return np.einsum('...i,...i->...', A, B)


def ray_inter1(position, pixel, line_a, line_b):
    v0 = pixel - position
    v1 = line_b - line_a

    eqs = np.column_stack([v0, -v1])
    base = line_a - position

    return np.linalg.solve(eqs, base)


@njit(fastmath=True)
def ray_inter(position, pixel, line_a, line_b):
    v0 = pixel - position
    v1 = line_b - line_a

    eqs = np.array([[v0[0][0], -v1[0][0]],
                    [v0[1][0], -v1[1][0]]])
    base = line_a - position

    return np.linalg.solve(eqs, base)
