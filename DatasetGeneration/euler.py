import math
import numpy as np

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0),
    'sxyx': (0, 0, 1, 0),
    'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0),
    'syzx': (1, 0, 0, 0),
    'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0),
    'syxy': (1, 1, 1, 0),
    'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0),
    'szyx': (2, 1, 0, 0),
    'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1),
    'rxyx': (0, 0, 1, 1),
    'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1),
    'rxzy': (1, 0, 0, 1),
    'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1),
    'ryxy': (1, 1, 1, 1),
    'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1),
    'rxyz': (2, 1, 0, 1),
    'rzyz': (2, 1, 1, 1)
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def euler_angles_to_rotation_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    rotation_matrix = np.identity(4)
    if repetition:
        rotation_matrix[i, i] = cj
        rotation_matrix[i, j] = sj * si
        rotation_matrix[i, k] = sj * ci
        rotation_matrix[j, i] = sj * sk
        rotation_matrix[j, j] = -cj * ss + cc
        rotation_matrix[j, k] = -cj * cs - sc
        rotation_matrix[k, i] = -sj * ck
        rotation_matrix[k, j] = cj * sc + cs
        rotation_matrix[k, k] = cj * cc - ss
    else:
        rotation_matrix[i, i] = cj * ck
        rotation_matrix[i, j] = sj * sc - cs
        rotation_matrix[i, k] = sj * cc + ss
        rotation_matrix[j, i] = cj * sk
        rotation_matrix[j, j] = sj * ss + cc
        rotation_matrix[j, k] = sj * cs - sc
        rotation_matrix[k, i] = -sj
        rotation_matrix[k, j] = cj * si
        rotation_matrix[k, k] = cj * ci
    return rotation_matrix


def rotation_matrix_to_euler_angles(rotation_matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    rot_mat = np.array(rotation_matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(rot_mat[i, j] * rot_mat[i, j] +
                       rot_mat[i, k] * rot_mat[i, k])
        if sy > _EPS:
            ax = math.atan2(rot_mat[i, j], rot_mat[i, k])
            ay = math.atan2(sy, rot_mat[i, i])
            az = math.atan2(rot_mat[j, i], -rot_mat[k, i])
        else:
            ax = math.atan2(-rot_mat[j, k], rot_mat[j, j])
            ay = math.atan2(sy, rot_mat[i, i])
            az = 0.0
    else:
        cy = math.sqrt(rot_mat[i, i] * rot_mat[i, i] +
                       rot_mat[j, i] * rot_mat[j, i])
        if cy > _EPS:
            ax = math.atan2(rot_mat[k, j], rot_mat[k, k])
            ay = math.atan2(-rot_mat[k, i], cy)
            az = math.atan2(rot_mat[j, i], rot_mat[i, i])
        else:
            ax = math.atan2(-rot_mat[j, k], rot_mat[j, j])
            ay = math.atan2(-rot_mat[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def test():
    R = euler_angles_to_rotation_matrix(1, 2, 3, 'syxz')
    assert np.allclose(np.sum(R[0]), -1.34786452)
    R = euler_angles_to_rotation_matrix(1, 2, 3, (0, 1, 0, 1))
    assert np.allclose(np.sum(R[0]), -0.383436184)

    ax1, ay1, az1 = 4 * math.pi * (np.random.random(3) - 0.5)
    R1 = euler_angles_to_rotation_matrix(ax1, ay1, az1, 'syxz')
    ax2, ay2, az2 = rotation_matrix_to_euler_angles(R1, 'syxz')
    R2 = euler_angles_to_rotation_matrix(ax2, ay2, az2, 'syxz')
    assert np.allclose(R1, R2),\
        'Euler angles to/from rotation matrix conversion failed.'

    for _axes, _tuple in _AXES2TUPLE.items():
        ax, ay, az = 4 * math.pi * (np.random.random(3) - 0.5)
        R1 = euler_angles_to_rotation_matrix(ax, ay, az, _axes)
        R2 = euler_angles_to_rotation_matrix(ax, ay, az, _tuple)
        assert np.allclose(R1, R2), 'axes {} failed'.format(_axes)

    for axes in _AXES2TUPLE.keys():
        angles1 = 4 * math.pi * (np.random.random(3) - 0.5)
        R1 = euler_angles_to_rotation_matrix(axes=axes, *angles1)
        angles2 = rotation_matrix_to_euler_angles(R1, axes)
        R2 = euler_angles_to_rotation_matrix(axes=axes, *angles2)
        assert np.allclose(R1, R2), 'axes {} failed'.format(axes)


if __name__ == "__main__":
    test()
