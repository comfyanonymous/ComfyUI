# Adapted from facebookresearch/MHR (Apache 2.0):
#   https://github.com/facebookresearch/MHR/blob/main/mhr/mhr.py
# Skinning ops follow facebookincubator/momentum (Apache 2.0) — formulas
# verbatim from the upstream mhr_model.pt
# (pymomentum.{skel_state,quaternion,backend.skel_state_backend}).
# Original Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.sparse.check_sparse_tensor_invariants.disable() # silence the "implicitly disabled" UserWarning.

from .mhr_utils import batch6DFromXYZ

_LN2 = 0.6931471824645996

def _euler_xyz_to_quat(angles):
    """(roll, pitch, yaw) -> quaternion (x, y, z, w). Matches pymomentum.quaternion.euler_xyz_to_quaternion."""
    roll, pitch, yaw = angles.unbind(-1)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return torch.stack([x, y, z, w], dim=-1)


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def _quat_rotate(q, v):
    """Rotate v by unit quaternion q (xyzw). v + 2 * (axis x v * w + axis x (axis x v))."""
    axis = q[..., :3]
    r = q[..., 3:4]
    av = torch.cross(axis, v, dim=-1)
    aav = torch.cross(axis, av, dim=-1)
    return v + 2.0 * (av * r + aav)


def _skel_multiply(s1, s2):
    """Compose two skel states (..., 8). Returns parent ∘ child.

    Mirrors pymomentum.skel_state.multiply: both quaternions are renormalized
    before composition. With many FK levels the previously-normalized quats
    drift in ULPs; upstream renormalizes defensively, so we do too to stay
    bit-close to its outputs.
    """
    t1, sc1 = s1[..., :3], s1[..., 7:8]
    t2, sc2 = s2[..., :3], s2[..., 7:8]
    q1 = F.normalize(s1[..., 3:7], p=2, dim=-1, eps=1e-12)
    q2 = F.normalize(s2[..., 3:7], p=2, dim=-1, eps=1e-12)
    t_res = t1 + sc1 * _quat_rotate(q1, t2)
    q_res = _quat_multiply(q1, q2)
    s_res = sc1 * sc2
    return torch.cat([t_res, q_res, s_res], dim=-1)


def _skel_transform_points(skel_state, points):
    """Apply skel_state (..., 8) to points (..., 3): t + q * (s * points).

    Assumes the quaternion in skel_state is already unit-norm. Callers that
    can't guarantee that should normalize first.
    """
    t = skel_state[..., :3]
    q = skel_state[..., 3:7]
    s = skel_state[..., 7:8]
    return t + _quat_rotate(q, s * points)


def _global_skel_state_from_local(local, pmi_levels):
    """FK walk in fp64 (matches upstream's use_double_precision=True path).

    `pmi_levels` is a precomputed list of (source_idx, target_idx) tensor pairs,
    one per BFS level. Avoids per-call torch.split + tolist() sync.
    """
    orig_dtype = local.dtype
    g = local.to(torch.float64).clone()
    for source, target in pmi_levels:
        parent = g.index_select(-2, target)
        child  = g.index_select(-2, source)
        g.index_copy_(-2, source, _skel_multiply(parent, child))
    return g.to(orig_dtype)


class MHRRig(nn.Module):
    """Plain-PyTorch reimplementation of Meta's MHR rig.

    All math runs in fp32 (FK upcast to fp64 internally, matching upstream's
    use_double_precision=True backend) regardless of the host model's dtype.
    """

    NUM_VERTS = 18439
    NUM_JOINTS = 127
    NUM_LBS_TRIPLETS = 51337
    NUM_IDENTITY = 45
    NUM_EXPR = 72
    PARAM_TRANSFORM_IN = 249    # = model_parameters(204) + identity_coeffs(45)
    PARAM_TRANSFORM_OUT = 889   # = NUM_JOINTS * 7
    POSE_CORR_IN = 750          # = (NUM_JOINTS - 2) * 6
    POSE_CORR_HIDDEN = 3000
    POSE_CORR_SPARSE_NNZ = 53136

    def __init__(self, device=None):
        super().__init__()

        # All buffers are populated by load_state_dict from the `mhr.*` keys
        def _p(*shape, dtype=torch.float32):
            return nn.Parameter(torch.empty(*shape, dtype=dtype, device=device), requires_grad=False)
        def _b(name, *shape, dtype):
            self.register_buffer(name, torch.empty(*shape, dtype=dtype, device=device))

        self.base_shape = _p(self.NUM_VERTS, 3)
        self.identity_basis = _p(self.NUM_IDENTITY, self.NUM_VERTS, 3)
        self.expr_basis = _p(self.NUM_EXPR, self.NUM_VERTS, 3)
        self.param_transform = _p(self.PARAM_TRANSFORM_OUT, self.PARAM_TRANSFORM_IN)

        self.skel_joint_translation_offsets = _p(self.NUM_JOINTS, 3)
        self.skel_joint_prerotations = _p(self.NUM_JOINTS, 4)
        _b("skel_joint_parents", self.NUM_JOINTS, dtype=torch.int32)
        _b("skel_pmi", 2, 266, dtype=torch.int64)
        _b("skel_pmi_buffer_sizes", 4, dtype=torch.int64)

        self.lbs_inverse_bind_pose = _p(self.NUM_JOINTS, 8)
        self.lbs_skin_weights = _p(self.NUM_LBS_TRIPLETS)
        _b("lbs_skin_indices", self.NUM_LBS_TRIPLETS, dtype=torch.int32)
        _b("lbs_vert_indices", self.NUM_LBS_TRIPLETS, dtype=torch.int64)

        _b("pose_corr_sparse_indices", 2, self.POSE_CORR_SPARSE_NNZ, dtype=torch.int64)
        self.pose_corr_sparse_weight = _p(self.POSE_CORR_SPARSE_NNZ)

        self.register_buffer("pose_corr_sparse_shape", torch.tensor([self.POSE_CORR_HIDDEN, self.POSE_CORR_IN], dtype=torch.int64, device=device))
        self.pose_corr_weight = _p(self.NUM_VERTS * 3, self.POSE_CORR_HIDDEN)
        self.pose_corr_bias = None
        self._pose_corr_sparse_cache = None
        self._pmi_levels_cache = None

    def forward(self, identity_coeffs, model_parameters, expr_coeffs, apply_correctives: bool = True):
        dtype = self.base_shape.dtype
        identity_coeffs = identity_coeffs.to(dtype)
        model_parameters = model_parameters.to(dtype)
        expr_coeffs = expr_coeffs.to(dtype)
        B = identity_coeffs.shape[0]

        identity_rest = self.base_shape + torch.einsum("nvd,bn->bvd", self.identity_basis, identity_coeffs)

        cat_in = torch.cat([model_parameters, torch.zeros_like(identity_coeffs)], dim=1)
        joint_parameters = torch.einsum("dn,bn->bd", self.param_transform, cat_in)

        jp = joint_parameters.view(B, self.NUM_JOINTS, 7)
        local_t = jp[..., :3] + self.skel_joint_translation_offsets.unsqueeze(0)
        local_q = _euler_xyz_to_quat(jp[..., 3:6])
        local_q = _quat_multiply(self.skel_joint_prerotations.unsqueeze(0), local_q)
        local_s = torch.exp(jp[..., 6:7] * _LN2)
        local_state = torch.cat([local_t, local_q, local_s], dim=-1)

        skel_state = _global_skel_state_from_local(local_state, self._pmi_levels())

        face_expr = torch.einsum("nvd,bn->bvd", self.expr_basis, expr_coeffs)
        unposed = identity_rest + face_expr
        if apply_correctives:
            unposed = unposed + self._pose_correctives(joint_parameters)

        verts = self._skin(skel_state, unposed)
        return verts, skel_state

    def _pose_correctives(self, joint_parameters):
        B = joint_parameters.shape[0]
        jp = joint_parameters.view(B, self.NUM_JOINTS, 7)
        # Joints [2:] only — root and one more skipped. Take Euler XYZ (cols 3:6).
        feat = batch6DFromXYZ(jp[:, 2:, 3:6], return_9D=False)  # (B, 125, 6)
        feat[..., 0] -= 1.0
        feat[..., 4] -= 1.0
        feat = feat.flatten(1, 2)  # (B, 750)

        h = (self._sparse_w() @ feat.T).T                # (B, 3000)
        h = F.relu(h)
        out = F.linear(h, self.pose_corr_weight, self.pose_corr_bias)  # (B, 55317)
        return out.view(B, self.NUM_VERTS, 3)

    def _pmi_levels(self):
        cached = self._pmi_levels_cache
        pmi = self.skel_pmi
        if cached is not None and cached[0][0].device == pmi.device:
            return cached
        sizes = self.skel_pmi_buffer_sizes.tolist()
        parts = torch.split(pmi, sizes, dim=1)
        levels = [(p[0], p[1]) for p in parts]
        self._pmi_levels_cache = levels
        return levels

    def _sparse_w(self):
        cached = self._pose_corr_sparse_cache
        w = self.pose_corr_sparse_weight
        if cached is not None and cached.device == w.device and cached.dtype == w.dtype:
            return cached
        sparse = torch.sparse_coo_tensor(
            self.pose_corr_sparse_indices,
            w,
            tuple(self.pose_corr_sparse_shape.tolist()),
            check_invariants=False,
        ).coalesce()
        self._pose_corr_sparse_cache = sparse
        return sparse

    def _skin(self, skel_state, rest_verts):
        B = skel_state.shape[0]
        ibp = self.lbs_inverse_bind_pose.unsqueeze(0).expand(B, self.NUM_JOINTS, 8)
        joint_xform = _skel_multiply(skel_state, ibp)

        norm_q = F.normalize(joint_xform[..., 3:7], p=2, dim=-1, eps=1e-12)
        joint_xform = torch.cat([joint_xform[..., :3], norm_q, joint_xform[..., 7:8]], dim=-1)

        sk_idx = self.lbs_skin_indices.long()
        v_idx = self.lbs_vert_indices
        w = self.lbs_skin_weights

        per_triplet_xform = joint_xform.index_select(-2, sk_idx)   # (B, 51337, 8)
        per_triplet_rest = rest_verts.index_select(-2, v_idx)      # (B, 51337, 3)
        contrib = _skel_transform_points(per_triplet_xform, per_triplet_rest) * w.unsqueeze(0).unsqueeze(-1)

        out = torch.zeros(B, self.NUM_VERTS, 3, dtype=rest_verts.dtype, device=rest_verts.device)
        out.index_add_(-2, v_idx, contrib)
        return out
