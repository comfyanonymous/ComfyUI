# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copied from Nvidia Cosmos code.

import torch
from torch import Tensor
from typing import Callable, List, Tuple, Optional, Any
import math
from tqdm.auto import trange


def common_broadcast(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    ndims1 = x.ndim
    ndims2 = y.ndim

    if ndims1 < ndims2:
        x = x.reshape(x.shape + (1,) * (ndims2 - ndims1))
    elif ndims2 < ndims1:
        y = y.reshape(y.shape + (1,) * (ndims1 - ndims2))

    return x, y


def batch_mul(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x * y


def phi1(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the first order phi function: (exp(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi1 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float32)
    return (torch.expm1(t) / t).to(dtype=input_dtype)


def phi2(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the second order phi function: (phi1(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi2 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float32)
    return ((phi1(t) - 1.0) / t).to(dtype=input_dtype)


def res_x0_rk2_step(
    x_s: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    x0_s: torch.Tensor,
    s1: torch.Tensor,
    x0_s1: torch.Tensor,
) -> torch.Tensor:
    """
    Perform a residual-based 2nd order Runge-Kutta step.

    Args:
        x_s: Current state tensor.
        t: Target time tensor.
        s: Current time tensor.
        x0_s: Prediction at current time.
        s1: Intermediate time tensor.
        x0_s1: Prediction at intermediate time.

    Returns:
        Tensor: Updated state tensor.

    Raises:
        AssertionError: If step size is too small.
    """
    s = -torch.log(s)
    t = -torch.log(t)
    m = -torch.log(s1)

    dt = t - s
    assert not torch.any(torch.isclose(dt, torch.zeros_like(dt), atol=1e-6)), "Step size is too small"
    assert not torch.any(torch.isclose(m - s, torch.zeros_like(dt), atol=1e-6)), "Step size is too small"

    c2 = (m - s) / dt
    phi1_val, phi2_val = phi1(-dt), phi2(-dt)

    # Handle edge case where t = s = m
    b1 = torch.nan_to_num(phi1_val - 1.0 / c2 * phi2_val, nan=0.0)
    b2 = torch.nan_to_num(1.0 / c2 * phi2_val, nan=0.0)

    return batch_mul(torch.exp(-dt), x_s) + batch_mul(dt, batch_mul(b1, x0_s) + batch_mul(b2, x0_s1))


def reg_x0_euler_step(
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    x0_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a regularized Euler step based on x0 prediction.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_s: Prediction at current time.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current prediction.
    """
    coef_x0 = (s - t) / s
    coef_xs = t / s
    return batch_mul(coef_x0, x0_s) + batch_mul(coef_xs, x_s), x0_s


def order2_fn(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_s: torch.Tensor, x0_preds: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    impl the second order multistep method in https://arxiv.org/pdf/2308.02157
    Adams Bashforth approach!
    """
    if x0_preds:
        x0_s1, s1 = x0_preds[0]
        x_t = res_x0_rk2_step(x_s, t, s, x0_s, s1, x0_s1)
    else:
        x_t = reg_x0_euler_step(x_s, s, t, x0_s)[0]
    return x_t, [(x0_s, s)]


class SolverConfig:
    is_multi: bool = True
    rk: str = "2mid"
    multistep: str = "2ab"
    s_churn: float = 0.0
    s_t_max: float = float("inf")
    s_t_min: float = 0.0
    s_noise: float = 1.0


def fori_loop(lower: int, upper: int, body_fun: Callable[[int, Any], Any], init_val: Any, disable=None) -> Any:
    """
    Implements a for loop with a function.

    Args:
        lower: Lower bound of the loop (inclusive).
        upper: Upper bound of the loop (exclusive).
        body_fun: Function to be applied in each iteration.
        init_val: Initial value for the loop.

    Returns:
        The final result after all iterations.
    """
    val = init_val
    for i in trange(lower, upper, disable=disable):
        val = body_fun(i, val)
    return val


def differential_equation_solver(
    x0_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sigmas_L: torch.Tensor,
    solver_cfg: SolverConfig,
    noise_sampler,
    callback=None,
    disable=None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a differential equation solver function.

    Args:
        x0_fn: Function to compute x0 prediction.
        sigmas_L: Tensor of sigma values with shape [L,].
        solver_cfg: Configuration for the solver.

    Returns:
        A function that solves the differential equation.
    """
    num_step = len(sigmas_L) - 1

    # if solver_cfg.is_multi:
    #     update_step_fn = get_multi_step_fn(solver_cfg.multistep)
    # else:
    #     update_step_fn = get_runge_kutta_fn(solver_cfg.rk)
    update_step_fn = order2_fn

    eta = min(solver_cfg.s_churn / (num_step + 1), math.sqrt(1.2) - 1)

    def sample_fn(input_xT_B_StateShape: torch.Tensor) -> torch.Tensor:
        """
        Samples from the differential equation.

        Args:
            input_xT_B_StateShape: Input tensor with shape [B, StateShape].

        Returns:
            Output tensor with shape [B, StateShape].
        """
        ones_B = torch.ones(input_xT_B_StateShape.size(0), device=input_xT_B_StateShape.device, dtype=torch.float32)

        def step_fn(
            i_th: int, state: Tuple[torch.Tensor, Optional[List[torch.Tensor]]]
        ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
            input_x_B_StateShape, x0_preds = state
            sigma_cur_0, sigma_next_0 = sigmas_L[i_th], sigmas_L[i_th + 1]

            if sigma_next_0 == 0:
                output_x_B_StateShape = x0_pred_B_StateShape = x0_fn(input_x_B_StateShape, sigma_cur_0 * ones_B)
            else:
                # algorithm 2: line 4-6
                if solver_cfg.s_t_min < sigma_cur_0 < solver_cfg.s_t_max and eta > 0:
                    hat_sigma_cur_0 = sigma_cur_0 + eta * sigma_cur_0
                    input_x_B_StateShape = input_x_B_StateShape + (
                        hat_sigma_cur_0**2 - sigma_cur_0**2
                    ).sqrt() * solver_cfg.s_noise * noise_sampler(sigma_cur_0, sigma_next_0)  # torch.randn_like(input_x_B_StateShape)
                    sigma_cur_0 = hat_sigma_cur_0

                if solver_cfg.is_multi:
                    x0_pred_B_StateShape = x0_fn(input_x_B_StateShape, sigma_cur_0 * ones_B)
                    output_x_B_StateShape, x0_preds = update_step_fn(
                        input_x_B_StateShape, sigma_cur_0 * ones_B, sigma_next_0 * ones_B, x0_pred_B_StateShape, x0_preds
                    )
                else:
                    output_x_B_StateShape, x0_preds = update_step_fn(
                        input_x_B_StateShape, sigma_cur_0 * ones_B, sigma_next_0 * ones_B, x0_fn
                    )

            if callback is not None:
                callback({'x': input_x_B_StateShape, 'i': i_th, 'sigma': sigma_cur_0, 'sigma_hat': sigma_cur_0, 'denoised': x0_pred_B_StateShape})

            return output_x_B_StateShape, x0_preds

        x_at_eps, _ = fori_loop(0, num_step, step_fn, [input_xT_B_StateShape, None], disable=disable)
        return x_at_eps

    return sample_fn
