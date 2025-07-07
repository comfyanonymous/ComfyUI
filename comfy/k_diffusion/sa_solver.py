# Modify from: https://github.com/scxue/SA-Solver
# MIT license
from typing import Union, Callable
import torch


def get_coefficients_exponential_positive(order, interval_start, interval_end, tau):
    """
    Calculate the integral of exp(x(1+tau^2)) * x^order dx from interval_start to interval_end
    For calculating the coefficient of gradient terms after the lagrange interpolation,
    see Eq.(15) and Eq.(18) in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
    For data_prediction formula.
    """
    assert order in [0, 1, 2, 3], "order is only supported for 0, 1, 2 and 3"

    # after change of variable(cov)
    interval_end_cov = (1 + tau ** 2) * interval_end
    interval_start_cov = (1 + tau ** 2) * interval_start

    if order == 0:
        return (torch.exp(interval_end_cov)
                * (1 - torch.exp(-(interval_end_cov - interval_start_cov)))
                / ((1 + tau ** 2))
                )
    elif order == 1:
        return (torch.exp(interval_end_cov)
                * ((interval_end_cov - 1) - (interval_start_cov - 1) * torch.exp(-(interval_end_cov - interval_start_cov)))
                / ((1 + tau ** 2) ** 2)
                )
    elif order == 2:
        return (torch.exp(interval_end_cov)
                * ((interval_end_cov ** 2 - 2 * interval_end_cov + 2)
                    - (interval_start_cov ** 2 - 2 * interval_start_cov + 2)
                    * torch.exp(-(interval_end_cov - interval_start_cov))
                  )
                / ((1 + tau ** 2) ** 3)
                )
    elif order == 3:
        return (torch.exp(interval_end_cov)
                * ((interval_end_cov ** 3 - 3 * interval_end_cov ** 2 + 6 * interval_end_cov - 6)
                   - (interval_start_cov ** 3 - 3 * interval_start_cov ** 2 + 6 * interval_start_cov - 6)
                   * torch.exp(-(interval_end_cov - interval_start_cov))
                  )
                / ((1 + tau ** 2) ** 4)
                )


def lagrange_polynomial_coefficient(order, lambda_list):
    """
    Calculate the coefficient of lagrange polynomial
    For lagrange interpolation
    """
    assert order in [0, 1, 2, 3]
    assert order == len(lambda_list) - 1
    if order == 0:
        return [[1.0]]
    elif order == 1:
        return [[1.0 / (lambda_list[0] - lambda_list[1]), -lambda_list[1] / (lambda_list[0] - lambda_list[1])],
                [1.0 / (lambda_list[1] - lambda_list[0]), -lambda_list[0] / (lambda_list[1] - lambda_list[0])]]
    elif order == 2:
        denominator1 = (lambda_list[0] - lambda_list[1]) * (lambda_list[0] - lambda_list[2])
        denominator2 = (lambda_list[1] - lambda_list[0]) * (lambda_list[1] - lambda_list[2])
        denominator3 = (lambda_list[2] - lambda_list[0]) * (lambda_list[2] - lambda_list[1])
        return [[1.0 / denominator1, (-lambda_list[1] - lambda_list[2]) / denominator1, lambda_list[1] * lambda_list[2] / denominator1],
                [1.0 / denominator2, (-lambda_list[0] - lambda_list[2]) / denominator2, lambda_list[0] * lambda_list[2] / denominator2],
                [1.0 / denominator3, (-lambda_list[0] - lambda_list[1]) / denominator3, lambda_list[0] * lambda_list[1] / denominator3]
                ]
    elif order == 3:
        denominator1 = (lambda_list[0] - lambda_list[1]) * (lambda_list[0] - lambda_list[2]) * (lambda_list[0] - lambda_list[3])
        denominator2 = (lambda_list[1] - lambda_list[0]) * (lambda_list[1] - lambda_list[2]) * (lambda_list[1] - lambda_list[3])
        denominator3 = (lambda_list[2] - lambda_list[0]) * (lambda_list[2] - lambda_list[1]) * (lambda_list[2] - lambda_list[3])
        denominator4 = (lambda_list[3] - lambda_list[0]) * (lambda_list[3] - lambda_list[1]) * (lambda_list[3] - lambda_list[2])
        return [[1.0 / denominator1,
                 (-lambda_list[1] - lambda_list[2] - lambda_list[3]) / denominator1,
                 (lambda_list[1] * lambda_list[2] + lambda_list[1] * lambda_list[3] + lambda_list[2] * lambda_list[3]) / denominator1,
                 (-lambda_list[1] * lambda_list[2] * lambda_list[3]) / denominator1],

                [1.0 / denominator2,
                 (-lambda_list[0] - lambda_list[2] - lambda_list[3]) / denominator2,
                 (lambda_list[0] * lambda_list[2] + lambda_list[0] * lambda_list[3] + lambda_list[2] * lambda_list[3]) / denominator2,
                 (-lambda_list[0] * lambda_list[2] * lambda_list[3]) / denominator2],

                [1.0 / denominator3,
                 (-lambda_list[0] - lambda_list[1] - lambda_list[3]) / denominator3,
                 (lambda_list[0] * lambda_list[1] + lambda_list[0] * lambda_list[3] + lambda_list[1] * lambda_list[3]) / denominator3,
                 (-lambda_list[0] * lambda_list[1] * lambda_list[3]) / denominator3],

                [1.0 / denominator4,
                 (-lambda_list[0] - lambda_list[1] - lambda_list[2]) / denominator4,
                 (lambda_list[0] * lambda_list[1] + lambda_list[0] * lambda_list[2] + lambda_list[1] * lambda_list[2]) / denominator4,
                 (-lambda_list[0] * lambda_list[1] * lambda_list[2]) / denominator4]
                ]


def get_coefficients_fn(order, interval_start, interval_end, lambda_list, tau):
    """
    Calculate the coefficient of gradients.
    """
    assert order in [1, 2, 3, 4]
    assert order == len(lambda_list), 'the length of lambda list must be equal to the order'
    lagrange_coefficient = lagrange_polynomial_coefficient(order - 1, lambda_list)
    coefficients = [sum(lagrange_coefficient[i][j] * get_coefficients_exponential_positive(order - 1 - j, interval_start, interval_end, tau)
                    for j in range(order))
                    for i in range(order)]
    assert len(coefficients) == order, 'the length of coefficients does not match the order'
    return coefficients


def adams_bashforth_update_few_steps(order, x, tau, model_prev_list, sigma_prev_list, noise, sigma):
    """
    SA-Predictor, with the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
    """

    assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"
    t_fn = lambda sigma: sigma.log().neg()
    sigma_prev = sigma_prev_list[-1]
    gradient_part = torch.zeros_like(x)
    lambda_list = [t_fn(sigma_prev_list[-(i + 1)]) for i in range(order)]
    lambda_t = t_fn(sigma)
    lambda_prev = lambda_list[0]
    h = lambda_t - lambda_prev
    gradient_coefficients = get_coefficients_fn(order, lambda_prev, lambda_t, lambda_list, tau)

    if order == 2:  ## if order = 2 we do a modification that does not influence the convergence order similar to unipc. Note: This is used only for few steps sampling.
        # The added term is O(h^3). Empirically we find it will slightly improve the image quality.
        # ODE case
        # gradient_coefficients[0] += 1.0 * torch.exp(lambda_t) * (h ** 2 / 2 - (h - 1 + torch.exp(-h))) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
        # gradient_coefficients[1] -= 1.0 * torch.exp(lambda_t) * (h ** 2 / 2 - (h - 1 + torch.exp(-h))) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
        gradient_coefficients[0] += (1.0 * torch.exp((1 + tau ** 2) * lambda_t)
                                     * (h ** 2 / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h))) / ((1 + tau ** 2) ** 2))
                                     / (lambda_prev - lambda_list[1])
                                    )
        gradient_coefficients[1] -= (1.0 * torch.exp((1 + tau ** 2) * lambda_t)
                                     * (h ** 2 / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h))) / ((1 + tau ** 2) ** 2))
                                     / (lambda_prev - lambda_list[1])
                                    )

    for i in range(order):
        gradient_part += gradient_coefficients[i] * model_prev_list[-(i + 1)]
    gradient_part *= (1 + tau ** 2) * sigma * torch.exp(- tau ** 2 * lambda_t)
    noise_part = 0 if tau == 0 else sigma * torch.sqrt(1. - torch.exp(-2 * tau ** 2 * h)) * noise
    return torch.exp(-tau ** 2 * h) * (sigma / sigma_prev) * x + gradient_part + noise_part


def adams_moulton_update_few_steps(order, x, tau, model_prev_list, sigma_prev_list, noise, sigma):
    """
    SA-Corrector, with the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
    """

    assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"
    t_fn = lambda sigma: sigma.log().neg()
    sigma_prev = sigma_prev_list[-1]
    gradient_part = torch.zeros_like(x)
    sigma_list = sigma_prev_list + [sigma]
    lambda_list = [t_fn(sigma_list[-(i + 1)]) for i in range(order)]
    lambda_t = lambda_list[0]
    lambda_prev = lambda_list[1] if order >= 2 else t_fn(sigma_prev)
    h = lambda_t - lambda_prev
    gradient_coefficients = get_coefficients_fn(order, lambda_prev, lambda_t, lambda_list, tau)

    if order == 2:  ## if order = 2 we do a modification that does not influence the convergence order similar to UniPC. Note: This is used only for few steps sampling.
        # The added term is O(h^3). Empirically we find it will slightly improve the image quality.
        # ODE case
        # gradient_coefficients[0] += 1.0 * torch.exp(lambda_t) * (h / 2 - (h - 1 + torch.exp(-h)) / h)
        # gradient_coefficients[1] -= 1.0 * torch.exp(lambda_t) * (h / 2 - (h - 1 + torch.exp(-h)) / h)
        gradient_coefficients[0] += (1.0 * torch.exp((1 + tau ** 2) * lambda_t)
                                     * (h / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h)))
                                     / ((1 + tau ** 2) ** 2 * h))
                                    )
        gradient_coefficients[1] -= (1.0 * torch.exp((1 + tau ** 2) * lambda_t)
                                     * (h / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h)))
                                     / ((1 + tau ** 2) ** 2 * h))
                                    )

    for i in range(order):
        gradient_part += gradient_coefficients[i] * model_prev_list[-(i + 1)]
    gradient_part *= (1 + tau ** 2) * sigma * torch.exp(- tau ** 2 * lambda_t)
    noise_part = 0 if tau == 0 else sigma * torch.sqrt(1. - torch.exp(-2 * tau ** 2 * h)) * noise
    return torch.exp(-tau ** 2 * h) * (sigma / sigma_prev) * x + gradient_part + noise_part


def get_tau_interval_func(start_sigma: float, end_sigma: float, eta: float = 1.0) -> Callable[[torch.Tensor], float]:
    """Get the function that controls the stochasticity of SA-Solver.

    When eta = 0, SA-Solver runs as ODE.
    The official implementation uses t for determining the SDE interval, while this uses sigma instead.
    See https://github.com/scxue/SA-Solver/blob/main/README.md
    """

    def tau_func(sigma: Union[torch.Tensor, float]) -> float:
        if eta <= 0:
            return 0.0  # ODE

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.item()
        return eta if start_sigma >= sigma >= end_sigma else 0.0

    return tau_func
