#Taken from: https://github.com/zju-pi/diff-sampler/blob/main/gits-main/solver_utils.py
#under Apache 2 license
import torch
import numpy as np

# A pytorch reimplementation of DEIS (https://github.com/qsh-zh/deis).
#############################
### Utils for DEIS solver ###
#############################
#----------------------------------------------------------------------------
# Transfer from the input time (sigma) used in EDM to that (t) used in DEIS.

def edm2t(edm_steps, epsilon_s=1e-3, sigma_min=0.002, sigma_max=80):
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    vp_beta_d = 2 * (np.log(torch.tensor(sigma_min).cpu() ** 2 + 1) / epsilon_s - np.log(torch.tensor(sigma_max).cpu() ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(torch.tensor(sigma_max).cpu() ** 2 + 1) - 0.5 * vp_beta_d
    t_steps = vp_sigma_inv(vp_beta_d.clone().detach().cpu(), vp_beta_min.clone().detach().cpu())(edm_steps.clone().detach().cpu())
    return t_steps, vp_beta_min, vp_beta_d + vp_beta_min

#----------------------------------------------------------------------------

def cal_poly(prev_t, j, taus):
    poly = 1
    for k in range(prev_t.shape[0]):
        if k == j:
            continue
        poly *= (taus - prev_t[k]) / (prev_t[j] - prev_t[k])
    return poly

#----------------------------------------------------------------------------
# Transfer from t to alpha_t.

def t2alpha_fn(beta_0, beta_1, t):
    return torch.exp(-0.5 * t ** 2 * (beta_1 - beta_0) - t * beta_0)

#----------------------------------------------------------------------------

def cal_intergrand(beta_0, beta_1, taus):
    with torch.inference_mode(mode=False):
        taus = taus.clone()
        beta_0 = beta_0.clone()
        beta_1 = beta_1.clone()
        with torch.enable_grad():
            taus.requires_grad_(True)
            alpha = t2alpha_fn(beta_0, beta_1, taus)
            log_alpha = alpha.log()
            log_alpha.sum().backward()
            d_log_alpha_dtau = taus.grad
    integrand = -0.5 * d_log_alpha_dtau / torch.sqrt(alpha * (1 - alpha))
    return integrand

#----------------------------------------------------------------------------

def get_deis_coeff_list(t_steps, max_order, N=10000, deis_mode='tab'):
    """
    Get the coefficient list for DEIS sampling.

    Args:
        t_steps: A pytorch tensor. The time steps for sampling.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
        N: A `int`. Use how many points to perform the numerical integration when deis_mode=='tab'.
        deis_mode: A `str`. Select between 'tab' and 'rhoab'. Type of DEIS.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """
    if deis_mode == 'tab':
        t_steps, beta_0, beta_1 = edm2t(t_steps)
        C = []
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            order = min(i+1, max_order)
            if order == 1:
                C.append([])
            else:
                taus = torch.linspace(t_cur, t_next, N)   # split the interval for integral appximation
                dtau = (t_next - t_cur) / N
                prev_t = t_steps[[i - k for k in range(order)]]
                coeff_temp = []
                integrand = cal_intergrand(beta_0, beta_1, taus)
                for j in range(order):
                    poly = cal_poly(prev_t, j, taus)
                    coeff_temp.append(torch.sum(integrand * poly) * dtau)
                C.append(coeff_temp)

    elif deis_mode == 'rhoab':
        # Analytical solution, second order
        def get_def_intergral_2(a, b, start, end, c):
            coeff = (end**3 - start**3) / 3 - (end**2 - start**2) * (a + b) / 2 + (end - start) * a * b
            return coeff / ((c - a) * (c - b))

        # Analytical solution, third order
        def get_def_intergral_3(a, b, c, start, end, d):
            coeff = (end**4 - start**4) / 4 - (end**3 - start**3) * (a + b + c) / 3 \
                    + (end**2 - start**2) * (a*b + a*c + b*c) / 2 - (end - start) * a * b * c
            return coeff / ((d - a) * (d - b) * (d - c))

        C = []
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            order = min(i, max_order)
            if order == 0:
                C.append([])
            else:
                prev_t = t_steps[[i - k for k in range(order+1)]]
                if order == 1:
                    coeff_cur = ((t_next - prev_t[1])**2 - (t_cur - prev_t[1])**2) / (2 * (t_cur - prev_t[1]))
                    coeff_prev1 = (t_next - t_cur)**2 / (2 * (prev_t[1] - t_cur))
                    coeff_temp = [coeff_cur, coeff_prev1]
                elif order == 2:
                    coeff_cur = get_def_intergral_2(prev_t[1], prev_t[2], t_cur, t_next, t_cur)
                    coeff_prev1 = get_def_intergral_2(t_cur, prev_t[2], t_cur, t_next, prev_t[1])
                    coeff_prev2 = get_def_intergral_2(t_cur, prev_t[1], t_cur, t_next, prev_t[2])
                    coeff_temp = [coeff_cur, coeff_prev1, coeff_prev2]
                elif order == 3:
                    coeff_cur = get_def_intergral_3(prev_t[1], prev_t[2], prev_t[3], t_cur, t_next, t_cur)
                    coeff_prev1 = get_def_intergral_3(t_cur, prev_t[2], prev_t[3], t_cur, t_next, prev_t[1])
                    coeff_prev2 = get_def_intergral_3(t_cur, prev_t[1], prev_t[3], t_cur, t_next, prev_t[2])
                    coeff_prev3 = get_def_intergral_3(t_cur, prev_t[1], prev_t[2], t_cur, t_next, prev_t[3])
                    coeff_temp = [coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3]
                C.append(coeff_temp)
    return C

