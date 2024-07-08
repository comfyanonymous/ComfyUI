from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    *,
    steps=1000,
    learn_sigma=True,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_type='epsilon',
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    mse_loss_weight_type='constant',
    beta_start=0.0001,
    beta_end=0.02,
    noise_offset=0.0,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps, beta_start, beta_end)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [steps]
    mean_type = gd.predict_type_dict[predict_type]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=mean_type,
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        mse_loss_weight_type=mse_loss_weight_type,
        noise_offset=noise_offset,
    )
