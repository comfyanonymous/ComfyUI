import torch

class EulerScheduler(torch.nn.Module):
    def __init__(self, num_training_timesteps: int = 1_000, shift: float = 1, 
                 num_inference_timesteps: int = 50, inference: bool = True, device: str = "cuda"):
        super(EulerScheduler, self).__init__()

        # compute timestep values so we can index into them later
        timesteps = torch.linspace(1, num_training_timesteps, num_training_timesteps).to(torch.float32)

        # normalize between 0 and 1
        sigmas = timesteps / num_training_timesteps

        # staticaly shift (fixed image size assumed)
        self.sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)

        # get timesteps after shifting
        self.timesteps = self.sigmas * num_training_timesteps

        self.num_training_timesteps = num_training_timesteps
        self.num_inference_timesteps = num_inference_timesteps

        if inference:

            sigmas = torch.linspace(0, 1, num_inference_timesteps, dtype = torch.float32, device = device)
            timesteps = sigmas * self.num_training_timesteps

            self.timesteps = timesteps.to(device = device)
            self.sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])

        self._step_index = 0

    def sigma_to_timestep(self, sigma):
        return sigma * self.num_training_timesteps
    
    def index_for_timestep(self, timestep, schedule_timesteps = None):

        indices = (schedule_timesteps == timestep).nonzero()

        return indices[0].item()

        
    def add_noise(self, image: torch.FloatTensor,  timestep: float):

        noise = torch.randn_like(image)
        
        if image.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(image.device, dtype=torch.float32)
            timestep = timestep.to(image.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(image.device)
            timestep = timestep.to(image.device)
        
        # supports a list and a float
        if not isinstance(timestep, torch.Tensor) or timestep.ndim == 0:
            step_indices = [self.index_for_timestep(timestep, schedule_timesteps)]
        else:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]

        sigma = self.sigmas[step_indices].flatten().to(dtype = image.dtype, device = image.device)

        while len(sigma.shape) < len(image.shape):
            sigma = sigma.unsqueeze(-1)

        noised_image = (1.0 - sigma) * image + noise * sigma

        return noised_image
    
    @torch.no_grad()
    def step(self, model_output: torch.FloatTensor, sample: torch.FloatTensor,):
  
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1

        return prev_sample
    
def test_scheduler():

    scheduler = EulerScheduler()
    
    torch.manual_seed(2025)
    image = torch.rand(1, 224, 224, dtype = torch.float32)
    latent = torch.rand(1, 224, 224, dtype = torch.float32)

    output = scheduler.add_noise(image, timestep = torch.tensor([scheduler.timesteps[15]]))
    output = scheduler.reverse_flow(model_output = image, current_sample = latent)

    print(output)

if __name__ == "__main__":
    test_scheduler()