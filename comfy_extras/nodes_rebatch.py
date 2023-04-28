import torch

class LatentRebatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latents": ("LATENT",),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 1000}),
                              }}
    RETURN_TYPES = ("LATENT",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )

    FUNCTION = "rebatch"

    CATEGORY = "latent"

    def get_batch(self, latent, i):
        samples = latent[i]['samples']
        shape = samples.shape
        mask = latent[i]['noise_mask'] if 'noise_mask' in latent[i] else torch.ones((shape[0], 1, shape[2]*8, shape[3]*8), device='cpu')
        if mask.shape[-1] != shape[-1] * 8 or mask.shape[-2] != shape[-2]:
            torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[-2]*8, shape[-1]*8), mode="bilinear")
        if mask.shape[0] < samples.shape[0]:
            mask = mask.repeat((shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
        return samples, mask

    def get_slices(self, tensors, num, batch_size):
        slices = []
        for i in range(num):
            slices.append(tensors[i*batch_size:(i+1)*batch_size])
        if num * batch_size < tensors.shape[0]:
            return slices, tensors[num * batch_size:]
        else:
            return slices, None

    def rebatch(self, latents, batch_size):
        batch_size = batch_size[0]

        output_list = []
        current_samples = None
        current_masks = None

        for i in range(len(latents)):
            # fetch new entry of list
            samples, masks = self.get_batch(latents, i)
            # set to current if current is None
            if current_samples is None:
                current_samples = samples
                current_masks = masks
            # add previous to list if dimensions do not match
            elif samples.shape[-1] != current_samples.shape[-1] or samples.shape[-2] != current_samples.shape[-2]:
                s = dict()
                sample_slices, _ = self.get_slices(current_samples, 1, batch_size)
                mask_slices, _ = self.get_slices(current_masks, 1, batch_size)
                output_list.append({'samples': sample_slices[0], 'noise_mask': mask_slices[0]})
                current_samples = samples
                current_masks = masks
            # cat if everything checks out
            else:
                current_samples = torch.cat((current_samples, samples))
                current_masks = torch.cat((current_masks, masks))

            # add to list if dimensions gone above target batch size
            if current_samples.shape[0] > batch_size:
                num = current_samples.shape[0] // batch_size
                sample_slices, latent_remainder = self.get_slices(current_samples, num, batch_size)
                mask_slices, mask_remainder = self.get_slices(current_masks, num, batch_size)
                
                for i in range(num):
                    output_list.append({'samples': sample_slices[i], 'noise_mask': mask_slices[i]})

                current_samples = latent_remainder
                current_masks = mask_remainder
        
        #add remainder
        if current_samples is not None:
            sample_slices, _ = self.get_slices(current_samples, 1, batch_size)
            mask_slices, _ = self.get_slices(current_masks, 1, batch_size)
            output_list.append({'samples': sample_slices[0], 'noise_mask': mask_slices[0]})

        return (output_list,)

NODE_CLASS_MAPPINGS = {
    "RebatchLatents": LatentRebatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebatchLatents": "Rebatch Latents",
}