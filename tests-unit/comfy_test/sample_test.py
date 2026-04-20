import unittest

import torch

import comfy.sample


class TestPrepareNoiseInnerTrellis(unittest.TestCase):
    def test_coord_counts_noise_matches_per_index_prefix_draws(self):
        latent = torch.zeros(2, 4, 5, 1)
        latent.trellis_coord_counts = torch.tensor([3, 5], dtype=torch.int64)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(123)
        noise = comfy.sample.prepare_noise_inner(latent, generator)

        expected = torch.zeros_like(noise, dtype=torch.float32)
        row0 = torch.Generator(device="cpu")
        row0.manual_seed(123)
        expected[0, :, :3, :] = torch.randn(1, 4, 3, 1, generator=row0)[0]
        row1 = torch.Generator(device="cpu")
        row1.manual_seed(124)
        expected[1] = torch.randn(1, 4, 5, 1, generator=row1)[0]

        self.assertTrue(torch.equal(noise.float(), expected))
        self.assertTrue(torch.equal(noise[0, :, 3:, :], torch.zeros_like(noise[0, :, 3:, :])))

    def test_coord_counts_noise_inds_share_prefixes_for_duplicates(self):
        latent = torch.zeros(2, 4, 5, 1)
        latent.trellis_coord_counts = torch.tensor([3, 5], dtype=torch.int64)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(456)
        noise = comfy.sample.prepare_noise_inner(latent, generator, noise_inds=[7, 7])

        replay = torch.Generator(device="cpu")
        replay.manual_seed(463)
        expected1 = torch.randn(1, 4, 5, 1, generator=replay)
        expected0 = expected1[:, :, :3, :]

        self.assertTrue(torch.equal(noise[0:1, :, :3, :], expected0))
        self.assertTrue(torch.equal(noise[1:2, :, :5, :], expected1))
        self.assertTrue(torch.equal(noise[0, :, 3:, :], torch.zeros_like(noise[0, :, 3:, :])))


if __name__ == "__main__":
    unittest.main()
