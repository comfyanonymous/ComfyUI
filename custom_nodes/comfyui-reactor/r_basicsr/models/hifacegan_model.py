import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from r_basicsr.archs import build_network
from r_basicsr.losses import build_loss
from r_basicsr.metrics import calculate_metric
from r_basicsr.utils import imwrite, tensor2img
from r_basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class HiFaceGANModel(SRModel):
    """HiFaceGAN model for generic-purpose face restoration.
    No prior modeling required, works for any degradations.
    Currently doesn't support EMA for inference.
    """

    def init_training_settings(self):

        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            raise (NotImplementedError('HiFaceGAN does not support EMA now. Pass'))

        self.net_g.train()

        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # define losses
        # HiFaceGAN does not use pixel loss by default
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('feature_matching_opt'):
            self.cri_feat = build_loss(train_opt['feature_matching_opt']).to(self.device)
        else:
            self.cri_feat = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def discriminate(self, input_lq, output, ground_truth):
        """
        This is a conditional (on the input) discriminator
        In Batch Normalization, the fake and real images are
        recommended to be in the same batch to avoid disparate
        statistics in fake and real images.
        So both fake and real images are fed to D all at once.
        """
        h, w = output.shape[-2:]
        if output.shape[-2:] != input_lq.shape[-2:]:
            lq = torch.nn.functional.interpolate(input_lq, (h, w))
            real = torch.nn.functional.interpolate(ground_truth, (h, w))
            fake_concat = torch.cat([lq, output], dim=1)
            real_concat = torch.cat([lq, real], dim=1)
        else:
            fake_concat = torch.cat([input_lq, output], dim=1)
            real_concat = torch.cat([input_lq, ground_truth], dim=1)

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.net_d(fake_and_real)
        pred_fake, pred_real = self._divide_pred(discriminator_out)
        return pred_fake, pred_real

    @staticmethod
    def _divide_pred(pred):
        """
        Take the prediction of fake and real images from the combined batch.
        The prediction contains the intermediate outputs of multiscale GAN,
        so it's usually a list
        """
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # Requires real prediction for feature matching loss
            pred_fake, pred_real = self.discriminate(self.lq, self.output, self.gt)
            l_g_gan = self.cri_gan(pred_fake, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            # feature matching loss
            if self.cri_feat:
                l_g_feat = self.cri_feat(pred_fake, pred_real)
                l_g_total += l_g_feat
                loss_dict['l_g_feat'] = l_g_feat

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # TODO: Benchmark test between HiFaceGAN and SRGAN implementation:
        # SRGAN use the same fake output for discriminator update
        # while HiFaceGAN regenerate a new output using updated net_g
        # This should not make too much difference though. Stick to SRGAN now.
        # -------------------------------------------------------------------
        # ---------- Below are original HiFaceGAN code snippet --------------
        # -------------------------------------------------------------------
        # with torch.no_grad():
        #    fake_image = self.net_g(self.lq)
        #    fake_image = fake_image.detach()
        #    fake_image.requires_grad_()
        #    pred_fake, pred_real = self.discriminate(self.lq, fake_image, self.gt)

        # real
        pred_fake, pred_real = self.discriminate(self.lq, self.output.detach(), self.gt)
        l_d_real = self.cri_gan(pred_real, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        # fake
        l_d_fake = self.cri_gan(pred_fake, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake

        l_d_total = (l_d_real + l_d_fake) / 2
        l_d_total.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            print('HiFaceGAN does not support EMA now. pass')

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        """
        Warning: HiFaceGAN requires train() mode even for validation
        For more info, see https://github.com/Lotayou/Face-Renovation/issues/31

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """

        if self.opt['network_g']['type'] in ('HiFaceGAN', 'SPADEGenerator'):
            self.net_g.train()

        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            print('In HiFaceGANModel: The new metrics package is under development.' +
                  'Using super method now (Only PSNR & SSIM are supported)')
            super().nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """
        TODO: Validation using updated metric system
        The metrics are now evaluated after all images have been tested
        This allows batch processing, and also allows evaluation of
        distributional metrics, such as:

        @ Frechet Inception Distance: FID
        @ Maximum Mean Discrepancy: MMD

        Warning:
            Need careful batch management for different inference settings.

        """
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = dict()  # {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            sr_tensors = []
            gt_tensors = []

        pbar = tqdm(total=len(dataloader), unit='image')
        for val_data in dataloader:
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()  # detached cpu tensor, non-squeeze
            sr_tensors.append(visuals['result'])
            if 'gt' in visuals:
                gt_tensors.append(visuals['gt'])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')

                imwrite(tensor2img(visuals['result']), save_img_path)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            sr_pack = torch.cat(sr_tensors, dim=0)
            gt_pack = torch.cat(gt_tensors, dim=0)
            # calculate metrics
            for name, opt_ in self.opt['val']['metrics'].items():
                # The new metric caller automatically returns mean value
                # FIXME: ERROR: calculate_metric only supports two arguments. Now the codes cannot be successfully run
                self.metric_results[name] = calculate_metric(dict(sr_pack=sr_pack, gt_pack=gt_pack), opt_)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            print('HiFaceGAN does not support EMA now. Fallback to normal mode.')

        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
