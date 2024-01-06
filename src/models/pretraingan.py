from __future__ import annotations

__all__ = ["pretrainGan"]

import random
from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Tuple, Dict

import os

import einops
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from cleanfid import fid
from einops import rearrange, repeat
from matplotlib import cm
from torch import nn, Tensor
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torchvision.utils import make_grid, save_image
from tqdm import trange

import random

from torchvision import transforms

from models import networks
from models.base import BaseModule
from utils import FromConfig, run_at_step, get_D_stats, G_path_loss, D_R1_loss, freeze, is_rank_zero, accumulate, \
    mixing_noise, pyramid_resize, splat_features_from_scores, rotation_matrix, print_once
import utils

from lpips import LPIPS

import math
PI = math.pi

# SPLAT_KEYS = ['spatial_style', 'xs', 'ys', 'covs', 'sizes']
SPLAT_KEYS = ['spatial_style', 'scores_pyramid']
_ = Image
_ = make_grid


@dataclass
class randomAugmentation:
    # for 360
    randomHorizontalFlip: bool = True
    randomHorizontalTranslation: bool = True
    imgWidth: int = 256
    rect_scaler: int = 1
    batchSize: int = 8
    randomSeed: int = 10

    def __post_init__(self):
        super().__init__()
        self.imgWidth *= self.rect_scaler
        random.seed(self.randomSeed)
        def HorFlip(imgs, flipbool):
            if flipbool:
                return transforms.functional.hflip(imgs)
            else:
                return imgs
        self.HorFlip = HorFlip

        def HorTranAtPosition(imgs, location):
            return torch.cat((imgs[..., location:], imgs[..., :location]), -1)
        self.HorTran = HorTranAtPosition

    def getAugResults(self, imgs: list = None):
        if self.randomHorizontalFlip:
            flipbools = [bool(random.getrandbits(1)) for _ in range(self.batchSize)]
            imgs = [torch.stack([self.HorFlip(bs, flipbools[k]) for k, bs in enumerate(ii)])
                    if ii is not None else None for ii in imgs]
        if self.randomHorizontalTranslation:
            locations = [random.randint(0, self.imgWidth - 1) for _ in range(self.batchSize)]
            imgs = [torch.stack([self.HorTran(bs, locations[k]) for k, bs in enumerate(ii)])
                    if ii is not None else None for ii in imgs]

        return imgs


@dataclass
class Lossλs:
    D_f2e_real: float = 1
    D_f2e_fake: float = 1
    D_f2e_R1: float = 5
    G_f2e: float = 1

    Cycle_Empty: float = 10

    def __getitem__(self, key):
        return super().__getattribute__(key)


@dataclass(eq=False)
class pretrainGan(BaseModule):
    # Modules
    generator_f2e: FromConfig[nn.Module]
    discriminator_f2e: FromConfig[nn.Module]
    # Data Augmentation
    randomAugmentation: FromConfig[randomAugmentation]
    # Module parameters
    dim: int = 256
    noise_dim: int = 512
    resolution: int = 128
    rect_scaler: int = 1
    p_mixing_noise: float = 0.0
    n_ema_sample: int = 8
    freeze_G: bool = False
    # Optimization
    lr: float = 1e-3
    eps: float = 1e-5
    # Regularization
    D_reg_every: int = 16
    G_reg_every: int = 4
    path_len: float = 0
    # Loss parameters
    λ: FromConfig[Lossλs] = None
    # Logging
    log_images_every_n_steps: Optional[int] = 500
    log_timing_every_n_steps: Optional[int] = -1
    log_fid_every_n_steps: Optional[int] = -1
    log_grads_every_n_steps: Optional[int] = -1
    log_fid_every_epoch: bool = True
    fid_n_imgs: Optional[int] = 50000
    fid_stats_name: Optional[str] = None
    flush_cache_every_n_steps: Optional[int] = -1
    fid_num_workers: Optional[int] = 24
    valtest_log_all: bool = False
    accumulate: bool = False
    validate_gradients: bool = False
    ipdb_on_nan: bool = False
    # Input feature generation
    n_features_min: int = 10
    n_features_max: int = 10
    feature_splat_temp: int = 2
    spatial_style: bool = False
    ab_norm: float = 0.02
    feature_jitter_xy: float = 0.0
    feature_jitter_shift: float = 0.0
    feature_jitter_angle: float = 0.0
    bg_loss: str = "zero"
    bg_inject_way: str = "after_mlp"

    # for 360
    blob_pano_aware: bool = False
    use_edge_loss: bool = False
    use_bg_encoder: bool = False

    # fid/kid
    fid_kid_save_path: Optional[str] = None
    fid_kid_dataset_path: Optional[str] = None

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.discriminator_f2e = networks.get_network(**self.discriminator_f2e)
        self.generator_f2e = networks.get_network(**self.generator_f2e)
        if self.freeze_G:
            self.generator_f2e.eval()
            freeze(self.generator_f2e)

        self.λ = Lossλs(**self.λ)

        self.randomAugmentation = randomAugmentation(**self.randomAugmentation)

        self.sample_z = torch.randn(self.n_ema_sample, self.noise_dim)

        self.loss_fn_vgg = LPIPS(net='vgg', verbose=False)
        freeze(self.loss_fn_vgg)
        if self.bg_loss != "lpips":
            del self.loss_fn_vgg

        if self.log_fid_every_epoch:
            self.fid_kid_save_path = os.path.join("fid_kid_out", self.fid_kid_save_path)
            print(self.fid_kid_save_path)
            if not os.path.exists(self.fid_kid_save_path):
                os.makedirs(self.fid_kid_save_path)


    # Initialization and state management
    def on_train_start(self):
        super().on_train_start()
        # Validate parameters w.r.t. trainer (must be done here since trainer is not attached as property yet in init)
        assert self.log_images_every_n_steps % self.trainer.log_every_n_steps == 0, \
            '`model.log_images_every_n_steps` must be divisible by `trainer.log_every_n_steps` without remainder. ' \
            f'Got {self.log_images_every_n_steps} and {self.trainer.log_every_n_steps}.'
        assert self.log_timing_every_n_steps < 0 or self.log_timing_every_n_steps % self.trainer.log_every_n_steps == 0, \
            '`model.log_images_every_n_steps` must be divisible by `trainer.log_every_n_steps` without remainder'
        assert self.log_fid_every_n_steps < 0 or self.log_fid_every_n_steps % self.trainer.log_every_n_steps == 0, \
            '`model.log_fid_every_n_steps` must be divisible by `trainer.log_every_n_steps` without remainder'
        #assert not ((self.log_fid_every_n_steps > -1 or self.log_fid_every_epoch) and (not self.fid_stats_name)), \
        #    'Cannot compute FID without name of statistics file to use.'


    def configure_optimizers(self) -> Union[optim, List[optim]]:
        G_reg_ratio = self.G_reg_every / ((self.G_reg_every + 1) or -1)
        D_reg_ratio = self.D_reg_every / ((self.D_reg_every + 1) or -1)
        req_grad = lambda l: [p for p in l if p.requires_grad]
        decay_params = []

        '''
        for name, param in self.loss_fn_vgg.named_parameters():
            if not param.requires_grad:
                print("freezing weights of -- {}".format(name))
            else:
                print("training weights of -- {}".format(name))
        '''

        G_params = [{'params': req_grad(self.generator_f2e.parameters()), 'weight_decay': 0}, {
            'params': [],
            'weight_decay': 0  # Legacy, dont remove :(
            }]

        D_params = [{'params': req_grad(self.discriminator_f2e.parameters())}]
        G_optim = torch.optim.AdamW(G_params, lr=self.lr * G_reg_ratio,
                                    betas=(0 ** G_reg_ratio, 0.99 ** G_reg_ratio), eps=self.eps, weight_decay=0)
        D_optim = torch.optim.AdamW(D_params, lr=self.lr * D_reg_ratio,
                                    betas=(0 ** D_reg_ratio, 0.99 ** D_reg_ratio), eps=self.eps, weight_decay=0)
        print_once(f'Optimizing {sum([p.numel() for grp in G_params for p in grp["params"]]) / 1e6:.3f}M params for G '
                   f'and {sum([p.numel() for grp in D_params for p in grp["params"]]) / 1e6:.3f}M params for D ')
        if self.freeze_G:
            return D_optim
        else:
            return G_optim, D_optim

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: Optimizer = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
    ):
        self.batch_idx = batch_idx
        optimizer.step(closure=optimizer_closure)

    def validation_epoch_end(self, *args, **kwargs):
        if self.log_fid_every_epoch and False:
            self.log_fid("validate")


    def gen(self, z=None, bg=None, batch_input=None, batch_real=None, layout=None, ema=False, norm_img=False, ret_layout=False, ret_latents=False, noise=None,
            **kwargs):
        G_f2e = self.generator_f2e_ema if ema else self.generator_f2e

        gen_input_f2e = {
            'full_size_input': batch_real
        }
        fake_empty = G_f2e(**gen_input_f2e)
        return fake_empty


    @torch.no_grad()
    def log_fid(self, mode, **kwargs):
        if is_rank_zero():
            out_path = os.path.join(self.fid_kid_save_path, str(self.current_epoch))
            fid_score = fid.compute_fid(fdir1=out_path, fdir2=self.fid_kid_dataset_path,
                                        dataset_res=self.resolution, device='cpu',
                                        num_workers=self.fid_num_workers, verbose=False)
            kid_score = fid.compute_kid(fdir1=out_path, fdir2=self.fid_kid_dataset_path,
                                        dataset_res=self.resolution, device='cpu',
                                        num_workers=self.fid_num_workers)
            self.log_scalars({'fid': fid_score}, mode)
            self.log_scalars({'kid': kid_score}, mode)
        else:
            fid_score = 0.0
            kid_score = 0.0

        return fid_score, kid_score


    def shared_step(self, batch, batch_idx: int,
                    optimizer_idx: Optional[int] = None, mode: str = 'train') -> Optional[Union[Tensor, dict]]:
        """
        Args:
            batch: tuple of tensor of shape N x C x H x W of images and a dictionary of batch metadata/labels
            batch_idx: pytorch lightning training loop batch index
            optimizer_idx: pytorch lightning optimizer index (0 = G, 1 = D)
            mode:
                `train` returns the total loss and logs losses and images/profiling info.
                `validate`/`test` log total loss and return images
        Returns: see description for `mode` above
        """
        torch.cuda.empty_cache()
        if run_at_step(self.trainer.global_step, self.flush_cache_every_n_steps):
            torch.cuda.empty_cache()
        # Set up modules and data
        train = mode == 'train'
        train_G = train and optimizer_idx == 0 and not self.freeze_G
        train_D = train and (optimizer_idx == 1 or self.freeze_G)

        if mode == 'train':
            batch_input, batch_input_labels = batch[0]
            batch_real, batch_labels = batch[1]
            batch_edge = None
            if self.use_edge_loss == True:
                batch_edge, batch_edge_labels = batch[2]

            [batch_input, batch_real, batch_edge] = self.randomAugmentation.getAugResults(
                [batch_input, batch_real, batch_edge])

        elif mode == 'test' or mode == 'validate':
            batch_input, batch_input_labels = batch['bg']
            batch_real, batch_labels = batch['gt']
            batch_edge = None
            if self.use_edge_loss == True:
                batch_edge, batch_edge_labels = batch['edge']


        z = None
        bg = None

        info = dict()
        losses = dict()

        log_images = run_at_step(self.trainer.global_step, self.log_images_every_n_steps)
        fake_empty = self.gen(z, bg, batch_input, batch_real, ret_layout=True, ret_latents=True, viz=log_images)

        if mode == 'test' or mode == 'validate':
            out_path = os.path.join(self.fid_kid_save_path, str(self.current_epoch))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            for img, name in zip(fake_empty, batch_labels['filenames']):
                save_image(img.add(1).div_(2).clamp(min=0, max=1),
                           os.path.join(out_path, name.split('/')[-1]),
                           nrow=1, padding=0, normalize=False, range=None, scale_each=False)
            return None

        # Compute various losses
        if train:
            logits_f2e_fake = self.discriminator_f2e(fake_empty)
        if train_G:
            losses['Cycle_Empty'] = F.mse_loss(batch_input, fake_empty).mean()
            # Log
            losses['G_f2e'] = F.softplus(-logits_f2e_fake).mean()

        if train_D:
            # Discriminate real images
            logits_f2e_real = self.discriminator_f2e(batch_input)
            # Log
            losses['D_f2e_real'] = F.softplus(-logits_f2e_real).mean()
            losses['D_f2e_fake'] = F.softplus(logits_f2e_fake).mean()
            info.update(get_D_stats('f2e_fake', logits_f2e_fake, gt=False))
            info.update(get_D_stats('f2e_real', logits_f2e_real, gt=True))

        # Save images
        if train:
            imgs = {
                'real_imgs': batch_real,
                'input_imgs': batch_input,
                'fake_empty': fake_empty,
            }
        # Compute train regularization loss

        if train_D and run_at_step(batch_idx, self.D_reg_every):
            if self.λ.D_f2e_R1:
                with autocast(enabled=False):
                    batch_input.requires_grad = True
                    logits_f2e_real = self.discriminator_f2e(batch_input)
                    R1 = D_R1_loss(logits_f2e_real, batch_input)
                    info['D_f2e_R1_unscaled'] = R1
                    losses['D_f2e_R1'] = R1 * self.D_reg_every



        # Compute final loss and log
        total_loss = f'total_loss_{"G" if train_G else "D"}'

        losses[total_loss] = sum(map(lambda k: losses[k] * self.λ[k], losses))
        isnan = self.alert_nan_loss(losses[total_loss], batch_idx)
        if self.all_gather(isnan).any():
            if self.ipdb_on_nan and is_rank_zero():
                import ipdb
                ipdb.set_trace()
            return
        self.log_scalars(losses, mode)
        self.log_scalars(info, mode)
        # Further logging and terminate
        if mode == "train":
            if train_G:
                if self.accumulate:
                    accumulate(self.generator_f2e_ema, self.generator_f2e, 0.5 ** (32 / (10 * 1000)))
                    accumulate(self.layout_net_ema, self.layout_net, 0.5 ** (32 / (10 * 1000)))
                if log_images and is_rank_zero():
                    if self.accumulate and self.n_ema_sample:
                        with torch.no_grad():
                            z = self.sample_z.to(self.device)
                            layout, imgs['fake_full_ema'] = self.gen(z, ema=True, viz=True, ret_layout=True)
                            imgs['feature_imgs_ema'] = layout['feature_img']
                    imgs = {k: v.clone().detach().float().cpu() for k, v in imgs.items() if v is not None}
                    self._log_image_dict(imgs, mode, square_grid=False, ncol=len(batch_real))
                #if run_at_step(self.trainer.global_step, self.log_fid_every_n_steps) and train_G:
                #    self.log_fid(mode)
                self._log_profiler()

            return losses[total_loss]
        else:
            if self.valtest_log_all:
                imgs = self.gather_tensor_dict(imgs)
            return None
