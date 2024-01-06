from __future__ import annotations

__all__ = ["v2_cycleBlobGAN"]

import random
from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Tuple, Dict

import os
import numpy as np
import einops
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D

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

import dnnlib
from torch_utils import misc

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
    D_e2f_real: float = 1
    D_e2f_fake: float = 1
    D_e2f_fake_half_translation: float = 1
    D_e2f_R1: float = 5

    D_e2f_false_real: float = 1

    G_e2f: float = 1
    G_e2f_path: float = 2

    Translation_Consistency: float = 1

    G_e2f_feature_mean: float = 10
    G_e2f_feature_variance: float = 10

    G_bg: float = 1

    D_f2e_real: float = 1
    D_f2e_fake: float = 1

    D_f2e_R1: float = 5
    G_f2e: float = 1

    Cycle_Empty: float = 10
    Cycle_Full: float = 10

    Object_Encourage: float = 10


    def __getitem__(self, key):
        return super().__getattribute__(key)


@dataclass(eq=False)
class v2_cycleBlobGAN(BaseModule):
    # Modules
    bgEncoder: FromConfig[nn.Module]
    generator_e2f: FromConfig[nn.Module]
    generator_f2e: FromConfig[nn.Module]
    layout_net: FromConfig[nn.Module]
    discriminator_e2f: FromConfig[nn.Module]
    discriminator_f2e: FromConfig[nn.Module]

    # Data Augmentation
    randomAugmentation: FromConfig[randomAugmentation]
    # pretrain ckpt
    usePretrainEmptier: bool = False
    freezeEmptier: bool = False
    pretrainEmptierCkpt: str = None
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
    flush_cache_every_n_steps: Optional[int] = 1000
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

    # for 360 aware
    blob_pano_aware: bool = False
    new_blob_pano_aware: bool = False
    old_blob_pano_aware: bool = False

    use_edge_loss: bool = False
    use_bg_encoder: bool = False
    use_gaussian_location: bool = False

    # 2d layoutnet
    use_twoD_layout_net: bool = True
    twoD_noise_dim: int = 3
    fix_blob_size: bool = False
    use_blob_specific_cycle_loss: bool = False
    #use_half_translation_loss: bool = False
    use_translation_gan_loss: bool = False
    use_translation_consistency_loss: bool = False
    translation_gan_loss_divide_pieces: float = 10
    use_pairwise_l2_loss: bool = False

    # fid/kid
    fid_kid_save_path: Optional[str] = None
    fid_kid_dataset_path: Optional[str] = None

    # ablation
    ablation_without_layoutnet: bool = False

    # for save time in living room
    hc_resize_liv: bool = False

    # ada aug
    apply_ada_aug: bool = False

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters()

        #self.bgEncoder = networks.get_network(**self.bgEncoder)
        self.discriminator_e2f = networks.get_network(**self.discriminator_e2f)
        #self.discriminator_f2e = networks.get_network(**self.discriminator_f2e)
        #self.generator_e2f_ema = networks.get_network(**self.generator_e2f)
        self.generator_e2f = networks.get_network(**self.generator_e2f)
        self.generator_f2e = networks.get_network(**self.generator_f2e)
        #self.layout_net_ema = networks.get_network(**self.layout_net)
        self.layout_net = networks.get_network(**self.layout_net)

        # ada aug
        if self.apply_ada_aug:
            augpipe_specs = {
                'blit': dict(xflip=1, rotate90=1, xint=1),
                'geom': dict(scale=1, rotate=1, aniso=1, xfrac=1),
                'color': dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
                'filter': dict(imgfilter=1),
                'noise': dict(noise=1),
                'cutout': dict(cutout=1),
                'bg': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
                'bgc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                            lumaflip=1, hue=1, saturation=1),
                'bgcf': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                             lumaflip=1, hue=1, saturation=1, imgfilter=1),
                'bgcfn': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1,
                              contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
                'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1,
                               contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
            }
            augment_kwargs = dnnlib.EasyDict(class_name='models.networks.AugmentPipe', **augpipe_specs['bgc'])
            augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False)  # subclass of torch.nn.Module
            augment_pipe.p.copy_(torch.as_tensor(0.))
            self.augment_pipe = augment_pipe
            self.ada_target = 0.6


        if self.freezeEmptier:
            self.generator_f2e.eval()
            freeze(self.generator_f2e)
            #del self.discriminator_f2e
        else:
            self.discriminator_f2e = networks.get_network(**self.discriminator_f2e)

        if self.freeze_G:
            self.generator_e2f.eval()
            freeze(self.generator_e2f)
        if self.accumulate:
            self.generator_e2f_ema = networks.get_network(**self.generator_e2f)
            self.layout_net_ema = networks.get_network(**self.layout_net)
            self.generator_e2f_ema.eval()
            freeze(self.generator_e2f_ema)
            accumulate(self.generator_e2f_ema, self.generator_e2f, 0)
            self.layout_net_ema.eval()
            freeze(self.layout_net_ema)
            accumulate(self.layout_net_ema, self.layout_net, 0)

        self.λ = Lossλs(**self.λ)

        self.randomAugmentation = randomAugmentation(**self.randomAugmentation)

        self.sample_z = torch.randn(self.n_ema_sample, self.noise_dim)

        if self.use_bg_encoder:
            self.bgEncoder = networks.get_network(**self.bgEncoder)

        if self.bg_loss == "lpips":
            self.loss_fn_vgg = LPIPS(net='vgg', verbose=False)
            freeze(self.loss_fn_vgg)
            #del self.loss_fn_vgg

        if self.log_fid_every_epoch:
            self.fid_kid_save_path = os.path.join("fid_kid_out", self.fid_kid_save_path)
            try:
                os.makedirs(self.fid_kid_save_path)
            except:
                pass

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
        if self.usePretrainEmptier:
            checkpoint = torch.load(self.pretrainEmptierCkpt, map_location=self.device)
            keys = [key for key in checkpoint['state_dict']]
            for key in keys:
                key_prefix = key.split('.')[0]
                if key_prefix == "generator_f2e":
                    new_key = key.split(key_prefix + '.')[1]
                    checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)
                else:
                    checkpoint['state_dict'].pop(key)
            self.generator_f2e.load_state_dict(checkpoint['state_dict'])
            del checkpoint


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

        G_params = [{'params': req_grad(self.generator_e2f.parameters()), 'weight_decay': 0}, {
            'params': [],
            'weight_decay': 0  # Legacy, dont remove :(
            },
                    {
                        'params': req_grad(
                            [p for p in self.layout_net.parameters() if not any([p is pp for pp in decay_params])]),
                        'weight_decay': 0
                    }]
        if not (self.usePretrainEmptier or self.freezeEmptier):
            G_params.append({'params': req_grad(self.generator_f2e.parameters()), 'weight_decay': 0})

        if self.use_bg_encoder:
            G_params.append({
                        'params': req_grad(self.bgEncoder.parameters()),
                        'weight_decay': 0
                    })
        if self.bg_loss == 'lpips':
            G_params.append({
                        'params': req_grad(self.loss_fn_vgg.parameters()),
                        'weight_decay': 0
                    })

        D_params = [{'params': req_grad(self.discriminator_e2f.parameters())}]
        if not (self.usePretrainEmptier or self.freezeEmptier):
            D_params.append({'params': req_grad(self.discriminator_f2e.parameters())})

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


    def gen(self, z=None, twoD_z=None, bg=None, batch_input=None, batch_real=None, layout=None, ema=False, norm_img=False, ret_layout=False, ret_latents=False, noise=None,
            **kwargs):
        assert not (z is None and twoD_z is None and layout is None)
        if layout is not None and 'covs_raw' not in kwargs:
            kwargs['covs_raw'] = False

        layout = self.generate_layout(z=z, twoD_z=twoD_z, batch_input=batch_input, bg=bg, layout=layout, ret_layout=ret_layout, ema=ema, **kwargs)

        G_e2f = self.generator_e2f_ema if ema else self.generator_e2f
        G_f2e = self.generator_f2e_ema if ema else self.generator_f2e


        gen_input_e2f = {
            'input': layout['feature_grid'],
            'styles': {k: layout[k] for k in SPLAT_KEYS} if self.spatial_style else z,
            'return_image_only': not ret_latents,
            'return_latents': ret_latents,
            'noise': noise
        }

        batch_input.requires_grad = True
        gen_input_e2f['styles'].update({'input_pyramid': pyramid_resize(batch_input, cutoff=G_e2f.size_in)})

        fake_full = G_e2f(**gen_input_e2f)


        full_size_input_for_gen_input_f2e = fake_full[0] if ret_latents else fake_full
        if self.hc_resize_liv:
            full_size_input_for_gen_input_f2e = F.interpolate(full_size_input_for_gen_input_f2e, size=(int(self.resolution/2), int(self.resolution)))
        gen_input_f2e = {
            'full_size_input': full_size_input_for_gen_input_f2e
        }
        fake_empty = G_f2e(**gen_input_f2e)
        if self.hc_resize_liv:
            fake_empty = F.interpolate(fake_empty, size=(int(self.resolution), int(self.resolution)))

        fake_empty_from_real = fake_empty
        if (not self.freezeEmptier) and (batch_real is not None):
            batch_real.requires_grad = True
            fake_empty_from_real = G_f2e(**{'full_size_input': batch_real})


        if norm_img:
            img_full = fake_full[0] if ret_latents else fake_full
            img_full.add_(1).div_(2).mul_(255)
            img_empty = fake_empty
            img_empty.add_(1).div_(2).mul_(255)
            img_empty_from_real = fake_empty_from_real
            img_empty_from_real.add_(1).div_(2).mul_(255)
        if ret_layout:
            if not ret_latents:
                fake_full = [fake_full]
                fake_empty = [fake_empty]
            return [layout, *fake_full, fake_empty, fake_empty_from_real]
        else:
            return fake_full, fake_empty, fake_empty_from_real



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

    # Training and evaluation
    @torch.no_grad()
    def visualize_features(self, xs, ys, viz_size, features=None, scores=None, feature_img=None,
                           c_border=-1, c_fill=1, sz=5, viz_entropy=False, viz_centers=False, viz_colors=None,
                           feature_center_mask=None, **kwargs) -> Dict[str, Tensor]:
        #print(torch.mean(scores))
        if feature_img is None:
            rand_colors = viz_colors is None
            viz_colors = (viz_colors if not rand_colors else torch.rand_like(features[..., :3])).to(xs.device)
            if viz_colors.ndim == 2:
                # viz colors should be [Kmax, 3]
                viz_colors = viz_colors[:features.size(1)][None].repeat_interleave(len(features), 0)
            elif viz_colors.ndim == 3:
                # viz colors should be [Nbatch, Kmax, 3]
                viz_colors = viz_colors[:, :features.size(1)]
            else:
                viz_colors = torch.rand_like(features[..., :3])
            img = splat_features_from_scores(scores, viz_colors, viz_size)
            if rand_colors:
                imax = img.amax((2, 3))[:, :, None, None]
                imin = img.amin((2, 3))[:, :, None, None]
                feature_img = img.sub(imin).div((imax - imin).clamp(min=1e-5)).mul(2).sub(1)
            else:
                feature_img = img
        imgs_flat = rearrange(feature_img, 'n c h w -> n c (h w)')
        if viz_centers:
            centers = torch.stack((xs, ys), -1).mul(viz_size).round()
            centers[..., 1].mul_(viz_size)
            centers = centers.sum(-1).long()
            if feature_center_mask is not None:
                fill_center = centers[torch.arange(len(centers)), feature_center_mask.int().argmax(1)]
                centers[~feature_center_mask] = fill_center.repeat_interleave((~feature_center_mask).sum(1), dim=0)
            offsets = (-sz // 2, sz // 2 + 1)
            offsets = (torch.arange(*offsets)[None] + torch.arange(*offsets).mul(viz_size)[:, None])
            border_mask = torch.zeros_like(offsets).to(bool)
            border_mask[[0, -1]] = border_mask[:, [0, -1]] = True
            offsets_border = offsets[border_mask].flatten()
            offsets_center = offsets[~border_mask].flatten()
            nonzero_features = scores[..., :-1].amax((1, 2)) > 0
            # draw center
            pixels = (centers[..., None] + offsets_center[None, None].to(self.device)) \
                .clamp(min=0, max=imgs_flat.size(-1) - 1)
            pixels = pixels.flatten(start_dim=1)
            pixels = repeat(pixels, 'n m -> n c m', c=3)
            empty_img = torch.ones_like(imgs_flat)
            imgs_flat.scatter_(dim=-1, index=pixels, value=c_fill)
            empty_img.scatter_(dim=-1, index=pixels, value=c_fill)
            # draw borders
            pixels = (centers[..., None] + offsets_border[None, None].to(self.device)) \
                .clamp(min=0, max=imgs_flat.size(-1) - 1)
            pixels = pixels.flatten(start_dim=1)
            pixels = repeat(pixels, 'n m -> n c m', c=3)
            imgs_flat.scatter_(dim=-1, index=pixels, value=c_border)
            empty_img.scatter_(dim=-1, index=pixels, value=c_border)
        out = {
            'feature_img': imgs_flat.reshape_as(feature_img)
        }
        if viz_centers:
            out['just_centers'] = empty_img.reshape_as(feature_img)
        if scores is not None and viz_entropy:
            img = (-scores.log2() * scores).sum(-1).nan_to_num(0)
            imax = img.amax((1, 2))[:, None, None]
            imin = img.amin((1, 2))[:, None, None]
            img = img.sub(imin).div((imax - imin).clamp(min=1e-5)).mul(256).int().cpu()
            h = w = img.size(-1)
            img = torch.from_numpy(cm.plasma(img.flatten())).mul(2).sub(1)[:, :-1]
            out['entropy_img'] = rearrange(img, '(n h w) c -> n c h w', h=h, w=w)
        return out

    def splat_features(self, xs: Tensor, ys: Tensor, features: Tensor, covs: Tensor, sizes: Tensor, size: int,
                       score_size: int, viz_size: Optional[int] = None, viz: bool = False,
                       ret_layout: bool = True,
                       covs_raw: bool = True, pyramid: bool = True, no_jitter: bool = False,
                       no_splat: bool = False, viz_score_fn=None,
                       **kwargs) -> Dict:
        """
        Args:
            xs: [N, M] X-coord location in [0,1]
            ys: [N, M] Y-coord location in [0,1]
            features: [N, M+1, dim] feature vectors to splat (and bg feature vector)
            covs: [N, M, 2, 2] xy covariance matrices for each feature
            sizes: [N, M+1] distributions of per feature (and bg) weights
            size: output grid size
            score_size: size at which to render score grid before downsampling to size
            viz_size: visualized grid in RGB dimension
            viz: whether to visualize
            covs_raw: whether covs already processed or not
            ret_layout: whether to return dict with layout info
            viz_score_fn: map from raw score to new raw score for generating blob maps. if you want to artificially enlarge blob borders, e.g., you can send in lambda s: s*1.5
            no_splat: return without computing scores, can be useful for visualizing
            no_jitter: manually disable jittering. useful for consistent results at test if model trained with jitter
            pyramid: generate score pyramid
            **kwargs: unused

        Returns: dict with requested information
        """
        if self.feature_jitter_xy and not no_jitter:
            xs = xs + torch.empty_like(xs).uniform_(-self.feature_jitter_xy, self.feature_jitter_xy)
            ys = ys + torch.empty_like(ys).uniform_(-self.feature_jitter_xy, self.feature_jitter_xy)

        if self.blob_pano_aware and self.use_gaussian_location:
            xs_variance, ys_variance = covs[..., :2].sigmoid().unbind(-1)

            xs_gaussian = D.normal.Normal(xs, xs_variance)
            ys_gaussian = D.normal.Normal(ys, ys_variance)
            xs_sample = xs_gaussian.rsample()
            ys_sample = ys_gaussian.rsample()

        elif self.blob_pano_aware:
            xs_sample = xs
            ys_sample = ys

        else:
            xs_sample = xs
            ys_sample = ys

            if covs_raw:
                a, b = covs[..., :2].sigmoid().unbind(-1)
                ab_norm = 1
                if self.ab_norm is not None:
                    ab_norm = self.ab_norm * (a * b).rsqrt()
                basis_i = covs[..., 2:]
                basis_i = F.normalize(basis_i, p=2, dim=-1)
                if self.feature_jitter_angle and not no_jitter:
                    with torch.no_grad():
                        theta = basis_i[..., 0].arccos()
                        theta = theta + torch.empty_like(theta).uniform_(-self.feature_jitter_angle,
                                                                         self.feature_jitter_angle)
                        basis_i_jitter = (rotation_matrix(theta)[..., 0] - basis_i).detach()
                    basis_i = basis_i + basis_i_jitter
                basis_j = torch.stack((-basis_i[..., 1], basis_i[..., 0]), -1)
                R = torch.stack((basis_i, basis_j), -1)
                covs = torch.zeros_like(R)
                covs[..., 0, 0] = a * ab_norm
                covs[..., -1, -1] = b * ab_norm
                covs = torch.einsum('...ij,...jk,...lk->...il', R, covs, R)
                covs = covs + torch.eye(2)[None, None].to(covs.device) * 1e-5

        if no_splat:
            return {'xs': xs_sample, 'ys': ys_sample, 'covs': covs, 'sizes': sizes, 'features': features}

        if self.new_blob_pano_aware:
            eps = 0.000001
            distance_norm = 8
            e_upper_bound = 0.7

            # print(['covs[..., :2]', covs[..., :2].shape])

            alpha = xs_sample  # [bs, num_blobs, 1]
            alpha = torch.unsqueeze(alpha.mul(2 * PI), -1)  # [bs, num_blobs, 1, 1]
            #print(['alpha', alpha.shape, torch.max(alpha), torch.min(alpha), torch.mean(alpha)])

            beta = ys_sample  # [bs, num_blobs, 1]
            beta = torch.unsqueeze(beta.mul(PI).add(-PI / 2), -1)  # [bs, num_blobs, 1, 1]
            #print(['beta', beta.shape, torch.max(beta), torch.min(beta), torch.mean(beta)])

            gamma = covs[..., 0] # [bs, num_blobs, 1]
            if self.feature_jitter_xy and not no_jitter:
                gamma = gamma + torch.empty_like(gamma).uniform_(-self.feature_jitter_xy, self.feature_jitter_xy)
            gamma = torch.unsqueeze(gamma.mul(2 * PI), -1)  # [bs, num_blobs, 1, 1]
            #print(['gamma', gamma.shape, torch.max(gamma), torch.min(gamma), torch.mean(gamma)])

            e = covs[..., 1]  # [bs, num_blobs, 1]
            if self.feature_jitter_angle and not no_jitter:
                e = e + torch.empty_like(e).uniform_(-self.feature_jitter_angle, self.feature_jitter_angle)
            e = torch.unsqueeze(torch.clamp(e.mul(e_upper_bound), min=0., max=e_upper_bound), -1)  # [bs, num_blobs, 1, 1]
            # print(['e', e.shape, torch.max(e), torch.min(e)])

            grid_coords = torch.stack(
                (torch.arange(start=-PI + eps, end=PI - eps, step=(2*PI-2*eps) / (score_size * self.rect_scaler)).repeat(score_size),
                 torch.arange(start=0 + eps, end=PI - eps, step=(PI-2*eps) / score_size).repeat_interleave(
                     score_size * self.rect_scaler))
            ).to(xs_sample.device)  # [2(x,y), size*size*rect_scaler]
            # print(['grid_coords', grid_coords.shape])

            feature_coords = torch.stack((torch.zeros_like(xs_sample), torch.zeros_like(xs_sample)),
                                         -1).to(xs_sample.device)  # [bs, num_blobs, 2]

            # print(['grid_coords', grid_coords.shape])
            delta = grid_coords[None, None] - feature_coords[..., None]  # [bs, num_blobs, 2(x,y), size*size]
            #print(['delta', delta.shape, torch.max(delta), torch.min(delta), torch.mean(delta)])

            theta_pano = delta[:, :, 0, :]
            phi_pano = delta[:, :, 1, :]

            # z y z rotation
            # z rotate

            theta_pano = theta_pano + alpha

            # y rotate
            cos_for_phi = torch.sin(beta).mul(-1) * torch.sin(phi_pano) * torch.cos(theta_pano) \
                          + torch.cos(beta) * torch.cos(phi_pano)
            phi_pano_temp = torch.acos(torch.clamp(cos_for_phi, min=-1. + eps, max=1. - eps))

            y_for_atan2 = torch.sin(phi_pano) * torch.sin(theta_pano)

            x_for_atan2 = torch.cos(beta) * torch.sin(phi_pano) * torch.cos(theta_pano) \
                          + torch.sin(beta) * torch.cos(phi_pano)

            theta_pano_temp = torch.atan2(y_for_atan2, x_for_atan2)

            phi_pano = phi_pano_temp
            theta_pano = theta_pano_temp

            # z rotate
            #gamma = gamma.mul(0).add(PI / 4)
            #theta_pano = theta_pano + gamma

            # project the sphere to polar coordinate system
            phi_pano = phi_pano.mul(-1).add(PI/2)
            cos_r = torch.clamp(torch.cos(theta_pano) * torch.cos(phi_pano), min=-1. + eps, max=1. - eps)
            # print(['cos_r', cos_r.shape, torch.max(cos_r), torch.min(cos_r)])
            # print(['cos_r', cos_r])

            polar_r = torch.acos(cos_r) # [bs, num_blobs, 1, size*size]
            #polar_r = polar_r * polar_r
            #polar_r = polar_r+1
            #polar_r = polar_r * polar_r * polar_r
            #PI_PLUS_ONE = PI+1
            #distance_norm = 100
            #polar_r = (polar_r - 1) / (PI_PLUS_ONE * PI_PLUS_ONE * PI_PLUS_ONE - 1)
            #sq_mahalanobis = polar_r
            #print(['polar_r_inside', polar_r.shape, torch.max(polar_r), torch.min(polar_r), torch.mean(polar_r)])

            #tan_omega = torch.div(torch.sin(phi_pano), torch.sin(theta_pano) * torch.cos(phi_pano))
            #omega = torch.atan(torch.clamp(tan_omega, min=-1., max=1.))
            omega = torch.atan2(torch.sin(phi_pano), torch.sin(theta_pano) * torch.cos(phi_pano))
            #gamma = gamma.mul(0).add(PI/4)
            omega = omega + gamma

            #omega = omega.add(PI)
            #print(['omega', omega.shape, torch.max(omega), torch.min(omega)])
            #if (e > 1).any():
            #    raise Exception("e > 1 !!!")
            #if (e < 0).any():
            #    raise Exception("e < 0 !!!")
            #e = e.mul(0).add(0.7)
            sq_mahalanobis = torch.sqrt(torch.div((e * e).add(-1.) * polar_r * polar_r,
                                       (e * e * torch.cos(omega) * torch.cos(omega)).add(-1.)
                                       ))
            #sq_mahalanobis = sq_mahalanobis * sq_mahalanobis
            #sq_mahalanobis = sq_mahalanobis * polar_r
            # print(['sq_mahalanobis', sq_mahalanobis.shape, torch.max(sq_mahalanobis), torch.min(sq_mahalanobis)])
            #sq_mahalanobis = sq_mahalanobis.mul(distance_norm)
            sq_mahalanobis = einops.rearrange(sq_mahalanobis, 'n m (s1 s2) -> n s1 s2 m', s1=score_size)

        elif self.old_blob_pano_aware:
            eps = 0.000001
            distance_norm = 8
            e_upper_bound = 0.7

            # print(['covs[..., :2]', covs[..., :2].shape])

            alpha = xs_sample  # [bs, num_blobs, 1]
            alpha = torch.unsqueeze(alpha.mul(2 * PI), -1)  # [bs, num_blobs, 1, 1]
            # print(['alpha', alpha.shape, torch.max(alpha), torch.min(alpha), torch.mean(alpha)])

            beta = ys_sample  # [bs, num_blobs, 1]
            beta = torch.unsqueeze(beta.mul(PI).add(-PI / 2), -1)  # [bs, num_blobs, 1, 1]
            # print(['beta', beta.shape, torch.max(beta), torch.min(beta), torch.mean(beta)])

            gamma = covs[..., 0]  # [bs, num_blobs, 1]
            gamma = torch.unsqueeze(gamma.mul(2 * PI), -1)  # [bs, num_blobs, 1, 1]
            # print(['gamma', gamma.shape, torch.max(gamma), torch.min(gamma), torch.mean(gamma)])

            e = torch.unsqueeze(torch.clamp(covs[..., 1].mul(e_upper_bound), min=0., max=e_upper_bound),
                                -1)  # [bs, num_blobs, 1, 1]
            # print(['e', e.shape, torch.max(e), torch.min(e)])

            grid_coords = torch.stack(
                (torch.arange(start=-PI + eps, end=PI - eps,
                              step=(2 * PI - 2 * eps) / (score_size * self.rect_scaler)).repeat(score_size),
                 torch.arange(start=0 + eps, end=PI - eps, step=(PI - 2 * eps) / score_size).repeat_interleave(
                     score_size * self.rect_scaler))
            ).to(xs_sample.device)  # [2(x,y), size*size*rect_scaler]
            # print(['grid_coords', grid_coords.shape])

            feature_coords = torch.stack((torch.zeros_like(xs_sample), torch.zeros_like(xs_sample)),
                                         -1).to(xs_sample.device)  # [bs, num_blobs, 2]

            # print(['grid_coords', grid_coords.shape])
            delta = grid_coords[None, None] - feature_coords[..., None]  # [bs, num_blobs, 2(x,y), size*size]
            # print(['delta', delta.shape])

            theta_pano = delta[:, :, 0, :]
            phi_pano = delta[:, :, 1, :]

            # z y z rotation
            # z rotate

            theta_pano = theta_pano + alpha

            # y rotate
            cos_for_phi = torch.sin(beta).mul(-1) * torch.sin(phi_pano) * torch.cos(theta_pano) \
                          + torch.cos(beta) * torch.cos(phi_pano)
            phi_pano_temp = torch.acos(torch.clamp(cos_for_phi, min=-1. + eps, max=1. - eps))

            y_for_atan2 = torch.sin(phi_pano) * torch.sin(theta_pano)

            x_for_atan2 = torch.cos(beta) * torch.sin(phi_pano) * torch.cos(theta_pano) \
                          + torch.sin(beta) * torch.cos(phi_pano)

            theta_pano_temp = torch.atan2(y_for_atan2, x_for_atan2)

            phi_pano = phi_pano_temp
            theta_pano = theta_pano_temp

            # z rotate
            # gamma = gamma.mul(0).add(PI / 4)
            # theta_pano = theta_pano + gamma

            # project the sphere to polar coordinate system
            phi_pano = phi_pano.mul(-1).add(PI / 2)
            cos_r = torch.clamp(torch.cos(theta_pano) * torch.cos(phi_pano), min=-1. + eps, max=1. - eps)
            # print(['cos_r', cos_r.shape, torch.max(cos_r), torch.min(cos_r)])
            # print(['cos_r', cos_r])

            polar_r = torch.acos(cos_r)  # [bs, num_blobs, 1, size*size]
            polar_r = polar_r * polar_r
            # sq_mahalanobis = polar_r
            # print(['polar_r', polar_r.shape, torch.max(polar_r), torch.min(polar_r)])

            # tan_omega = torch.div(torch.sin(phi_pano), torch.sin(theta_pano) * torch.cos(phi_pano))
            # omega = torch.atan(torch.clamp(tan_omega, min=-1., max=1.))
            omega = torch.atan2(torch.sin(phi_pano), torch.sin(theta_pano) * torch.cos(phi_pano))
            # gamma = gamma.mul(0).add(PI/4)
            omega = omega + gamma

            # omega = omega.add(PI)
            # print(['omega', omega.shape, torch.max(omega), torch.min(omega)])
            # if (e > 1).any():
            #    raise Exception("e > 1 !!!")
            # if (e < 0).any():
            #    raise Exception("e < 0 !!!")
            # e = e.mul(0).add(0.7)
            sq_mahalanobis = torch.sqrt(torch.div((e * e).add(-1.) * polar_r * polar_r,
                                                  (e * e * torch.cos(omega) * torch.cos(omega)).add(-1.)
                                                  ))
            # sq_mahalanobis = sq_mahalanobis * sq_mahalanobis
            # sq_mahalanobis = sq_mahalanobis * polar_r
            # print(['sq_mahalanobis', sq_mahalanobis.shape, torch.max(sq_mahalanobis), torch.min(sq_mahalanobis)])
            sq_mahalanobis = sq_mahalanobis.mul(distance_norm)
            sq_mahalanobis = einops.rearrange(sq_mahalanobis, 'n m (s1 s2) -> n s1 s2 m', s1=score_size)


        elif self.blob_pano_aware:
            feature_coords = self.feature_uv2xyzTransform(score_size, self.rect_scaler, xs_sample.mul(score_size * self.rect_scaler), ys_sample.mul(score_size), coor_mode='feature')
            lanlat_grid_coords = torch.stack(
                (torch.arange(start=-PI, end=PI, step=2 * PI / (score_size * self.rect_scaler)).repeat(score_size),
                 torch.arange(start=PI / 2, end=-PI / 2, step=-PI / score_size).repeat_interleave(score_size * self.rect_scaler))).to(
                xs_sample.device)
            grid_coords = self.feature_uv2xyzTransform(score_size, self.rect_scaler, lanlat_grid_coords[0, :], lanlat_grid_coords[1, :], coor_mode='grid')  # [2, size*size]
            delta = (grid_coords[None, None] - feature_coords[..., None]).div(2.) # [n, m, 2, size*size]
            sq_mahalanobis = (delta * delta).sum(2).mul(50.)
            sq_mahalanobis = einops.rearrange(sq_mahalanobis, 'n m (s1 s2) -> n s1 s2 m', s1=score_size)



        else:
            feature_coords = torch.stack((xs_sample.mul(score_size * self.rect_scaler), ys_sample.mul(score_size)), -1).to(xs_sample.device)  # [n, m, 2]
            grid_coords = torch.stack(
                (torch.arange(score_size * self.rect_scaler).repeat(score_size),
                 torch.arange(score_size).repeat_interleave(score_size * self.rect_scaler))).to(
                xs_sample.device)  # [2, size*size*rect_scaler]

            delta = (grid_coords[None, None] - feature_coords[..., None]).div(score_size)  # [n, m, 2, size*size]
            #print("delta.shape")
            #print(delta.shape)
            #print([torch.min(delta), torch.max(delta)])
            #print([covs.device, feature_coords.device, grid_coords.device, torch.linalg.solve(covs, delta).device])
            #sq_mahalanobis = (delta * torch.linalg.solve(covs, delta).to(xs_sample.device)).sum(2)
            sq_mahalanobis = (delta * delta).sum(2).mul(50.)
            #print("sq_mahalanobis.shape")
            #print(sq_mahalanobis.shape)
            sq_mahalanobis = einops.rearrange(sq_mahalanobis, 'n m (s1 s2) -> n s1 s2 m', s1=score_size).to(xs_sample.device)

            #print("sq_mahalanobis.shape")
            #print(sq_mahalanobis.shape)
            #print([torch.min(sq_mahalanobis), torch.max(sq_mahalanobis)])

        # [n, h, w, m]
        #print(sq_mahalanobis.shape)
        #print(sizes.shape)
        if self.ablation_without_layoutnet:
            shift = torch.zeros_like(sizes[:, None, None, 1:]) - 1000.
        elif self.fix_blob_size:
            shift = sizes[:, None, None, 1:] * 0.
            shift[..., 0:2] += 2.
            shift[..., 2:] -= 1
            #shift[..., 8:] -= 1
        else:
            shift = sizes[:, None, None, 1:]
        #print(shift.shape)
        if self.feature_jitter_shift and not no_jitter:
            shift = shift + torch.empty_like(shift).uniform_(-self.feature_jitter_shift, self.feature_jitter_shift)

        if self.new_blob_pano_aware:
            thickness = 50  # higher the thicker
            max_size = 1/3  # 0~1, 1 means covering the whole image
            start_learn_size = 1/4  # 0 means size 0, 1 means max size
            changing_rate = 1.  # how fast to learn the sizes, closer to thickness means faster

            #shift = sizes[:, None, None, 1:].mul(0).add(PI * max_size * (1 - start_learn_size) * thickness)
            shift = sizes[:, None, None, 1:].mul(changing_rate)
            shift = torch.clamp(shift, max=PI * max_size * (1 - start_learn_size) * thickness)
            sq_mahalanobis = sq_mahalanobis.add(-PI * max_size * start_learn_size) * thickness
            scores = sq_mahalanobis.div(-1).add(shift).sigmoid()
        elif self.old_blob_pano_aware:
            #shift = sizes[:, None, None, 1:] * 0. + 5
            shift = torch.clamp(shift, max=5)
            scores = sq_mahalanobis.div(-1).add(shift).sigmoid()
        else:
            scores = sq_mahalanobis.div(-1).add(shift).sigmoid()

        bg_scores = torch.zeros_like(scores[..., :1])
        scores = torch.cat((bg_scores, scores), -1)  # [n, h, w, m+1]
        #print(['scores.shape', scores.shape])

        # alpha composite
        rev = list(range(scores.size(-1) - 1, -1, -1))  # flip, but without copy
        d_scores = (1 - scores[..., rev]).cumprod(-1)[..., rev].roll(-1, -1) * scores
        d_scores[..., -1] = scores[..., -1]

        ret = {}
        if pyramid and (not 'scores_pyramid' in ret):
            score_img = einops.rearrange(d_scores, 'n h w m -> n m h w')
            #print(['score_img.shape', score_img.shape])
            ret['scores_pyramid'] = pyramid_resize(score_img, cutoff=size)


        feature_grid = splat_features_from_scores(ret['scores_pyramid'][size], features, size, channels_last=False)
        #print(['feature_grid.shape', feature_grid.shape])
        ret.update({'feature_grid': feature_grid, 'feature_img': None, 'entropy_img': None})
        if ret_layout:
            layout = {'xs': xs_sample, 'ys': ys_sample, 'covs': covs, 'raw_scores': scores, 'sizes': sizes,
                        'composed_scores': d_scores, 'features': features}
            ret.update(layout)
        if viz or self.use_blob_specific_cycle_loss:
            if viz_score_fn is not None:
                viz_posterior = viz_score_fn(scores)
                scores_viz = (1 - viz_posterior[..., rev]).cumprod(-1)[..., rev].roll(-1, -1) * viz_posterior
                scores_viz[..., -1] = viz_posterior[..., -1]
            else:
                scores_viz = d_scores
            ret.update(self.visualize_features(xs_sample, ys_sample, viz_size, features, scores_viz, **kwargs))
        return ret

    def generate_layout(self, z: Optional[Tensor] = None,
                        twoD_z: Optional[Tensor] = None,
                        bg: Optional[Tensor] = None,
                        batch_input: Optional[Tensor] = None,
                        ret_layout: bool = False, ema: bool = False,
                        size: Optional[int] = None, viz: bool = False,
                        num_features: Optional[int] = None,
                        layout: Optional[Dict[str, Tensor]] = None,
                        mlp_idx: Optional[int] = None,
                        score_size: Optional[int] = None,
                        viz_size: Optional[int] = None,
                        truncate: Optional[float] = None,
                        **kwargs) -> Dict[str, Tensor]:
        """
        Args:
            z: [N x D] tensor of noise
            bg: 1d feature vector of input_batch
            batch_input: input condition image
            mlp_idx: idx at which to split layout net MLP used for truncating
            num_features: how many features if not drawn randomly
            ema: use EMA version or not
            size: H, W output for feature grid
            viz: return RGB viz of feature grid
            ret_layout: if true, return an RGB image demonstrating feature placement
            score_size: size at which to render score grid before downsampling to size
            viz_size: visualized grid in RGB dimension
            truncate: if not None, use this as factor for computing truncation. requires self.mean_latent to be set. 0 = no truncation. 1 = full truncation.
            layout: output in format returned by ret_layout, can be used to generate instead of fwd pass
        Returns: [N x C x H x W] tensor of input, optionally [N x 3 x H_out x W_out] visualization of feature spread
        """
        if num_features is None:
            num_features = random.randint(self.n_features_min, self.n_features_max)
        if layout is None:
            assert not (z is None and twoD_z is None)
            layout_net = self.layout_net_ema if ema else self.layout_net

            if self.use_bg_encoder and self.bg_inject_way == "before_mlp":
                #bg = torch.unsqueeze(bg, 1)

                bg = bg - torch.min(bg)
                bg = bg / torch.max(bg)

                z = torch.cat((z, bg), 1)

            if self.use_twoD_layout_net:
                batch_input.requires_grad = True
                twoD_z.requires_grad = True
                layout = layout_net(inp=batch_input, twoD_z=twoD_z, n_features=num_features)
            else:
                z.requires_grad = True
                layout = layout_net(z=z, n_features=num_features, mlp_idx=mlp_idx)
        try:
            G = self.generator_e2f
        except AttributeError:
            G = self.generator_e2f_ema

        if (not bg is None) and self.bg_inject_way == "after_mlp":
            bg = torch.unsqueeze(bg, 1)

            bg = bg - torch.min(bg)
            bg = bg/torch.max(bg)

            layout['features'][:, 0, :] = bg[:, 0, :768]
            layout['spatial_style'][:, 0, :] = bg[:, 0, 768:]

        ret = self.splat_features(**layout, size=size or G.size_in, viz_size=viz_size or G.size,
                                  viz=viz, ret_layout=ret_layout, score_size=score_size or (size or G.size),
                                  pyramid=True,
                                  **kwargs)

        if self.spatial_style:
            ret['spatial_style'] = layout['spatial_style']
        if 'noise' in layout:
            ret['noise'] = layout['noise']
        if 'h_stdev' in layout:
            ret['h_stdev'] = layout['h_stdev']
        return ret


    def get_mean_latent(self, n_trunc: int = 10000, ema=True):
        Z = torch.randn((n_trunc, 512)).to(self.device)
        layout_net = self.layout_net_ema if ema else self.layout_net
        latents = [layout_net.mlp[:-1](Z[_]) for _ in trange(n_trunc, desc='Computing mean latent')]
        mean_latent = self.mean_latent = torch.stack(latents, 0).mean(0)
        return mean_latent


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
        if run_at_step(self.trainer.global_step, self.flush_cache_every_n_steps):
            torch.cuda.empty_cache()
        # Set up modules and data
        train = mode == 'train'
        train_G = train and optimizer_idx == 0 and not self.freeze_G
        train_D = train and (optimizer_idx == 1 or self.freeze_G)
        train_Emptier = train and not self.freezeEmptier

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
        twoD_z = None
        if self.use_twoD_layout_net:
            twoD_z = torch.randn(len(batch_real), self.twoD_noise_dim, self.resolution, self.resolution * self.rect_scaler).type_as(batch_input)
        else:
            z = torch.randn(len(batch_real), self.noise_dim).type_as(batch_real)
        bg = None
        if self.use_bg_encoder:
            bg = self.bgEncoder(batch_input)['bg']
        info = dict()
        losses = dict()

        log_images = train and run_at_step(self.trainer.global_step, self.log_images_every_n_steps)

        if train:
            if self.use_translation_consistency_loss or self.use_translation_gan_loss:
                devide_point = random.randint(0, self.resolution * self.rect_scaler)
            if self.use_translation_consistency_loss:
                batch_input_translation = torch.cat((batch_input[..., devide_point:], batch_input[..., :devide_point]),
                                                    -1)
                _, fake_full_translation, _, _, _ = self.gen(z, twoD_z, None, batch_input_translation, None,
                                                             ret_layout=True,
                                                             ret_latents=True,
                                                             viz=(log_images or not train))
                fake_full_translation_back = torch.cat(
                    (fake_full_translation[..., devide_point:], fake_full_translation[..., :devide_point]),
                    -1)

        layout, fake_full, latents, fake_empty, fake_empty_from_real = self.gen(z, twoD_z, bg, batch_input, batch_real, ret_layout=True, ret_latents=True, viz=(log_images or not train))

        if mode == 'test' or mode == 'validate':
            if int(self.current_epoch) % 10 == 0:
                out_path = os.path.join(self.fid_kid_save_path, str(self.current_epoch))
                out_fea_path = os.path.join(self.fid_kid_save_path, str(self.current_epoch) + '_feature')
                try:
                    os.makedirs(out_path)
                except:
                    pass
                try:
                    os.makedirs(out_fea_path)
                except:
                    pass

                if 'floor' in batch_input_labels['filenames'][0]:
                    labels = batch_input_labels['filenames']
                else:
                    labels = batch_labels['filenames']
                for img, fea_img, name in zip(fake_full, layout['feature_img'], labels):
                    save_image(img.add(1).div_(2).clamp(min=0, max=1),
                               os.path.join(out_path, name.split('/')[-1]),
                               nrow=1, padding=0, normalize=False, value_range=None, scale_each=False)

                    save_image(fea_img.add(1).div_(2).clamp(min=0, max=1),
                               os.path.join(out_fea_path, name.split('/')[-1]),
                               nrow=1, padding=0, normalize=False, value_range=None, scale_each=False)

            return None


        # Compute various losses
        if train:
            if train_G:
                if self.use_translation_consistency_loss:
                    losses['Translation_Consistency'] = F.mse_loss(fake_full, fake_full_translation_back).mean()

            if self.apply_ada_aug:
                logits_e2f_fake = self.discriminator_e2f(self.augment_pipe(fake_full))
            else:
                logits_e2f_fake = self.discriminator_e2f(fake_full)

            if self.use_translation_gan_loss:
                logits_e2f_fake = logits_e2f_fake + self.discriminator_e2f(
                    torch.cat((fake_full[..., devide_point:], fake_full[..., :devide_point]), -1))
                logits_e2f_fake.mul_(0.5)
                #eatch_piece_weight = 1. / self.translation_gan_loss_divide_pieces
                #logits_e2f_fake.mul_(eatch_piece_weight)
                #for i in range(1, self.translation_gan_loss_divide_pieces):
                #    devide_point = int(self.resolution * self.rect_scaler / self.translation_gan_loss_divide_pieces * i)
                #    logits_e2f_fake = logits_e2f_fake + (self.discriminator_e2f(torch.cat((fake_full[..., devide_point:], fake_full[..., :devide_point]), -1)).mul_(eatch_piece_weight))


        if train_G:
            if latents is not None and not self.spatial_style:
                if latents.ndim == 3:
                    latents = latents[0]
                info['latent_norm'] = latents.norm(2, 1).mean()
                info['latent_stdev'] = latents.std(0).mean()
            if self.use_blob_specific_cycle_loss:
                loss_map = layout['feature_img']
                #print(loss_map.shape)
                loss_map = torch.sum(loss_map, dim=1)
                loss_map[loss_map > -2.8] = 1
                loss_map[loss_map <= -2.8] = 0
                #print([torch.max(loss_map), torch.min(loss_map)])
                #print(torch.sum(loss_map))
                #print(loss_map.shape)
                loss_map = torch.unsqueeze(loss_map, 1)
                loss_map = loss_map.expand(-1, 3, -1, -1)
                #print(loss_map.shape)

                if False:
                    out_path = os.path.join(self.fid_kid_save_path, str(self.current_epoch) + '_lossmap')
                    out_fea_path = os.path.join(self.fid_kid_save_path, str(self.current_epoch) + '_feature')
                    try:
                        os.makedirs(out_path)
                    except:
                        pass
                    try:
                        os.makedirs(out_fea_path)
                    except:
                        pass
                    for lm, fea, name in zip(loss_map, layout['feature_img'], batch_labels['filenames']):
                        save_image(lm.clamp(min=0, max=1),
                                   os.path.join(out_path, name.split('/')[-1]),
                                   nrow=1, padding=0, normalize=False, value_range=None, scale_each=False)
                        save_image(fea.add(1).div_(2).clamp(min=0, max=1),
                                   os.path.join(out_fea_path, name.split('/')[-1]),
                                   nrow=1, padding=0, normalize=False, value_range=None, scale_each=False)

                #print([torch.max(batch_input), torch.min(batch_input)])
                losses['Cycle_Empty'] = F.mse_loss(batch_input, fake_empty).mean()
                losses['Object_Encourage'] = (1 - torch.sigmoid(F.mse_loss(batch_input * loss_map, fake_full * loss_map).mean()))
                #print(losses['Cycle_Empty'])
            elif self.use_pairwise_l2_loss:
                losses['Cycle_Empty'] = F.mse_loss(batch_real, fake_full).mean()
            else:
                losses['Cycle_Empty'] = F.mse_loss(batch_input, fake_empty).mean()
            # Log
            losses['G_e2f'] = F.softplus(-logits_e2f_fake).mean()


            if self.use_bg_encoder:
                if self.bg_loss == "zero":
                    losses['G_bg'] = torch.zeros_like((1, 1)).mean()
                elif self.bg_loss == "mse":
                    #losses['G_bg'] = F.mse_loss(batch_input, fake_full).mean()
                    losses['G_bg'] = torch.zeros_like((1, 1)).mean()
                    if self.use_edge_loss:
                        losses['G_bg'] += F.mse_loss(batch_input*(batch_edge+1)/2, fake_full*(batch_edge+1)/2).mean()
                elif self.bg_loss == "lpips":
                    losses['G_bg'] = self.loss_fn_vgg(fake_full, batch_input).mean()


            if run_at_step(self.trainer.global_step, self.trainer.log_every_n_steps):
                with torch.no_grad():
                    coords = torch.stack((layout['xs'], layout['ys']), -1)
                    centroids = coords.mean(1, keepdim=True)
                    # only consider spread of elements being used
                    coord_mask = layout['sizes'][:, 1:] > -5
                    info.update({'coord_spread': (coords - centroids)[coord_mask].norm(2, -1).mean()})
                    shift = layout['sizes'][:, 1:]
                    info.update({
                        'shift_mean': shift.mean(),
                        'shift_std': shift.std(-1).mean()
                    })
        if train_D:

            # Discriminate real images
            if self.apply_ada_aug:
                logits_e2f_real = self.discriminator_e2f(self.augment_pipe(batch_real))
            else:
                logits_e2f_real = self.discriminator_e2f(batch_real)

            #logits_e2f_false_real = self.discriminator_e2f(batch_input)
            #losses['D_e2f_false_real'] = F.softplus(logits_e2f_false_real).mean()
            # Log
            losses['D_e2f_real'] = F.softplus(-logits_e2f_real).mean()
            losses['D_e2f_fake'] = F.softplus(logits_e2f_fake).mean()

            info.update(get_D_stats('e2f_fake', logits_e2f_fake, gt=False))
            info.update(get_D_stats('e2f_real', logits_e2f_real, gt=True))

        if train:
            imgs = {
                'real_imgs': batch_real,
                'input_imgs': batch_input,
                'fake_full': fake_full,
                'fake_empty': fake_empty,
                'feature_imgs': layout['feature_img'],
            }
            if self.use_edge_loss:
                imgs['edge_imgs'] = fake_full * (batch_edge + 1) / 2

        if train_Emptier and train:
            logits_f2e_fake = self.discriminator_f2e(fake_empty)
            logits_f2e_fake_from_real = self.discriminator_f2e(fake_empty_from_real)
            logits_f2e_real = self.discriminator_f2e(batch_input)
            # Log
            losses['G_f2e'] = F.softplus(-logits_f2e_fake).mean() \
                              + F.softplus(-logits_f2e_fake_from_real).mean() \
                              + F.mse_loss(batch_input, fake_empty_from_real).mean()

            losses['D_f2e_real'] = F.softplus(-logits_f2e_real).mean()
            losses['D_f2e_fake'] = F.softplus(logits_f2e_fake).mean()
            info.update(get_D_stats('f2e_fake', logits_f2e_fake, gt=False))
            info.update(get_D_stats('f2e_real', logits_f2e_real, gt=True))

            imgs['fake_empty_from_real'] = fake_empty_from_real

        # Save images

        # Compute train regularization loss
        if train_G and run_at_step(batch_idx, self.G_reg_every):
            pass
        elif train_D and run_at_step(batch_idx, self.D_reg_every):
            if self.λ.D_e2f_R1:
                with autocast(enabled=False):
                    batch_real.requires_grad = True
                    if self.apply_ada_aug and False:
                        logits_e2f_real = self.discriminator_e2f(self.augment_pipe(batch_real))
                    else:
                        logits_e2f_real = self.discriminator_e2f(batch_real)
                    R1 = D_R1_loss(logits_e2f_real, batch_real)
                    info['D_e2f_R1_unscaled'] = R1
                    losses['D_e2f_R1'] = R1 * self.D_reg_every

            if train_Emptier and self.λ.D_f2e_R1:
                with autocast(enabled=False):
                    batch_input.requires_grad = True
                    logits_f2e_real = self.discriminator_f2e(batch_input)
                    R1 = D_R1_loss(logits_f2e_real, batch_input)
                    info['D_f2e_R1_unscaled'] = R1
                    losses['D_f2e_R1'] = R1 * self.D_reg_every

        if train_D:
            if self.apply_ada_aug:
                adjust = np.sign(logits_e2f_real.clone().detach().mean().cpu() - self.ada_target) * (batch_real.shape[0] * 1) / (
                            30 * 1000)
                self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(misc.constant(0, device=batch_real.device)))
                #print(self.augment_pipe.p)
                info['augment_pipe.probability'] = self.augment_pipe.p
                if run_at_step(self.trainer.global_step, 200):
                    if is_rank_zero():
                        print(self.augment_pipe.p)

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
                    accumulate(self.generator_e2f_ema, self.generator_e2f, 0.5 ** (32 / (10 * 1000)))
                    accumulate(self.generator_f2e_ema, self.generator_f2e, 0.5 ** (32 / (10 * 1000)))
                    accumulate(self.layout_net_ema, self.layout_net, 0.5 ** (32 / (10 * 1000)))
                if log_images and is_rank_zero():
                    imgs = {k: v.clone().detach().float().cpu() for k, v in imgs.items() if v is not None}
                    self._log_image_dict(imgs, mode, square_grid=False, ncol=len(batch_real))
                #if run_at_step(self.trainer.global_step, self.log_fid_every_n_steps) and train_G:
                #    self.log_fid(mode)
                self._log_profiler()
            return losses[total_loss]
        else:
            #if self.valtest_log_all:
            #    imgs = self.gather_tensor_dict(imgs)
            return None

    def feature_uv2xyzTransform(self, score_size, rect_scaler, us, vs, coor_mode):
        if coor_mode == 'feature':
            xyz = torch.empty_like(torch.stack((us, us, us), -1))
            # print('feature.shape')
            # print(xyz.shape)
            w = score_size * rect_scaler
            h = score_size
            lans = 2 * PI / w * us - PI
            lats = -PI / h * vs + PI / 2
            sin_lans = torch.sin(lans)
            cos_lans = torch.cos(lans)
            sin_lats = torch.sin(lats)
            cos_lats = torch.cos(lats)

            xyz[:, :, 0] = cos_lats * sin_lans
            xyz[:, :, 1] = -sin_lats
            xyz[:, :, 2] = cos_lats * cos_lans
        elif coor_mode == 'grid':
            xyz = torch.empty_like(torch.stack((us, us, us)))
            # print('grid.xyz')
            # print(xyz.shape)
            lans = us
            lats = vs
            sin_lans = torch.sin(lans)
            cos_lans = torch.cos(lans)
            sin_lats = torch.sin(lats)
            cos_lats = torch.cos(lats)

            xyz[0, :] = cos_lats * sin_lans
            xyz[1, :] = -sin_lats
            xyz[2, :] = cos_lats * cos_lans
        return xyz