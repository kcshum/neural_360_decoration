# Conditional 360-degree Image Synthesis for Immersive Indoor Scene Decoration
Code repository for ICCV2023 paper "Conditional 360-degree Image Synthesis for Immersive Indoor Scene Decoration".

# Setup
Install below dependencies:
```bash
git clone https://github.com/kcshum/neural_360_decoration.git
conda create -y -n n360dec python=3.9
conda activate n360dec
conda install -y pytorch torchvision torchaudio cudatoolkit
conda install -y cudatoolkit-dev
pip install tqdm==4.64.0 hydra-core==1.1.2 omegaconf==2.1.2 clean-fid==0.1.23 wandb==0.12.11 ipdb==0.13.9 lpips==0.1.4 einops==0.4.1 inputimeout==1.0.4 matplotlib==3.5.2 "mpl_interactions[jupyter]==0.21.0" protobuf~=3.19.0 moviepy==1.0.3
pip install pytorch-lightning==1.5.10
cd n360dec
```

Install ninja:
```bash
wget -q --show-progress https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip
sudo unzip -q ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```

# Download Dataset & Pretrained Emptier weights
Google drive link: https://drive.google.com/drive/folders/1uIZek4dzMScZbXoTPAoZtHmKKbkhjSw6?usp=sharing

Download the data folder to be `./data`, and extract the inside `.zip` files.

Download the ckpt files folder to be `./models`.


# Training
```bash
python src/run.py \
+experiment=[v2_2dlayoutnet_n360dec,local_v2_n360dec,jitter] \
"wandb.name=512x512 20-cycleBlobGAN -CIRCULAR PAD- -ADA DIS- -REVISED bedroom- -UPDATED 360 blob NO Gaussian- -grid 4- -cycle coeffi 5-" \
dataset=inandoutimagefolder +dataset.path=data/bedroom_full_only_remove_wrongs +dataset.bgpath=data/bedroom_empty_only_remove_wrongs \
model.pretrainEmptierCkpt=models/pix2pixHD_emptier/512x512_epoch_1114-step_407999.ckpt model.n_features=20 model.blob_pano_aware=true \
+model.use_gaussian_location=false model.generator_e2f.size_in=4 model.twoD_noise_dim=0 model.layout_net.n_downsampling=4 \
+model.layout_net.max_downsample_dim=16 model.lambda.Cycle_Empty=5 model.lambda.D_e2f_real=1 model.lambda.D_e2f_fake=1 model.lambda.G_e2f=1 \
model.resolution=512 model.rect_scaler=1 \
dataset.dataloader.num_workers=4 trainer.gpus=4 dataset.dataloader.batch_size=4 trainer.limit_val_batches=20 \
checkpoint.every_n_train_steps=0 +checkpoint.every_n_epochs=50 checkpoint.save_top_k=30 logger=false \
model.discriminator_e2f.name=StyleGAN2_ADA_Discriminator model.use_horizontal_circular_padding=true model.new_blob_pano_aware=true
```
**Important Notes before Training:**
1. Make sure you have enough disk space to save test images, for 256x256 training, it takes ~80 MBs for each epoch.
2. By default, the pipeline will save `320` test images to `./fid_kid_out/{wandb.name}/{epoch_num}` after each train epoch.
3. Make sure you are training with even number of `trainer.gpus` and `dataset.dataloader.batch_size`.

# Qualitative and Quantitative Evaluation on the each-epoch Results
1. Observe the testing images in `./fid_kid_out/{wandb.name}`.
2. Run following command to compute FID & KID score:
```bash
python fid_kid_out/cal_fid_kid_results.py \
--epoch_main_path={fid_kid_out/{wandb.name}} \
--dataset_src_path=data/bedroom_full_only/train \
--resolution=256
```

# Acknowledgement
We thank [BlobGAN](https://github.com/dave-epstein/blobgan) for their nice implementation.