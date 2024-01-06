import os
from cleanfid import fid
from matplotlib import pyplot as plt


def cal_fid_kid(epoch_main_path, dataset_src_path, resolution, device):

    cur_epoch = 0

    epoch_list = []
    fid_score_list = []
    kid_score_list = []

    while(True):
        print("calculating epoch {}...".format(cur_epoch))
        cur_epoch_path = os.path.join(epoch_main_path, str(cur_epoch))
        drop_last_path = os.path.join(epoch_main_path, str(cur_epoch+1))
        if os.path.exists(cur_epoch_path) and os.path.exists(drop_last_path):

            # num_workers=fid_num_workers
            epoch_list.append(cur_epoch)
            fid_score_list.append(fid.compute_fid(fdir1=cur_epoch_path, fdir2=dataset_src_path,
                                        dataset_res=resolution, device=device, verbose=False))
            kid_score_list.append(fid.compute_kid(fdir1=cur_epoch_path, fdir2=dataset_src_path,
                                        dataset_res=resolution, device=device))
            cur_epoch += 1

        else:
            break

        figure, axis = plt.subplots(2, 1)
        axis[0].plot(epoch_list, fid_score_list)
        axis[0].set_title("FID Score")
        axis[1].plot(epoch_list, kid_score_list)
        axis[1].set_title("KID Score")
        plt.savefig(args.epoch_main_path + '_fid_kid.png')



if __name__ == "__main__":


    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resolution", default=256,
                        help="resolution")
    parser.add_argument("--device", default='cuda',
                        help="device")
    parser.add_argument("--epoch_main_path", default='256x512 20-cycleBlobGAN bedroom 360 aware -3 noise 2d layoutnet- -pre_pix2pixhd cycle coeffi 1',
                        help="epoch_main_path")
    parser.add_argument("--dataset_src_path", default='../data/bedroom_full_only/train',
                        help="dataset_src_path")

    args = parser.parse_args()

    cal_fid_kid(epoch_main_path=args.epoch_main_path, dataset_src_path=args.dataset_src_path, resolution=args.resolution, device=args.device)