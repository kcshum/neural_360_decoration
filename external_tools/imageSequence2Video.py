import os
import glob
from PIL import Image


def pic2video(folder, duration):
    duration = float(duration)
    # filepaths
    fp_in = os.path.join(folder, '*')
    fp_out = os.path.join(folder, 'out.gif')

    fps = sorted(glob.glob(fp_in))
    imgs = (Image.open(f) for f in fps)
    img = next(imgs)  # extract first image from iterator
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=max(1., duration * len(fps))*100, loop=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-folder", "--img_folder",
                        help="path to image folder")
    parser.add_argument("-time", "--each_image_duration", default=0.1,
                        help="duration of each image to show in video, in second")

    args = parser.parse_args()

    pic2video(folder=args.img_folder, duration=args.each_image_duration)