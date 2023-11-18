from PIL import Image
import glob
import os

in_dir = "/media/jill/Offload/Jill/VR_img_cmps/VR_Dataset/"
out_dir = "./processed/"
new_size = (960, 540)

to_process = glob.glob(in_dir+"*.png")
for infile, idx in zip(to_process, range(len(to_process))):
    with Image.open(infile) as im:
        print(f"Processing {infile} [{idx+1}/{len(to_process)}]")
        im.resize(new_size, resample=Image.Resampling.BILINEAR).save(out_dir+os.path.basename(infile))
