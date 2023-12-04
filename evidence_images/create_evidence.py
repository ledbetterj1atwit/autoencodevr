import matplotlib.pyplot as plt
from PIL import Image

images = ["961", "1232", "7245", "9476", "9946", "12337", "15878", "18815", "19566", "34516"]
dirs = ["originals", "ai", "jpeg", "webp"]
extensions = ["", "png", "png", "jpg", "webp"]

for im_id in images:
    plt.figure()
    for im_type, im_no in zip(dirs, range(1, 5)):
        ax = plt.subplot(2, 2, im_no)
        ax.get_xaxis().set_visible(False)  # Disable axes
        ax.get_yaxis().set_visible(False)
        ax.margins(tight=True)  # Tighen Margins
        ax.set_title(im_type, fontstyle='oblique', fontfamily='serif', fontsize='medium')

        im_data = Image.open(f"./{im_type}/img{im_id}.{extensions[im_no]}")
        plt.imshow(im_data)

    plt.savefig(f"./combined/img{im_id}.png", dpi=500, bbox_inches='tight')
