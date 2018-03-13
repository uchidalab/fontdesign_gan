from glob import glob

import numpy as np
import imageio

ALPHABET_CAPS = list(chr(i) for i in range(65, 65 + 26))
HIRAGANA_SEION = list('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわゐゑをん')


def concat_imgs(src_imgs, row_n, col_n):
    """Concatenate images

    Load src_imgs and concatenate them.
    The result's # of rows is row_n, and # of columns is col_n.

    Args:
        src_imgs: Source images.
        row_n: # of rows.
        col_n: # of columns.
    """
    concated_img = np.empty((0, src_imgs.shape[1] * col_n, src_imgs.shape[3]))
    white_img = np.ones((src_imgs.shape[1], src_imgs.shape[2], src_imgs.shape[3]))
    for row_i in range(row_n):
        concated_row_img = np.empty((src_imgs.shape[1], 0, src_imgs.shape[3]))
        for col_i in range(col_n):
            count = row_i * col_n + col_i
            if count < len(src_imgs):
                concated_row_img = np.concatenate((concated_row_img, src_imgs[count]), axis=1)
            else:
                concated_row_img = np.concatenate((concated_row_img, white_img), axis=1)
        concated_img = np.concatenate((concated_img, concated_row_img), axis=0)
    return concated_img


def set_chars_type(chars_type):
    """Set characters type

    Set characters you want to generate.

    Args:
        chars_type: Type of characters. "caps" or "hiragana". If you want both of them, set "caps,hiragana".
    """
    chars = list()
    if 'caps' in chars_type:
        chars.extend(ALPHABET_CAPS)
    if 'hiragana' in chars_type:
        chars.extend(HIRAGANA_SEION)
    return chars


def make_gif(src_imgs_dir_path, dst_img_path):
    """Make a gif animation

    Read images from src_imgs_dir_path, make a gif animation and save at dst_img_path.

    Args:
        src_imgs_dir_path: Path of source images' directory.
        dst_img_path: Path of destination image.
    """
    img_paths = sorted(glob('{}/*.png'.format(src_imgs_dir_path)))
    imgs = [imageio.imread(f) for f in img_paths]
    imageio.mimsave(dst_img_path, imgs)
