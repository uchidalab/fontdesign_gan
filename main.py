import os
import tensorflow as tf
from datetime import datetime
import subprocess

FLAGS = tf.app.flags.FLAGS


def get_gpu_n():
    """ Get # of GPUs

    Count rows of 'nvidia-smi -L' result.

    Returns:
        # of GPUs. If CUDA is not installed, return 0.
    """
    result = subprocess.run('nvidia-smi -L | wc -l', shell=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        return 0
    return int(result.stdout)


def define_flags():
    now_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')

    # Mode
    tf.app.flags.DEFINE_boolean('ttf2png', False, 'Convert font files into images')
    tf.app.flags.DEFINE_boolean('png2h5', False, 'Pack images into HDF5 format file')
    tf.app.flags.DEFINE_boolean('train', False, 'Train GAN')
    tf.app.flags.DEFINE_boolean('generate', False, 'Generate images')
    tf.app.flags.DEFINE_boolean('generate_walk', False, 'Generate images with random walking')

    # Paths
    tf.app.flags.DEFINE_string('font_ttfs', '', 'Path of font files\' directory')
    tf.app.flags.DEFINE_string('font_pngs', 'src/pngs/' + now_str, 'Path of font images\' directory')
    tf.app.flags.DEFINE_string('font_h5', '', 'Path of HDF5 file')
    tf.app.flags.DEFINE_string('gan_dir', 'result/' + now_str, 'Path of result\'s destination')
    tf.app.flags.DEFINE_string('ids', '', 'Path of input IDs settings\' JSON file')
    tf.app.flags.DEFINE_string('gen_name', now_str, 'Filename of saving image')
    tf.app.flags.DEFINE_integer('char_img_n', 256, '# of frames for generate_walk mode')

    # Other options
    tf.app.flags.DEFINE_integer('img_width', 64, 'Image\'s width')
    tf.app.flags.DEFINE_integer('img_height', 64, 'Image\'\'s height')
    tf.app.flags.DEFINE_integer('img_dim', 3, 'Image\'s dimention')
    tf.app.flags.DEFINE_string('chars_type', 'caps', 'Types of characters')
    tf.app.flags.DEFINE_string('gpu_ids', ', '.join([str(i) for i in range(get_gpu_n())]), 'GPU IDs to use')
    tf.app.flags.DEFINE_integer('batch_size', 256, 'Batch size')
    tf.app.flags.DEFINE_string('arch', 'DCGAN', 'Architecture of GAN')
    tf.app.flags.DEFINE_integer('style_ids_n', 256, '# of style IDs')
    tf.app.flags.DEFINE_integer('style_z_size', 100, 'z\'s size')
    tf.app.flags.DEFINE_integer('gan_epoch_n', 10000, '# of epochs for training GAN')
    tf.app.flags.DEFINE_integer('critic_n', 5, '# of critics to approximate wasserstein distance')
    tf.app.flags.DEFINE_integer('sample_imgs_interval', 10, 'Interval epochs of saving images')
    tf.app.flags.DEFINE_integer('sample_col_n', 26, '# of sample images\' columns')
    tf.app.flags.DEFINE_integer('keep_ckpt_interval', 250, 'Interval of keeping TensorFlow\'s dumps')
    tf.app.flags.DEFINE_boolean('run_tensorboard', True, 'Run tensorboard or not')
    tf.app.flags.DEFINE_integer('tensorboard_port', 6006, 'Port of tensorboard')


def main(argv=None):
    if FLAGS.ttf2png:
        assert FLAGS.font_ttfs != '', 'have to set --font_ttfs'
        from font2img.font2img import font2img
        if 'hiragana' in FLAGS.chars_type:
            src_chars_txt_path = 'font2img/src_chars_txt/hiragana_seion.txt'
        else:
            src_chars_txt_path = 'font2img/src_chars_txt/alphabets_hankaku_caps.txt'
        if not os.path.exists(FLAGS.font_pngs):
            os.makedirs(FLAGS.font_pngs)
        f2i = font2img(src_font_dir_path=FLAGS.font_ttfs,
                       src_chars_txt_path=src_chars_txt_path,
                       dst_dir_path=FLAGS.font_pngs,
                       canvas_size=FLAGS.img_height,
                       font_size=0,
                       output_ext='png',
                       is_center=True,
                       is_maximum=False,
                       is_binary=False,
                       is_unicode=False,
                       is_by_char=True,
                       is_recursive=True)
        f2i.run()
    if FLAGS.png2h5:
        assert FLAGS.font_h5 != '', 'have to set --font_h5'
        assert FLAGS.font_pngs != '', 'have to set --font_pngs'
        from dataset import Dataset
        dataset = Dataset(FLAGS.font_h5, 'w', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        dataset.load_imgs_into_h5(FLAGS.font_pngs)
        del dataset
    if FLAGS.train:
        assert FLAGS.font_h5 != '', 'have to set --font_h5'
        from train import TrainingFontDesignGAN
        gan = TrainingFontDesignGAN()
        gan.train()
        del gan
    if FLAGS.generate:
        assert FLAGS.gan_dir != '', 'have to set --gan_dir'
        assert FLAGS.ids != '', 'have to set --ids'
        from generate import GeneratingFontDesignGAN
        gan = GeneratingFontDesignGAN()
        gan.generate(filename=FLAGS.gen_name)
        del gan
    if FLAGS.generate_walk:
        assert FLAGS.gan_dir != '', 'have to set --gan_dir'
        from generate import GeneratingFontDesignGAN
        gan = GeneratingFontDesignGAN()
        gan.generate_random_walking()
        del gan


if __name__ == '__main__':
    define_flags()
    tf.app.run()
