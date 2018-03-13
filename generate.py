import os
import json
import math

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset import Dataset
from models import GeneratorDCGAN, GeneratorResNet
from utils import set_chars_type, concat_imgs, make_gif

FLAGS = tf.app.flags.FLAGS


def construct_ids(ids):
    """Construct ID from JSON file

    Prepare IDs (that will converted z) from a JSON file.
    You can use some operand:
        "-" means serial IDs. ("0-3" -> 0, 1, 2, 3)
        "*" means # of IDs. ("2*5" -> 2, 2, 2, 2, 2)
        "..:" means step of IDs. ("5..2:4" -> between 5 and 2 with 4 steps)
    You can connect some equations.
    ("0-2", "5*4" -> 0, 1, 2, 5, 5, 5, 5)
    Also read sample JSON file.
    """
    ids_x = np.array([], dtype=np.int32)
    ids_y = np.array([], dtype=np.int32)
    ids_alpha = np.array([], dtype=np.float32)
    for id_str in ids:
        if '-' in id_str:
            id_nums = id_str.split('-')
            for i in range(int(id_nums[0]), int(id_nums[1]) + 1):
                ids_x = np.append(ids_x, i)
                ids_y = np.append(ids_y, i)
                ids_alpha = np.append(ids_alpha, 0.)
        elif '*' in id_str:
            id_nums = id_str.split('*')
            for i in range(int(id_nums[1])):
                ids_x = np.append(ids_x, int(id_nums[0]))
                ids_y = np.append(ids_y, int(id_nums[0]))
                ids_alpha = np.append(ids_alpha, 0.)
        elif '..' in id_str and ':' in id_str:
            tmp, step = id_str.split(':')
            id_nums = tmp.split('..')
            for i in range(int(step)):
                ids_x = np.append(ids_x, int(id_nums[0]))
                ids_y = np.append(ids_y, int(id_nums[1]))
                ids_alpha = np.append(ids_alpha, 1. / float(step) * i)
        else:
            ids_x = np.append(ids_x, int(id_str))
            ids_y = np.append(ids_y, int(id_str))
            ids_alpha = np.append(ids_alpha, 0.)
    return ids_x, ids_y, ids_alpha


class GeneratingFontDesignGAN():
    """Generating font design GAN

    This class is only for generating fonts.
    """

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        self._setup_params()
        self._setup_embedding_chars()
        if FLAGS.generate_walk:
            self.batch_size = FLAGS.batch_size
            while ((FLAGS.char_img_n * self.char_embedding_n) % self.batch_size != 0) or (self.batch_size % self.char_embedding_n != 0):
                self.batch_size -= 1
            print('batch_size: {}'.format(self.batch_size))
            if FLAGS.generate_walk:
                self.walk_step = self.batch_size // self.char_embedding_n
                print('walk_step: {}'.format(self.walk_step))
            self._load_dataset()
        else:
            self._setup_inputs()
        self._prepare_generating()

    def _setup_dirs(self):
        """Setup output directories

        If destinations are not existed, make directories like this:
            FLAGS.gan_dir
            ├ generated
            └ random_walking
        """
        self.src_log = os.path.join(FLAGS.gan_dir, 'log')
        self.dst_generated = os.path.join(FLAGS.gan_dir, 'generated')
        if not os.path.exists(self.dst_generated):
            os.mkdir(self.dst_generated)
        if FLAGS.generate_walk:
            self.dst_walk = os.path.join(FLAGS.gan_dir, 'random_walking')
            if not os.path.exists(self.dst_walk):
                os.makedirs(self.dst_walk)

    def _setup_params(self):
        """Setup paramaters

        To setup GAN, read JSON file and set as attribute (self.~).
        JSON file's path is "FLAGS.gan_dir/log/flags.json".
        """
        with open(os.path.join(self.src_log, 'flags.json'), 'r') as json_file:
            json_dict = json.load(json_file)
        keys = ['chars_type', 'img_width', 'img_height', 'img_dim', 'style_z_size', 'font_h5',
                'style_ids_n', 'arch']
        for key in keys:
            setattr(self, key, json_dict[key])

    def _setup_embedding_chars(self):
        """Setup embedding characters

        Setup generating characters, like alphabets or hiragana.
        """
        self.embedding_chars = set_chars_type(self.chars_type)
        assert self.embedding_chars != [], 'embedding_chars is empty'
        self.char_embedding_n = len(self.embedding_chars)

    def _setup_inputs(self):
        """Setup inputs

        Setup generating inputs, batchsize and others.
        """
        assert os.path.exists(FLAGS.ids), '{} is not found'.format(FLAGS.ids)
        with open(FLAGS.ids, 'r') as json_file:
            json_dict = json.load(json_file)
        self.style_gen_ids_x, self.style_gen_ids_y, self.style_gen_ids_alpha = construct_ids(json_dict['style_ids'])
        self.char_gen_ids_x, self.char_gen_ids_y, self.char_gen_ids_alpha = construct_ids(json_dict['char_ids'])
        assert self.style_gen_ids_x.shape[0] == self.char_gen_ids_x.shape[0], \
            'style_ids.shape is not equal char_ids.shape'
        self.batch_size = self.style_gen_ids_x.shape[0]
        self.col_n = json_dict['col_n']
        self.row_n = math.ceil(self.batch_size / self.col_n)

    def _load_dataset(self):
        """Load dataset

        Setup dataset.
        """
        self.real_dataset = Dataset(self.font_h5, 'r', self.img_width, self.img_height, self.img_dim)
        self.real_dataset.set_load_data()

    def _prepare_generating(self):
        """Prepare generating

        Make tensorflow's graph.
        """
        self.z_size = self.style_z_size + self.char_embedding_n

        if self.arch == 'DCGAN':
            generator = GeneratorDCGAN(img_size=(self.img_width, self.img_height),
                                       img_dim=self.img_dim,
                                       z_size=self.z_size,
                                       layer_n=4,
                                       k_size=3,
                                       smallest_hidden_unit_n=64,
                                       is_bn=False)
        elif self.arch == 'ResNet':
            generator = GeneratorResNet(k_size=3, smallest_unit_n=64)

        if FLAGS.generate_walk:
            style_embedding_np = np.random.uniform(-1, 1, (FLAGS.char_img_n // self.walk_step, self.style_z_size)).astype(np.float32)
        else:
            style_embedding_np = np.random.uniform(-1, 1, (self.style_ids_n, self.style_z_size)).astype(np.float32)

        with tf.variable_scope('embeddings'):
            style_embedding = tf.Variable(style_embedding_np, name='style_embedding')
        self.style_ids_x = tf.placeholder(tf.int32, (self.batch_size,), name='style_ids_x')
        self.style_ids_y = tf.placeholder(tf.int32, (self.batch_size,), name='style_ids_y')
        self.style_ids_alpha = tf.placeholder(tf.float32, (self.batch_size,), name='style_ids_alpha')
        self.char_ids_x = tf.placeholder(tf.int32, (self.batch_size,), name='char_ids_x')
        self.char_ids_y = tf.placeholder(tf.int32, (self.batch_size,), name='char_ids_y')
        self.char_ids_alpha = tf.placeholder(tf.float32, (self.batch_size,), name='char_ids_alpha')

        # If sum of (style/char)_ids is less than -1, z is generated from uniform distribution
        style_z_x = tf.cond(tf.less(tf.reduce_sum(self.style_ids_x), 0),
                            lambda: tf.random_uniform((self.batch_size, self.style_z_size), -1, 1),
                            lambda: tf.nn.embedding_lookup(style_embedding, self.style_ids_x))
        style_z_y = tf.cond(tf.less(tf.reduce_sum(self.style_ids_y), 0),
                            lambda: tf.random_uniform((self.batch_size, self.style_z_size), -1, 1),
                            lambda: tf.nn.embedding_lookup(style_embedding, self.style_ids_y))
        style_z = style_z_x * tf.expand_dims(1. - self.style_ids_alpha, 1) \
            + style_z_y * tf.expand_dims(self.style_ids_alpha, 1)
        char_z_x = tf.one_hot(self.char_ids_x, self.char_embedding_n)
        char_z_y = tf.one_hot(self.char_ids_y, self.char_embedding_n)
        char_z = char_z_x * tf.expand_dims(1. - self.char_ids_alpha, 1) \
            + char_z_y * tf.expand_dims(self.char_ids_alpha, 1)

        z = tf.concat([style_z, char_z], axis=1)

        self.generated_imgs = generator(z, is_train=False)

        if FLAGS.gpu_ids == "":
            sess_config = tf.ConfigProto(
                device_count={"GPU": 0},
                log_device_placement=True
            )
        else:
            sess_config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
            )
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

        if FLAGS.generate_walk:
            var_list = [var for var in tf.global_variables() if 'embedding' not in var.name]
        else:
            var_list = [var for var in tf.global_variables()]
        pretrained_saver = tf.train.Saver(var_list=var_list)
        checkpoint = tf.train.get_checkpoint_state(self.src_log)
        assert checkpoint, 'cannot get checkpoint: {}'.format(self.src_log)
        pretrained_saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def _concat_and_save_imgs(self, src_imgs, dst_path):
        """Concatenate and save images

        Connect some images and save at dst_path.

        Args:
            src_imgs: Images that will be saved.
            dst_path: Destination path of image.
        """
        concated_img = concat_imgs(src_imgs, self.row_n, self.col_n)
        concated_img = (concated_img + 1.) * 127.5
        if self.img_dim == 1:
            concated_img = np.reshape(concated_img, (-1, self.col_n * self.img_height))
        else:
            concated_img = np.reshape(concated_img, (-1, self.col_n * self.img_height, self.img_dim))
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.save(dst_path)

    def generate(self, filename='generated'):
        """Generate fonts

        Generate fonts from JSON input.
        """
        generated_imgs = self.sess.run(self.generated_imgs,
                                       feed_dict={self.style_ids_x: self.style_gen_ids_x,
                                                  self.style_ids_y: self.style_gen_ids_y,
                                                  self.style_ids_alpha: self.style_gen_ids_alpha,
                                                  self.char_ids_x: self.char_gen_ids_x,
                                                  self.char_ids_y: self.char_gen_ids_y,
                                                  self.char_ids_alpha: self.char_gen_ids_alpha})
        self._concat_and_save_imgs(generated_imgs, os.path.join(self.dst_generated, '{}.png'.format(filename)))

    def generate_random_walking(self):
        """Generate fonts with random walking

        Generate fonts from random walking inputs.
        Results are changed gradually.
        """
        for c in self.embedding_chars:
            dst_dir = os.path.join(self.dst_walk, c)
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
        batch_n = (self.char_embedding_n * FLAGS.char_img_n) // self.batch_size
        c_ids = self.real_dataset.get_ids_from_labels(self.embedding_chars)
        for batch_i in tqdm(range(batch_n)):
            style_id_start = batch_i
            if batch_i == batch_n - 1:
                style_id_end = 0
            else:
                style_id_end = batch_i + 1
            generated_imgs = self.sess.run(self.generated_imgs,
                                           feed_dict={self.style_ids_x: np.ones(self.batch_size) * style_id_start,
                                                      self.style_ids_y: np.ones(self.batch_size) * style_id_end,
                                                      self.style_ids_alpha: np.repeat(np.linspace(0., 1., num=self.walk_step, endpoint=False), self.char_embedding_n),
                                                      self.char_ids_x: np.tile(c_ids, self.batch_size // self.char_embedding_n),
                                                      self.char_ids_y: np.tile(c_ids, self.batch_size // self.char_embedding_n),
                                                      self.char_ids_alpha: np.zeros(self.batch_size)})
            for img_i in range(generated_imgs.shape[0]):
                img = generated_imgs[img_i]
                img = (img + 1.) * 127.5
                pil_img = Image.fromarray(np.uint8(img))
                pil_img.save(os.path.join(self.dst_walk,
                             str(self.embedding_chars[img_i % self.char_embedding_n]),
                             '{:05d}.png'.format((batch_i * self.batch_size + img_i) // self.char_embedding_n)))
        print('making gif animations...')
        for i in range(self.char_embedding_n):
            make_gif(os.path.join(self.dst_walk, self.embedding_chars[i]),
                     os.path.join(self.dst_walk, self.embedding_chars[i] + '.gif'))
