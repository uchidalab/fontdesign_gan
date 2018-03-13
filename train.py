import os
import shutil
from glob import glob
import json
import time
from subprocess import Popen, PIPE

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset import Dataset
from models import GeneratorDCGAN, DiscriminatorDCGAN, GeneratorResNet, DiscriminatorResNet
from ops import average_gradients
from utils import set_chars_type, concat_imgs

FLAGS = tf.app.flags.FLAGS


class TrainingFontDesignGAN():
    """Training font design GAN.

    This is main part of our programs.
    """

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        self._save_flags()
        self._prepare_training()
        self._load_dataset()

    def _setup_dirs(self):
        """Setup output directories

        If destinations are not existed, make directories like this:
            FLAGS.gan_dir
            ├ log
            │ └ keep
            └ sample
        """
        if not os.path.exists(FLAGS.gan_dir):
            os.makedirs(FLAGS.gan_dir)
        self.dst_log = os.path.join(FLAGS.gan_dir, 'log')
        self.dst_samples = os.path.join(FLAGS.gan_dir, 'sample')
        if not os.path.exists(self.dst_log):
            os.mkdir(self.dst_log)
        self.dst_log_keep = os.path.join(self.dst_log, 'keep')
        if not os.path.exists(self.dst_log_keep):
            os.mkdir(self.dst_log_keep)
        if not os.path.exists(self.dst_samples):
            os.mkdir(self.dst_samples)

    def _save_flags(self):
        """Save FLAGS as JSON

        Write FLAGS paramaters as 'FLAGS.gan_dir/log/flsgs.json'.
        """
        with open(os.path.join(self.dst_log, 'flags.json'), 'w') as f:
            json.dump(FLAGS.__dict__['__flags'], f, indent=4)

    def _load_dataset(self):
        """Load dataset

        Set up dataset. All of data is for training, and they are shuffled.
        """
        self.real_dataset = Dataset(FLAGS.font_h5, 'r', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        self.real_dataset.set_load_data()
        self.real_dataset.shuffle()

    def _prepare_training(self):
        """Prepare Training

        Make tensorflow's graph.
        To support Multi-GPU, divide mini-batch.
        And this program has resume function.
        If there is checkpoint file in FLAGS.gan_dir/log, load checkpoint file and restart training.
        """
        assert FLAGS.batch_size >= FLAGS.style_ids_n, 'batch_size must be greater equal than style_ids_n'
        self.gpu_n = len(FLAGS.gpu_ids.split(','))
        self.embedding_chars = set_chars_type(FLAGS.chars_type)
        assert self.embedding_chars != [], 'embedding_chars is empty'
        self.char_embedding_n = len(self.embedding_chars)
        self.z_size = FLAGS.style_z_size + self.char_embedding_n

        with tf.device('/cpu:0'):
            # Set embeddings from uniform distribution
            style_embedding_np = np.random.uniform(-1, 1, (FLAGS.style_ids_n, FLAGS.style_z_size)).astype(np.float32)
            with tf.variable_scope('embeddings'):
                self.style_embedding = tf.Variable(style_embedding_np, name='style_embedding')

            self.style_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='style_ids')
            self.char_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='char_ids')
            self.is_train = tf.placeholder(tf.bool, name='is_train')
            self.real_imgs = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim), name='real_imgs')
            self.labels = tf.placeholder(tf.float32, (FLAGS.batch_size, self.char_embedding_n), name='labels')

            d_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0., beta2=0.9)
            g_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0., beta2=0.9)

        # Initialize lists for multi gpu
        fake_imgs = [0] * self.gpu_n
        d_loss = [0] * self.gpu_n
        g_loss = [0] * self.gpu_n

        d_grads = [0] * self.gpu_n
        g_grads = [0] * self.gpu_n

        divided_batch_size = FLAGS.batch_size // self.gpu_n
        is_not_first = False

        # Build graph
        for i in range(self.gpu_n):
            batch_start = i * divided_batch_size
            batch_end = (i + 1) * divided_batch_size
            with tf.device('/gpu:{}'.format(i)):
                if FLAGS.arch == 'DCGAN':
                    generator = GeneratorDCGAN(img_size=(FLAGS.img_width, FLAGS.img_height),
                                               img_dim=FLAGS.img_dim,
                                               z_size=self.z_size,
                                               layer_n=4,
                                               k_size=3,
                                               smallest_hidden_unit_n=64,
                                               is_bn=False)
                    discriminator = DiscriminatorDCGAN(img_size=(FLAGS.img_width, FLAGS.img_height),
                                                       img_dim=FLAGS.img_dim,
                                                       layer_n=4,
                                                       k_size=3,
                                                       smallest_hidden_unit_n=64,
                                                       is_bn=False)
                elif FLAGS.arch == 'ResNet':
                    generator = GeneratorResNet(k_size=3, smallest_unit_n=64)
                    discriminator = DiscriminatorResNet(k_size=3, smallest_unit_n=64)

                # If sum of (style/char)_ids is less than -1, z is generated from uniform distribution
                style_z = tf.cond(tf.less(tf.reduce_sum(self.style_ids[batch_start:batch_end]), 0),
                                  lambda: tf.random_uniform((divided_batch_size, FLAGS.style_z_size), -1, 1),
                                  lambda: tf.nn.embedding_lookup(self.style_embedding, self.style_ids[batch_start:batch_end]))
                char_z = tf.one_hot(self.char_ids[batch_start:batch_end], self.char_embedding_n)
                z = tf.concat([style_z, char_z], axis=1)

                # Generate fake images
                fake_imgs[i] = generator(z, is_reuse=is_not_first, is_train=self.is_train)

                # Calculate loss
                d_real = discriminator(self.real_imgs[batch_start:batch_end], is_reuse=is_not_first, is_train=self.is_train)
                d_fake = discriminator(fake_imgs[i], is_reuse=True, is_train=self.is_train)
                d_loss[i] = - (tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))
                g_loss[i] = - tf.reduce_mean(d_fake)

                # Calculate gradient Penalty
                epsilon = tf.random_uniform((divided_batch_size, 1, 1, 1), minval=0., maxval=1.)
                interp = self.real_imgs[batch_start:batch_end] + epsilon * (fake_imgs[i] - self.real_imgs[batch_start:batch_end])
                d_interp = discriminator(interp, is_reuse=True, is_train=self.is_train)
                grads = tf.gradients(d_interp, [interp])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[-1]))
                grad_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                d_loss[i] += 10 * grad_penalty

                # Get trainable variables
                d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
                g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

                d_grads[i] = d_opt.compute_gradients(d_loss[i], var_list=d_vars)
                g_grads[i] = g_opt.compute_gradients(g_loss[i], var_list=g_vars)

            is_not_first = True

        with tf.device('/cpu:0'):
            self.fake_imgs = tf.concat(fake_imgs, axis=0)
            avg_d_grads = average_gradients(d_grads)
            avg_g_grads = average_gradients(g_grads)
            self.d_train = d_opt.apply_gradients(avg_d_grads)
            self.g_train = g_opt.apply_gradients(avg_g_grads)

        # Calculate summary for tensorboard
        tf.summary.scalar('d_loss', -(sum(d_loss) / len(d_loss)))
        tf.summary.scalar('g_loss', -(sum(g_loss) / len(g_loss)))
        self.summary = tf.summary.merge_all()

        # Setup session
        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
        )
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=5)

        # If checkpoint is found, restart training
        checkpoint = tf.train.get_checkpoint_state(self.dst_log)
        if checkpoint:
            saver_resume = tf.train.Saver()
            saver_resume.restore(self.sess, checkpoint.model_checkpoint_path)
            self.epoch_start = int(checkpoint.model_checkpoint_path.split('-')[-1])
            print('restore ckpt')
        else:
            self.sess.run(tf.global_variables_initializer())
            self.epoch_start = 0

        # Setup writer for tensorboard
        self.writer = tf.summary.FileWriter(self.dst_log)

    def _get_ids(self, char_selector=''):
        """Get IDs for Generator's input.

        Generator's input 'z' is made from style_z and char_z.
        style_z is always given from random uniform distribution.
        char_z is one-hot encoded shape. It correspond with its character.
        In this function, prepare IDs(style_ids, char_ids) for style_z and char_z.
        Ids will converted style_z and char_z in _prepare_training().

        Args:
            char_selector: If this is only 1 character, set char_ids of this character.
                           Else, char_ids will be random IDs.
        """
        # All ids are -1 -> z is generated from uniform distribution when calculate graph
        style_ids = np.ones(FLAGS.batch_size) * -1
        if type(char_selector) == str and len(char_selector) == 1:
            char_ids = np.repeat(self.real_dataset.get_ids_from_labels(char_selector)[0], FLAGS.batch_size).astype(np.int32)
        else:
            char_ids = np.random.randint(0, self.char_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        return style_ids, char_ids

    def train(self):
        """Train GAN

        Run training GAN program.
        """
        # Start tensorboard
        if FLAGS.run_tensorboard:
            self._run_tensorboard()

        for epoch_i in tqdm(range(self.epoch_start, FLAGS.gan_epoch_n), initial=self.epoch_start, total=FLAGS.gan_epoch_n):
            for embedding_char in self.embedding_chars:
                # Calculate wasserstein distance
                for critic_i in range(FLAGS.critic_n):
                    real_imgs = self.real_dataset.get_random_by_labels(FLAGS.batch_size, [embedding_char])
                    style_ids, char_ids = self._get_ids(embedding_char)
                    self.sess.run(self.d_train, feed_dict={self.style_ids: style_ids,
                                                           self.char_ids: char_ids,
                                                           self.real_imgs: real_imgs,
                                                           self.is_train: True})

                # Minimize wasserstein distance
                style_ids, char_ids = self._get_ids(embedding_char)
                self.sess.run(self.g_train, feed_dict={self.style_ids: style_ids,
                                                       self.char_ids: char_ids,
                                                       self.is_train: True})

            # Calculate losses for tensorboard
            real_imgs = self.real_dataset.get_random(FLAGS.batch_size, is_label=False)
            style_ids, char_ids = self._get_ids()
            summary = self.sess.run(self.summary, feed_dict={self.style_ids: style_ids,
                                                             self.char_ids: char_ids,
                                                             self.real_imgs: real_imgs,
                                                             self.is_train: True})

            self.writer.add_summary(summary, epoch_i)

            # Save model weights
            dst_model_path = os.path.join(self.dst_log, 'result.ckpt')
            global_step = epoch_i + 1
            self.saver.save(self.sess, dst_model_path, global_step=global_step)
            if global_step % FLAGS.keep_ckpt_interval == 0:
                for f in glob(dst_model_path + '-' + str(global_step) + '.*'):
                    shutil.copy(f, self.dst_log_keep)

            # Save sample images
            if (epoch_i + 1) % FLAGS.sample_imgs_interval == 0:
                self._save_sample_imgs(epoch_i + 1)

    def _run_tensorboard(self):
        """Run tensorboard

        Run tensorboard for visualization of losses.
        To show progress-bar clearly in command line, sleep only 1 sec.
        """
        Popen(['tensorboard', '--logdir', '{}'.format(os.path.realpath(self.dst_log)), '--port', '{}'.format(FLAGS.tensorboard_port)], stdout=PIPE)
        time.sleep(1)

    def _generate_img(self, style_ids, char_ids, row_n, col_n):
        """Generate image

        This function is used for generating samples.

        Args:
            style_ids: ID of style_z. This paramaters are initialized when training started.
            char_ids: ID of char_z. ex. A->0, B->1...
            row_n: # of images in 1 row.
            col_n: # of images in 1 column.
        """
        feed = {self.style_ids: style_ids, self.char_ids: char_ids, self.is_train: False}
        generated_imgs = self.sess.run(self.fake_imgs, feed_dict=feed)
        combined_img = concat_imgs(generated_imgs, row_n, col_n)
        combined_img = (combined_img + 1.) * 127.5
        if FLAGS.img_dim == 1:
            combined_img = np.reshape(combined_img, (-1, col_n * FLAGS.img_height))
        else:
            combined_img = np.reshape(combined_img, (-1, col_n * FLAGS.img_height, FLAGS.img_dim))
        return Image.fromarray(np.uint8(combined_img))

    def _init_sample_imgs_inputs(self):
        """Initialize inputs for generating sample images

        Sample images are generated once every FLAGS.sample_imgs_interval times.
        These' inputs are given by this method.
        """
        self.sample_row_n = FLAGS.batch_size // FLAGS.sample_col_n
        self.sample_style_ids = np.repeat(np.arange(0, FLAGS.style_ids_n), self.char_embedding_n)[:FLAGS.batch_size]
        self.sample_char_ids = np.tile(np.arange(0, self.char_embedding_n), FLAGS.style_ids_n)[:FLAGS.batch_size]

    def _save_sample_imgs(self, epoch_i):
        """Save sample images

        Generate and save sample images in 'FLAGS.gan_dir/sample'.
        """
        if not hasattr(self, 'sample_style_ids'):
            self._init_sample_imgs_inputs()
        concated_img = self._generate_img(self.sample_style_ids, self.sample_char_ids,
                                          self.sample_row_n, FLAGS.sample_col_n)
        concated_img.save(os.path.join(self.dst_samples, '{}.png'.format(epoch_i)))
