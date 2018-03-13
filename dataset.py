import os
import sys
import random

import numpy as np
from PIL import Image
import h5py
from glob import glob
from tqdm import tqdm


class Dataset():

    def __init__(self, h5_path, mode, img_width, img_height, img_dim, is_mem=True):
        """For inputting/outputting font images' dataset.

        Use hdf5 files.
        Notes:
            self.mode == 'r' -> read mode
                         'w' -> write mode
            self.is_mem == True -> put data on memory. Very fast, but use a lot of memory space.
                           False -> read data from storage. Very slow, not recommended.
        """
        self.mode = mode
        self.img_width = img_width
        self.img_height = img_height
        self.img_dim = img_dim
        self.is_mem = is_mem

        assert mode == 'w' or mode == 'r', 'mode must be \'w\' or \'r\''
        if self.mode == 'w':
            if os.path.exists(h5_path):
                while True:
                    inp = input('overwrite {}? (y/n)\n'.format(h5_path))
                    if inp == 'y' or inp == 'n':
                        break
                if inp == 'n':
                    print('canceled')
                    sys.exit()
            self.h5file = h5py.File(h5_path, mode)
        if self.mode == 'r':
            assert os.path.exists(h5_path), 'hdf5 file is not found: {}'.format(h5_path)
            self.h5file = h5py.File(h5_path, mode)
            if self.is_mem:
                self._get = self._get_from_mem
            else:
                self._get = self._get_from_file

    def load_imgs_into_h5(self, src_dir_path):
        """Load png images, and save into hdf5 file.

        Load png images.
        Directory tree have to be like this:
            src_dir_path
            ├ A
            │ ├ foo.png
            │ ├ bar.png
            │ └ baz.png
            ├ B
            │ ├ foo.png
            │ └ bar.png
            └ C
              ├ foo.png
              ├ bar.png
              └ baz.png
        Don't have to put all character's image.
        but image's size have to be same.

        Args:
            src_dir_path: source directory
        """
        dir_paths = sorted(glob('{}/*'.format(src_dir_path)))
        for dir_path in tqdm(dir_paths):
            if not os.path.isdir(dir_path):
                continue
            img_paths = sorted(glob('{}/*.png'.format(dir_path)))
            imgs = np.empty((len(img_paths), self.img_width, self.img_height, self.img_dim), dtype=np.float32)
            fontnames = np.empty((len(img_paths)), dtype=object)
            for i, img_path in enumerate(img_paths):
                pil_img = Image.open(img_path)
                np_img = np.asarray(pil_img)
                np_img = (np_img.astype(np.float32) / 127.5) - 1.
                if len(np_img.shape) == 2:
                    np_img = np_img[np.newaxis, :, :, np.newaxis]
                    if self.img_dim == 3:
                        np_img = np.repeat(np_img, 3, axis=3)
                elif len(np_img.shape) == 3:
                    np_img = np_img[np.newaxis, :, :, :]
                imgs[i] = np_img
                fontnames[i] = os.path.basename(img_path).replace('.png', '')
            self._save(os.path.basename(dir_path), imgs, fontnames)

    def _save(self, char, imgs, fontnames):
        """Save images into hdf5 file.

        Args:
            char: character name
            imgs: image data
            fontname: font names
        """
        self.h5file.create_group(char)
        self.h5file.create_dataset(char + '/imgs', data=imgs)
        self.h5file.create_dataset(char + '/fontnames', data=fontnames, dtype=h5py.special_dtype(vlen=str))
        self.h5file.flush()

    def set_load_data(self, train_rate=1.):
        """Setup data for outputting.

        Make data queue for training(testing).
        also make label_to_id dictionary.

        Args:
            train_rate: Rate of training data.
                        If train_rate == 1., testing data aren't prepared.
        """
        print('preparing dataset...')
        self.keys_queue_train = list()
        self.label_to_id = dict()
        fontnames_list = list()
        all_fontnames = set()
        for i, (key, val) in enumerate(self.h5file.items()):
            fontnames = list()
            for fontname in val['fontnames'].value:
                fontnames.append(fontname)
                all_fontnames.add(fontname)
            fontnames_list.append(fontnames)
            font_n = len(val['imgs'])
            for j in range(font_n):
                self.keys_queue_train.append((key, j))
            self.label_to_id[key] = i
        self.font_n = len(all_fontnames)
        self.label_n = len(self.label_to_id)
        if train_rate != 1.:
            for i in range(self.label_n):
                for fontname in all_fontnames:
                    assert fontname in fontnames_list[i], 'If you want to divide train/test, all of fonts must have same characters'
            train_n = int(self.font_n * train_rate)
            train_ids = random.sample(range(0, self.font_n), train_n)
            self.keys_queue_test = list(filter(lambda x: x[1] not in train_ids, self.keys_queue_train))
            self.keys_queue_train = list(filter(lambda x: x[1] in train_ids, self.keys_queue_train))
        if self.is_mem:
            self._put_on_mem()

    def shuffle(self, is_test=False):
        """Shuffle data queue.

        Args:
            is_test: If you want to shuffle test data queue, set True.
        """
        if is_test:
            random.shuffle(self.keys_queue_test)
        else:
            random.shuffle(self.keys_queue_train)

    def get_data_n(self, is_test=False):
        """Get # of data.

        Args:
            is_test: If you want to get # of test data queue, set True.
        """
        if is_test:
            return len(self.keys_queue_test)
        return len(self.keys_queue_train)

    def get_data_n_by_labels(self, labels, is_test=False):
        """Get # of data of selected labels.

        Args:
            labels: List of label names
            is_test: If you want to get # of test data queue, set True.
        """
        if is_test:
            keys_queue = self.keys_queue_test
        else:
            keys_queue = self.keys_queue_train
        filtered_keys_queue = list(filter(lambda x: x[0] in labels, keys_queue))
        return len(filtered_keys_queue)

    def get_ids_from_labels(self, labels):
        """Get label's id from selected labels.

        Args:
            labels: List of label names.
        """
        ids = list()
        for label in labels:
            ids.append(self.label_to_id[label])
        return ids

    def get_batch(self, batch_i, batch_size, is_test=False, is_label=False):
        """Get data of a batch.

        Divide data by batch_size, and get batch_i/batch_size data.

        Args:
            batch_i: index of batches.
            batch_size: Batch size.
            is_test: If you want to get from test data, set True.
            is_label: If you want labels too, set True.
        """
        keys_list = list()
        for i in range(batch_i * batch_size, (batch_i + 1) * batch_size):
            if is_test:
                keys_list.append(self.keys_queue_test[i])
            else:
                keys_list.append(self.keys_queue_train[i])
        return self._get(keys_list, is_label)

    def get_batch_by_labels(self, batch_i, batch_size, labels, is_test=False, is_label=False):
        """Get data of a batch, from selected labels.

        Divide data by batch_size, and get batch_i/batch_size data.
        But only get from selected labels.

        Args:
            batch_i: index of batches.
            batch_size: Batch size.
            labels: List of label names.
            is_test: If you want to get from test data, set True.
            is_label: If you want labels too, set True.
        """
        if is_test:
            keys_queue = self.keys_queue_test
        else:
            keys_queue = self.keys_queue_train
        filtered_keys_queue = list(filter(lambda x: x[0] in labels, keys_queue))
        keys_list = list()
        for i in range(batch_i * batch_size, (batch_i + 1) * batch_size):
            keys_list.append(filtered_keys_queue[i])
        return self._get(keys_list, is_label)

    def get_random(self, batch_size, is_test=False, is_label=False):
        """Get data randomly.

        Args:
            batch_size: Batch size.
            is_test: If you want to get from test data, set True.
            is_label: If you want labels too, set True.
        """
        keys_list = list()
        for _ in range(batch_size):
            if is_test:
                keys_list.append(random.choice(self.keys_queue_test))
            else:
                keys_list.append(random.choice(self.keys_queue_train))
        return self._get(keys_list, is_label)

    def get_random_by_labels(self, batch_size, labels, is_test=False, is_label=False):
        """Get data randomly, from selected labels.

        Args:
            batch_size: Batch size.
            labels: List of label names.
            is_test: If you want to get from test data, set True.
            is_label: If you want labels too, set True.
        """
        if is_test:
            keys_queue = self.keys_queue_test
        else:
            keys_queue = self.keys_queue_train
        filtered_keys_queue = list(filter(lambda x: x[0] in labels, keys_queue))
        keys_list = list()
        for _ in range(batch_size):
            keys_list.append(random.choice(filtered_keys_queue))
        return self._get(keys_list, is_label)

    def get_fontname_by_label_id(self, label, index):
        """Get fontname by label id.

        Args:
            label: String of label.
            index: index of fontnames.
        """
        assert self.is_mem, 'Sorry, this function is only available is_mem==True'
        return str(self.fontnames[self.label_to_id[label]][index])

    def _get_from_file(self, keys_list, is_label=False):
        """Get data from file in storage.

        If self.is_mem == False, this function is called.

        Args:
            keys_list: List of keys that you get.
            is_label: If this is true, you also get labels.
        """
        imgs = np.empty((len(keys_list), self.img_width, self.img_height, self.img_dim), np.float32)
        labels = list()
        for i, keys in enumerate(keys_list):
            img = self.h5file[keys[0] + '/imgs'].value[keys[1]]
            imgs[i] = img[np.newaxis, :]
            labels.append(keys[0])
        if is_label:
            return imgs, labels
        return imgs

    def _put_on_mem(self):
        """Put data on RAM.

        If self.is_mem == True, this function is called.
        """
        print('putting data on RAM...')
        self.imgs = np.empty((self.label_n, self.font_n, self.img_width, self.img_height, self.img_dim), np.float32)
        self.fontnames = np.empty((self.label_n, self.font_n), np.object)
        self.label_to_font_n = dict()
        for i, key in enumerate(self.h5file.keys()):
            val = self.h5file[key + '/imgs'].value
            if len(val) < self.font_n:
                white_imgs = np.ones((self.font_n - len(val), self.img_width, self.img_height, self.img_dim), np.float32)
                val = np.concatenate((val, white_imgs), axis=0)
            self.imgs[i] = val
            self.fontnames[i] = self.h5file[key + '/fontnames'].value
            self.label_to_font_n[key] = len(self.imgs[i])

    def _get_from_mem(self, keys_list, is_label=False):
        """Get data from RAM.

        If self.is_mem == True, this function is called.

        Args:
            keys_list: List of keys that you get.
            is_label: If this is true, you also get labels.
        """
        imgs = np.empty((len(keys_list), self.img_width, self.img_height, self.img_dim), np.float32)
        labels = list()
        for i, keys in enumerate(keys_list):
            assert keys[1] < self.label_to_font_n[keys[0]], 'Image is out of range'
            img = self.imgs[self.label_to_id[keys[0]]][keys[1]]
            imgs[i] = img[np.newaxis, :]
            labels.append(keys[0])
        if is_label:
            return imgs, labels
        return imgs
