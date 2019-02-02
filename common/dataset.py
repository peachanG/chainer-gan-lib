import os

import numpy as np
from PIL import Image
import chainer
from chainer.dataset import dataset_mixin


class Cifar10Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, test=False):
        d_train, d_test = chainer.datasets.get_cifar10(ndim=3, withlabel=False, scale=1.0)
        if test:
            self.ims = d_test
        else:
            self.ims = d_train
        self.ims = self.ims * 2 - 1.0  # [-1.0, 1.0]
        print("load cifar-10.  shape: ", self.ims.shape)

    def __len__(self):
        return self.ims.shape[0]

    def get_example(self, i):
        return self.ims[i]


def image_to_np(img, dtype):
    img = img.convert('RGB')
    img = np.asarray(img, dtype=np.uint8)
    img = img.transpose((2, 0, 1)).astype(dtype)
    if img.shape[0] == 1:
        img = np.broadcast_to(img, (3, img.shape[1], img.shape[2]))
    img = (img - 127.5)/127.5
    return img


def preprocess_image(img, crop_width=256, img2np=True):
    wid = min(img.size[0], img.size[1])
    ratio = crop_width / wid + 1e-4
    img = img.resize((int(ratio * img.size[0]), int(ratio * img.size[1])), Image.BILINEAR)
    x_l = (img.size[0]) // 2 - crop_width // 2
    x_r = x_l + crop_width
    y_u = 0
    y_d = y_u + crop_width
    img = img.crop((x_l, y_u, x_r, y_d))

    if img2np:
        img = image_to_np(img)
    return img


def find_all_files(directory):
    """http://qiita.com/suin/items/cdef17e447ceeff6e79d"""
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)


class ImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, root, one_class_flag=False, dtype=None, label_dtype=np.int32):
        extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        if one_class_flag:
            classes = [os.path.basename(root)]
            class_to_idx = {}
            self.pairs = self._make_one_class_dataset(root, extensions)
        else:
            classes, class_to_idx = self._find_classes(root)
            self.pairs = self._make_dataset(root, class_to_idx, extensions)

        if len(self.pairs) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.root = root
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self._dtype = chainer.get_dtype(dtype)
        self._label_dtype = label_dtype

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):
        path, int_label = self.pairs[i]
        img = Image.open(path)
        img = image_to_np(img, self._dtype)
        label = np.array(int_label, dtype=self._label_dtype)
        return img, label

    def _make_dataset(self, dir, class_to_idx, extensions):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self._has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
        return images

    def _make_one_class_dataset(self, dir, extensions):
        data_list = []
        dir = os.path.expanduser(dir)
        assert os.path.isdir(dir), '{} in make_dataset function is not directory'.format(dir)
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if self._has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, 0)
                    data_list.append(item)
        return data_list

    def _has_file_allowed_extension(self, filename, extensions):
        """Checks if a file is an allowed extension.
        Args:
            filename (string): path to a file
        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class ImagenetDataset(dataset_mixin.DatasetMixin):
    def __init__(self, file_list, crop_width=256):
        self.crop_width = crop_width
        self.image_files = file_list
        print(len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def get_example(self, i):
        np.random.seed()
        img = None

        while img is None:
            # print(i,id)
            try:
                fn = "%s" % (self.image_files[i])
                img = Image.open(fn)
            except Exception as e:
                print(i, fn, str(e))
        return preprocess_image(img, crop_width=self.crop_width)
