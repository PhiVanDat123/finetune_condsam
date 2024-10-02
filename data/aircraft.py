import numpy as np
import os
import torch
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset


class Aircraft(VisionDataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True, class_type='variant', transform=None,
                 target_transform=None, download=False):
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images

class Imagenet_Val(VisionDataset):
    """`Load imagenet validation set with folder of images and class in a file.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, classes_file='valprep.sh', transform=None,
                 target_transform=None, class_to_idx=None):
        super(Imagenet_Val, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.classes_file = classes_file

        # if download:
        #     self.download()
        self.class_to_idx = class_to_idx
        (image_ids, targets, classes) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(self.root) and os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                if ".JPEG" in line:
                    _, img_n, cls_n = line.split(" ")
                    image_ids.append(img_n)
                    cls_n = cls_n.split("/")[0]
                    targets.append(cls_n)
        # import pdb; pdb.set_trace()
        # index class names
        classes = np.unique(targets)
        if self.class_to_idx is None:
            self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [self.class_to_idx[c] for c in targets]

        return image_ids, targets, classes

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, image_ids[i]), targets[i])
            images.append(item)
        return images


class AirCraft_dataset:
    def __init__(self, batch_size, threads, img_size=224, root=None, dataset="cifar10", pre_trained=False, bz_test=None, custom={},
                 train_transform=None, test_transform=None, total_val=500, bz_val=30, num_chunk_val=1, correct_split=False):
        """

        :param batch_size:
        :param threads:
        :param root:
        :param dataset:
        :param pre_trained:
        :param bz_test:
        :param custom: {'type': "fft", "data": [ ]}
        """
        if bz_test is None:
            bz_test = batch_size
        if not root:
            root = './dataset'
        self.classes = None
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.Resize(size=(int(img_size / 0.875), int(img_size / 0.875))),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.126, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
                )
        if test_transform is None:
            test_transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                )
        if dataset == "aircraft":
            full_train_set = Aircraft(root, train=True, download=True, transform=train_transform)
            test_dataset = Aircraft(root, train=False, download=True, transform=test_transform)
        elif dataset == "imagenette":
            full_train_set = torchvision.datasets.Imagenette(root, split="train", download=True, transform=train_transform)
            test_dataset = torchvision.datasets.Imagenette(root, split="val", download=True, transform=test_transform)
        elif dataset == "OxfordIIITPet":
            full_train_set = torchvision.datasets.OxfordIIITPet(root, split="trainval", download=True, transform=train_transform)
            test_dataset = torchvision.datasets.OxfordIIITPet(root, split="test", download=True, transform=test_transform)
        elif dataset == "Flowers102":
            train_set = torchvision.datasets.Flowers102(root, split="train", download=True, transform=train_transform)
            val_set = torchvision.datasets.Flowers102(root, split="val", download=True, transform=train_transform)
            full_train_set = ConcatDataset([train_set, val_set])
            test_dataset = torchvision.datasets.Flowers102(root, split="test", download=True, transform=test_transform)
            self.classes = torch.arange(102)
        elif dataset == "Country211":
            # Hard code root for dataset
            if not os.path.exists(os.path.join(root, "country211")):
                root = "~/pb90_scratch/juliaN/dataset/"
            train_set = torchvision.datasets.Country211(root, split="train", download=True, transform=train_transform)
            val_set = torchvision.datasets.Country211(root, split="valid", download=True, transform=train_transform)
            full_train_set = ConcatDataset([train_set, val_set])
            test_dataset = torchvision.datasets.Country211(root, split="test", download=True, transform=test_transform)
            self.classes = train_set.classes
        elif dataset == "StanfordCars":
            full_train_set = torchvision.datasets.StanfordCars(root, split="train", download=True, transform=train_transform)
            test_dataset = torchvision.datasets.StanfordCars(root, split="test", download=True, transform=test_transform)
        elif dataset == "Food101":
            # Hard code root for dataset
            if not os.path.exists(os.path.join(root, "food-101")):
                root = "~/pb90_scratch/juliaN/dataset/"
            full_train_set = torchvision.datasets.Food101(root, split="train", download=True, transform=train_transform)
            test_dataset = torchvision.datasets.Food101(root, split="test", download=True, transform=test_transform)
        elif dataset == "imagenet":
            train_folder = os.path.join(root, "training_images_task_1and2")
            val_folder = os.path.join(root, "validation_images_all_tasks")
            # the `valprep.sh` should be saved in this folder 
            test_cls_file = os.path.join(os.path.dirname(__file__), "valprep.sh")
            full_train_set = torchvision.datasets.ImageFolder(train_folder, transform=train_transform)
            test_dataset = Imagenet_Val(val_folder, classes_file=test_cls_file,
                                        class_to_idx=full_train_set.class_to_idx, transform=test_transform)

        else:
            print("NO DATASET")
        self.train = torch.utils.data.DataLoader(full_train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.val = torch.utils.data.DataLoader(full_train_set, batch_size=bz_val, shuffle=True, num_workers=threads)
        
        print("Len full dataset", len(full_train_set))
        self.test = torch.utils.data.DataLoader(test_dataset, batch_size=bz_test, shuffle=False, num_workers=threads)
        if self.classes is None:
            self.classes = full_train_set.classes

if __name__ == '__main__':
    # train_dataset = Aircraft('./aircraft', train=True, download=False)
    # test_dataset = Aircraft('./aircraft', train=False, download=False)
    # train_dataset = Aircraft('../../dataset', train=True, download=True)
    # test_dataset = Aircraft('../../dataset', train=False, download=True)
    import pdb; pdb.set_trace()
    # dataset = AirCraft_dataset(128, 8, img_size=224, root="../../dataset", dataset="StanfordCars")
    dataset = AirCraft_dataset(128, 8, img_size=224, root="/mnt/reference/imagenet-2012/imagenet-2012_20210302", dataset="imagenet")
    import pdb; pdb.set_trace()