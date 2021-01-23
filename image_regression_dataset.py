import os

from torch.utils.data import Dataset
import SimpleITK as sitk
import torch
from helpers import resample_image
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


class ImageRegressionDataset(Dataset):

    def __init__(self, data_dir, mode="train", smoothen=None,
                 edge_detect=False, max_workers=0):
        self.data, self.targets = None, None
        self.data_dir = data_dir
        if smoothen is None:
            smoothen = 0
        self.smoothen = smoothen
        self.edge_detect = edge_detect
        self.max_workers = max_workers

        if mode == "train":
            self.load_train_meta_files()
        elif mode == "test":
            self.load_test_meta_files()
        else:
            raise ValueError(f"Expected mode to be: train or test "
                             f"but got {mode}")
        self.load_dataset()

    def get_filename(self, patient_id):
        return f"sub-{patient_id}_T1w_unbiased.nii.gz"

    def load_train_meta_files(self):
        train_meta_csv_path = os.path.join(self.data_dir, "meta",
                                           "meta_data_regression_train.csv")
        self._load_meta_files(train_meta_csv_path)

    def load_test_meta_files(self):
        test_meta_csv_path = os.path.join(self.data_dir, "meta",
                                          "meta_data_regression_test.csv")
        self._load_meta_files(test_meta_csv_path)

    def _load_meta_files(self, fn):
        meta_data = pd.read_csv(fn)
        self.ids = meta_data["subject_id"].tolist()
        ages = meta_data["age"].tolist()
        self.targets = torch.tensor(ages).unsqueeze(-1)

    def load_image(self, patient_id):
        fn = self.get_filename(patient_id)
        filepath = os.path.join(self.data_dir, "images", fn)
        img = sitk.ReadImage(filepath, sitk.sitkFloat32)
        img = resample_image(img, [3, 3, 3], [60, 60, 50])
        img = sitk.DiscreteGaussian(img, self.smoothen)
        if self.edge_detect:
            img = sitk.SobelEdgeDetection(img)
        img = sitk.GetArrayFromImage(img)
        img = torch.from_numpy(img)
        img.unsqueeze_(0)
        return img

    def load_dataset(self):
        self.data = []
        if self.max_workers > 0:
            with ProcessPoolExecutor(self.max_workers) as e:
                img_iter = tqdm(e.map(self.load_image, self.ids, chunksize=32),
                                total=len(self))
                img_iter.set_description("Loading Images")
                self.data = [img for img in img_iter]
        else:
            ids_iter = tqdm(self.ids)
            ids_iter.set_description("Loading Images")
            self.data = [img for img in map(self.load_image, ids_iter)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]
