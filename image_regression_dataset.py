from torch.utils.data import Dataset
import SimpleITK as sitk
import torch
from helpers import resample_image

########################################
# Create Dataset Class:
########################################

class ImageRegressionDataset(Dataset):
    """Dataset for image segmentation."""

    def __init__(self, data_dir, selected_ids, id_ages, smoothen=None, edgen=False):
        if smoothen is None:
            smoothen = 0
        print("Initialising Dataset")
        self.ids = selected_ids
        if edgen:  # resample_image(dts[0][0], [3, 3, 3], [60, 60, 50])
            self.samples = [torch.from_numpy(sitk.GetArrayFromImage(sitk.SobelEdgeDetection(sitk.DiscreteGaussian(
                resample_image(sitk.ReadImage(f"{data_dir}/greymatter/wc1sub-{ID}_T1w.nii.gz", sitk.sitkFloat32),
                               [3, 3, 3], [60, 60, 50]), smoothen)))).unsqueeze(0) for ID in self.ids]
        else:
            self.samples = [torch.from_numpy(sitk.GetArrayFromImage(sitk.DiscreteGaussian(
                resample_image(sitk.ReadImage(f"{data_dir}/greymatter/wc1sub-{ID}_T1w.nii.gz", sitk.sitkFloat32),
                               [3, 3, 3], [60, 60, 50]), smoothen))).unsqueeze(0) for ID in self.ids]

        # self.samples = [(sitk.DiscreteGaussian(sitk.ReadImage(f"{data_dir}/greymatter/wc1sub-{ID}_T1w.nii.gz", sitk.sitkFloat32), smoothen)) for ID in self.ids]
        self.targets = torch.tensor(id_ages, dtype=torch.float).view((-1, 1))
        print("Initialisation complete")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.samples[item], self.targets[item]
