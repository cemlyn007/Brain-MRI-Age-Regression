from torch.nn import Module, Conv3d, ConvTranspose3d, Linear, ReLU, Sequential, Linear, Flatten, L1Loss, BatchNorm3d, \
    Dropout, BatchNorm1d


class MRIRegressor(Module):
    """
    Neural Network for part 3.
    """

    def __init__(self, feats, dropout_p):
        super(MRIRegressor, self).__init__()
        self.model = Sequential(
            # 50, 60, 60
            Conv3d(1, feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(feats),
            ReLU(),
            Conv3d(feats, feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(feats),
            ReLU(),
            Conv3d(feats, 2 * feats, padding=0, kernel_size=2, stride=2, bias=True),
            Dropout(p=dropout_p),

            # 23, 28, 28
            Conv3d(2 * feats, 2 * feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(2 * feats),
            ReLU(),
            Conv3d(2 * feats, 2 * feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(2 * feats),
            ReLU(),
            Conv3d(2 * feats, 2 * 2 * feats, padding=0, kernel_size=2, stride=2, bias=True),

            # 9, 12, 12
            Conv3d(2 * 2 * feats, 2 * 2 * feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(2 * 2 * feats),
            ReLU(),
            Conv3d(2 * 2 * feats, 2 * 2 * feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(2 * 2 * feats),
            ReLU(),
            Conv3d(2 * 2 * feats, 2 * 2 * 2 * feats, padding=0, kernel_size=1, stride=1, bias=True),
            Dropout(p=dropout_p),

            # 5, 8, 8
            Conv3d(2 * 2 * 2 * feats, 2 * 2 * 2 * feats, padding=0, kernel_size=3, stride=1, bias=True),
            # 3, 6, 6
            BatchNorm3d(2 * 2 * 2 * feats),
            ReLU(),
            Conv3d(2 * 2 * 2 * feats, 2 * 2 * 2 * feats, padding=0, kernel_size=3, stride=1, bias=True),
            # 1, 4, 4
            BatchNorm3d(2 * 2 * 2 * feats),
            ReLU(),
            Conv3d(2 * 2 * 2 * feats, 2 * 2 * 2 * 2 * feats, padding=0, kernel_size=(1, 2, 2), stride=1, bias=True),
            Dropout(p=dropout_p),
            #  1, 3, 3
            Flatten(start_dim=1),  # Output: 1
            Linear(2 * 2 * 2 * 2 * feats * (1 * 3 * 3), 1),
        )

    def forward(self, x):
        return self.model(x)
