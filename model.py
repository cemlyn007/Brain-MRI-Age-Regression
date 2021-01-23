from torch.nn import Module, Conv3d, ReLU, Sequential, Linear, Flatten, \
    BatchNorm3d, Dropout


class MRIRegressor(Module):

    def __init__(self, feats, dropout_p):
        super(MRIRegressor, self).__init__()
        self.model = Sequential(
            Conv3d(1, feats, 3, 1, 0),
            BatchNorm3d(feats),
            ReLU(inplace=True),
            Conv3d(feats, feats, 3, 1, 0),
            BatchNorm3d(feats),
            ReLU(inplace=True),
            Conv3d(feats, 2 * feats, 2, 2, 0),
            Dropout(inplace=True, p=dropout_p),
            Conv3d(2 * feats, 2 * feats, 3, 1, 0),
            BatchNorm3d(2 * feats),
            ReLU(inplace=True),
            Conv3d(2 * feats, 2 * feats, 3, 1, 0),
            BatchNorm3d(2 * feats),
            ReLU(inplace=True),
            Conv3d(2 * feats, 2 * 2 * feats, 2, 2, 0),
            Conv3d(2 * 2 * feats, 2 * 2 * feats, 3, 1, 0),
            BatchNorm3d(2 * 2 * feats),
            ReLU(inplace=True),
            Conv3d(2 * 2 * feats, 2 * 2 * feats, 3, 1, 0),
            BatchNorm3d(2 * 2 * feats),
            ReLU(inplace=True),
            Conv3d(2 * 2 * feats, 2 * 2 * 2 * feats, 1, 1, 0),
            Dropout(inplace=True, p=dropout_p),
            Conv3d(2 * 2 * 2 * feats, 2 * 2 * 2 * feats, 3, 1, 0),
            BatchNorm3d(2 * 2 * 2 * feats),
            ReLU(inplace=True),
            Conv3d(2 * 2 * 2 * feats, 2 * 2 * 2 * feats, 3, 1, 0),
            BatchNorm3d(2 * 2 * 2 * feats),
            ReLU(inplace=True),
            Conv3d(2 * 2 * 2 * feats, 2 * 2 * 2 * 2 * feats, (1, 2, 2), 1, 0),
            Dropout(inplace=True, p=dropout_p),
            Flatten(start_dim=1),
            Linear(2 * 2 * 2 * 2 * feats * (1 * 3 * 3), 1),
        )

    def forward(self, x):
        return self.model(x)
