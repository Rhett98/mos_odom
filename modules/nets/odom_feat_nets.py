from torch import nn
from torch.nn import functional as F

class OdomFeatFC(nn.Module):
    def __init__(self, in_features):
        super(OdomFeatFC, self).__init__()
        self.input_size = in_features
        self.hidden_size = [1024, 512, 256]
        self.p = 0.
        num_layers = len(self.hidden_size)

        layers = [nn.Linear(self.input_size, self.hidden_size[0])]
        for i in range(1, num_layers):
            layers.append(nn.Linear(self.hidden_size[i - 1], self.hidden_size[i]))
        if self.p > 0.:
            self.drop = nn.Dropout(self.p)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """

        :param x: input of dim [BxN]
        :return:
        """
        b, n = x.shape
        x = x.view(b, n)
        for i, layer in enumerate(self.layers):
            x = F.leaky_relu(layer(x), negative_slope=0.01, inplace=True)

        if self.p > 0.:
            x = self.drop(x)

        x = x.view(b, -1)
        return x
