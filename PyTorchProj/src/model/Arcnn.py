import torch
import torch.nn as nn
import torch.nn.init as init


class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            # nn.PReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            # nn.PReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            # nn.PReLU()
            nn.LeakyReLU(),
        )
        self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.constant_(m.bias, val=0.0)
#     elif isinstance(m, nn.BatchNorm2d):
#         init.constant_(m.weight, val=1.0)
#         init.constant_(m.bias, val=0.0)
#     elif isinstance(m, torch.nn.Linear):
#         init.xavier_normal_(m.weight)
#         if layer.bias is not None:
#             init.constant_(m.bias, val=0.0)


# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         init.normal_(m.weight, std=0.001)
