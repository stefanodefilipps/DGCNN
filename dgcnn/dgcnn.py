import torch
import torch.nn as nn
import torch.nn.functional as F

from dgcnn.dgcnn_utils import get_graph_feature

class DGCNNSegmentation(nn.Module):
    def __init__(self, num_classes, input_dim=3, k=20):
        super().__init__()
        self.k = k
        self.input_dim = input_dim

        # EdgeConv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * input_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * 64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * 64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * 128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Batch norm for concatenated local features
        self.bn_cat = nn.BatchNorm1d(64 + 64 + 128 + 256)

        # Global feature projection: 512 -> 1024
        self.global_conv = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Final per-point classifier:
        # input per point will be [local 512 + global 1024] = 1536 channels
        self.classifier = nn.Sequential(
            nn.Conv1d(512 + 1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.5),
            nn.Conv1d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        """
        x: (B, C, N)
        returns: logits (B, num_classes, N)
        """
        B, C, N = x.shape

        # EdgeConv 1
        x1 = get_graph_feature(x, k=self.k)  # (B,2*C,N,k)
        x1 = self.conv1(x1)                  # (B,64,N,k)
        x1 = x1.max(dim=-1)[0]               # (B,64,N)

        # EdgeConv 2
        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)                  # (B,64,N,k)
        x2 = x2.max(dim=-1)[0]               # (B,64,N)

        # EdgeConv 3
        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)                  # (B,128,N,k)
        x3 = x3.max(dim=-1)[0]               # (B,128,N)

        # EdgeConv 4
        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4)                  # (B,256,N,k)
        x4 = x4.max(dim=-1)[0]               # (B,256,N)

        # Concatenate multi-scale local features
        x_local = torch.cat((x1, x2, x3, x4), dim=1)  # (B,512,N)
        x_local = self.bn_cat(x_local)

        # Global feature: max over points, then project to 1024 channels
        x_global = self.global_conv(x_local)          # (B,1024,N)
        x_global = torch.max(x_global, dim=-1, keepdim=True)[0]  # (B,1024,1)

        # Tile global feature for each point
        x_global_expanded = x_global.repeat(1, 1, N)  # (B,1024,N)

        # Concatenate local + global for each point
        x_cat = torch.cat((x_local, x_global_expanded), dim=1)  # (B,512+1024=1536,N)

        # Final per-point classifier
        logits = self.classifier(x_cat)  # (B,num_classes,N)
        return logits, x_local, x_global

    def compute_loss(self, points, labels):
        """
        points: (B, C, N)
        labels: (B, N)
        """
        logits, _, _ = self(points)
        loss = F.cross_entropy(logits, labels.to(points.device))

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()

        logs = {
            "loss": loss.detach(),
            "acc": acc.detach(),
            "ce": loss.detach(),
            "reg": loss.detach(),  # placeholder if you later add graph regularization
        }
        return loss, logs
