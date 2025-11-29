from datasets.airplane_dataset import PrecomputedAirplaneSurfaceDataset
from datasets.shapenetparts import ShapeNetPartDataset
from dgcnn import DGCNNSegmentation
import torch
from torch.utils.data import DataLoader

from dgcnn._plots import plot_gt_vs_pred
from training.trainer import evaluate, fit

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = "/home/stefano/stefano_repo/data/archive/PartAnnotation"  # folder containing shapenetcore_partanno_segmentation_benchmark_v0

    train_dataset = ShapeNetPartDataset(
        root=root,
        split="train",
        num_points=2048,
        class_choice=["Airplane"],   # or None for all
        normalize=True,
    )

    val_dataset = ShapeNetPartDataset(
        root=root,
        split="test",
        num_points=2048,
        class_choice=["Airplane"],   # or None for all
        normalize=True,
    )

    train_dataset.view_sample(0)  # Optional: visualize a sample
    plot_gt_vs_pred(
        model=DGCNNSegmentation(num_classes=train_dataset.num_seg_classes, input_dim=3, k=20).to(device),
        device=device,
        dataset=train_dataset,
        idx=0
    )  # Optional: visualize predictions before training

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

    num_classes = train_dataset.num_seg_classes
    model = DGCNNSegmentation(num_classes=num_classes, input_dim=3, k=20).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    _= evaluate(
        model,
        val_loader,
        device,
        0
    ) # Initial evaluation before training


    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=100,
        checkpoint_path="checkpoints/best_model_dgcnn_airplanes.pt",
        es_min_delta=0.0,
        es_patience=50
    )

    plot_gt_vs_pred(
        model=model,
        device=device,
        dataset=train_dataset,
        idx=0
    )  # Optional: visualize predictions before training

if __name__ == "__main__":
    main()
