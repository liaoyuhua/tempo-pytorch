import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim import lr_scheduler
from .utils import EarlyStopping
from .loss import mae_loss


class Trainer:
    """
    Default training loss is MAE. If you want to change it, you need
    to change it in the trainer.
    """

    def __init__(
        self,
        model: nn.Module,
        use_amp: bool = False,
        features: str = "M",
        inverse: bool = False,
        num_workers: int = 0,
    ) -> None:
        """
        Args:
            model: the model to train
            use_amp: whether to use automatic mixed precision training
            features: the features to use, either M (multivariate predict multivariate),
                S (univariate predict univariate) or MS (multivariate predict univariate)
            inverse: whether to inverse the target sequence
            num_workers: number of workers for data loader
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.use_amp = use_amp  # automatic mixed precision training

        self.ealry_stopping = EarlyStopping(patience=10, verbose=True)

        self.f_dim = -1 if features == "MS" else 0  # -1 for MS, 0 for M and S
        self.inverse = inverse
        self.num_workers = num_workers

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        max_epochs: int,
        lr: float,
        pct_start: float = 0.3,
    ):
        # save meta data
        self.train_info = {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "lr": lr,
            "pct_start": pct_start,
            "num_workers": self.num_workers,
            "use_amp": self.use_amp,
        }
        self.max_epochs = max_epochs

        # initialize data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        train_steps = len(train_loader)
        self.train_info.update({"train_steps": train_steps})

        # initialize optimizer
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )

        # initialize lr scheduler
        self.scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=pct_start,
            epochs=max_epochs,
            max_lr=lr,
        )

        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        for epoch in range(max_epochs):
            start_time = time.time()
            train_loss = self._train_epoch(train_loader, scaler)
            val_loss = self._val_epoch(val_loader)

            self.ealry_stopping(val_loss, self.model, "checkpoint")

            if self.ealry_stopping.early_stop:
                print("Early stopping")
                break

            print(
                f"Epoch: {epoch+1}/{self.max_epochs} | Train loss: {train_loss} | Val loss: {val_loss} | Time: {time.time()-start_time: .3f}s"
            )

    def _train_epoch(self, train_loader, scaler=None):
        self.model.train()

        total_loss = 0
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred, y_true = self._run_on_batch(train_loader.dataset, x, y)

            loss = mae_loss(pred, y_true)

            if self.use_amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _val_epoch(self, val_loader):
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred, y_true = self._run_on_batch(val_loader.dataset, x, y)

                loss = mae_loss(pred, y_true)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _run_on_batch(
        self,
        dataset_object: Dataset,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor = None,
    ):
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(x_batch)
        else:
            outputs = self.model(x_batch)

        if self.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        if y_batch is not None:
            y_batch = y_batch[:, self.f_dim :, :].to(self.device)  # B, N (target), Y

        outputs = outputs[:, self.f_dim :, :]  # B, N (target), Y

        return outputs, y_batch

    def test(self, test_dataset, batch_size):
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.model.load_state_dict(torch.load("checkpoint/checkpoint.pth"))

        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred, y_true = self._run_on_batch(test_loader.dataset, x, y)

                loss = mae_loss(pred, y_true)

                total_loss += loss.item()

        return total_loss / len(test_loader)

    def predict(self, dataset, batch_size):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.model.load_state_dict(torch.load("checkpoint/checkpoint.pth"))

        self.model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred, y_true = self._run_on_batch(loader.dataset, x, y)

        return pred, y_true
