import torch
from abc import abstractmethod
from numpy import inf
import time
import copy


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self, model, criterion, optimizer, max_epochs, device, patience, min_delta
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.device = device

        self.max_epochs = max_epochs

        self.early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    def fit(self):
        """
        Full training logic
        """

        for epoch in range(self.max_epochs + 1):
            start_time = time.time()

            train_result = self._train_epoch(epoch)
            val_result = self._validation_epoch(epoch)

            # early stopping
            if self.early_stopper.check_early_stop(epoch, val_result, self.model):
                print(
                    f"Restoring model weights from the end of the best epoch {self.early_stopper.best_epoch}: val_loss = {self.early_stopper.min_validation_loss:.5f}"
                )
                self.model.load_state_dict(self.early_stopper.best_model_state)
                break

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"Epoch {epoch:3d}/{self.max_epochs:2d}\n"
                f"  {elapsed_time:.1f}s - train_loss: {train_result:.5f} - val_loss: {val_result:.5f}"
            )

    @abstractmethod
    def _train_epoch(self):
        """
        Train an epoch

        """
        raise NotImplementedError

    @abstractmethod
    def _validation_epoch(self):
        """
        Validate after training an epoch

        """
        raise NotImplementedError


class EarlyStopping:
    """
    Base class for early stopping.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_model_state = None
        self.best_epoch = None

    def check_early_stop(self, epoch, validation_loss, model):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0

            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
