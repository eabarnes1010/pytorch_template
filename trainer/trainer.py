import numpy as np
import torch
from base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        max_epochs,
        device,
        data_loader,
        validation_data_loader,
        settings,
    ):
        super().__init__(
            model,
            criterion,
            optimizer,
            max_epochs,
            device,
            settings["patience"],
            settings["min_delta"],
        )

        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.data_loader):
            input, input_unit, target = (
                data[0].to(self.device),
                data[1].to(self.device),
                target.to(self.device),
            )
            # input, target = (
            #     data.to(self.device),
            #     target.to(self.device),
            # )

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            output = self.model(input, input_unit)
            # output = self.model(input)

            # Compute the loss and its gradients
            loss = self.criterion(output, target)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        # return running_loss / len(self.trainloader.sampler)
        return running_loss / self.data_loader.__len__()

    def _validation_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.validation_data_loader):
                # input, target = data.to(self.device), target.to(self.device)
                input, input_unit, target = (
                    data[0].to(self.device),
                    data[1].to(self.device),
                    target.to(self.device),
                )

                output = self.model(input, input_unit)
                loss = self.criterion(output, target)

                running_loss += loss.item()

        return running_loss / self.validation_data_loader.__len__()
