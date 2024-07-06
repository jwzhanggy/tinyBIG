# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
Error back propagation algorithm based learner.

This module implements the backward learner, which can be used to
train, validate and test the RPN model within the tinyBIG toolkit.
"""
from tqdm import tqdm
import time
import torch

from tinybig.util.util import get_obj_from_str
from tinybig.learner.base_learner import learner
from tinybig.model.base_model import model as tinybig_model
from tinybig.data.base_data import dataloader as tinybig_dataloader
from tinybig.metric.base_metric import metric as tinybig_metric

class backward_learner(learner):
    """
    The backward learner defined based on the error back propagation algorithm.

    Attributes
    ----------
    n_epochs: int, default = 100
        Number of training epochs in the backward learner.
    loss: torch.nn.Module
        The loss function for RPN prediction evaluation.
    optimizer: torch.optim.Optimizer
        The optimizer for parameter gradient calculation
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
        The learning rate scheduler of the optimizer.

    Methods
    ----------
    __init__
        The backward learner initialization method.

    train
        The training method of the backward learner.

    test
        The testing method of the backward learner.
    """
    def __init__(
            self,
            name='error_backward_propagation_algorithm',
            n_epochs=100,
            lr_scheduler=None,
            optimizer=None,
            loss=None,
            lr_scheduler_configs=None,
            optimizer_configs=None,
            loss_configs=None,
            *args, **kwargs
    ):
        """
        The initialization method of the backward learner.

        It initializes the backward learner object, and initializes the loss function, optimizer,
        and lr_scheduler that will be used for the RPN model training.

        Specifically, the loss function, optimizer, and lr_scheduler can be initialized with the provided
        parameter loss, optimizer and lr_scheduler directly.
        Another initialization approach is to define the corresponding configurations, and initialize them
        based on the configuration descriptions instead.

        Parameters
        ----------
        name: str, default = 'error_backward_propagation_algorithm'
            Name of the backward learner.
        n_epochs: int, default = 100
            Number of training epochs in the backward learner.

        loss: torch.nn.Module
            The loss function for RPN prediction evaluation.
        optimizer: torch.optim.Optimizer
            The optimizer for parameter gradient calculation
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler
            The learning rate scheduler of the optimizer.

        loss_configs: dict, default = None
            The loss function configuration, which can also be used to initialize the loss function.
        optimizer_configs: dict, default = None
            The optimizer configuration, which can also be used to initialize the optimizer.
        lr_scheduler_configs: dict, default = None
            The configuration of the lr_scheduler, which can also
            be used to initialize the learning rate scheduler.

        Returns
        ----------
        object
            The backward learner object initialized with the parameters.
        """
        super().__init__(name=name)
        self.n_epochs = n_epochs

        # initialize of the lr_scheduler
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        elif lr_scheduler_configs is not None:
            self.lr_scheduler = lr_scheduler_configs['lr_scheduler_class']
            self.lr_scheduler_parameters = lr_scheduler_configs['lr_scheduler_parameters'] if 'lr_scheduler_parameters' in lr_scheduler_configs else {}
        else:
            self.lr_scheduler = None

        # initialize of the optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        elif optimizer_configs is not None:
            self.optimizer = optimizer_configs['optimizer_class']
            self.optimizer_parameters = optimizer_configs['optimizer_parameters'] if 'optimizer_parameters' in optimizer_configs else {}
        else:
            self.optimizer = None

        # initialize the loss function
        if loss is not None:
            self.loss = loss
        elif loss_configs is not None:
            loss_class = loss_configs['loss_class']
            parameters = loss_configs['loss_parameters'] if 'loss_parameters' in loss_configs else {}
            self.loss = get_obj_from_str(loss_class)(**parameters)
        else:
            self.loss = None

    def train(
            self,
            model: tinybig_model,
            data_loader: tinybig_dataloader,
            device: str = 'cpu',
            metric: tinybig_metric = None,
            test_check: bool = True,
            disable_tqdm: bool = False,
            display_step: int = 1
    ):
        """
        The backward learner training method for RPN model.

        It trains the RPN model with the provided training dataset. Based on the provided parameters,
        this method will also display information about the training process for each of the epochs,
        like the current epochs, time cost, training loss, training scores, and testing loss and testing scores.

        Parameters
        ----------
        model: tinybig.model.model
            The RPN model to be trained.
        data_loader: tinybig.data.dataloader
            The training data_loader.
        device: str, default = 'cpu'
            The device used for the model training.
        metric: tinybig.metric.metric, default = None
            The evaluation metric used to display the training process.
        test_check: bool, default = True
            Boolean tag indicating whether to display the testing performance or not during training.
        disable_tqdm: bool, default = False
            Boolean tag indicating whether to disable the tqdm progress bar or not.
        display_step: int, default = 1
            How often this method will display the training progress information,
            e.g., display_step=10, the training information will be displayed every 10 epochs.

        Returns
        -------
        dict
            The training record of the RPN model, covering information like the time cost,
            training loss, training scores, and testing loss and testing scores, etc.
        """
        #----------------------------
        training_record_dict = {
            'preparation': {},
            'training': {}
        }
        start_time = time.time()
        # ----------------------------

        train_loader = data_loader['train_loader']
        test_loader = data_loader['test_loader']

        model.to(device)

        criterion = self.loss

        # ----------------------------
        if type(self.optimizer) is str:
            self.optimizer_parameters['params'] = model.parameters()
            optimizer = get_obj_from_str(self.optimizer)(**self.optimizer_parameters)
        else:
            assert self.optimizer is not None
            optimizer = self.optimizer

        if type(self.lr_scheduler) is str:
            self.lr_scheduler_parameters['optimizer'] = optimizer
            lr_scheduler = get_obj_from_str(self.lr_scheduler)(**self.lr_scheduler_parameters)
        else:
            if self.lr_scheduler is not None:
                lr_scheduler = self.lr_scheduler
            else:
                lr_scheduler = None
        # ----------------------------

        # ----------------------------
        training_record_dict['preparation'] = {
            'preparation_time_cost': time.time() - start_time
        }
        # ----------------------------

        for epoch in range(self.n_epochs):
            # ----------------------------
            epoch_start_time = time.time()
            training_record_dict['training'][epoch] = {
                'start_time': epoch_start_time,
                'batch_records': {},
            }
            # ----------------------------

            model.train()
            with tqdm(train_loader, disable=disable_tqdm) as pbar:
                for i, (features, labels) in enumerate(pbar):
                    batch_start_time = time.time()

                    optimizer.zero_grad()
                    y_score = model(features.to(device), device=device)
                    y_pred = y_score.argmax(dim=1).tolist()
                    loss = criterion(y_score, labels.to(device))

                    if metric is not None:
                        metric_name = metric.__class__.__name__
                        score = metric.evaluate(y_pred=y_pred, y_true=labels, y_score=y_score.tolist())
                    else:
                        metric_name = None
                        score = None

                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(time=time.time()-start_time, epoch="{}/{}".format(epoch, self.n_epochs), loss=loss.item(), metric_score=score, lr=optimizer.param_groups[0]['lr'])

                    # ----------------------------
                    training_record_dict['training'][epoch]['batch_records'][i] = {
                        'time': time.time(),
                        'time_cost': time.time()-batch_start_time,
                        'loss': loss.item(),
                        'score': score,
                        'metric_name': metric_name
                    }
                    # ----------------------------
            # ----------------------------
            training_record_dict['training'][epoch]['end_time'] = time.time()
            training_record_dict['training'][epoch]['time_cost'] = time.time() - epoch_start_time
            # ----------------------------
            if test_check:
                test_result = self.test(model, test_loader, device=device, metric=metric, return_full_result=False)
                # ----------------------------
                training_record_dict['training'][epoch]['test_result'] = test_result
                # ----------------------------
                if epoch % display_step == 0:
                    print(f"Epoch: {epoch}, Test Loss: {test_result['test_loss']}, Test Score: {test_result['test_score']}, Time Cost: {test_result['time_cost']}")
            if lr_scheduler is not None:
                lr_scheduler.step()

        return training_record_dict

    def test(
            self,
            model: tinybig_model,
            test_loader: tinybig_dataloader,
            device: str = 'cpu',
            metric: tinybig_metric = None,
            return_full_result: bool = True
    ):
        """
        The testing method of the backward learning for RPN performance testing.

        It applies the RPN model to the provided testing set,
        and return the generated prediction results on the testing set.

        Parameters
        ----------
        model: tinybig.model.model
            The RPN model to be tested.
        test_loader: tinybig.data.dataloader
            The testing set dataloader.
        device: str, default = 'cpu'
            Device used for the testing method.
        metric: tinybig.metric.metric, default = None
            Evaluation metric used for evaluating the testing performance.
        return_full_result: bool, default = True
            The boolean tag indicating whether the full result should be returned.
            Since this test method will also be called in the train method for training
            performance display, which don't require the full testing results actually.

        Returns
        -------
        dict
            The testing results together with testing performance records.
        """

        start_time = time.time()

        model.eval()
        criterion = self.loss

        test_loss = 0.0
        y_pred = []
        y_true = []
        y_score = []
        with torch.no_grad():
            for features, labels in test_loader:
                output = model(features.to(device), device=device)
                max_class = output.argmax(dim=1).tolist()

                loss = criterion(output, labels.to(device))
                test_loss += loss.item()

                y_pred.extend(max_class)
                y_true.extend(labels.tolist())
                y_score.extend(output.tolist())


        test_loss /= len(test_loader)

        if metric is not None:
            metric_name = metric.__class__.__name__
            score = metric.evaluate(y_pred=y_pred, y_true=y_true, y_score=y_score)
        else:
            metric_name = None
            score = None

        if return_full_result:
            return {
                'y_pred': y_pred,
                'y_true': y_true,
                'y_score': y_score,
                'test_loss': test_loss,
                'test_score': score,
                'time_cost': time.time()-start_time,
                'metric_name': metric_name
            }
        else:
            return {
                'test_loss': test_loss,
                'test_score': score,
                'time_cost': time.time() - start_time
            }




