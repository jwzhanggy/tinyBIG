# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from tqdm import tqdm
import time
import torch

from tinybig.util.util import get_obj_from_str
from tinybig.learner.learner import learner


class backward_learner(learner):
    def __init__(self, name='error_backward_propagation_algorithm', n_epochs=100,
                 lr_scheduler=None, optimizer=None, loss=None,
                 lr_scheduler_class=None, optimizer_class=None, loss_class=None, *args, **kwargs):
        super().__init__(name=name)
        self.n_epochs = n_epochs

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        elif lr_scheduler_class is not None:
            self.lr_scheduler = lr_scheduler_class
            self.lr_scheduler_parameters = kwargs['lr_scheduler_parameters'] if 'lr_scheduler_parameters' in kwargs else {}
        else:
            self.lr_scheduler = None

        if optimizer is not None:
            self.optimizer = optimizer
        elif optimizer_class is not None:
            self.optimizer = optimizer_class
            self.optimizer_parameters = kwargs['optimizer_parameters'] if 'optimizer_parameters' in kwargs else {}
        else:
            self.optimizer = None

        if loss is not None:
            self.loss = loss
        if loss_class is not None:
            parameters = kwargs['loss_parameters'] if 'loss_parameters' in kwargs else {}
            self.loss = get_obj_from_str(loss_class)(**parameters)
        else:
            self.loss = None

    def train(self, model, data_loader, device='cpu', metric=None, test_check=True, disable_tqdm: bool = False, display_step: int = 1):
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
            optimizer = self.optimizer(params=model.parameters())

        if type(self.lr_scheduler) is str:
            self.lr_scheduler_parameters['optimizer'] = optimizer
            lr_scheduler = get_obj_from_str(self.lr_scheduler)(**self.lr_scheduler_parameters)
        else:
            if self.lr_scheduler is not None:
                lr_scheduler = self.optimizer(optimizer=optimizer)
            else:
                lr_scheduler = self.lr_scheduler
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

    def test(self, model, test_loader, device='cpu', metric=None, return_full_result=True):
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




