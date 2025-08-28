import os
import time
import numpy as np
import pandas as pd
import random
import gc
import copy

from collections import defaultdict

import torch
import torch as t
from torch import optim
from pathlib import Path
from functools import partial

from nbeats_lstm_model import NBeats, NBeatsBlock, IdentityBasis, TrendBasis, SeasonalityBasis
from nbeats_lstm_model import ExogenousBasisInterpretable, ExogenousBasisWavenet, ExogenousBasisTCN, ExogenousBasisLSTM
from nbeatsx_lstm_loader import TimeSeriesLoader
from nbeatsx_lstm_losses import MAPELoss, MASELoss, SMAPELoss, MSELoss, MAELoss, PinballLoss
from nbeatsx_lstm_metrics import *

def init_weights(module, initialization):
    if type(module) == t.nn.Linear:
        if initialization == 'orthogonal':
            t.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            t.nn.init.kaiming_uniform_(module.weight)
        elif initialization == 'he_normal':
            t.nn.init.kaiming_normal_(module.weight)
        elif initialization == 'glorot_uniform':
            t.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            t.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            pass
        else:
            assert 1<0, f'Initialization {initialization} not found'

class Nbeats(object):
    """
    Future documentation
    """
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    IDENTITY_BLOCK = 'identity'
    EXOGENOUS_BLOCK = 'exogenous'
    EXOGENOUS_TCN_BLOCK = 'exogenous_tcn'
    EXOGENOUS_WAVENET_BLOCK = 'exogenous_wavenet'
    EXOGENOUS_LSTM_BLOCK = 'exogenous_lstm'  # New block type

    def __init__(self,
                 input_size_multiplier,
                 output_size,
                 shared_weights,
                 activation,
                 initialization,
                 stack_types,
                 n_blocks,
                 n_layers,
                 n_hidden,
                 n_harmonics,
                 n_polynomials,
                 exogenous_n_channels,
                 include_var_dict,
                 t_cols,
                 batch_normalization,
                 dropout_prob_theta,
                 dropout_prob_exogenous,
                 x_s_n_hidden,
                 learning_rate,
                 lr_decay,
                 n_lr_decay_steps,
                 weight_decay,
                 l1_theta,
                 n_iterations,
                 early_stopping,
                 loss,
                 loss_hypar,
                 val_loss,
                 random_seed,
                 seasonality,
                 device=None):
        super(Nbeats, self).__init__()


        """
        N-BEATS model.
        Parameters
        ----------
        input_size_multiplier: int
            Multiplier to get insample size.
            Insample size = input_size_multiplier * output_size
        output_size: int
            Forecast horizon.
        shared_weights: bool
            If True, repeats first block.
        activation: str
            Activation function.
            An item from ['relu', 'softplus', 'tanh', 'selu', 'lrelu', 'prelu', 'sigmoid'].
        initialization: str
            Initialization function.
            An item from ['orthogonal', 'he_uniform', 'glorot_uniform', 'glorot_normal', 'lecun_normal'].
        stack_types: List[str]
            List of stack types.
            Subset from ['seasonality', 'trend', 'identity', 'exogenous', 'exogenous_tcn', 'exogenous_wavenet'].
        n_blocks: List[int]
            Number of blocks for each stack type.
            Note that len(n_blocks) = len(stack_types).
        n_layers: List[int]
            Number of layers for each stack type.
            Note that len(n_layers) = len(stack_types).
        n_hidden: List[List[int]]
            Structure of hidden layers for each stack type.
            Each internal list should contain the number of units of each hidden layer.
            Note that len(n_hidden) = len(stack_types).
        n_harmonics: List[int]
            Number of harmonic terms for each stack type.
            Note that len(n_harmonics) = len(stack_types).
        n_polynomials: List[int]
            Number of polynomial terms for each stack type.
            Note that len(n_polynomials) = len(stack_types).
        exogenous_n_channels:
            Exogenous channels for non-interpretable exogenous basis.
        include_var_dict: Dict[str, List[int]]
            Exogenous terms to add.
        t_cols: List
            Ordered list of ['y'] + X_cols + ['available_mask', 'sample_mask'].
            Can be taken from the dataset.
        batch_normalization: bool
            Whether perform batch normalization.
        dropout_prob_theta: float
            Float between (0, 1).
            Dropout for Nbeats basis.
        dropout_prob_exogenous: float
            Float between (0, 1).
            Dropout for exogenous basis.
        x_s_n_hidden: int
            Number of encoded static features to calculate.
        learning_rate: float
            Learning rate between (0, 1).
        lr_decay: float
            Decreasing multiplier for the learning rate.
        n_lr_decay_steps: int
            Period for each lerning rate decay.
        weight_decay: float
            L2 penalty for optimizer.
        l1_theta: float
            L1 regularization for the loss function.
        n_iterations: int
            Number of training steps.
        early_stopping: int
            Early stopping interations.
        loss: str
            Loss to optimize.
            An item from ['MAPE', 'MASE', 'SMAPE', 'MSE', 'MAE', 'PINBALL'].
        loss_hypar:
            Hyperparameter for chosen loss.
        val_loss:
            Validation loss.
            An item from ['MAPE', 'MASE', 'SMAPE', 'RMSE', 'MAE', 'PINBALL'].
        random_seed: int
            random_seed for pseudo random pytorch initializer and
            numpy random generator.
        seasonality: int
            Time series seasonality.
            Usually 7 for daily data, 12 for monthly data and 4 for weekly data.
        device: Optional[str]
            If None checks 'cuda' availability.
            An item from ['cuda', 'cpu'].
        """

        if activation == 'selu':
            initialization = 'lecun_normal'

        self.input_size_multiplier = input_size_multiplier
        self.input_size = int(input_size_multiplier * output_size)
        self.output_size = output_size
        self.shared_weights = shared_weights
        self.activation = activation
        self.initialization = initialization
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        self.exogenous_n_channels = exogenous_n_channels

        self.batch_normalization = batch_normalization
        self.dropout_prob_theta = dropout_prob_theta
        self.dropout_prob_exogenous = dropout_prob_exogenous
        self.x_s_n_hidden = x_s_n_hidden
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.n_lr_decay_steps = n_lr_decay_steps
        self.weight_decay = weight_decay
        self.n_iterations = n_iterations
        self.early_stopping = early_stopping
        self.loss = loss
        self.loss_hypar = loss_hypar
        self.val_loss = val_loss
        self.l1_theta = l1_theta
        self.l1_conv = 1e-3
        self.random_seed = random_seed

        self.seasonality = seasonality
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols

        if device is None:
            device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.device = device

        self._is_instantiated = False



    def get_params(self, deep=True):
            # 返回模型的参数作为字典
        return {
            'input_size_multiplier': self.input_size_multiplier,
            'output_size': self.output_size,
            'shared_weights': self.shared_weights,
            'activation': self.activation,
            'initialization': self.initialization,
            'stack_types': self.stack_types,
            'n_blocks': self.n_blocks,
            'n_layers': self.n_layers,
            'n_hidden': self.n_hidden,
            'n_harmonics': self.n_harmonics,
            'n_polynomials': self.n_polynomials,
            'exogenous_n_channels': self.exogenous_n_channels,
            'include_var_dict': self.include_var_dict,
            't_cols': self.t_cols,
            'batch_normalization': self.batch_normalization,
            'dropout_prob_theta': self.dropout_prob_theta,
            'dropout_prob_exogenous': self.dropout_prob_exogenous,
            'x_s_n_hidden': self.x_s_n_hidden,
            'learning_rate': self.learning_rate,
            'lr_decay': self.lr_decay,
            'n_lr_decay_steps': self.n_lr_decay_steps,
            'weight_decay': self.weight_decay,
            'l1_theta': self.l1_theta,
            'n_iterations': self.n_iterations,
            'early_stopping': self.early_stopping,
            'loss': self.loss,
            'loss_hypar': self.loss_hypar,
            'val_loss': self.val_loss,
            'random_seed': self.random_seed,
            'seasonality': self.seasonality,
            'device': self.device
        }

    def set_params(self, **params):
        # 根据传递的参数字典更新模型的超参数
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def create_stack(self):
        if self.include_var_dict is not None:
            x_t_n_inputs = self.output_size * int(sum([len(x) for x in self.include_var_dict.values()]))
            #print(x_t_n_inputs)

            if len(self.include_var_dict['week_day']) > 0:
                x_t_n_inputs = x_t_n_inputs - self.output_size + 1
        else:
            x_t_n_inputs = self.input_size

        block_list = []
        self.blocks_regularizer = []
        for i in range(len(self.stack_types)):
            for block_id in range(self.n_blocks[i]):

                if (len(block_list) == 0) and (self.batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                self.blocks_regularizer += [0]

                if self.shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    if self.stack_types[i] == self.SEASONALITY_BLOCK:
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=4 * int(
                                                       np.ceil(self.n_harmonics / 2 * self.output_size) - (
                                                               self.n_harmonics - 1)),
                                                   basis=SeasonalityBasis(harmonics=self.n_harmonics,
                                                                          backcast_size=self.input_size,
                                                                          forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == self.TREND_BLOCK:
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * (self.n_polynomials + 1),
                                                   basis=TrendBasis(degree_of_polynomial=self.n_polynomials,
                                                                    backcast_size=self.input_size,
                                                                    forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == self.IDENTITY_BLOCK:
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=self.input_size + self.output_size,
                                                   basis=IdentityBasis(backcast_size=self.input_size,
                                                                       forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == self.EXOGENOUS_BLOCK:
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * self.n_x_t,
                                                   basis=ExogenousBasisInterpretable(),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == self.EXOGENOUS_TCN_BLOCK:
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * self.exogenous_n_channels,
                                                   basis=ExogenousBasisTCN(self.exogenous_n_channels, self.n_x_t),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == self.EXOGENOUS_WAVENET_BLOCK:
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * self.exogenous_n_channels,
                                                   basis=ExogenousBasisWavenet(self.exogenous_n_channels, self.n_x_t),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                        self.blocks_regularizer[-1] = 1
                    elif self.stack_types[i] == self.EXOGENOUS_LSTM_BLOCK:  # New block type
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * self.exogenous_n_channels,
                                                   basis=ExogenousBasisLSTM(input_size=self.n_x_t,
                                                                            #hidden_size=64,
                                                                            #num_layers=2,
                                                                            hidden_size= self.n_hidden[i][0],
                                                                            num_layers=self.n_layers[i],
                                                                            dropout_prob=self.dropout_prob_theta,
                                                                            theta_size= 2 * self.exogenous_n_channels
                                                                            ),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                        self.blocks_regularizer[-1] = 1
                    else:
                        assert False, f'Block type not found!'
                init_function = partial(init_weights, initialization=self.initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def __loss_fn(self, loss_name: str):
        def loss(x, loss_hypar, forecast, target, mask):
            if loss_name == 'MAPE':
                return MAPELoss(y=target, y_hat=forecast, mask=mask) + \
                       self.loss_l1_conv_layers() + self.loss_l1_theta()
            elif loss_name == 'MASE':
                return MASELoss(y=target, y_hat=forecast, y_insample=x, seasonality=loss_hypar, mask=mask) + \
                       self.loss_l1_conv_layers() + self.loss_l1_theta()
            elif loss_name == 'SMAPE':
                return SMAPELoss(y=target, y_hat=forecast, mask=mask) + \
                       self.loss_l1_conv_layers() + self.loss_l1_theta()
            elif loss_name == 'MSE':
                return MSELoss(y=target, y_hat=forecast, mask=mask) + \
                       self.loss_l1_conv_layers() + self.loss_l1_theta()
            elif loss_name == 'MAE':
                return MAELoss(y=target, y_hat=forecast, mask=mask) + \
                       self.loss_l1_conv_layers() + self.loss_l1_theta()
            elif loss_name == 'PINBALL':
                return PinballLoss(y=target, y_hat=forecast, mask=mask, tau=loss_hypar) + \
                       self.loss_l1_conv_layers() + self.loss_l1_theta()
            else:
                raise Exception(f'Unknown loss function: {loss_name}')
        return loss

    def __val_loss_fn(self, loss_name='MAE'):
        def loss(forecast, target, weights):
            if loss_name == 'MAPE':
                return mape(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'SMAPE':
                return smape(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'MSE':
                return mse(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'RMSE':
                return rmse(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'MAE':
                return mae(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'PINBALL':
                return pinball_loss(y=target, y_hat=forecast, weights=weights, tau=0.5)
            else:
                raise Exception(f'Unknown loss function: {loss_name}')
        return loss

    '''
    def loss_l1_conv_layers(self):
        loss_l1 = 0
        for i, indicator in enumerate(self.blocks_regularizer):
            if indicator:
                loss_l1 += self.l1_conv * t.sum(t.abs(self.model.blocks[i].basis.weight))
        return loss_l1
    '''

    def loss_l1_conv_layers(self):
        loss_l1 = 0
        for i, indicator in enumerate(self.blocks_regularizer):
            if indicator:
                block = self.model.blocks[i].basis
                # 确保weight是可迭代的并且包含tensor
                if hasattr(block, 'weight') and isinstance(block.weight, list):
                    for w in block.weight:
                        loss_l1 += self.l1_conv * t.sum(t.abs(w))
                else:
                    # 这里处理非列表形式的权重（单个张量）
                    loss_l1 += self.l1_conv * t.sum(t.abs(block.weight))
        return loss_l1

    def loss_l1_theta(self):
        loss_l1 = 0
        for block in self.model.blocks:
            for layer in block.modules():
                if isinstance(layer, t.nn.Linear):
                    loss_l1 += self.l1_theta * layer.weight.abs().sum()
        return loss_l1

    def to_tensor(self, x: np.ndarray) -> t.Tensor:
        tensor = t.as_tensor(x, dtype=t.float32).to(self.device)
        return tensor

    def fit(self, train_ts_loader, val_ts_loader=None, n_iterations=None, verbose=True, eval_steps=1):
        # TODO: Indexes hardcoded, information duplicated in train and val datasets
        assert (self.input_size)==train_ts_loader.input_size, \
            f'model input_size {self.input_size} data input_size {train_ts_loader.input_size}'

        # Random Seeds (model initialization)
        t.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Attributes of ts_dataset
        self.n_x_t, self.n_x_s = train_ts_loader.get_n_variables()

        # Instantiate model
        if not self._is_instantiated:
            block_list = self.create_stack()
            self.model = NBeats(t.nn.ModuleList(block_list)).to(self.device)
            self._is_instantiated = True

        # Overwrite n_iterations and train datasets
        if n_iterations is None:
            n_iterations = self.n_iterations

        lr_decay_steps = n_iterations // self.n_lr_decay_steps
        if lr_decay_steps == 0:
            lr_decay_steps = 1

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=self.lr_decay)
        training_loss_fn = self.__loss_fn(self.loss)
        validation_loss_fn = self.__val_loss_fn(self.val_loss)

        print('\n')
        print('='*30+' Start fitting '+'='*30)

        start = time.time()
        self.trajectories = {'iteration':[],'train_loss':[], 'val_loss':[]}
        self.final_insample_loss = None
        self.final_outsample_loss = None

        # Training Loop
        early_stopping_counter = 0
        best_val_loss = np.inf
        best_state_dict = copy.deepcopy(self.model.state_dict())
        break_flag = False
        iteration = 0
        epoch = 0
        while (iteration < n_iterations) and (not break_flag):
            epoch +=1
            for batch in iter(train_ts_loader):
                iteration += 1
                if (iteration > n_iterations) or (break_flag):
                    continue

                self.model.train()
                # Parse batch
                insample_y     = self.to_tensor(batch['insample_y'])
                insample_x     = self.to_tensor(batch['insample_x'])
                insample_mask  = self.to_tensor(batch['insample_mask'])
                outsample_x    = self.to_tensor(batch['outsample_x'])
                outsample_y    = self.to_tensor(batch['outsample_y'])
                outsample_mask = self.to_tensor(batch['outsample_mask'])
                s_matrix       = self.to_tensor(batch['s_matrix'])

                optimizer.zero_grad()
                forecast   = self.model(x_s=s_matrix, insample_y=insample_y,
                                        insample_x_t=insample_x, outsample_x_t=outsample_x,
                                        insample_mask=insample_mask)

                training_loss = training_loss_fn(x=insample_y, loss_hypar=self.loss_hypar, forecast=forecast,
                                                 target=outsample_y, mask=outsample_mask)


                # Protection if exploding gradients
                # 获取训练损失的数值
                training_loss_value = training_loss.detach().cpu().item()

                if not np.isnan(training_loss_value):
                    # 反向传播
                    training_loss.backward()

                    # 计算并打印总的梯度范数
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    #print(f"Total gradient norm: {total_norm}")

                    # 梯度裁剪（可选）
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                else:
                    early_stopping_counter = self.early_stopping

                lr_scheduler.step()
                if (iteration % eval_steps == 0):
                    display_string = 'Step: {}, Time: {:03.3f}, Insample {}: {:.5f}'.format(iteration,
                                                                                            time.time()-start,
                                                                                            self.loss,
                                                                                            training_loss.cpu().data.numpy())
                    self.trajectories['iteration'].append(iteration)
                    self.trajectories['train_loss'].append(np.float(training_loss.cpu().data.numpy()))

                    if val_ts_loader is not None:
                        loss = self.evaluate_performance(ts_loader=val_ts_loader,
                                                         validation_loss_fn=validation_loss_fn)
                        display_string += ", Outsample {}: {:.5f}".format(self.val_loss, loss)
                        self.trajectories['val_loss'].append(loss)

                        if self.early_stopping:
                            if loss < best_val_loss:
                                # Save current model if improves outsample loss
                                best_state_dict = copy.deepcopy(self.model.state_dict())
                                best_insample_loss = training_loss.cpu().data.numpy()
                                early_stopping_counter = 0
                                best_val_loss = loss
                            else:
                                early_stopping_counter += 1
                            if early_stopping_counter >= self.early_stopping:
                                break_flag = True

                    print(display_string)

                    self.model.train()

                if break_flag:
                    print('\n')
                    print(19*'-',' Stopped training by early stopping', 19*'-')
                    self.model.load_state_dict(best_state_dict)
                    break

        #End of fitting
        if n_iterations >0:
            # This is batch loss!
            self.final_insample_loss = np.float(training_loss.cpu().data.numpy()) if not break_flag else best_insample_loss
            string = 'Step: {}, Time: {:03.3f}, Insample {}: {:.5f}'.format(iteration,
                                                                            time.time()-start,
                                                                            self.loss,
                                                                            self.final_insample_loss)
            if val_ts_loader is not None:
                self.final_outsample_loss = self.evaluate_performance(ts_loader=val_ts_loader,
                                                                      validation_loss_fn=validation_loss_fn)
                string += ", Outsample {}: {:.5f}".format(self.val_loss, self.final_outsample_loss)
            print(string)
            print('='*30+'  End fitting  '+'='*30)
            print('\n')

    def predict(self, ts_loader, X_test=None, return_decomposition=False):
        self.model.eval()
        assert not ts_loader.shuffle, 'ts_loader must have shuffle as False.'

        forecasts = []
        block_forecasts = []
        outsample_ys = []
        outsample_masks = []
        with t.no_grad():
            for batch in iter(ts_loader):
                insample_y     = self.to_tensor(batch['insample_y'])
                insample_x     = self.to_tensor(batch['insample_x'])
                insample_mask  = self.to_tensor(batch['insample_mask'])
                outsample_x    = self.to_tensor(batch['outsample_x'])
                s_matrix       = self.to_tensor(batch['s_matrix'])

                forecast, block_forecast = self.model(insample_y=insample_y, insample_x_t=insample_x,
                                                      insample_mask=insample_mask, outsample_x_t=outsample_x,
                                                      x_s=s_matrix, return_decomposition=True) # always return decomposition
                forecasts.append(forecast.cpu().data.numpy())
                #print("forecast is:", forecast)
                #("block_forecast is:", block_forecast)
                block_forecasts.append(block_forecast.cpu().data.numpy())
                outsample_ys.append(batch['outsample_y'])
                outsample_masks.append(batch['outsample_mask'])

        forecasts = np.vstack(forecasts)
        block_forecasts = np.vstack(block_forecasts)
        outsample_ys = np.vstack(outsample_ys)
        outsample_masks = np.vstack(outsample_masks)

        self.model.train()
        if return_decomposition:
            return outsample_ys, forecasts, block_forecasts, outsample_masks
        else:
            return outsample_ys, forecasts, outsample_masks




    def evaluate_performance(self, ts_loader, validation_loss_fn):
        self.model.eval()

        print("validation_loss_fn is :", validation_loss_fn)

        target, forecast, outsample_mask = self.predict(ts_loader=ts_loader)

        complete_loss = validation_loss_fn(target=target, forecast=forecast, weights=outsample_mask)

        print("complete_loss is :", complete_loss)

        self.model.train()
        return complete_loss

    def save(self, model_dir, model_id, state_dict = None):

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if state_dict is None:
            state_dict = self.model.state_dict()

        model_file = os.path.join(model_dir, f"model_{model_id}.pth")
        print('Saving model to:\n {}'.format(model_file)+'\n')
        #t.save({'model_state_dict': state_dict}, model_file)
        t.save(self, model_file)

    #@staticmethod
    def load(self, model_dir, model_id):
        model_file = os.path.join(model_dir, f"model_{model_id}.pth")
        print(model_file)
        path = Path(model_file)

        assert path.is_file(), 'No model_*.pth file found in this path!'

        print('Loading model from:\n {}'.format(model_file) + '\n')

        checkpoint = t.load(model_file, map_location=self.device)

        # 从 checkpoint 中恢复 n_x_t 和 n_x_s
        self.n_x_t = checkpoint.get('n_x_t')
        self.n_x_s = checkpoint.get('n_x_s')

        # 如果还没有实例化内部模型，则先实例化
        if not self._is_instantiated:
            block_list = self.create_stack()  # create_stack() 会用到 self.n_x_s 等参数
            self.model = NBeats(torch.nn.ModuleList(block_list)).to(self.device)
            self._is_instantiated = True

        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)


