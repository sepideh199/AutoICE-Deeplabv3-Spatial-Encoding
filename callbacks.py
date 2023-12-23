import numpy as np

class EarlyStopper:
    """Early stopping callback 
    modified from 
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    Args:
        mode: min or max
        patience: number of epochs the trainer accepts divergence in metrics
        min_delta: smallest value to be considered 
    """
    def __init__(self, mode='min', patience=10, min_delta=1e-8, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

        self.mode = 'min'
        self.verbose = verbose
        
        self.best_metric = np.inf if mode=='min' else -np.inf

    def early_stop(self, current_metric):
        if self.mode == 'min':
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.counter = 0
            elif current_metric > (self.best_metric + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    if self.verbose:
                        print(f'Signaling model to stop training, best metric was {self.best_metric}')
                    return True

        elif self.mode == 'max':
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.counter = 0
            elif current_metric < (self.best_metric - self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    if self.verbose:
                        print(f'Signaling model to stop training, best metric was {self.best_metric}')
                    return True

        return False
