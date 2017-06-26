#####################################
## Keras classes ported to Lasagne ##
#####################################

import numpy as np

class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    # Example
        ```python
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.001)
        model.fit(X_train, Y_train, callbacks=[reduce_lr])
        ```
    # Arguments
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    #def __init__(self, monitor='val_loss', factor=0.1, patience=10,
    #             verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0):
    def __init__(self, learning_rate, factor=0.1, patience=10,
                 verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0):
        #self.monitor = monitor
        self.learning_rate = learning_rate # store the theano variable
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, monitor, epoch, logs=None):
        logs = logs or {}
        #logs['lr'] = K.get_value(self.model.optimizer.lr)
        #current = logs.get(self.monitor)
        current = monitor
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires %s available!' %
                          self.monitor, RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    old_lr = float(self.learning_rate.get_value())
                    if old_lr > self.min_lr + self.lr_epsilon:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.learning_rate.set_value(new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0

if __name__ == '__main__':
    import theano
    lr = theano.shared(0.01)
    cb = ReduceLROnPlateau(lr,verbose=1)
    cb.on_train_begin()
    cb.on_epoch_end(1.45, 1)
    cb.on_epoch_end(1.43, 2)
    cb.on_epoch_end(1.41, 3)
    for i in range(1, 10):
        cb.on_epoch_end(1.41, 3+i)
    print lr.get_value()
