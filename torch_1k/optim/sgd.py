from .optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, parameters, lr=1e-2):
        Optimizer.__init__(self, parameters)
        self.lr = lr

    def update_one(self, parameter):
        #print(f'{type(self.lr)=}')
        #print(f'{type(parameter.grad.data)=}')
        #print(f'{type(parameter.data)=}')
        # parameter.data -= self.lr * parameter.grad.data
        parameter.data = parameter.data - self.lr * parameter.grad.data
        #print(parameter)

    def state_dict(self):
        return {
            'state': {},
            'param_groups': [self._param_group(lr=self.lr)],
        }

    def load_state_dict(self, state_dict):
        group = self._validate_state_dict(state_dict)
        self.lr = group.get('lr', self.lr)
