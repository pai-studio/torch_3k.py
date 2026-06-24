from .optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, parameters, lr=1e-2):
        Optimizer.__init__(self, parameters, defaults={'lr': lr})
        self._refresh_attributes_from_first_group()

    def _refresh_attributes_from_first_group(self):
        self.lr = self.param_groups[0].get('lr', self.lr if hasattr(self, 'lr') else 1e-2)

    def update_one(self, parameter, group):
        #print(f'{type(self.lr)=}')
        #print(f'{type(parameter.grad.data)=}')
        #print(f'{type(parameter.data)=}')
        # parameter.data -= self.lr * parameter.grad.data
        parameter.data = parameter.data - group['lr'] * parameter.grad.data
        #print(parameter)

    def state_dict(self):
        return {
            'state': {},
            'param_groups': self._state_param_groups(),
        }

    def load_state_dict(self, state_dict):
        self._load_param_groups(state_dict)
