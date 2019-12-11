class Engine(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, iterator, maxepoch, optimizer_list):
        state = {
            'network': network,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer_list': optimizer_list,
            'epoch': 0,
            't': 0,
            'train': True,
        }

        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                # def closure():
                #     loss, output = state['network'](state['sample'])
                #     state['output'] = output
                #     state['loss'] = loss
                #     loss.backward()
                #     self.hook('on_forward', state)
                #     # to free memory in save_for_backward
                #     state['output'] = None
                #     state['loss'] = None
                #     return loss
                # state['optimizer'].zero_grad()
                # state['optimizer'].step(closure)

                def closure():
                    loss_list, output = state['network'](state['sample'])
                    loss = loss_list[i]
                    loss.backward()
                    if i == len(loss_list)-1:
                        state['loss'] = loss
                        state['output'] = output
                        self.hook('on_forward', state)
                        state['output'] = None
                        state['loss'] = None
                    return loss

                for optimzer in state['optimizer_list']:
                    optimzer.zero_grad()

                for i, optimzer in enumerate(state['optimizer_list']):
                    optimzer.step(closure=closure)

                # loss_list, output = state['network'](state['sample'])
                # assert len(state['optimizer_list']) == len(loss_list)
                # for i, optimzer in enumerate(state['optimizer_list']):
                #     optimzer.zero_grad()
                #     loss_list[i].backward(retain_graph=True)
                #     optimzer.step()
                #
                # state['output'] = output
                # state['loss'] = loss_list[-1]
                # self.hook('on_forward', state)
                # state['output'] = None
                # state['loss'] = None

                self.hook('on_update', state)
                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state


    def test(self, network, iterator):
        state = {
            'network': network,
            'iterator': iterator,
            't': 0,
            'train': False,
        }

        self.hook('on_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_sample', state)

            # def closure():
            #     loss, output = state['network'](state['sample'])
            #     state['output'] = output
            #     state['loss'] = loss
            #     self.hook('on_forward', state)
            #     # to free memory in save_for_backward
            #     state['output'] = None
            #     state['loss'] = None
            def closure():
                loss_list, output = state['network'](state['sample'])
                loss = loss_list[-1]
                state['output'] = output
                state['loss'] = loss
                self.hook('on_forward', state)
                state['output'] = None
                state['loss'] = None

            closure()
            state['t'] += 1
        self.hook('on_end', state)
        return state