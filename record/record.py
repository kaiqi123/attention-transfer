# part1
def closure():
    loss_list, output = state['network'](state['sample'])
    loss = loss_list[i]
    loss.backward()
    if i == len(loss_list) - 1:
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

# part2
loss_list, output = state['network'](state['sample'])
assert len(state['optimizer_list']) == len(loss_list)
for optimzer in state['optimizer_list']:
    optimzer.zero_grad()

for i, optimzer in enumerate(state['optimizer_list']):
    # optimzer.zero_grad()
    retain_graph = False if i == len(loss_list) - 1 else True
    loss_list[i].backward(retain_graph=retain_graph)
    optimzer.step()

state['output'] = output
state['loss'] = loss_list[-1]
self.hook('on_forward', state)
state['output'] = None
state['loss'] = None

# part3
loss_list, output = state['network'](state['sample'])
assert len(state['optimizer_list']) == len(loss_list)
for optimzer in state['optimizer_list']:
    optimzer.zero_grad()

loss = loss_list[0] + loss_list[1] + loss_list[2] + loss_list[3]
loss.backward()
for i, optimzer in enumerate(state['optimizer_list']):
    optimzer.step()

state['output'] = output
state['loss'] = loss_list[-1]
self.hook('on_forward', state)
state['output'] = None
state['loss'] = None