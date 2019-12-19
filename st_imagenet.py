import argparse
import os
import re
import json
import numpy as np
from torch.optim import SGD
from tqdm import tqdm
import torch
import torchnet as tnt
from torchnet.engine import Engine
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
from collections import OrderedDict
from torch.autograd import Variable
import time

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--imagenetpath', default='/home/zagoruys/ILSVRC2012', type=str)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--teacher_params', default='', type=str)
parser.add_argument('--teacher_id', default='', type=str)

# Training options
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=100, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epoch_step', default='[30,60,90]', type=str,help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.1, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--temperature', default=4, type=float)
parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--beta', default=0, type=float)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,help='id(s) for CUDA_VISIBLE_DEVICES')

# KT methods options
parser.add_argument('--kt_method', default='#', type=str, help="at,st,kd")


def get_iterator(imagenetpath, batch_size, nthread, mode):
    imagenetpath = os.path.expanduser(imagenetpath)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("| setting up data loader...")
    if mode:
        traindir = os.path.join(imagenetpath, 'train')
        ds = ImageFolder(traindir, T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]))
    else:
        valdir = os.path.join(imagenetpath, 'val')
        ds = ImageFolder(valdir, T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]))

    return DataLoader(ds, batch_size=batch_size, shuffle=mode,num_workers=nthread, pin_memory=True)


def define_teacher(params_file):
    """
    pretrained wide-resnet-50-2: wget https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth
    """
    params = torch.load(params_file)
    for k, v in sorted(params.items()):
        # print(k, v.shape)
        params[k] = Variable(v, requires_grad=False)
    print('\nTeacher wide-resnet-50-2 Total parameters:', sum(v.numel() for v in params.values()))

    # [3,4,6,3]
    blocks = [sum([re.match('group%d.block\d+.conv0.weight' % j, k) is not None
                   for k in params.keys()]) for j in range(4)]

    # select the same architechure with wide-resnet-14-2
    # params = {k: v for k, v in params.items() if 'block0' in k or "fc" in k or k=="conv0.weight" or k=="conv0.bias"}
    # for k, v in sorted(params.items()): print(k, v.shape)
    # blocks = [1, 1, 1, 1]
    # print('\nwide-resnet-14-2 (including conv biases) Total parameters:', sum(v.numel() for v in params.values()))

    def conv2d(input, params, base, stride=1, pad=0):
        return F.conv2d(input, params[base + '.weight'], params[base + '.bias'], stride, pad)

    def group(input, params, base, stride, n):
        o = input
        for i in range(0, n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0')
            o = F.relu(o)
            o = conv2d(o, params, b_base + '1', stride=i == 0 and stride or 1, pad=1)
            o = F.relu(o)
            o = conv2d(o, params, b_base + '2')
            if i == 0:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = F.relu(o)
        return o

    def f(input, params, pr=''):
        # o = F.conv2d(input, params['conv0.weight'], params['conv0.bias'], 2, 3)
        o = conv2d(input, params, pr+'conv0', 2, 3)
        o = F.relu(o)
        o = F.max_pool2d(o, 3, 2, 1)
        o_g0 = group(o, params, pr+'group0', 1, blocks[0])
        o_g1 = group(o_g0, params, pr+'group1', 2, blocks[1])
        o_g2 = group(o_g1, params, pr+'group2', 2, blocks[2])
        o_g3 = group(o_g2, params, pr+'group3', 2, blocks[3])
        o = F.avg_pool2d(o_g3, 7, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params[pr+'fc.weight'], params[pr+'fc.bias'])
        return o, (o_g0, o_g1, o_g2, o_g3)

    return f, params


def define_student(depth, width):
    # wide-resnet-14-2, 21530792
    definitions = {14: [1, 1, 1, 1]}
    assert depth in list(definitions.keys())
    widths = [int(w * width) for w in (64, 128, 256, 512)]
    blocks = definitions[depth]
    print("student model is resnet-{}-{}".format(depth, width))

    def gen_block_params(ni, nm, no):
        return {'conv0': utils.conv_params(ni, nm, 1),
                'conv1': utils.conv_params(nm, nm, 3),
                'conv2': utils.conv_params(nm, no, 1),
                'conv_dim': utils.conv_params(ni, no, 1) if ni != no else None,
                }

    def gen_group_params(ni, nm, no, count):
        return {'block%d' % i: gen_block_params(ni if i==0 else no, nm, no) for i in range(count)}

    flat_params = OrderedDict(utils.flatten({
        'conv0': utils.conv_params(3, 64, 7),
        'group0': gen_group_params(64, widths[0], widths[0]*2, blocks[0]),
        'group1': gen_group_params(widths[0]*2, widths[1],  widths[1]*2, blocks[1]),
        'group2': gen_group_params(widths[1]*2, widths[2],  widths[2]*2, blocks[2]),
        'group3': gen_group_params(widths[2]*2, widths[3],  widths[3]*2, blocks[3]),
        'fc': utils.linear_params(widths[3]*2, 1000),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def conv2d(input, params, base, stride=1, pad=0):
        # return F.conv2d(input, params[base + '.weight'], params[base + '.bias'], stride, pad)
        return F.conv2d(input, params[base], stride=stride, padding=pad)

    def group(input, params, base, stride, n):
        o = input
        for i in range(0, n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0')
            o = F.relu(o)
            o = conv2d(o, params, b_base + '1', stride=i == 0 and stride or 1, pad=1)
            o = F.relu(o)
            o = conv2d(o, params, b_base + '2')
            if i == 0:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = F.relu(o)
        return o

    def f(input, params, mode, pr=''):
        # o = F.conv2d(input, params['conv0.weight'], params['conv0.bias'], 2, 3)
        o = conv2d(input, params, pr + 'conv0', 2, 3)
        o = F.relu(o)
        o = F.max_pool2d(o, 3, 2, 1)
        o_g0 = group(o, params, pr + 'group0', 1, blocks[0])
        o_g1 = group(o_g0, params, pr + 'group1', 2, blocks[1])
        o_g2 = group(o_g1, params, pr + 'group2', 2, blocks[2])
        o_g3 = group(o_g2, params, pr + 'group3', 2, blocks[3])
        o = F.avg_pool2d(o_g3, 7, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params[pr + 'fc.weight'], params[pr + 'fc.bias'])
        return o, (o_g0, o_g1, o_g2, o_g3)

    return f, flat_params


def main():
    st = time.time()
    opt = parser.parse_args()
    epoch_step = json.loads(opt.epoch_step)
    print('parsed options:', vars(opt))

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    epoch_step = json.loads(opt.epoch_step)

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    f_s, params_s = define_student(opt.depth, opt.width)

    if opt.teacher_id:
        assert opt.teacher_id == "resnet-50-2"
        f_t, params_t = define_teacher(opt.teacher_params)

        # init student
        # params_temp = {k: v for k, v in params_t.items() if 'block0' in k and "weight" in k or "fc" in k or k=="conv0.weight"}
        # for k, v in sorted(params_temp.items()):
        #     t = re.findall(r'(.*?).weight', k)
        #     if t!=[] and 'conv' in t[0]:
        #         assert t[0] in params_s.keys()
        #         params_s[t[0]].data=v.data
        #         print(t[0], v.shape)
        # params_s['fc.weight'].data = params_t['fc.weight'].data
        # params_s['fc.bias'].data = params_t['fc.bias'].data
        # print('\nStudent wide-resnet-14-2 Total parameters:', sum(v.numel() for v in params_temp.values())) # 21530792

        params = {'student.'+k: v for k, v in params_s.items()}
        params.update({'teacher.'+k: v for k, v in params_t.items()})
        def f(inputs, params, mode):
            y_s, g_s = f_s(inputs, params, mode, 'student.')
            with torch.no_grad():
                y_t, g_t = f_t(inputs, params, 'teacher.')
            return y_s, y_t, [utils.at_loss(x, y) for x, y in zip(g_s, g_t)]
    else:
        f, params = f_s, params_s

    params = OrderedDict((k, p.cuda().detach().requires_grad_(p.requires_grad)) for k, p in params.items())

    optimizable = [v for v in params.values() if v.requires_grad]
    def create_optimizer(opt, lr):
        # print('creating optimizer with lr = ', lr)
        return SGD(optimizable, lr, momentum=0.9, weight_decay=opt.weight_decay)

    optimizer = create_optimizer(opt, opt.lr)

    iter_train = get_iterator(opt.imagenetpath, opt.batch_size, opt.nthread, True)
    iter_test = get_iterator(opt.imagenetpath, opt.batch_size, opt.nthread, False)
    # train_size = len(iter_train.dataset)
    # test_size = len(iter_test.dataset)
    # steps_per_epoch = round(train_size / opt.batch_size)
    # total_steps = opt.epochs * steps_per_epoch
    # print("train size: {}, test size: {}, steps per epoch: {}, total steps: {}".format(train_size, test_size, steps_per_epoch, total_steps))

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    utils.print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in optimizable)
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    meters_at = [tnt.meter.AverageValueMeter() for i in range(4)]

    # # check teacher test accuracy
    if opt.teacher_id != '':
        print("begin to check teacher and student init accuracy......")
        def add_accuracy(y, targets, classacc, test_acc_top1, test_acc_top5):
            classacc.add(y, targets)
            test_acc_top1.append(classacc.value()[0])
            test_acc_top5.append(classacc.value()[1])
            classacc.reset()
            return test_acc_top1, test_acc_top5

        classacc_s = classacc_t = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
        t_test_acc_top1, t_test_acc_top5, s_test_acc_top1, s_test_acc_top5 = [], [], [], []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(iter_test):
                inputs = inputs.cuda().detach()
                targets = targets.cuda().long().detach()
                y_s, y_t, _ = utils.data_parallel(f, inputs, params, False, range(opt.ngpu))
                t_test_acc_top1, t_test_acc_top5 = add_accuracy(y_t, targets, classacc_t, t_test_acc_top1, t_test_acc_top5)
                s_test_acc_top1, s_test_acc_top5 = add_accuracy(y_s, targets, classacc_s, s_test_acc_top1, s_test_acc_top5)
        print("teacher top1 test acc: {}, teacher top5 test acc: {}".format(np.mean(t_test_acc_top1), np.mean(t_test_acc_top5)))
        print("student top1 test acc: {}, student top5 test acc: {}".format(np.mean(s_test_acc_top1), np.mean(s_test_acc_top5)))

    def h(sample):
        inputs, targets, mode = sample
        inputs = inputs.cuda().detach()
        targets = targets.cuda().long().detach()
        if opt.teacher_id != '':
            if opt.kt_method == "at":
                y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, mode, range(opt.ngpu))
                loss_groups = [v.sum() for v in loss_groups]
                [m.add(v.item()) for m,v in zip(meters_at, loss_groups)]
                return utils.distillation(y_s, y_t, targets, opt.temperature, opt.alpha) + opt.beta * sum(loss_groups), y_s
            elif opt.kt_method == "st":
                y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))
                return torch.sqrt(torch.mean((y_s - y_t) ** 2)), y_s
        else:
            y = utils.data_parallel(f, inputs, params, mode, range(opt.ngpu))[0]
            return F.cross_entropy(y, targets), y

    def log(t, state):
        torch.save(dict(params={k: v.data for k, v in params.items()},
                        optimizer=state['optimizer'].state_dict(),
                        epoch=t['epoch']),
                   os.path.join(opt.save, 'model.pt7'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])
        # if state['sample'][2]:
        #     curr_lr = 0.5 * opt.lr * (1 + np.cos(np.pi * state['t'] / total_steps))
        #     state['optimizer'] = create_optimizer(opt, curr_lr)

    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1])
        loss = state['loss'].item()
        meter_loss.add(loss)
        if state['train']:
            state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        [meter.reset() for meter in meters_at]
        state['iterator'] = tqdm(iter_train, dynamic_ncols=True)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, iter_test)
        test_acc = classacc.value()
        print(log({
            "train_loss": train_loss[0],
            "train_acc": train_acc,
            "test_loss": meter_loss.value()[0],
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
            "at_losses": [m.value() for m in meters_at],
            "kt_method": opt.kt_method,
            "curr_lr": state['optimizer'].param_groups[0]['lr'],
        }, state))
        print('==> id: %s (%d/%d), test_top1_acc: \33[91m%.2f\033[0m, test_top5_acc: \33[91m%.2f\033[0m' %
              (opt.save, state['epoch'], opt.epochs, test_acc[0], test_acc[1]))

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, iter_train, opt.epochs, optimizer)
    print("total time: {}".format(time.time()-st))


if __name__ == '__main__':
    main()
