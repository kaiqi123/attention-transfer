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
import time
from wide_resnet_50_2 import define_model

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--depth', default=-1, type=int)
parser.add_argument('--width', default=-1, type=float)
parser.add_argument('--imagenetpath', default='#', type=str)
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
parser.add_argument('--save', default='#', type=str,help='save parameters and logs in this folder')
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


def main():
    st = time.time()
    opt = parser.parse_args()
    epoch_step = json.loads(opt.epoch_step)
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 1000
    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    # f_s, params_s = define_student(opt.depth, opt.width)
    f_s, params_s = wide_resnet(opt.depth, opt.width, num_classes)

    if opt.teacher_id:
        # f_t, params_t = define_teacher(opt.teacher_params)
        with open(os.path.join('logs', opt.teacher_id, 'log.txt'), 'r') as ff:
            line = ff.readline()
            r = line.find('json_stats')
            info = json.loads(line[r + 12:])
        f_t, _ = wide_resnet(info['depth'], info['width'], num_classes)
        model_data = torch.load(os.path.join('logs', opt.teacher_id, 'model.pt7'))
        params_t = model_data['params']

        params = {'student.'+k: v for k, v in params_s.items()}
        params.update({'teacher.'+k: v for k, v in params_t.items()})
        def f(inputs, params, mode):
            y_s, g_s, out_dict_s = f_s(inputs, params, mode, 'student.')
            with torch.no_grad():
                y_t, g_t, out_dict_t = f_t(inputs, params, 'teacher.')
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
    train_size = len(iter_train.dataset)
    test_size = len(iter_test.dataset)
    steps_per_epoch = round(train_size / opt.batch_size)
    total_steps = opt.epochs * steps_per_epoch
    print("train size: {}, test size: {}, steps per epoch: {}, total steps: {}".
          format(train_size, test_size, steps_per_epoch, total_steps))

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

    if opt.teacher_id != '':
        classacc_t = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
        t_test_acc_top1, t_test_acc_top5 = [], []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(iter_test):
                inputs = inputs.cuda().detach()
                targets = targets.cuda().long().detach()
                y_t, _, _ = f_t(inputs, params, 'teacher.')
                classacc_t.add(y_t, targets)
                t_test_acc_top1.append(classacc_t.value()[0])
                t_test_acc_top5.append(classacc_t.value()[1])
                classacc_t.reset()
        print("teacher top1 test acc: {}, teacher top5 test acc: {}".
              format(np.mean(t_test_acc_top1), np.mean(t_test_acc_top5)))

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
