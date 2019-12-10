import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torchvision.transforms as T
from torchvision import datasets
import torch.nn.functional as F
import torchnet as tnt
# from torchnet.engine import Engine
from engine_utils import Engine
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import utils
import time
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--teacher_id', default='', type=str)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str, help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--randomcrop_pad', default=4, type=float)
parser.add_argument('--temperature', default=4, type=float)
parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--beta', default=0, type=float)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str, help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int, help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# KT methods options
parser.add_argument('--kt_method', default='#', type=str, help="at,st,kd")

def create_dataset(opt, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    if train:
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])
    return getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=True, transform=transform)


def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_block_params(ni, no):
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def block(x, params, base, mode, stride, out_dict):
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            o = z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            o = z + x
        out_dict[base+'.relu1'] = o1
        out_dict[base+'.relu2'] = o2
        return o, out_dict

    def group(o, params, base, mode, stride, out_dict):
        for i in range(n):
            o, out_dict = block(o, params, '{}.block{}'.format(base, i), mode, stride if i == 0 else 1, out_dict)
        return o, out_dict

    def f(input, params, mode, base=''):
        out_dict = {}
        x = F.conv2d(input, params['{}conv0'.format(base)], padding=1)
        g0, out_dict = group(x, params, '{}group0'.format(base), mode, 1, out_dict)
        g1, out_dict = group(g0, params, '{}group1'.format(base), mode, 2, out_dict)
        g2, out_dict = group(g1, params, '{}group2'.format(base), mode, 2, out_dict)
        o = F.relu(utils.batch_norm(g2, params, '{}bn'.format(base), mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['{}fc.weight'.format(base)], params['{}fc.bias'.format(base)])
        return o, (g0, g1, g2), out_dict

    return f, flat_params

def main():
    st_total = time.time()
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)
    train_size = len(train_loader.dataset); test_size = len(test_loader.dataset)
    steps_per_epoch = round(train_size / opt.batch_size)
    total_steps = opt.epochs * steps_per_epoch
    print("train size: {}, test size: {}, steps per epoch: {}, total steps: {}".format(train_size, test_size, steps_per_epoch, total_steps))

    # deal with student first
    f_s, params_s = resnet(opt.depth, opt.width, num_classes)
    print(type(f_s), type(params_s))

    # deal with teacher
    if opt.teacher_id:
        with open(os.path.join('logs', opt.teacher_id, 'log.txt'), 'r') as ff:
            line = ff.readline()
            r = line.find('json_stats')
            info = json.loads(line[r + 12:])
        f_t, _ = resnet(info['depth'], info['width'], num_classes)
        model_data = torch.load(os.path.join('logs', opt.teacher_id, 'model.pt7'))
        params_t = model_data['params']

        # merge teacher and student params
        params = {'student.' + k: v for k, v in params_s.items()}
        for k, v in params_t.items():
            params['teacher.' + k] = v.detach().requires_grad_(False)

        if opt.kt_method == "at":
            def f(inputs, params, mode):
                y_s, g_s, out_dict_s = f_s(inputs, params, mode, 'student.')
                with torch.no_grad():
                    y_t, g_t, out_dict_t = f_t(inputs, params, False, 'teacher.')
                return y_s, y_t, [utils.at_loss(x, y) for x, y in zip(g_s, g_t)]
        elif opt.kt_method == "st":
            # print('Set st:')
            # print(relu_out_s.keys(), relu_out_t.keys())
            # relu_out_s = {'student.' + k: v for k, v in relu_out_s.items()}
            # relu_out_t = {'teacher.' + k: v for k, v in relu_out_t.items()}
            # for key, value in relu_out_s.items(): print(key, value)
            # for key, value in relu_out_t.items(): print(key, value)
            def f(inputs, params, mode):
                y_s, g_s, out_dict_s = f_s(inputs, params, mode, 'student.')
                with torch.no_grad():
                    y_t, g_t, out_dict_t = f_t(inputs, params, False, 'teacher.')
                for key, value in sorted(out_dict_s.items()): print(key, value.shape)
                for key, value in sorted(out_dict_t.items()): print(key, value.shape)
                return y_s, y_t, [utils.at_loss(x, y) for x, y in zip(g_s, g_t)]
        else:
            raise EOFError("Not found kt method.")

    else:
        f, params = f_s, params_s

    def create_optimizer(opt, lr):
        # print('creating optimizer with lr = ', lr)
        return SGD((v for v in params.values() if v.requires_grad), lr, momentum=0.9, weight_decay=opt.weight_decay)

    optimizer = create_optimizer(opt, opt.lr)

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

    n_parameters = sum(p.numel() for p in list(params_s.values()))
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        inputs = utils.cast(sample[0], opt.dtype).detach()
        targets = utils.cast(sample[1], 'long')
        if opt.teacher_id != '':
            if opt.kt_method == "at":
                y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))
                loss_groups = [v.sum() for v in loss_groups]
                [m.add(v.item()) for m, v in zip(meters_at, loss_groups)]
                return utils.distillation(y_s, y_t, targets, opt.temperature, opt.alpha) + opt.beta * sum(loss_groups), y_s
            elif opt.kt_method == "st":
                y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))
                # loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
                # return torch.sqrt(torch.nn.MSELoss(y_s, y_t) + 1e-6), y_s
                return torch.sqrt(torch.mean((y_s-y_t)**2)), y_s
        else:
            y = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))[0]
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
        # print(len(state['sample']), state['sample'][0].size(), state['sample'][1].size(), state['sample'][2])

    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(state['loss'].item())

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        [meter.reset() for meter in meters_at]
        state['iterator'] = tqdm(train_loader)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        train_loss = meter_loss.mean
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, test_loader)

        test_acc = classacc.value()
        print(log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": meter_loss.mean,
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
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
    engine.train(h, train_loader, opt.epochs, optimizer)

    print("total time (h): {}".format((time.time()-st_total)/3600.))


if __name__ == '__main__':
    main()
