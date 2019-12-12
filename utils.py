from nested_dict import nested_dict
from functools import partial
import torch
from torch.nn.init import kaiming_normal_
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.nn.functional as F


def norm_rmse_loss_cosine(s, t_dict):
    def cosineSimilarity(x1, x2):
        x1_sqrt = torch.sqrt(torch.sum(x1 ** 2))
        x2_sqrt = torch.sqrt(torch.sum(x2 ** 2))
        return torch.div(torch.sum(x1 * x2), max(x1_sqrt * x2_sqrt, 1e-8))

    # print(s.shape)
    s_norm = F.normalize(s, p=2, dim=0)
    t_norm_dict = {}
    simi_dict = {}
    for k, v in t_dict.items():
        if s_norm.shape == v.shape:
            v_norm = F.normalize(v, p=2, dim=0)
            t_norm_dict[k] = v_norm
            simi_dict[k] = cosineSimilarity(s_norm, v_norm)
            # print("t_dict: {}, {}".format(k, v.shape))
    max_key = sorted(simi_dict.items(), key=lambda x: x[1], reverse=True).pop()[0]
    t_norm = t_norm_dict[max_key]
    # for k, v in t_norm_dict.items(): print("t_norm_dict: {}, {}".format(k, v.shape))
    # for k, v in simi_dict.items(): print("simi_dict: {}, {}".format(k, v))
    # print("max similarity: {}, {}, {}".format(max_key, simi_dict[max_key], t_norm.shape))
    return torch.sqrt(torch.mean((s_norm-t_norm)**2))


def st_3relu_loss_cosine(out_dict_s, out_dict_t):
    # out_dict_s = {k: v for k, v in out_dict_s.items() if 'relu2' not in k}
    # for key, value in sorted(out_dict_s.items()): print(key, value.shape)
    # for key, value in sorted(out_dict_t.items()): print(key, value.shape)
    loss_group1 = norm_rmse_loss_cosine(out_dict_s['student.group1.block0.relu0'], out_dict_t)
    loss_group2 = norm_rmse_loss_cosine(out_dict_s['student.group2.block0.relu0'], out_dict_t)
    loss_last_relu = norm_rmse_loss_cosine(out_dict_s['student.last_relu'], out_dict_t)
    return [loss_group1, loss_group2, loss_last_relu]


def norm_rmse_loss(s, t):
    assert s.size() == t.size()
    s_norm = F.normalize(s, p=2, dim=0)
    t_norm = F.normalize(t, p=2, dim=0)
    return torch.sqrt(torch.mean((s_norm-t_norm)**2))


def st_3relu_loss(out_dict_s, out_dict_t):
    # out_dict_s = {k: v for k, v in out_dict_s.items() if 'relu2' not in k}
    # for key, value in sorted(out_dict_s.items()): print(key, value.shape)
    # for key, value in sorted(out_dict_t.items()): print(key, value.shape)
    loss_group1 = norm_rmse_loss(out_dict_s['student.group1.block0.relu0'], out_dict_t['teacher.group1.block0.relu0'])
    loss_group2 = norm_rmse_loss(out_dict_s['student.group2.block0.relu0'], out_dict_t['teacher.group2.block0.relu0'])
    loss_last_relu = norm_rmse_loss(out_dict_s['student.last_relu'], out_dict_t['teacher.last_relu'])
    return [loss_group1, loss_group2, loss_last_relu]


def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}


def data_parallel(f, input, params, mode, device_ids, output_device=None):
    device_ids = list(device_ids)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode) for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(sorted(params.items())):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True
