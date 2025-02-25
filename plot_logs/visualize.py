import os
import matplotlib.pyplot as plt
import pandas as pd
import sweeper


def plot_acc(log_name, frame, numParame, acc_type='test_acc', top_n='top_1'):
    acc = [e[0] for e in frame[acc_type]] if top_n == "top_1" else [e[1] for e in frame[acc_type]]
    label_type = acc_type + "_" + top_n
    label_name = "{}, {}={}, n_param.={}M".format(log_name, label_type, round(max(acc), 2), numParame)
    plt.plot(acc, label=label_name)
    print(len(acc), acc)
    print("{}, max_{}: {}".format(log_name, label_type, round(max(acc), 2)))


def plotLogs(dir, log_names):
    frames = [pd.DataFrame(sweeper.loadLog(dir + log + '/log.txt')) for log in log_names]

    for i, frame in enumerate(frames):
        log_name = log_names[i]
        numParame = round(float(frame["n_parameters"][0] / 1000000.), 2)
        plot_acc(log_name, frame, numParame, 'test_acc', 'top_1')
        # plot_acc(log_name, frame, numParame, 'test_acc', 'top_5')
        # plot_acc(log_name, frame, numParame, 'train_acc', 'top_1')
        # plot_acc(log_name, frame, numParame, 'train_acc', 'top_5')

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend().set_draggable(True)
    plt.show()


os.chdir(r'/home/local/ASUAD/kzhao27/Desktop/my_code/server/attention-transfer/')

# dir = "./logs/cifar10/lr_decay/bs128/complete_1gpu/"
# type = "dependent"
# filenames = [f for f in os.listdir(dir) if type in f]
# plotLogs(dir, filenames)

dir = "./logs/"
filenames = [
            "imagenet_dependent_resnet18_at",
            'imagenet_dependent_resnet18_onlyFc',
            "imagenet_independent_resnet18",
            # 'cifar10_teacher_d28w10_bs128_lrCosine',
            ]
plotLogs(dir, filenames)

# dir = "./logs/"
# filenames = [
#             "cifar10_dependent_d10w10_bs128_3reluFc_part22_updateWeightsAndBn_norm",
#             "cifar10_dependent_d10w10_bs128_3reluFc_part22_updateWeightsAndBn_norm_4gpu",
#             ]
# plotLogs(dir, filenames)

# dir = "./logs/imagenet/"
# type = ""
# filenames = [f for f in os.listdir(dir) if type in f]
# plotLogs(dir, filenames)
