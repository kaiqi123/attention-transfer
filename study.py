import numpy as np
import tensorflow as tf
import torch.nn.functional as F


# l2 norm by tensorflow and pytorch
def l2_norm():
    data = np.arange(72).reshape(4, 3, 3, 2)
    data = np.matrix([[0., 1.], [2., 3.]])
    print(data)
    input_data = tf.convert_to_tensor(data, tf.float64)
    output = tf.nn.l2_normalize(input_data, axis=0)
    print(output)
    with tf.Session() as sess:
        print(sess.run(output))

    x = torch.Tensor(data)
    x = F.normalize(x, p=2, dim=0)
    print(x)


# cosine similarity by tensorflow and pytorch
def cosine_similarity():
    def cosineSimilarity(data1, data2):
        with tf.variable_scope("cosine"):
            product12 = tf.reduce_sum(tf.multiply(data1, data2))
            data1_sqrt = tf.sqrt(tf.reduce_sum(tf.square(data1)))
            data2_sqrt = tf.sqrt(tf.reduce_sum(tf.square(data2)))
            cosine = tf.divide(product12, tf.multiply(data1_sqrt, data2_sqrt))
        return product12, cosine

    data1 = np.matrix([[0., 1.], [2., 3.]])
    data2 = np.matrix([[0., 1.], [10., 99.]])
    # data1 = np.arange(72).reshape(4, 3, 3, 2)
    # data2 = np.arange(72).reshape(4, 3, 3, 2)
    print(data1)
    print(data2)
    x_tf1 = tf.convert_to_tensor(data1, tf.float64)
    x_tf2 = tf.convert_to_tensor(data2, tf.float64)
    product12, c = cosineSimilarity(x_tf1, x_tf2)
    with tf.Session() as sess:
        print(sess.run([product12, c]))

    def cosineSimilarity(x1, x2):
        x1_sqrt = torch.sqrt(torch.sum(x1 ** 2))
        x2_sqrt = torch.sqrt(torch.sum(x2 ** 2))
        return torch.div(torch.sum(x1 * x2), max(x1_sqrt * x2_sqrt, 1e-8))

    x1 = torch.Tensor(data1)
    x2 = torch.Tensor(data2)
    c1 = F.cosine_similarity(x1, x2, dim=0)
    c2 = cosineSimilarity(x1, x2)
    print(c1, c1.size(), c2)