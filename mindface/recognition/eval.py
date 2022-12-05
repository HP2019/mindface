"""
Evaluation of lfw, calfw, cfp_fp, agedb_30, cplfw.
"""
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import datetime
import os
import pickle
from io import BytesIO

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy import interpolate
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

from .models import iresnet50, iresnet100, get_mbf


class LFold:
    """
    LFold.
    """
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        """
        split
        """
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    """
    Calculate roc.
    """
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrofpairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrofpairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    """
    Calculate acc.
    """
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    """
    Calculate val.
    """
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrofpairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrofpairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f_train = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f_train(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    """
    Calculate val, far.
    """
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    """
    Evaluate.
    """
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def load_bin(path, image_size):
    """
    Load evalset of .bin
    """
    try:
        with open(path, 'rb') as file:
            bins, issame_list = pickle.load(file)  # py2
    except UnicodeDecodeError as _:
        with open(path, 'rb') as file:
            bins, issame_list = pickle.load(file, encoding='bytes')  # py3
    data_list = []
    for _ in [0, 1]:
        data = np.zeros(
            (len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        bin_idx = bins[idx]
        img = plt.imread(BytesIO(bin_idx), "jpg")
        img = np.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = np.flip(img, axis=2)
            data_list[flip][idx][:] = img
    return data_list, issame_list


def test(data_set, backbone, batch_size, nfolds=10):
    """
    Test.
    """
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for data in data_list:
        embeddings = None
        b_a = 0
        while b_a < data.shape[0]:
            b_b = min(b_a + batch_size, data.shape[0])
            count = b_b - b_a
            num_data = data[b_b - batch_size: b_b]

            time0 = datetime.datetime.now()
            img = ((num_data / 255) - 0.5) / 0.5
            net_out = backbone(ms.Tensor(img, ms.float32))
            net_embeddings = net_out.asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], net_embeddings.shape[1]))
            embeddings[b_a:b_b, :] = net_embeddings[(batch_size - count):, :]
            b_a = b_b
        embeddings_list.append(embeddings)
    net_xnorm = 0.0
    net_xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            net_em = embed[i]
            net_norm = np.linalg.norm(net_em)
            net_xnorm += net_norm
            net_xnorm_cnt += 1
    net_xnorm /= net_xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    _, _, acc, _, _, _ = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc1 = np.mean(acc)
    std1 = np.std(acc)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, _, _, _ = evaluate(
        embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, net_xnorm, embeddings_list

def face_eval(model_name, ckpt_url, eval_url, num_features=512,
        target='lfw,cfp_fp,agedb_30,calfw,cplfw',
        device_id=0, device_target="GPU", batch_size=64, nfolds=10
    ):
    """
    The eval of arcface.

    Args:
        model_name (String): The name of backbone.
        ckpt_url (String): The The path of .ckpt
        eval_url (String): The The path of saved results.
        target (String): The eval datasets. Default: 'lfw,cfp_fp,agedb_30,calfw,cplfw'.
        device_id (Int): The id of eval device. Default: 0.
        device_target (String): The device target. Default: "GPU".
        batch_size (Int): The batch size of dataset. Default: 64.
        nfolds (Int): The eval folds. Default: 10.

    Examples:
        >>> model_name = "iresnet50"
        >>> ckpt_url = "/path/to/eval/ArcFace.ckpt"
        >>> eval_url = "/path/to/eval"
        >>> face_eval(model_name, ckpt_url, eval_url)
    """
    context.set_context(device_id=device_id, mode=context.GRAPH_MODE,
                        device_target=device_target)
    image_size = [112, 112]
    time0 = datetime.datetime.now()

    if model_name == "iresnet50":
        model = iresnet50(num_features=num_features)
    elif model_name == "iresnet100":
        model = iresnet100(num_features=num_features)
    elif model_name == "mobilefacenet":
        model = get_mbf(num_features=num_features)
    else:
        raise NotImplementedError

    param_dict = load_checkpoint(ckpt_url)
    load_param_into_net(model, param_dict)
    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())

    ver_list = []
    ver_name_list = []
    for name in target.split(','):
        path = os.path.join(eval_url, name + ".bin")
        if os.path.exists(path):
            print('loading.. ', name)
            data_set = load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)

    length = len(ver_list)
    for i in range(length):
        acc1, std1, acc2, std2, xnorm, _ = test(
            ver_list[i], model, batch_size, nfolds)
        print(f"[{ver_name_list[i]}]XNorm: {xnorm}")
        print(f"[{ver_name_list[i]}]Accuracy: {acc1}+-{std1}")
        print(f"[{ver_name_list[i]}]Accuracy-Flip: {acc2}+-{std2}")
