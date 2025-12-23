"""
GPU-enabled Broad Learning System (BLS) using CuPy.
This is a minimal port of broadnet_enhmap for fit/predict on GPU.
Incremental APIs are omitted for simplicity.
"""

from __future__ import annotations

import cupy as cp  # type: ignore
from typing import List


def _show_accuracy(pred, label):
    pred = cp.asarray(pred).reshape(-1)
    label = cp.asarray(label).reshape(-1)
    return float((pred == label).sum() / label.size)


class scalerGPU:
    def __init__(self, use_internal_scaler: bool = True):
        self._mean = None
        self._std = None
        self._use = use_internal_scaler

    def fit_transform(self, traindata):
        if not self._use:
            return traindata
        self._mean = traindata.mean(axis=0, keepdims=True)
        self._std = traindata.std(axis=0, keepdims=True) + 1e-8
        return (traindata - self._mean) / self._std
    
    def transform(self, testdata):
        if not self._use:
            return testdata
        return (testdata - self._mean) / self._std


class node_generatorGPU:
    def __init__(self, whiten: bool = False):
        self.Wlist: List[cp.ndarray] = []
        self.blist: List[cp.ndarray] = []
        self.nonlinear = None
        self.whiten = whiten

    def sigmoid(self, data):
        return 1.0 / (1 + cp.exp(-data))

    def linear(self, data):
        return data

    def tanh(self, data):
        return cp.tanh(data)

    def relu(self, data):
        return cp.maximum(data, 0)

    def leakyrelu(self, data, alpha=0.01):
        return cp.where(data > 0, data, alpha * data)

    def orth(self, W):
        # Gram-Schmidt on GPU
        for i in range(W.shape[1]):
            w = W[:, i : i + 1]
            for j in range(i):
                wj = W[:, j : j + 1]
                w = w - (w.T @ wj)[0, 0] * wj
            w = w / cp.sqrt((w.T @ w)[0, 0] + 1e-8)
            W[:, i : i + 1] = w
        return W

    def generator(self, shape, times):
        for _ in range(times):
            W = 2 * cp.random.random(size=shape, dtype=cp.float32) - 1
            if self.whiten:
                W = self.orth(W)
            b = 2 * cp.random.random(dtype=cp.float32) - 1
            yield (W, b)

    def generator_nodes(self, data, times, batchsize, nonlinear):
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]
        self.nonlinear = {
            "linear": self.linear,
            "sigmoid": self.sigmoid,
            "tanh": self.tanh,
            "relu": self.relu,
            "leakyrelu": self.leakyrelu,
        }[nonlinear]
        nodes = self.nonlinear(data @ self.Wlist[0] + self.blist[0])
        for i in range(1, len(self.Wlist)):
            nodes = cp.column_stack((nodes, self.nonlinear(data @ self.Wlist[i] + self.blist[i])))
        return nodes

    def transform(self, testdata):
        testnodes = self.nonlinear(testdata @ self.Wlist[0] + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = cp.column_stack((testnodes, self.nonlinear(testdata @ self.Wlist[i] + self.blist[i])))
        return testnodes

    def update(self, otherW, otherb):
        self.Wlist += otherW
        self.blist += otherb


class broadnet_enhmap_gpu:
    def __init__(
        self,
        maptimes=10,
        enhencetimes=10,
        traintimes=100,
        map_function="linear",
        enhence_function="linear",
        batchsize="auto",
        acc=1,
        mapstep=1,
        enhencestep=1,
        reg=0.001,
        map_whiten: bool = False,
        use_internal_scaler: bool = True,
        dtype: str = "float32",
    ):
        self._maptimes = maptimes
        self._enhencetimes = enhencetimes
        self._batchsize = batchsize
        self._traintimes = traintimes
        self._acc = acc
        self._mapstep = mapstep
        self._enhencestep = enhencestep
        self._reg = reg
        self._map_function = map_function
        self._enhence_function = enhence_function

        self.W = None
        self.pesuedoinverse = None
        self.num_classes = None
        self._dtype = cp.float32 if dtype == "float32" else cp.float64

        self.normalscaler = scalerGPU(use_internal_scaler=use_internal_scaler)
        self.mapping_generator = node_generatorGPU(whiten=map_whiten)
        self.enhence_generator = node_generatorGPU(whiten=True)
        self.local_mapgeneratorlist: List[node_generatorGPU] = []
        self.local_enhgeneratorlist: List[node_generatorGPU] = []

    def _one_hot(self, labels, fit: bool = False):
        labels = cp.asarray(labels).astype(cp.int32).reshape(-1)
        if fit or self.num_classes is None:
            self.num_classes = int(labels.max().get()) + 1
        eye = cp.eye(self.num_classes, dtype=self._dtype)
        return eye[labels]

    def fit(self, oridata, orilabel):
        oridata = cp.asarray(oridata, dtype=self._dtype)
        orilabel = cp.asarray(orilabel)
        if self._batchsize == "auto":
            self._batchsize = oridata.shape[1]
        data = self.normalscaler.fit_transform(oridata)
        label = self._one_hot(orilabel, fit=True)

        mappingdata = self.mapping_generator.generator_nodes(
            data, self._maptimes, self._batchsize, self._map_function
        )
        enhencedata = self.enhence_generator.generator_nodes(
            mappingdata, self._enhencetimes, self._batchsize, self._enhence_function
        )
        inputdata = cp.column_stack((mappingdata, enhencedata))

        self.pesuedoinverse = self.pinv(inputdata)
        self.W = self.pesuedoinverse @ label

        Y = self.predict(oridata)
        accuracy, i = self.accuracy(Y, orilabel), 0
        while i < self._traintimes and accuracy < self._acc:
            Y = self.adding_predict(oridata, orilabel, self._mapstep, self._enhencestep, self._batchsize)
            accuracy = self.accuracy(Y, orilabel)
            i += 1

    def pinv(self, A):
        return cp.linalg.inv(self._reg * cp.eye(A.shape[1], dtype=self._dtype) + A.T @ A) @ A.T

    def decode(self, Y_onehot):
        return cp.argmax(Y_onehot, axis=1)

    def accuracy(self, predictlabel, label):
        return _show_accuracy(predictlabel, label)

    def predict(self, testdata):
        testdata = cp.asarray(testdata, dtype=self._dtype)
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)
        preds = self.decode(test_inputdata @ self.W)
        return cp.asnumpy(preds)

    def transform(self, data):
        mappingdata = self.mapping_generator.transform(data)
        enhencedata = self.enhence_generator.transform(mappingdata)
        inputdata = cp.column_stack((mappingdata, enhencedata))
        for elem1, elem2 in zip(self.local_mapgeneratorlist, self.local_enhgeneratorlist):
            inputdata = cp.column_stack((inputdata, elem1.transform(data)))
            inputdata = cp.column_stack((inputdata, elem2.transform(mappingdata)))
        return inputdata

    def adding_nodes(self, data, label, mapstep=1, enhencestep=1, batchsize="auto"):
        if batchsize == "auto":
            batchsize = data.shape[1]

        mappingdata = self.mapping_generator.transform(data)
        inputdata = self.transform(data)

        localmap_generator = node_generatorGPU(whiten=self.mapping_generator.whiten)
        extramap_nodes = localmap_generator.generator_nodes(data, mapstep, batchsize, self._map_function)
        localenhence_generator = node_generatorGPU()
        extraenh_nodes = localenhence_generator.generator_nodes(mappingdata, enhencestep, batchsize, self._map_function)
        extra_nodes = cp.column_stack((extramap_nodes, extraenh_nodes))

        D = self.pesuedoinverse @ extra_nodes
        C = extra_nodes - inputdata @ D
        if (C == 0).any():
            BT = self.pinv(C)
        else:
            BT = cp.linalg.inv((D.T @ D + cp.eye(D.shape[1], dtype=cp.float32))) @ D.T @ self.pesuedoinverse

        self.W = cp.row_stack((self.W - D @ BT @ label, BT @ label))
        self.pesuedoinverse = cp.row_stack((self.pesuedoinverse - D @ BT, BT))
        self.local_mapgeneratorlist.append(localmap_generator)
        self.local_enhgeneratorlist.append(localenhence_generator)

    def adding_predict(self, data, label, mapstep=1, enhencestep=1, batchsize="auto"):
        data = cp.asarray(data, dtype=self._dtype)
        label = self._one_hot(label, fit=False)
        data = self.normalscaler.transform(data)
        self.adding_nodes(data, label, mapstep, enhencestep, batchsize)
        test_inputdata = self.transform(data)
        preds = self.decode(test_inputdata @ self.W)
        return cp.asnumpy(preds)


__all__ = ["broadnet_enhmap_gpu"]
