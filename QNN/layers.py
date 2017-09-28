import numpy as np
import math
from im2col import im2col_indices

def convert(qnn):
  new_qnn = []
  for L in qnn:
    if L.get_type() == "FullyConnectedLayer":
      Wnew = np.asarray(L.W, dtype=np.int8)
      new_qnn += [QNNFullyConnectedLayer(Wnew)]
    elif L.get_type() == "BipolarThresholdingLayer":
      new_qnn += [QNNBipolarThresholdingLayer(L.thresholds)]
    elif L.get_type() == "ThresholdingLayer":
      new_qnn += [QNNThresholdingLayer(L.thresholds)]
    elif L.get_type() == "LinearLayer":
      new_qnn += [QNNScaleShiftLayer(L.A, L.B)]
    elif L.get_type() == "PoolingLayer":
      new_qnn += [QNNPoolingLayer(L.idim, L.chans, L.k, L.s, L.poolFxn)]
    elif L.get_type() == "ConvolutionLayer":
      Wnew = L.W.reshape((L.ofm, L.ifm, L.k, L.k))
      Wnew = Wnew.astype(np.int8)
      new_qnn += [QNNConvolutionLayer(Wnew, L.idim, L.pad, L.stride, L.padVal)]
    elif L.get_type() == "SoftmaxLayer":
      new_qnn += [QNNSoftmaxLayer()]
    elif L.get_type() == "ReLULayer":
      new_qnn += [QNNReLULayer()]
    else:
     raise Exception("Unrecognized layer type")
  return new_qnn

# Quick and easy forward path definitions for a few layer types
# Despite the QNN in the class names, these layers will work for any numpy-supported
# data type, since the internals are mostly implemented with numpy function calls.
# Assumptions:

# 1. Data layout for the input image and all intermediate activations is [channels][rows][columns] in C style layout, that is to say, looks like this if laid out as a 1D array,
# assuming three columns per row, two rows per channel, two channels img[2][2][3]:
# pixel pixel pixel pixel pixel pixel pixel pixel pixel pixel pixel pixel 
# ^                 ^                 ^                 ^
# row 0             row 1             row 0             row 1 
# ^                                   ^
# channel 0                           channel 1

# 2. All data going into and going between layers is exchanged as a flattened numpy vector (a one-dimensional array). It is the next layer's responsibility to reshape the input data as needed.

# 3. Only a single image per execution (batch size = 1).

def predict(qnn, input_img):
  "Predict the class of input_img using a quantized neural network."
  activations = input_img
  for layer in qnn:
    activations = layer.execute(activations)
  return activations

class QNNLayer(object):
  "Base class for all layer types."
  
  def layerType(self):
    "Return the layer class name as a string."
    return self.__class__.__name__
    
  def execute(self, v):
    "Forward-propagate given flat vector v through this layer."
    return v

class QNNFullyConnectedLayer(QNNLayer):
    """
    Fully-connected network layers as matrix-vector multiplication.
    Note that bias is not implemented, this can be done by adding a LinearLayer
    with A=1 B=bias following the QNNFullyConnectedLayer.
    """
    def __init__(self, W):
        self.W = W

    def execute(self, v):
        return np.dot(self.W, v)

class QNNThresholdingLayer(QNNLayer):
    "Given a set of thresholds, return the number of thresholds crossed."
    def __init__(self, thresholds):
        # we expect the thresholds array in the following format:
        # thresholds = [levels][channels]
        if thresholds.ndim == 1:
            self.thresholds = thresholds.reshape((len(thresholds),-1))
        elif thresholds.ndim == 2:
            self.thresholds = thresholds
        else:
            raise Exception("Thresholds array must be 1- or 2-dimensional")

    def execute(self, v):
        # interpret as multi-channel image, where the number of channels is
        # decided as the number of threshold channels
        vr = v.reshape((self.thresholds.shape[1], -1))
        ret = np.zeros(vr.shape, dtype=np.int)
        for t in self.thresholds:
            for c in range(self.thresholds.shape[1]):
                ret[c] += map(lambda x: 1 if x == True else 0, vr[c] >= t[c])
        return ret.flatten()

class QNNBipolarThresholdingLayer(QNNThresholdingLayer):
    "A 1-level QNNThresholdingLayer that returns -1 and +1 instead of 0 and 1."
    def __init__(self, thresholds):
        super(QNNBipolarThresholdingLayer, self).__init__(thresholds)
        if self.thresholds.shape[0] != 1:
            raise Exception("BipolarThresholdingLayer can only have one level")

    def execute(self, v):
        # just the base implementation, but scaled by 2x-1 such that the output
        # is -1, +1 instead of 0, 1. this could have been done with a following
        # LinearLayer, but this way we keep the bipolar thresholding as a stand-
        # alone operation.
        ret = super(QNNBipolarThresholdingLayer, self).execute(v)
        return 2*ret - 1

class QNNScaleShiftLayer(QNNLayer):
    "Using 1D vectors A and B, apply Ax+B to incoming x."
    def __init__(self, A, B):
        if A.shape != B.shape:
            raise Exception("QNNScaleShiftLayer A and B shapes do not match")
        if A.ndim != 1:
            raise Exception("QNNScaleShiftLayer needs 1D vectors as parameters.")
        self.A = A
        self.B = B

    def execute(self, v):
        # the outermost dimension is the channel dimension
        # reshape as inner dimension to apply transform
        vr = v.reshape((self.A.shape[0], -1)).transpose()
        return (self.A*vr+self.B).transpose().flatten()

class QNNPaddingLayer(QNNLayer):
    "A layer that adds padding around the edges of the image."
    def __init__(self, inDim, inChans, padCount, padVal):
        self.dim = inDim          # input image dimension
        self.chans = inChans      # number of input channels
        self.padCount = padCount  # number of pixels to add on each edge
        self.padVal = padVal      # value of pixels to be added to each edge

    def execute(self, v):
        img = v.reshape((self.chans, self.dim, self.dim))
        padCounts = ((0, 0),
                     (self.padCount, self.padCount),
                     (self.padCount, self.padCount))
        img = np.pad(img, padCounts, "constant", constant_values=self.padVal)
        return img.flatten()

class QNNSlidingWindowLayer(QNNLayer):
    "Slide a window over a multichannel image (im2col)"
    def __init__(self, inDim, inChans, windowDim, stride=1):
        self.idim = inDim     # input image dimension
        self.chans = inChans  # channels in input image
        self.k = windowDim    # window size
        self.s = stride       # stride for next window

    def execute(self, v):
        # reshape the input vector into a 2D image
        img = v.reshape((1, self.chans, self.idim, self.idim))
        # call im2col to get the sliding window result
        res = im2col_indices(img, self.k, self.k, padding=0,
                            stride_y=self.s, stride_x=self.s)
        return res.flatten()

class QNNConvolutionLayer(QNNLayer):
    "Convolution via im2col and matrix-matrix multiplication"
    def __init__(self, W, inDim, pad, stride, padVal=0):
        self.ofm = W.shape[0] # number of output channels
        self.ifm = W.shape[1] # number of input channels
        self.k = W.shape[2]   # kernel (window) dimension
        self.idim = inDim     # input image dimension
        self.padded_idim = inDim + 2*pad  # input image dimension including padding
        self.odim = ((self.padded_idim - self.k) / stride) + 1  # output image dimension
        self.stride = stride  # stride for sliding window
        self.pad = pad        # number of padding pixels to add on each edge
        self.padVal = padVal  # value of padding pixels
        if(W.shape[2] != W.shape[3]):
            raise Exception("Only square conv filters supported for now")
        # instantiate internal layer components
        self.layers = []
        # add a padding layer, if padding is required
        if pad != 0:
            self.layers += [QNNPaddingLayer(self.idim, self.ifm, pad, padVal)]
        # add the sliding window layer
        self.layers += [QNNSlidingWindowLayer(self.padded_idim, self.ifm, self.k, self.stride)]
        # convert the kernel tensor to a matrix:
        # [ofm][ifm][k][k] to [ofm][ifm*k*k]
        self.W = W.reshape((self.ofm, self.ifm*self.k*self.k))

    def execute(self, v):
        # execute internal padding/sliding window layers first
        vn = v
        for l in self.layers:
            vn = l.execute(vn)
        # reconstruct image matrix
        vn = vn.reshape((self.ifm*self.k*self.k, self.odim*self.odim))
        # matrix-matrix multiply
        res = np.dot(self.W, vn)
        return res.flatten()

class QNNPoolingLayer(QNNLayer):
    "Perform either max or average pooling."
    def __init__(self, inDim, inChans, poolSize, strideSize, poolFxn = "MAX"):
        self.idim = inDim
        self.chans = inChans
        self.k = poolSize
        self.s = strideSize
        self.odim = ((self.idim - self.k) / self.s) + 1
        self.poolFxn = poolFxn

    def execute(self, v):
        img = v.reshape((self.chans, self.idim, self.idim))
        out_img = np.zeros((self.chans, self.odim*self.odim), dtype=np.float32)
        for c in range(self.chans):
            chan_img = img[c].reshape((1, 1, self.idim, self.idim))
            # extract parts of image with sliding window
            wnd = im2col_indices(chan_img, self.k, self.k, padding=0,
            stride_y=self.s, stride_x=self.s)
            # each window is a column -- get the reduction along columns
            if self.poolFxn == "MAX":
                out_img[c]=wnd.max(axis = 0).flatten()
            elif self.poolFxn == "AVE":
                out_img[c]=wnd.mean(axis = 0).flatten()
            else:
                raise Exception("Unsupported pooling function")
        return out_img.flatten()


class QNNSoftmaxLayer(QNNLayer):
    "Compute softmax values for each sets of scores."
    def execute(selv, v):
        e_x = np.exp(v - np.max(v))
        return e_x / e_x.sum()

class QNNReLULayer(QNNLayer):
    "Apply elementwise ReLU to the vector."
    def execute(self, v):
        return np.asarray(map(lambda x: x if x>0 else 0, v))

