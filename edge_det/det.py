import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

cv.dnn_registerLayer('Crop', CropLayer)

# Load the model.
prototxt = r'C:\Users\byx\Desktop\qk_maskrcnn_trs\edge_det\deploy.prototxt'
caffemodel =r'C:\Users\byx\Desktop\qk_maskrcnn_trs\edge_det\hed_pretrained_bsds.caffemodel'
net = cv.dnn.readNet(prototxt, caffemodel)

## Create a display window
kWinName = 'Holistically-Nested_Edge_Detection'
cv.namedWindow(kWinName, cv.WINDOW_AUTOSIZE) # Create a window using the flag CV_WINDOW_AUTOSIZE

img = cv.imread(r'./crop_mask/b6_18_v2_0_mask.jpg')
inp = cv.dnn.blobFromImage(img, scalefactor=1.0, size=(500, 500),
                            mean=(104.00698793, 116.66876762, 122.67891434),
                            swapRB=False, crop=False)

net.setInput(inp)
out = net.forward()
out = out[0, 0]
out = cv.resize(out, (img.shape[1], img.shape[0]))
out = 255 * out
out = out.astype(np.uint8)
out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
cv.imwrite('edge.png',out)
