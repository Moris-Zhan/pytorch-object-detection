import torch
from torch import nn
from collections import OrderedDict, Iterable
from distutils.version import LooseVersion
import math

torchversion = LooseVersion(torch.__version__)
version120 = LooseVersion("1.2.0")

class SelectiveSequential(nn.Sequential):
    """ Sequential that allows to select which layers are to be considered as output.
        See :class:`torch.nn.Sequential` for more information.
        Args:
            selection (list): names of the layers for which you want to get the output
            *args: Arguments that are passed to the Sequential init function
        Note:
            If your selection only contains one item, this layer will flatten the return output in a single list:
            >>> main_output, selected_output = layer(input)  # doctest: +SKIP
            However, if you have multiple outputs in your selection list, the outputs will be grouped in a dictionary:
            >>> main_output, selected_output_dict = layer(input)  # doctest: +SKIP
    """
    def __init__(self, selection, *args):
        super().__init__(*args)

        self.selection = [str(select) for select in selection]
        self.flatten = len(self.selection) == 1
        k = list(self._modules.keys())
        for sel in self.selection:
            if sel not in k:
                raise KeyError('Selection key not found in sequential [{sel}]')

    def extra_repr(self):
        return f'selection={self.selection}, flatten={self.flatten}'

    def forward(self, x):
        sel_output = {sel: None for sel in self.selection}

        for key, module in self._modules.items():
            x = module(x)
            if key in self.selection:
                sel_output[key] = x

        if self.flatten:
            return x, sel_output[self.selection[0]]
        else:
            return x, sel_output

class Residual(nn.Sequential):
    """ Residual block that runs like a Sequential, but then adds the original input to the output tensor.
        See :class:`torch.nn.Sequential` for more information.
        Warning:
            The dimension between the input and output of the module need to be the same
            or need to be broadcastable from one to the other!
    """
    def forward(self, x):
        y = super().forward(x)
        return x + y

class Reorg(nn.Module):
    """ This layer reorganizes a tensor according to a stride.
    The dimensions 2,3 will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)
    Args:
        stride (int): stride to divide the input tensor
    """
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
        if not isinstance(stride, int):
            raise TypeError(f'stride is not an int [{type(stride)}]')

    def extra_repr(self):
        return f'stride={self.stride}'

    def forward(self, x):
        assert(x.dim() == 4)
        B, C, H, W = x.size()

        if H % self.stride != 0:
            raise ValueError(f'Dimension mismatch: {H} is not divisible by {self.stride}')
        if W % self.stride != 0:
            raise ValueError(f'Dimension mismatch: {W} is not divisible by {self.stride}')

        # from: https://github.com/thtrieu/darkflow/issues/173#issuecomment-296048648
        x = x.view(B, C//(self.stride**2), H, self.stride, W, self.stride).contiguous()
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, -1, H//self.stride, W//self.stride)

        return x

class PaddedMaxPool2d(nn.Module):
    """ Maxpool layer with a replicating padding.
    Args:
        kernel_size (int or tuple): Kernel size for maxpooling
        stride (int or tuple, optional): The stride of the window; Default ``kernel_size``
        padding (tuple, optional): (left, right, top, bottom) padding; Default **None**
        dilation (int or tuple, optional): A parameter that controls the stride of elements in the window
    """
    def __init__(self, kernel_size, stride=None, padding=(0, 0, 0, 0), dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}'

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, self.padding, mode='replicate'), self.kernel_size, self.stride, 0, self.dilation)
        return x

class GlobalAvgPool2d(nn.Module):
    """ This layer averages each channel to a single number.
    Args:
        squeeze (boolean, optional): Whether to reduce dimensions to [batch, channels]; Default **True**
    Deprecated:
        This function is deprecated in favor of :class:`torch.nn.AdaptiveAvgPool2d`. |br|
        To replicate the behaviour with `squeeze` set to **True**, append a Flatten layer afterwards:
        >>> layer = torch.nn.Sequential(
        ...     torch.nn.AdaptiveAvgPool2d(1),
        ...     ln.network.layer.Flatten()
        ... )
    """
    def __init__(self, squeeze=True):
        super().__init__()
        self.squeeze = squeeze
        log.deprecated('The GlobalAvgPool2d layer is deprecated and will be removed in future version, please use "torch.nn.AdaptiveAvgPool2d"')

    def forward(self, x):
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))

        if self.squeeze:
            x = x.view(B, C)

        return x

class Flatten(nn.Module):
    """ Flatten tensor into single dimension.
    Args:
        batch (boolean, optional): If True, consider input to be batched and do not flatten first dim; Default **True**
    """
    def __init__(self, batch=True):
        super().__init__()
        self.batch = batch

    def forward(self, x):
        if self.batch:
            return x.view(x.size(0), -1)
        else:
            return x.view(-1)

class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        momentum (int, optional): momentum of the moving averages of the normalization; Default **0.01**
        relu (class, optional): Which ReLU to use; Default :class:`torch.nn.LeakyReLU`
    Note:
        If you require the `relu` class to get extra parameters, you can use a `lambda` or `functools.partial`:
        >>> conv = ln.layer.Conv2dBatchReLU(
        ...     in_c, out_c, kernel, stride, padding,
        ...     relu=functools.partial(torch.nn.LeakyReLU, 0.1)
        ... )   # doctest: +SKIP
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 momentum=0.01, relu=lambda: nn.LeakyReLU(0.1)):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.momentum = momentum

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels, momentum=self.momentum),
            relu()
        )

    def __repr__(self):
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, {relu})'
        return s.format(name=self.__class__.__name__, relu=self.layers[2], **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x

class YOLOv3(nn.Module):
    """ Yolo v3 implementation :cite:`yolo_v3`.
    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list, optional): 3D list with anchor values; Default **Yolo v3 anchors**
    Attributes:
        self.stride: Subsampling factors of the network (input dimensions should be a multiple of these numbers)
        self.remap_darknet53: Remapping rules for weights from the `~lightnet.models.Darknet53` model.
    Note:
        Unlike YoloV2, the anchors here are defined as multiples of the input dimensions and not as a multiple of the output dimensions!
        The anchor list also has one more dimension than the one from YoloV2, in order to differentiate which anchors belong to which stride.
    Warning:
        The :class:`~lightnet.network.loss.MultiScaleRegionLoss` and :class:`~lightnet.data.transform.GetMultiScaleBoundingBoxes`
        do not implement the overlapping class labels of the original implementation.
        Your weight files from darknet will thus not have the same accuracies as in darknet itself.
    """
    stride = (32, 16, 8)
    remap_darknet53 = [
        (r'^layers.([a-w]_)',   r'extractor.\1'),   # Residual layers
        (r'^layers.(\d_)',      r'extractor.\1'),   # layers 1, 2, 5
        (r'^layers.([124]\d_)', r'extractor.\1'),   # layers 10, 27, 44
    ]

    def __init__(self, num_classes=20, input_channels=3, anchors=[[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]]):
        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable) and not isinstance(anchors[0][0], Iterable):
            raise TypeError('Anchors need to be a 3D list of numbers')

        # Parameters
        self.num_classes = num_classes
        self.anchors = []   # YoloV3 defines anchors as a multiple of the input dimensions of the network as opposed to the output dimensions
        for i, s in enumerate(self.stride):
            self.anchors.append([(a[0] / s, a[1] / s) for a in anchors[i]])

        # Network
        self.extractor = SelectiveSequential(
            ['k_residual', 's_residual'],
            OrderedDict([
                ('1_convbatch',         Conv2dBatchReLU(input_channels, 32, 3, 1, 1)),
                ('2_convbatch',         Conv2dBatchReLU(32, 64, 3, 2, 1)),
                ('a_residual',          Residual(OrderedDict([
                    ('3_convbatch',     Conv2dBatchReLU(64, 32, 1, 1, 0)),
                    ('4_convbatch',     Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ]))),
                ('5_convbatch',         Conv2dBatchReLU(64, 128, 3, 2, 1)),
                ('b_residual',          Residual(OrderedDict([
                    ('6_convbatch',     Conv2dBatchReLU(128, 64, 1, 1, 0)),
                    ('7_convbatch',     Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ]))),
                ('c_residual',          Residual(OrderedDict([
                    ('8_convbatch',     Conv2dBatchReLU(128, 64, 1, 1, 0)),
                    ('9_convbatch',     Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ]))),
                ('10_convbatch',        Conv2dBatchReLU(128, 256, 3, 2, 1)),
                ('d_residual',          Residual(OrderedDict([
                    ('11_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('12_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('e_residual',          Residual(OrderedDict([
                    ('13_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('14_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('f_residual',          Residual(OrderedDict([
                    ('15_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('16_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('g_residual',          Residual(OrderedDict([
                    ('17_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('18_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('h_residual',          Residual(OrderedDict([
                    ('19_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('20_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('i_residual',          Residual(OrderedDict([
                    ('21_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('22_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('j_residual',          Residual(OrderedDict([
                    ('23_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('24_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('k_residual',          Residual(OrderedDict([
                    ('25_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('26_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('27_convbatch',        Conv2dBatchReLU(256, 512, 3, 2, 1)),
                ('l_residual',          Residual(OrderedDict([
                    ('28_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('29_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('m_residual',          Residual(OrderedDict([
                    ('30_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('31_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('n_residual',          Residual(OrderedDict([
                    ('32_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('33_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('o_residual',          Residual(OrderedDict([
                    ('34_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('35_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('p_residual',          Residual(OrderedDict([
                    ('36_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('37_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('q_residual',          Residual(OrderedDict([
                    ('38_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('39_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('r_residual',          Residual(OrderedDict([
                    ('40_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('41_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('s_residual',          Residual(OrderedDict([
                    ('42_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('43_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('44_convbatch',        Conv2dBatchReLU(512, 1024, 3, 2, 1)),
                ('t_residual',          Residual(OrderedDict([
                    ('45_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('46_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('u_residual',          Residual(OrderedDict([
                    ('47_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('48_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('v_residual',          Residual(OrderedDict([
                    ('49_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('50_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('w_residual',          Residual(OrderedDict([
                    ('51_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('52_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
            ]),
        )

        self.detector = nn.ModuleList([
            # Sequence 0 : input = extractor
            SelectiveSequential(
                ['57_convbatch'],
                OrderedDict([
                    ('53_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('54_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                    ('55_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('56_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                    ('57_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('58_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                    ('59_conv',         nn.Conv2d(1024, len(self.anchors[0])*(5+self.num_classes), 1, 1, 0)),
                ])
            ),

            # Sequence 1 : input = 57_convbatch
            nn.Sequential(
                OrderedDict([
                    ('60_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('61_upsample',     nn.Upsample(scale_factor=2, mode='nearest')),
                ])
            ),

            # Sequence 2 : input = 61_upsample and s_residual
            SelectiveSequential(
                ['66_convbatch'],
                OrderedDict([
                    ('62_convbatch',    Conv2dBatchReLU(256+512, 256, 1, 1, 0)),
                    ('63_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                    ('64_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('65_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                    ('66_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('67_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                    ('68_conv',         nn.Conv2d(512, len(self.anchors[1])*(5+self.num_classes), 1, 1, 0)),
                ])
            ),

            # Sequence 3 : input = 66_convbatch
            nn.Sequential(
                OrderedDict([
                    ('69_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('70_upsample',     nn.Upsample(scale_factor=2, mode='nearest')),
                ])
            ),

            # Sequence 4 : input = 70_upsample and k_residual
            nn.Sequential(
                OrderedDict([
                    ('71_convbatch',    Conv2dBatchReLU(128+256, 128, 1, 1, 0)),
                    ('72_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                    ('73_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('74_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                    ('75_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('76_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                    ('77_conv',         nn.Conv2d(256, len(self.anchors[2])*(5+self.num_classes), 1, 1, 0)),
                ])
            ),
        ])

    def forward(self, x):
        out = [None, None, None]

        # Feature extractor
        x, inter_features = self.extractor(x)

        # detector 0
        out[0], x = self.detector[0](x)

        # detector 1
        x = self.detector[1](x)
        out[1], x = self.detector[2](torch.cat((x, inter_features['s_residual']), 1))

        # detector 2
        x = self.detector[3](x)
        out[2] = self.detector[4](torch.cat((x, inter_features['k_residual']), 1))

        return out

class RegionLoss(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        # super(RegionLoss, self).__init__()
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        # self.img_dim = 416
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if len(targets) == 0:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            bwbh = pred_boxes[..., 2:4][obj_mask]    
            shape = min(len(bwbh), len(targets))        
            wh_loss = self.mse_loss(
                torch.sqrt(torch.abs(bwbh) + 1e-32)[:shape],
                torch.sqrt(torch.abs(targets[..., 3:5]) + 1e-32)[:shape],
            )

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }
            # loss.item(), obj_coord1_loss, obj_size1_loss, noobjness1_loss, objness1_loss, obj_class_loss
            return total_loss, (loss_x + loss_y), wh_loss, loss_conf, loss_cls, loss_conf_obj, loss_conf_noobj

class MultiScaleRegionLoss(RegionLoss):
    def __init__(self, anchors, num_classes, img_dim = 416, **kwargs):
        super().__init__(anchors, num_classes, img_dim, **kwargs)
        # self._anchors = torch.tensor(anchors, requires_grad=False)
        self._anchors = anchors

    def forward(self, output, target, seen=None):
        device = output[0].device
        loss = torch.tensor(0.0).to(device)
        loss_coord = torch.tensor(0.0).to(device)
        loss_size = torch.tensor(0.0).to(device)
        loss_conf = torch.tensor(0.0).to(device)
        loss_cls = torch.tensor(0.0).to(device)
        loss_conf_obj = torch.tensor(0.0).to(device)
        loss_conf_noobj = torch.tensor(0.0).to(device)
        # if seen is not None:
        #     self.seen = torch.tensor(seen)

        # Run loss at different scales and sum resulting loss values
        for i, out in enumerate(output):    
            self.anchors = self._anchors[i]
            self._num_anchors = len(self.anchors)
            # self.anchor_step = self.anchors.shape[1]
            # self.stride = self._stride[i]
            scale_loss, \
            scale_loss_coord, \
            scale_loss_size, \
            scale_loss_conf, \
            scale_loss_cls, \
            scale_loss_conf_obj, \
            scale_loss_conf_noobj = super().forward(out, target)

            loss_coord += scale_loss_coord
            loss_size += scale_loss_size
            loss_conf += scale_loss_conf
            loss_cls += scale_loss_cls
            loss_conf_obj += scale_loss_conf_obj
            loss_conf_noobj += scale_loss_conf_noobj
            loss += scale_loss


        # Overwrite loss values with avg
        self.loss_coord = loss_coord / len(output)
        self.loss_size = loss_size / len(output)
        self.loss_conf = loss_conf / len(output)
        self.loss_cls = loss_cls / len(output)
        self.loss_tot = loss / len(output)

        self.loss_conf_obj = loss_conf_obj / len(output)
        self.loss_conf_noobj = loss_conf_noobj / len(output)
        # return self.loss_tot, self.loss_coord, self.loss_conf, self.loss_cls     
        return self.loss_tot, self.loss_coord, self.loss_size, self.loss_conf, self.loss_cls, self.loss_conf_obj, self.loss_conf_noobj
       
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
    over_b = torch.sum(torch.Tensor([d>= obj_mask.shape[0] for d in b]))
    over_n = torch.sum(torch.Tensor([d>= obj_mask.shape[1] for d in best_n]))
    over_gj = torch.sum(torch.Tensor([d>= obj_mask.shape[2] for d in gj]))
    over_gi = torch.sum(torch.Tensor([d>= obj_mask.shape[3] for d in gi]))
    if over_b.item() + over_n.item() + over_gj.item() + over_gi.item() == 0:    
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
        if i >= len(b): continue
        if i >= len(gj): continue
        if i >= len(gi): continue
        if b[i] >= noobj_mask.shape[0]: continue
        if gj[i] >= noobj_mask.shape[2]: continue
        if gi[i] >= noobj_mask.shape[3]: continue
        
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    over_b = torch.sum(torch.Tensor([d>= tx.shape[0] for d in b]))
    over_n = torch.sum(torch.Tensor([d>= tx.shape[1] for d in best_n]))
    over_gj = torch.sum(torch.Tensor([d>= tx.shape[2] for d in gj]))
    over_gi = torch.sum(torch.Tensor([d>= tx.shape[3] for d in gi]))
    over_labels = torch.sum(torch.Tensor([d>= tcls.shape[4] for d in target_labels]))
    # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
    if over_b.item() + over_n.item() + over_gj.item() + over_gi.item() + over_labels.item() == 0:
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()
        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def to_cpu(tensor):
    return tensor.detach().cpu()

if __name__ == "__main__":  
    net = YOLOv3(num_classes=20)
    print(net)