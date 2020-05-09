import torch
from torchvision.models import resnet50
from torchvision.ops import RoIPool
from torchvision.ops import RoIAlign

class M_ROI_CLASSIFIER(torch.nn.Module):
    def __init__(self, num_classes, backbone=None):
        super(M_ROI_CLASSIFIER, self).__init__()

        self.num_classes = num_classes
        M_backbone = backbone
        if M_backbone is None:
            M_backbone = resnet50(pretrained=True, replace_stride_with_dilation=[True, True, True])

        M_conv_ = torch.nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        M_batchn_ = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        M_relu_ = torch.nn.ReLU(inplace=True)
        M_custom_layer = torch.nn.Sequential(
            M_conv_,
            M_batchn_,
            M_relu_
        )
        self.M_custom_backbone = torch.nn.Sequential(
            M_backbone.conv1,
            M_backbone.bn1,
            M_backbone.relu,
            M_backbone.maxpool,
            M_backbone.layer1,
            M_backbone.layer2,
            M_backbone.layer3,
            M_backbone.layer4,
            M_custom_layer
        )

        self.M_roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1, sampling_ratio=-1)
        self.M_flatten = torch.nn.Flatten()
        self.M_classifier = torch.nn.Linear(in_features=25088, out_features=self.num_classes, bias=True)

    def forward(self, inputs):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        T_image, T_roi_bbox = inputs
        T_extracted_features = self.M_custom_backbone(T_image)
        batch_size = T_roi_bbox.shape[0]
        T_roi_bbox = T_roi_bbox*T_extracted_features.shape[2]
        T_batch_indices = torch.Tensor(list(range(batch_size))).to("cuda")
        T_roi_bbox = torch.cat((T_batch_indices.view(batch_size, 1), T_roi_bbox), axis=1)
        T_roi_features = self.M_roi_align(T_extracted_features, T_roi_bbox)
        T_roi_features_vector = self.M_flatten(T_roi_features)
        return self.M_classifier(T_roi_features_vector)

        """
        batch_size = T_roi_bbox.shape[0]
        T_roi_bbox = T_roi_bbox*240;
        T_batch_indices = torch.Tensor(list(range(batch_size))).to("cuda")
        T_roi_bbox = torch.cat((T_batch_indices.view(batch_size, 1), T_roi_bbox), axis=1)
        T_roi_features = self.M_roi_align(T_image, T_roi_bbox)
        return T_roi_features
        """
