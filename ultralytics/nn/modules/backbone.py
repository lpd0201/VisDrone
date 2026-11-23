import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.segmentation.fcn import FCNHead
import timm
# class ShuffleNetV2Segmentation(nn.Module):
#     def __init__(self, num_classes, pretrained=False):
#         super().__init__()
#         # 1. Tùy chỉnh backbone ShuffleNetV2 x1.0 theo yêu cầu
#         weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        
#         # shuffle_net_v2_x1_0 trả về cấu trúc gồm nhiều stage, ta lấy phần features
#         net = models.shufflenet_v2_x1_0(weights=weights)
#         self.backbone = nn.Sequential(
#             net.conv1,
#             net.maxpool,
#             net.stage2,
#             net.stage3,
#             net.stage4,
#             net.conv5
#         )
        
#         # 2. ShuffleNetV2 x1.0 trả về 1024 channels ở lớp feature cuối cùng (conv5)
#         self.head = FCNHead(in_channels=1024, channels=num_classes)

#     def forward(self, x):
#         input_shape = x.shape[-2:] # Lấy kích thước ảnh gốc (ví dụ: 640x640)
        
#         # Đi qua ShuffleNetV2 backbone
#         features = self.backbone(x)
        
#         # Đưa qua FCN Head để phân loại pixel
#         x = self.head(features)
        
#         # Phóng to nội suy trở lại kích thước ảnh gốc
#         x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
#         return x

class MobileNetV3Backbone(nn.Module):
    def __init__(self, c1=3, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        
        # Sửa lỗi cú pháp: truyền đúng biến c1 hoặc 3 vào Conv2d
        if c1 != 3:
            model.features[0][0] = nn.Conv2d(
                c1, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
            
        self.backbone = model.features
        self.out_channels = 576  

    def forward(self, x):
        return self.backbone(x)

class ConvNeXtBackbone(nn.Module):
    """
    Backbone ConvNeXt-Tiny đã loại bỏ phần Head/Classifier,
    trả về Feature Maps dạng (B, 768, H', W') cho Head xử lý.
    """
    def __init__(self, in_channels=2, pretrained=False):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        
        # Sửa lớp Conv Stem đầu tiên nếu in_channels khác 3
        if in_channels != 3:
            model.features[0][0] = nn.Conv2d(
                in_channels, 96, kernel_size=4, stride=4
            )
            
        self.backbone = model.features
        self.out_channels = 768  # ConvNeXt-Tiny trả về 768 channels ở feature map cuối

    def forward(self, x):
        return self.backbone(x)

class EfficientNetB0Backbone(nn.Module):
    """
    Backbone EfficientNet-B0 siêu nhẹ cho Object Detection / Instance Segmentation.
    Trả về Feature Map dạng (B, 1280, H', W') cho Head xử lý.
    """
    def __init__(self, c1=3, pretrained=True):
        super().__init__()
        # 1. Tải trọng số chuẩn của EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        
        # 2. Sửa lớp Conv Stem đầu tiên nếu in_channels (c1) khác 3
        # Trong torchvision, lớp Conv2d của EfficientNet-B0 nằm sâu trong features[0][0]
        if c1 != 3:
            model.features[0][0] = nn.Conv2d(
                c1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            
        # 3. Trích xuất phần lõi (loại bỏ lớp Classifier phía sau)
        self.backbone = model.features
        
        # 4. Cố định số kênh đầu ra. 
        # Khối MBConv cuối cùng của bản B0 xuất ra 1280 kênh
        self.out_channels = 1280  

    def forward(self, x):
        return self.backbone(x)

class SqueezeNetBackbone(nn.Module):
    """
    Backbone SqueezeNet v1.1: Khoảng 1.2M tham số.
    Siêu tiết kiệm VRAM, lý tưởng làm base để test độ ổn định code trước.
    """
    def __init__(self, c1=3, pretrained=True):
        super().__init__()
        weights = models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        model = models.squeezenet1_1(weights=weights)
        
        if c1 != 3:
            # SqueezeNet dùng lớp Conv đầu tiên kích thước hạt nhân khá lạ
            model.features[0] = nn.Conv2d(c1, 64, kernel_size=3, stride=2)
            
        self.backbone = model.features
        self.out_channels = 512 

    def forward(self, x):
        return self.backbone(x)

import torch.nn as nn
import torchvision.models as models

class RegNetY400MFBackbone(nn.Module):
    """
    Backbone RegNetY-400MF: Khoảng 5.3M tham số.
    Cân bằng tuyệt vời giữa FLOPs, VRAM và tốc độ thực tế (Latency).
    Trả về Feature Map P5 với 432 channels.
    """
    def __init__(self, c1=3, c2=440,pretrained=True):
        super().__init__()
        weights = models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None
        model = models.regnet_y_400mf(weights=weights)
        
        if c1 != 3:
            # Sửa đổi lớp Stem đầu tiên nếu input không phải RGB
            model.stem[0] = nn.Conv2d(c1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            
        # RegNet trong torchvision phân tách rõ phần Stem và Trunk (các khối chính)
        self.stem = model.stem
        self.trunk_output = model.trunk_output
        self.out_channels = c2 # Chính là 432 cho bản 400MF

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)
        return x

class GhostNetBackbone(nn.Module):
    def __init__(self, *args, pretrained=True):
        super().__init__()
        # Mặc định an toàn
        c1 = 3
        c2 = 960  # Stride 32 (P5) của GhostNet-100 mặc định xuất ra 960 kênh
        
        # Bóc tách linh hoạt tham số truyền vào từ file YAML
        if len(args) >= 2:
            c1 = args[0]
            c2 = args[1]
        elif len(args) == 1:
            c2 = args[0]

        # Khởi tạo mô hình GhostNet-100 từ timm. 
        # in_chans tự động xử lý số kênh ảnh đầu vào (RGB = 3, Xám = 1)
        self.backbone = timm.create_model(
            'ghostnet_100', 
            pretrained=pretrained, 
            in_chans=c1
        )
        self.out_channels = c2

    def forward(self, x):
        # forward_features tự động xả ra lớp đặc trưng cuối cùng (P5) 
        # Output shape với ảnh 640x640: [Batch, 960, 20, 20]
        return self.backbone.forward_features(x)

class VanillaNetBackbone(nn.Module):
    """
    Backbone VanillaNet-5 (Small): Khoảng 1.5M tham số.
    Thiết kế "siêu nông" (shallow network), giảm thiểu tối đa Memory Access Cost (MAC).
    Siêu tiết kiệm VRAM, lý tưởng làm base để chạy tốc độ cao trên thiết bị nhúng.
    """
    def __init__(self, c1=3, pretrained=True):
        super().__init__()
        # Load VanillaNet-5 từ timm. Truyền luôn in_chans=c1 để thư viện 
        # tự động biên dịch cấu trúc an toàn nhất.
        model = timm.create_model('vanillanet_5', pretrained=pretrained, in_chans=c1)
        
        if c1 != 3:
            # Khác với SqueezeNet trong torchvision phải tự tìm và gán đè lớp Conv thủ công,
            # timm đã xử lý động (dynamic) việc đổi kênh c1 ngay ở hàm create_model trên.
            # Ta giữ khối if này để đồng nhất cấu trúc logic.
            pass
            
        self.backbone = model
        self.out_channels = 1024 

    def forward(self, x):
        # Dùng forward_features() để bỏ qua lớp phân loại (Classifier) ở cuối mạng,
        # chỉ trả về ma trận đặc trưng (Feature Map).
        return self.backbone.forward_features(x)

class MNASNetBackbone(nn.Module):
    """
    Backbone MNASNet (1.0 multiplier): Khoảng 4.2M tham số.
    Kiến trúc sinh ra từ AutoML của Google, tối ưu trực tiếp cho Latency (độ trễ) 
    trên thiết bị biên thay vì chỉ tối ưu FLOPs.
    Lý tưởng cho môi trường VRAM nhỏ và inference tốc độ cao.
    """
    def __init__(self, c1=3, pretrained=True):
        super().__init__()
        weights = models.MNASNet1_0_Weights.DEFAULT if pretrained else None
        model = models.mnasnet1_0(weights=weights)
        
        if c1 != 3:
            # Lớp Conv đầu tiên của MNASNet trong torchvision nằm ở layers[0]
            model.layers[0] = nn.Conv2d(c1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            
        self.backbone = model.layers
        self.out_channels = 1280  # Feature map cuối (P5) của MNASNet 1.0 có 1280 kênh

    def forward(self, x):
        return self.backbone(x)