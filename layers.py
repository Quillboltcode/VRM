import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module as described in CBAM paper."""
    
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x))
        # Max pooling path
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module as described in CBAM paper."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(combined)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    
    Reference: "CBAM: Convolutional Block Attention Module" (https://arxiv.org/abs/1807.06521)
    
    Combines channel attention and spatial attention in sequence.
    """
    
    def __init__(self, in_channels: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention first
        x = x * self.channel_attention(x)
        # Then apply spatial attention
        x = x * self.spatial_attention(x)
        return x


class CBAMConvBlock(nn.Module):
    """
    Convolutional block with CBAM attention.
    
    Standard conv block (Conv -> BN -> Activation) followed by CBAM.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, ratio: int = 16, 
                 kernel_size_cbam: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels, ratio, kernel_size_cbam)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        return x


class ResidualCBAMBlock(nn.Module):
    """
    Residual block with CBAM attention.
    
    Similar to RecursiveBlock but with CBAM added.
    """
    
    def __init__(self, channels: int, ratio: int = 16, kernel_size_cbam: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels, ratio, kernel_size_cbam)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        out += residual
        return self.relu(out)


# Example usage and testing
if __name__ == "__main__":
    # Test CBAM module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test input
    x = torch.randn(2, 64, 32, 32).to(device)
    
    # Test CBAM
    cbam = CBAM(64).to(device)
    out_cbam = cbam(x)
    print(f"CBAM input shape: {x.shape}")
    print(f"CBAM output shape: {out_cbam.shape}")
    
    # Test CBAMConvBlock
    conv_block = CBAMConvBlock(64, 128).to(device)
    out_conv = conv_block(x)
    print(f"CBAMConvBlock output shape: {out_conv.shape}")
    
    # Test ResidualCBAMBlock
    residual_block = ResidualCBAMBlock(128).to(device)
    out_residual = residual_block(out_conv)
    print(f"ResidualCBAMBlock output shape: {out_residual.shape}")
    
    print("All CBAM modules working correctly!")