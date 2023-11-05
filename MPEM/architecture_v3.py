'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23

Description:
The Architecture of the neural network used for pose estimation
'''

# AI-Lib
import torch
import torch.nn as nn
import torch.nn.functional as F

# Internal module
from UTILS.geometry_utils import LieEuclideanMapper, PoseOperator



def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    # From Pytorch3D
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    # PYTORCH3D FUNCTION
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
           F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
           ].reshape(batch_dim + (4,))


def motion_matrix_to_pose7(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert 4x4 motion matrix to a 7-element vector with 3 translation values and 4 quaternion values.

    Args:
        matrix: Motion matrices as tensor of shape (batch, 4, 4).

    Returns:
        Pose vector as tensor of shape (batch, 7).
    """
    if matrix.size(-1) != 4 or matrix.size(-2) != 4:
        raise ValueError(f"Invalid motion matrix shape {matrix.shape}.")

    # Extract translation (assuming matrix is in homogeneous coordinates)
    translation = matrix[..., :3, 3]

    # Extract rotation
    rotation = matrix[..., :3, :3]

    # Convert rotation matrix to quaternion
    quaternion = matrix_to_quaternion(rotation)

    # Combine translation and quaternion to get 7-element pose vector
    pose7 = torch.cat([translation, quaternion], dim=-1)

    return pose7


def normalize_quaternion(q):
    norm = torch.norm(q, p=2, dim=-1, keepdim=True)
    return q / norm


class MultiScaleMultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, num_scales=4):
        super(MultiScaleMultiHeadAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.num_scales = num_scales

        # Multi-Head Attention components
        self.queries = nn.ModuleList([nn.Conv2d(in_channels, self.head_dim, 1) for _ in range(num_heads)])
        self.keys = nn.ModuleList([nn.Conv2d(in_channels, self.head_dim, 1) for _ in range(num_heads)])
        self.values = nn.ModuleList([nn.Conv2d(in_channels, self.head_dim, 1) for _ in range(num_heads)])

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.PReLU(),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )

        # Projection Layer
        self.fc = nn.Conv2d(in_channels, in_channels, 1)

        # Parameterized Activation
        self.prelu = nn.PReLU()

        # Regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.size()

        # Multi-Scale processing
        multi_scale_outputs = [F.interpolate(self.multi_head_attention(F.interpolate(x, scale_factor=1.0 / (2 ** i), mode='bilinear')), size=(H, W), mode='bilinear') for i in range(self.num_scales)]

        out = sum(multi_scale_outputs)

        # Channel Attention
        avg = self.avg_pool(out).view(B, C)
        channel_weights = self.channel_fc(avg).view(B, C, 1, 1)
        out = out * channel_weights

        # Residual Connection
        out += x

        # Normalization
        out = F.layer_norm(out, out.size()[1:])

        # Parameterized Activation
        out = self.prelu(out)

        # Regularization
        out = self.dropout(out)

        return out

    def multi_head_attention(self, x):
        B, _, H, W = x.size()
        heads_output = []
        for i in range(self.num_heads):
            q = self.queries[i](x).view(B, -1, H * W).permute(0, 2, 1)
            k = self.keys[i](x).view(B, -1, H * W)
            v = self.values[i](x).view(B, -1, H * W)

            attention = F.softmax(torch.bmm(q, k), dim=-1)
            out = torch.bmm(v, attention.permute(0, 2, 1)).view(B, self.head_dim, H, W)
            heads_output.append(out)

        # Concatenate all the heads' outputs and project
        concatenated = torch.cat(heads_output, dim=1)
        return self.fc(concatenated)


class GaussianBlur(nn.Module):
    def __init__(self, channels):
        super(GaussianBlur, self).__init__()
        self.weight = nn.Parameter(data=torch.FloatTensor(channels, 1, 3, 3), requires_grad=False)
        self.weight.data.fill_(1.0 / 9.0)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1, groups=x.size(1))


class EnhancedAttention(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedAttention, self).__init__()
        self.gaussian = GaussianBlur(in_channels)
        self.attention = MultiScaleMultiHeadAttention(in_channels)

    def forward(self, x):
        x = self.gaussian(x)
        return self.attention(x)


class MultiTaskModel(nn.Module):
    def __init__(self, input_shape, device):
        super(MultiTaskModel, self).__init__()

        # Assuming LEM and PO are defined elsewhere in your code
        self.LEM = LieEuclideanMapper()
        self.PO = PoseOperator()
        self.device = device

        channels, height, width = input_shape
        self.output_shape = (1, height // 2 ** 5, width // 2 ** 5)
        stacked_channels = channels

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.shared_layers = nn.Sequential(
            *discriminator_block(stacked_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512)
        )

        self.discriminator_layers = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )


    def forward(self, stacked_frame=None, task='pose'):
        shared_output = self.shared_layers(stacked_frame)
        discriminator_output = self.discriminator_layers(shared_output)
        return discriminator_output


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features=None):
        super(ResidualBlock, self).__init__()

        out_features = out_features or in_features

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, out_features, 3),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_features, out_features, 3),
            nn.InstanceNorm2d(out_features)
        )

    def forward(self, x):
        return x + self.block(x)


class ConditionalGenerator(nn.Module):
    def __init__(self, device, input_shape=(6, 256, 256), num_residual_block=9, condition_dim=7):
        super(ConditionalGenerator, self).__init__()
        self.condition_dim = condition_dim
        channels = input_shape[0]
        self.device = device
        self.skip_linear = None

        self.reproject = nn.Conv2d(input_shape[1] + condition_dim, 256, kernel_size=1, stride=1, padding=0)

        # Initial Convolution Block
        out_features = 64
        self.initial_model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
        in_features = out_features

        # Downsampling
        downsample_layers = []
        for _ in range(2):
            out_features *= 2
            downsample_layers += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        self.downsampling = nn.Sequential(*downsample_layers)

        # Pose Estimation Tail

        self.pose_conv = nn.Sequential(
            nn.Conv2d(in_features, 512, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Dense part
        self.pose_dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7)  # 3 for translation, 4 for rotation (quaternion)
        )
        '''
        self.pose_estimation = nn.Sequential(
            nn.Conv2d(in_features, 512, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7)  # 3 for translation, 4 for rotation (quaternion)
        )
        '''

        # Adjust channel dimension for condition
        #out_features += condition_dim  # adding condition_dim channels for concatenation

        # Residual blocks
        residual_layers = []
        for _ in range(num_residual_block):
            residual_layers += [ResidualBlock(out_features, out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2  # adjust for upsampling
            residual_layers += [
                nn.Upsample(scale_factor=2),  # --> width*2, height*2
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Output Layer
        residual_layers += [nn.ReflectionPad2d(3),
                            nn.Conv2d(out_features, 3, 7),
                            nn.Tanh()
                            ]

        # Unpacking
        self.residual_and_upsampling = nn.Sequential(*residual_layers)

    def forward(self, x, c = None, mode="generate"):
        x = self.initial_model(x)
        x = self.downsampling(x)

        if mode == "pose":
            # pose = self.pose_estimation(x)
            conv_out = self.pose_conv(x)
            flattened = conv_out.view(conv_out.size(0), -1)

            # Skip connection
            concatenated = torch.cat([flattened, x.view(x.size(0), -1)], dim=1)  # concatenate the original input x with flattened output from conv

            # Initialize the linear layer if it hasn't been initialized
            if self.skip_linear is None:
                self.skip_linear = nn.Linear(concatenated.size(1), 7).to(self.device)

            skip_connection = self.skip_linear(concatenated)

            dense_out = self.pose_dense(conv_out)

            # Merge skip connection with the output of the dense layers
            pose = dense_out + skip_connection

            translation_part = pose[:, :3]
            rotation_part = normalize_quaternion(pose[:, 3:])
            rotation_matrix = quaternion_to_matrix(rotation_part)  # Assuming quaternion_to_matrix is defined elsewhere

            motion_matrix_SE3 = torch.eye(4).unsqueeze(0).repeat(rotation_matrix.shape[0], 1, 1)
            motion_matrix_SE3[:, :3, :3] = rotation_matrix
            motion_matrix_SE3[:, :3, 3] = translation_part

            return motion_matrix_SE3.to(self.device)

        elif mode == "generate":
            # Injecting condition after downsampling (bottleneck)
            #c = c.view(c.size(0), self.condition_dim, 1, 1)
            c = motion_matrix_to_pose7(c).view(c.size(0), self.condition_dim, 1, 1)
            c = c.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat([x, c], dim=1)
            x = self.reproject(x)
            x = self.residual_and_upsampling(x)
            return x

        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose between 'generate' and 'pose'.")

