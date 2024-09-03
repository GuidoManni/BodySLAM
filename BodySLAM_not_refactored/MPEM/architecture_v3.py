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
from UTILS.geometry_utils import PoseOperator
PO = PoseOperator()





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
    quaternion = PO.matrix_to_quaternion(rotation)

    # Combine translation and quaternion to get 7-element pose vector
    pose7 = torch.cat([translation, quaternion], dim=-1)

    return pose7




class MultiTaskModel(nn.Module):
    def __init__(self, input_shape, device):
        super(MultiTaskModel, self).__init__()
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

        self.reproject = nn.Conv2d(256 + condition_dim, 256, kernel_size=1, stride=1, padding=0)

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
            rotation_part = PO.normalize_quaternion(pose[:, 3:])
            rotation_matrix = PO.quaternion_to_matrix(rotation_part)  # Assuming quaternion_to_matrix is defined elsewhere

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



if __name__ == "__main__":
    disc = MultiTaskModel(input_shape=(6, 128, 128), device='cpu').to('cpu')
    gen = ConditionalGenerator(input_shape=(6, 128, 128), device='cpu')

    test = torch.randn(12, 3, 128, 128).to('cpu')

    test_test = torch.cat([test, test], dim=1)

    disc_output = disc(test_test)

    gen_pose = gen(test_test, mode="pose")

    gen_img = gen(test_test, gen_pose, mode="generate")


    print(f"gen_img.shape: {gen_img.shape}")
    print(f"gen_pose.shape: {gen_pose.shape}")
    print(f"disc.shape: {disc_output.shape}")

    disc_output = (disc.output_shape[0], 2 * disc.output_shape[1], 2 * disc.output_shape[2])
    print(disc_output)