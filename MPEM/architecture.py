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

# Internal module
from UTILS.geometry_utils import LieEuclideanMapper

class MultiTaskModel(nn.Module):
    def __init__(self, input_shape):
        super(MultiTaskModel, self).__init__()

        self.LEM = LieEuclideanMapper()

        channels, height, width = input_shape
        self.output_shape = (1, height // 2 ** 5, width // 2 ** 5)

        # Define common layers
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.shared_layers = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512)
        )

        # Define specialized discriminator layers
        self.discriminator_layers = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

        '''
        # Define pose prediction layers
        self.pose_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # average pooling to transform feature map to 1x1 size
            nn.Flatten(),  # flatten the output for FC layer
            nn.Linear(512 * 2, 128),  # Linear layer, input dimension multiplied by 4 because of the concatenation
            nn.ReLU(inplace=True),  # Activation function
            nn.Linear(128, 6)  # Output layer to output 6 DOF pose
        )
        '''
        self.lstm_input_size = 512 * 2
        self.hidden_size = 128
        self.num_layers = 1

        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 6)  # output 6 DOF pose

    def forward(self, prev_frame = None, curr_frame = None):
        # Define an initial hidden state

        if prev_frame is not None and curr_frame is not None:
            h0 = torch.zeros(self.num_layers, prev_frame.size(0), self.hidden_size).to(prev_frame.device)
            c0 = torch.zeros(self.num_layers, prev_frame.size(0), self.hidden_size).to(prev_frame.device)
            prev_frame_output = self.shared_layers(prev_frame)
            curr_frame_output = self.shared_layers(curr_frame)
            shared_output = torch.cat([prev_frame_output, curr_frame_output], dim=1)
            #shared_output = shared_output.view(shared_output.size(0), 1, -1)  # reshape for LSTM input
            shared_output = torch.mean(shared_output, dim=[2, 3]).view(shared_output.size(0), 1, -1)  # Apply GAP and reshape for LSTM

            lstm_out, _ = self.lstm(shared_output, (h0, c0))
            pose_output = self.fc(lstm_out[:, -1, :])  # only use the last sequence output

            # get the SE3 representation
            translation_vector = pose_output[:, :3]
            rotation_vector = pose_output[:, 3:]

            rotation_matrix = self.LEM.convert_euler_angles_to_rotation_matrix(rotation_vector)

            # create the SE(3) matrix
            motion_matrix_SE3 = torch.eye(4).unsqueeze(0).repeat(rotation_vector.shape[0], 1, 1)
            motion_matrix_SE3[:, :3, :3] = rotation_matrix
            motion_matrix_SE3[:, :3, 3] = translation_vector
            return motion_matrix_SE3, pose_output
        elif prev_frame is not None and curr_frame is None:
            prev_frame_output = self.shared_layers(prev_frame)
            discriminator_output = self.discriminator_layers(prev_frame_output)
            return discriminator_output

        elif prev_frame is None and curr_frame is not None:
            curr_frame_output = self.shared_layers(curr_frame)
            discriminator_output = self.discriminator_layers(curr_frame_output)
            return discriminator_output


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),  # Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class ConditionalGenerator(nn.Module):
    def __init__(self, input_shape = (3, 256, 256), num_residual_block = 9, condition_dim = 6):
        super(ConditionalGenerator, self).__init__()
        self.condition_dim = condition_dim
        channels = input_shape[0] + condition_dim  # channels will now include condition_dim

        # Initial Convolution Block
        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),  # --> width*2, heigh*2
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Output Layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(out_features, 3, 7),
                  nn.Tanh()
                  ]

        # Unpacking
        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        # Repeat condition and concatenate along the channel dimension.
        c = c.view(c.size(0), self.condition_dim, 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        return self.model(x)