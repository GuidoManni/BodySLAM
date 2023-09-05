'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23

Description:
It's the train script used to train the pose network
'''
import gc
# Python standard lib
import sys
import os
import itertools

# Numerical lib
import numpy as np

# Computer Vision lib
import cv2

# AI-lib
import torch
from torchvision.utils import save_image
from torch.optim import Adam

# Stat lib
import wandb

# Internal Module
from architecture import MultiTaskModel, ConditionalGenerator
from training_utils import TrainingLoss
from UTILS.io_utils import DatasetsIO, ModelIO

datasetIO = DatasetsIO()
modelIO = ModelIO()

def train_model(training_dataset_path, tr_dataset_type, testing_dataset_path, te_dataset_type, num_epoch, batch_size, path_to_the_model, load_model, num_worker = 10, lr = 0.0002, best_loss = float("inf"), input_shape = (3, 256, 256), standard = False):
    # step 0: detect the DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # step 1: we need to load the datasets
    training_root_content = datasetIO.load_UCBM(training_dataset_path)
    testing_root_content = datasetIO.load_EndoSlam(testing_dataset_path)

    # step 2: we initialize the models
    G_AB = ConditionalGenerator(input_shape=input_shape).to(DEVICE) # the generator that from A generates B
    G_BA = ConditionalGenerator(input_shape=input_shape).to(DEVICE) # the generator that from B generates A
    PaD_A = MultiTaskModel(input_shape=input_shape).to(DEVICE) # the Pose estimator and the discriminator of A
    PaD_B = MultiTaskModel(input_shape=input_shape).to(DEVICE) # the Pose estimator and the discriminator of B

    PaD_shape = MultiTaskModel(input_shape=input_shape)

    # step 3: we initialize the optimizers
    optimizer_G = Adam(params=itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(0.5, 0.999)) # optimizer for the GANs
    optimizer_PaD_A = Adam(params=PaD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_PaD_B = Adam(params=PaD_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # step 4: we initialize the losses
    losses = TrainingLoss()

    # step 5: we load the model (if we need to load it)
    if load_model:
        path_to_G_AB = path_to_the_model + "G_AB.pth"
        path_to_G_BA = path_to_the_model + "G_BA.pth"
        path_to_PaD_A = path_to_the_model + "PaD_A.pth"
        path_to_PaD_B = path_to_the_model + "PaD_B.pth"

        # load G_AB
        G_AB, _, training_var = modelIO.load_pose_model(path_to_G_AB, G_AB, optimizer_G)
        # load G_BA
        G_BA, optimizer_G, _ = modelIO.load_pose_model(path_to_G_BA, G_BA, optimizer_G)
        # load PaD_A
        PaD_A, optimizer_PaD_A, _ = modelIO.load_pose_model(path_to_PaD_A, PaD_A, optimizer_PaD_A)
        # load PaD_B
        PaD_B, optimizer_PaD_B, _ = modelIO.load_pose_model(path_to_PaD_B, PaD_B, optimizer_PaD_B)

    scaler = torch.cuda.amp.GradScaler()
    NUM_ACCUMULATION_STEPS = 10
    i_folder = 0

    # step 6: the train loop
    for epoch in range(num_epoch):
        print(f"Epoch: {epoch}/{num_epoch}")
        train_loader = datasetIO.ucbm_dataloader(training_root_content, batch_size=batch_size, num_worker=num_worker, i_folder=i_folder)

        # initialize local losses for the current epoch
        loss_G_epoch = 0.0  # Initialize loss for this epoch
        loss_identity_epoch = 0.0
        loss_GAN_epoch = 0.0
        loss_D_epoch = 0.0
        loss_cycle_epoch = 0.0

        for batch, data in enumerate(train_loader):
            real_fr1 = data["rgb1"].to(DEVICE)
            real_fr2 = data["rgb2"].to(DEVICE)

            valid = torch.Tensor(np.ones((fr1.size(0), *PaD_shape.output_shape))).to(DEVICE)  # requires_grad = False. Default.
            fake = torch.Tensor(np.zeros((fr1.size(0), *PaD_shape.output_shape))).to(DEVICE)  # requires_grad = False. Default.

            # Training the Generator and Pose Network

            # Automatic Tensor Casting
            with torch.cuda.amp.autocast():
                # Estimate the pose
                estimated_pose_AB = PaD_B(real_fr1, real_fr2)
                estimated_pose_BA = PaD_A(real_fr2, real_fr1)

                # Identity Loss
                identity_motion = torch.zeros(estimated_pose_AB.shape[0], estimated_pose_AB.shape[1]).to(DEVICE)

                if standard:
                    # we compute the standard identity loss of the cyclegan
                    identity_fr1 = G_BA(real_fr1, identity_motion)
                    identity_fr2 = G_AB(real_fr2, identity_motion)
                    total_identity_loss = losses.standard_total_cycle_loss(identity_fr1, fr1, identity_fr2, fr2)

                else:
                    # we compute our custom identity loss
                    identity_fr1 = G_BA(real_fr1, identity_motion)
                    identity_fr2 = G_AB(real_fr2, identity_motion)
                    identity_p1 = PaD_A(identity_fr1, real_fr1)
                    identity_p2 = PaD_B(identity_fr2, real_fr2)
                    weights_identity_loss = [0.5, 0.5, 0.5, 0.5]

                    total_identity_loss = losses.custom_total_identity_loss(identity_fr1, real_fr1, identity_p1, identity_motion, identity_fr2, real_fr2, identity_p2, identity_motion, weights_identity_loss)

                # GAN loss
                fake_fr2 = G_AB(real_fr1, estimated_pose_AB)
                fake_fr1 = G_BA(real_fr2, estimated_pose_BA)
                loss_GAN = losses.standard_total_GAN_loss(PaD_B(curr_frame = fake_fr2), valid, PaD_A(curr_frame = fake_fr1), valid)

                # Cycle loss
                if standard:
                    # we compute the standard cycle loss of the cyclegan
                    recov_fr1 = G_BA(fake_fr2, estimated_pose_BA)
                    recov_fr2 = G_AB(fake_fr1, estimated_pose_AB)
                    total_cycle_loss = losses.standard_total_cycle_loss(recov_fr1, real_fr1, recov_fr2, real_fr2)
                else:
                    # we compute our custom cycle loss
                    recov_fr1 = G_BA(fake_fr2, estimated_pose_BA)
                    recov_fr2 = G_AB(fake_fr1, estimated_pose_AB)
                    recov_P12 = PaD_B(recov_fr1, recov_fr2)
                    recov_P21 = PaD_B(recov_fr2, recov_fr1)
                    weights_cycle_loss = [0.5, 0.5, 0.5, 0.5]
                    total_cycle_loss = losses.custom_total_cycle_loss(recov_fr1, real_fr1, recov_P12, estimated_pose_AB, recov_fr2, real_fr2, recov_P21, estimated_pose_BA, weights_cycle_loss)

            # Automatic Gradient Scaling
            scaler.scale(total_identity_loss).backward()
            scaler.scale(loss_GAN).backward()
            scaler.scale(total_cycle_loss).backward()

            # Normalize the gradients
            total_identity_loss = total_identity_loss / NUM_ACCUMULATION_STEPS
            loss_GAN = loss_GAN / NUM_ACCUMULATION_STEPS
            total_cycle_loss = total_cycle_loss / NUM_ACCUMULATION_STEPS

            # Gradient Accumulation
            if ((batch + 1) % NUM_ACCUMULATION_STEPS == 0) or (batch + 1 == len(train_loader)):
                scaler.step(optimizer_G)
                scaler.step(optimizer_PaD_A)
                scaler.step(optimizer_PaD_B)
                scaler.update()
                optimizer_G.zero_grad()
                optimizer_PaD_B.zero_grad()
                optimizer_PaD_A.zero_grad()

            # Garbage Collection
            torch.cuda.empty_cache()
            _ = gc.collect()

            # Training the Discriminator A

            # Automatic Tensor Casting
            with torch.cuda.amp.autocast():
                loss_DA = losses.standard_discriminator_loss(PaD_A(prev_frame = real_fr1), valid, PaD_A(prev_frame = fake_fr1.detach()), fake)

            # Automatic Gradient Scaling
            scaler.scale(loss_DA).backward()

            # Normalize the gradients
            loss_DA = loss_DA / NUM_ACCUMULATION_STEPS

            # Gradient Accumulation
            if ((batch + 1) % NUM_ACCUMULATION_STEPS == 0) or (batch + 1 == len(train_loader)):
                scaler.step(optimizer_PaD_A)
                scaler.update()
                optimizer_PaD_A.zero_grad()

            # Garbage Collection
            torch.cuda.empty_cache()
            _ = gc.collect()

            # Training the Discriminator B

            # Automatic Tensor Casting
            with torch.cuda.amp.autocast():
                loss_DB = losses.standard_discriminator_loss(PaD_B(curr_frame=real_fr2), valid,
                                                             PaD_A(curr_frame=fake_fr2.detach()), fake)

            # Automatic Gradient Scaling
            scaler.scale(loss_DB).backward()

            # Normalize the gradients
            loss_DB = loss_DB / NUM_ACCUMULATION_STEPS

            # Gradient Accumulation
            if ((batch + 1) % NUM_ACCUMULATION_STEPS == 0) or (batch + 1 == len(train_loader)):
                scaler.step(optimizer_PaD_B)
                scaler.update()
                optimizer_PaD_B.zero_grad()

            # Garbage Collection
            torch.cuda.empty_cache()
            _ = gc.collect()

            # TODO: implementa registrazione metriche
            # TODO: implementa salvataggio best model and normal model
            # TODO: implementa testing loop
















