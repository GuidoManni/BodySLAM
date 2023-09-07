'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 07/09/23

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

def train_model(training_dataset_path, testing_dataset_path, num_epoch, batch_size, path_to_the_model, load_model, num_worker = 10, lr = 0.0002, input_shape = (3, 256, 256), standard = False):
    # step 0: detect the DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    best_metrics = {'ATE': float('inf'), 'ARE': float('inf'), 'RTE': float('inf'), 'RRE': float('inf')}

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

    # step 6: the training loop
    for epoch in range(num_epoch):
        print(f"Epoch: {epoch}/{num_epoch}")
        train_loader, i_folder = datasetIO.ucbm_dataloader(training_root_content, batch_size=batch_size, num_worker=num_worker, i_folder=i_folder)

        # initialize local losses for the current epoch
        loss_G_epoch = 0.0  # Initialize loss for this epoch
        loss_identity_epoch = 0.0
        loss_D_epoch = 0.0
        loss_cycle_epoch = 0.0
        num_batches = 0

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

            # total discriminator loss (not backwarded! -> used only for tracking)
            loss_D = (loss_DA + loss_DB)/2

            # Garbage Collection
            torch.cuda.empty_cache()
            _ = gc.collect()

            loss_G_epoch += loss_GAN.item()
            loss_D_epoch += loss_D.item()
            loss_cycle_epoch += total_cycle_loss.item()
            loss_identity_epoch += total_identity_loss.item()
            num_batches += 1

        wandb.log({"training_loss_G": loss_G_epoch/num_batches,
                   "training_loss_D": loss_D_epoch/num_batches,
                   "training_loss_cycle": loss_cycle_epoch/num_batches,
                   "training_loss_identity": loss_identity_epoch/num_batches,
                   })

        # step 7: testing loop
        total_testing_pose_loss = 0.0
        loss_testing_G_epoch = 0.0
        loss_testing_identity_epoch = 0.0
        loss_testing_D_epoch = 0.0
        loss_testing_cycle_epoch = 0.0
        num_batches = 0

        ground_truth_pose = []
        predictions = []

        print("[INFO]: evaluating the model...")
        for i in range(len(testing_root_content)):
            testing_loader = datasetIO.endoslam_dataloader(testing_root_content, batch_size=batch_size, num_worker=num_worker, i=i)

            G_AB.eval()
            G_BA.eval()
            PaD_A.eval()
            PaD_B.eval()

            gt = []
            pd = []

            with torch.no_grad():
                for batch, data in enumerate(testing_loader):
                    real_fr1 = data["rgb1"].to(DEVICE)
                    real_fr2 = data["rgb2"].to(DEVICE)
                    target = data["target"].to(DEVICE).float()

                    # Evaluate GAN & POSE

                    # Adversarial ground truths
                    valid = torch.Tensor(np.ones((real_fr1.size(0), *PaD_shape.output_shape))).to(DEVICE)  # requires_grad = False. Default.
                    fake = torch.Tensor(np.zeros((real_fr1.size(0), *PaD_shape.output_shape))).to(DEVICE)  # requires_grad = False. Default.

                    # Estimate the pose
                    estimated_pose_AB = PaD_B(real_fr1, real_fr2)
                    estimated_pose_BA = PaD_A(real_fr2, real_fr1)

                    gt.append(target.numpy().cpu())
                    pd.append(estimated_pose_AB.numpy().cpu())

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

                        total_identity_loss = losses.custom_total_identity_loss(identity_fr1, real_fr1, identity_p1,
                                                                                identity_motion, identity_fr2, real_fr2,
                                                                                identity_p2, identity_motion,
                                                                                weights_identity_loss)

                    # GAN loss
                    fake_fr2 = G_AB(real_fr1, estimated_pose_AB)
                    fake_fr1 = G_BA(real_fr2, estimated_pose_BA)
                    loss_GAN = losses.standard_total_GAN_loss(PaD_B(curr_frame=fake_fr2), valid, PaD_A(curr_frame=fake_fr1),
                                                              valid)

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
                        total_cycle_loss = losses.custom_total_cycle_loss(recov_fr1, real_fr1, recov_P12, estimated_pose_AB,
                                                                          recov_fr2, real_fr2, recov_P21, estimated_pose_BA,
                                                                          weights_cycle_loss)
                    # Evaluate Discriminators
                    loss_DA = losses.standard_discriminator_loss(PaD_A(prev_frame=real_fr1), valid,
                                                                 PaD_A(prev_frame=fake_fr1.detach()), fake)

                    loss_DB = losses.standard_discriminator_loss(PaD_B(curr_frame=real_fr2), valid,
                                                                 PaD_A(curr_frame=fake_fr2.detach()), fake)

                    # total discriminator loss (not backwarded! -> used only for tracking)
                    loss_D = (loss_DA + loss_DB) / 2

                    loss_testing_G_epoch += loss_GAN.item()
                    loss_testing_D_epoch += loss_D.item()
                    loss_testing_cycle_epoch += total_cycle_loss.item()
                    loss_testing_identity_epoch += total_identity_loss.item()

                    num_batches += 1

            ground_truth_pose.append(gt)
            predictions.append(pd)

        # compute ATE and ARE
        ate, are = losses.absolute_pose_error(ground_truth_pose, predictions)

        # compute RTE and RRE
        rre, rte = losses.relative_pose_error(ground_truth_pose, predictions)

        # compute the other metrics
        wandb.log({"testing_loss_G": loss_testing_G_epoch / num_batches,
                   "testing_loss_D": loss_testing_D_epoch / num_batches,
                   "testing_loss_cycle": loss_testing_cycle_epoch / num_batches,
                   "testing_loss_identity": loss_testing_identity_epoch / num_batches,
                   "ATE": ate,
                   "ARE": are,
                   "RRE": rre,
                   "RTE": rte,
                   })

        # Check if the current metrics are the best so far
        if ate < best_metrics['ATE'] and are < best_metrics['ARE'] and rte < best_metrics['RTE'] and rre < \
                best_metrics['RRE']:
            best_metrics.update({'ATE': ate, 'ARE': are, 'RTE': rte, 'RRE': rre})
            print("[INFO]: saving the best models")
            saving_path = ""
            training_var = {'epoch': epoch,
                            'iter_on_ucbm': i_folder,
                            'ate': ate,
                            'are': are,
                            'rre': rre,
                            'rte': rte,}
            # save generators
            modelIO.save_pose_model(saving_path, G_AB, optimizer_G, training_var, best_model=True)
            modelIO.save_pose_model(saving_path, G_BA, optimizer_G, training_var, best_model=True)
            # save pose and discriminator model
            modelIO.save_pose_model(saving_path, PaD_A, optimizer_PaD_A, training_var, best_model=True)
            modelIO.save_pose_model(saving_path, PaD_B, optimizer_PaD_B, training_var, best_model=True)
        else:
            # we save the model as a normal one:
            print("[INFO]: saving the models")
            saving_path = ""
            training_var = {'epoch': epoch,
                            'iter_on_ucbm': i_folder,
                            'ate': ate,
                            'are': are,
                            'rre': rre,
                            'rte': rte, }
            # save generators
            modelIO.save_pose_model(saving_path, G_AB, optimizer_G, training_var, best_model=False)
            modelIO.save_pose_model(saving_path, G_BA, optimizer_G, training_var, best_model=False)
            # save pose and discriminator model
            modelIO.save_pose_model(saving_path, PaD_A, optimizer_PaD_A, training_var, best_model=False)
            modelIO.save_pose_model(saving_path, PaD_B, optimizer_PaD_B, training_var, best_model=False)






wandb.init(
            project="Pose Estimator",

            config={
                "learning_rate": 0.001,
                "architecture": "CycleGan",
                "dataset": "Kitti",
                "epoch": 100,
                "standard": True,
                "weights": "[0.5, 0.5, 0.5, 0.5]"
            }
        )


training_datset_path = "/home/gvide/Dataset/EndoSlam"
testing_dataset_path = "/home/gvide/Dataset/EndoSlam_testing/"
train_model()








