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
import argparse
from tqdm import tqdm

# Numerical lib
import numpy as np

# Computer Vision lib
import cv2

# Metric lib
from evo.core import metrics
from evo.core import units
from evo.tools import file_interface

# Stat lib
import wandb

# AI-lib
import torch
from torch.optim import Adam



# Internal Module
from architecture_v3 import MultiTaskModel, ConditionalGenerator
from training_utils import TrainingLoss, LearnableScaleConsistencyLoss
from UTILS.io_utils import ModelIO, TXTIO
from dataloader import DatasetsIO
from UTILS.geometry_utils import PoseOperator


datasetIO = DatasetsIO()
modelIO = ModelIO()
poseOperator = PoseOperator()
txtIO = TXTIO()

def check_value(value):
    assert value == 1 or value == 0, "value must be 0 or 1"
    if value == 1:
        return True
    elif value == 0:
        return False



def train_model(training_dataset_path, testing_dataset_path, num_epoch, batch_size,
                path_to_the_model, load_model, standard_cycle, standard_identity,
                num_worker = 10, lr = 0.0002, input_shape = (1, 256, 256), weights_cycle_loss = [0.5, 0.5, 0.5, 0.5],
                weights_identity_loss = [0.5, 0.5, 0.5, 0.5], id = "0", id_wandb_run = ''):

    load_model = check_value(load_model)
    standard_cycle = check_value(standard_cycle)
    standard_identity = check_value(standard_identity)

    print(f"Num Epoch: {num_epoch}")
    print(f"Batch Size: {batch_size}")
    print(f"standard_cycle: {standard_cycle}")
    print(f"standard_identity: {standard_identity}")
    print(f"Load Model: {load_model}")
    print(f"weights_cycle_loss: {weights_cycle_loss}")
    print(f"weights_identity_loss: {weights_identity_loss}")
    print(f"id: {id}")
    print(f"wandb id run: {id_wandb_run}")

    if load_model:
        # we resume the run
        #wandb.init(project="Pose Estimator", id=id_wandb_run, resume='must')  # id 1
        pass

    else:
        # we start a new run
        wandb.init(
            project="CyclePoseV2",

            config={
                "learning_rate": 0.0002,
                "architecture": "CycleGan",
                "epoch": 200,
            }
        )



    
    # step 0: detect the DEVICE
    #DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #print(DEVICE)
    DEVICE = torch.device("cuda")

    best_metrics = {'ATE': float('inf'), 'ARE': float('inf'), 'RTE': float('inf'), 'RRE': float('inf')}

    # step 1: we need to load the datasets
    training_root_content = datasetIO.load_UCBM(training_dataset_path)
    testing_root_content = datasetIO.load_EndoSlam(testing_dataset_path)

    # step 2: we initialize the models
    G_AB = ConditionalGenerator(device=DEVICE, input_shape=input_shape).to(DEVICE) # the generator that from A generates B
    G_BA = ConditionalGenerator(device=DEVICE, input_shape=input_shape).to(DEVICE) # the generator that from B generates A
    PaD_A = MultiTaskModel(input_shape=input_shape, device = DEVICE).to(DEVICE) # the Pose estimator and the discriminator of A
    PaD_B = MultiTaskModel(input_shape=input_shape, device = DEVICE).to(DEVICE) # the Pose estimator and the discriminator of B

    PaD_shape = MultiTaskModel(input_shape=input_shape, device= DEVICE)

    # step 3: we initialize the optimizers
    optimizer_G = Adam(params=itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(0.5, 0.999)) # optimizer for the GANs
    optimizer_PaD_A = Adam(params=PaD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_PaD_B = Adam(params=PaD_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # step 4: we initialize the losses
    losses = TrainingLoss()
    scale_consistency_loss = LearnableScaleConsistencyLoss()

    # step 5: we load the model (if we need to load it)
    if load_model:
        path_to_G_AB = path_to_the_model + id + "_gen_ab.pth"
        path_to_G_BA = path_to_the_model + id + "_gen_ba.pth"
        path_to_PaD_A = path_to_the_model + id + "_PaD_A.pth"
        path_to_PaD_B = path_to_the_model + id + "_PaD_B.pth"

        # load G_AB
        G_AB, _, training_var = modelIO.load_pose_model(path_to_G_AB, G_AB, optimizer_G)
        # load G_BA
        G_BA, optimizer_G, _ = modelIO.load_pose_model(path_to_G_BA, G_BA, optimizer_G)
        # load PaD_A
        PaD_A, optimizer_PaD_A, _ = modelIO.load_pose_model(path_to_PaD_A, PaD_A, optimizer_PaD_A)
        # load PaD_B
        PaD_B, optimizer_PaD_B, _ = modelIO.load_pose_model(path_to_PaD_B, PaD_B, optimizer_PaD_B)

    i_folder = 0

    # step 6: the training loop
    for epoch in range(num_epoch):
        print(f"Epoch: {epoch}/{num_epoch}")
        print("[INFO]: training the model")
        train_loader, i_folder = datasetIO.ucbm_dataloader(training_root_content, batch_size=batch_size, num_worker=num_worker, i_folder=i_folder)

        # initialize local losses for the current epoch
        loss_G_epoch = 0.0  # Initialize loss for this epoch
        loss_GAN_epoch = 0.0
        loss_identity_epoch = 0.0
        loss_D_epoch = 0.0
        loss_cycle_epoch = 0.0
        num_batches = 0

        for batch, data in enumerate(train_loader):
            real_rgb1 = data["rgb1"]
            real_rgb2 = data["rgb2"]
            #real_dp1 = data["dp1"].unsqueeze(1)
            #real_dp2 = data["dp2"].unsqueeze(1)

            real_fr1 = real_rgb1.to(DEVICE)
            real_fr2 = real_rgb2.to(DEVICE)
            stacked_frame11 = torch.cat([real_fr1, real_fr1], dim=1)
            stacked_frame22 = torch.cat([real_fr2, real_fr2], dim=1)

            disc_output = (PaD_shape.output_shape[0], 2*PaD_shape.output_shape[1], 2*PaD_shape.output_shape[2])
            valid = torch.Tensor(np.ones((real_fr1.size(0), *disc_output))).to(DEVICE)  # requires_grad = False. Default.
            fake = torch.Tensor(np.zeros((real_fr1.size(0), *disc_output))).to(DEVICE)  # requires_grad = False. Default.

            # Training the Generator and Pose Network
            G_AB.train()
            G_BA.train()
            PaD_A.train()
            PaD_B.train()

            optimizer_G.zero_grad()


            # Estimate the pose
            stacked_frame12 = torch.cat([real_fr1, real_fr2], dim=1)
            stacked_frame21 = torch.cat([real_fr2, real_fr1], dim=1)
            estimated_pose_AB_SE3 = G_AB(stacked_frame12, mode="pose")
            estimated_pose_BA_SE3 = G_BA(stacked_frame21, mode="pose")

            print(estimated_pose_AB_SE3.shape)

            # Identity Loss
            #identity_motion = torch.zeros(estimated_pose_AB_SE3.shape[0], estimated_pose_AB_SE3.shape[1]).to(DEVICE)
            identity_motion = torch.eye(4).unsqueeze(0).expand(estimated_pose_AB_SE3.shape[0], -1, -1).to(DEVICE)

            if standard_identity:
                print("standard_id")
                # we compute the standard identity loss of the cyclegan
                identity_fr1 = G_BA(stacked_frame11, identity_motion)
                identity_fr2 = G_AB(stacked_frame22, identity_motion)
                total_identity_loss = losses.standard_total_cycle_loss(identity_fr1, real_fr1, identity_fr2, real_fr2)

            else:
                print("not standard_id")
                # we compute our custom identity loss
                identity_fr1 = G_BA(stacked_frame11, identity_motion)
                identity_fr2 = G_AB(stacked_frame22, identity_motion)
                identity_stacked_fr1 = torch.cat([identity_fr1, real_fr1], dim = 1)
                identity_stacked_fr2 = torch.cat([identity_fr2, real_fr2], dim=1)
                identity_p1 = G_BA(identity_stacked_fr1, mode = "pose")
                identity_p2 = G_AB(identity_stacked_fr2, mode = "pose")

                total_identity_loss = losses.custom_total_identity_loss(identity_fr1, real_fr1, identity_p1, identity_motion, identity_fr2, real_fr2, identity_p2, identity_motion, weights_identity_loss)

            # GAN loss
            fake_fr2 = G_AB(stacked_frame11, estimated_pose_AB_SE3)
            fake_fr1 = G_BA(stacked_frame22, estimated_pose_BA_SE3)
            curr_frame_fake_2 = torch.cat([fake_fr2, fake_fr2], dim = 1)
            curr_frame_fake_1 = torch.cat([fake_fr1, fake_fr1], dim = 1)
            loss_GAN = losses.standard_total_GAN_loss(PaD_B(curr_frame_fake_2, task = "discriminator"), valid, PaD_A(curr_frame_fake_1, task = "discriminator"), valid)


            # Cycle loss
            if standard_cycle:
                print("standard_cycle")
                # we compute the standard cycle loss of the cyclegan
                recov_fr1 = G_BA(curr_frame_fake_2, estimated_pose_BA_SE3)
                recov_fr2 = G_AB(curr_frame_fake_1, estimated_pose_AB_SE3)
                total_cycle_loss = losses.standard_total_cycle_loss(recov_fr1, real_fr1, recov_fr2, real_fr2)
            else:
                print("not standard_cycle")
                # we compute our custom cycle loss
                recov_fr1 = G_BA(curr_frame_fake_2, estimated_pose_BA_SE3)
                recov_fr2 = G_AB(curr_frame_fake_1, estimated_pose_AB_SE3)
                recov_fr12 = torch.cat([recov_fr1, recov_fr2], dim = 1)
                recov_fr21 = torch.cat([recov_fr2, recov_fr1], dim=1)
                recov_P12 = G_BA(recov_fr12, mode = "pose")
                recov_P21 = G_AB(recov_fr21, mode = "pose")
                total_cycle_loss = losses.custom_total_cycle_loss(recov_fr1, real_fr1, recov_P12, estimated_pose_AB_SE3, recov_fr2, real_fr2, recov_P21, estimated_pose_BA_SE3, weights_cycle_loss)

            scale_loss_AB = scale_consistency_loss(estimated_pose_AB_SE3)
            scale_loss_BA = scale_consistency_loss(estimated_pose_BA_SE3)
            total_scale_loss = (scale_loss_AB + scale_loss_BA)/2

            loss_G = loss_GAN + (10.0 * total_cycle_loss) + (5.0 * total_identity_loss) + total_scale_loss
            loss_G.backward()
            optimizer_G.step()

            # Training the Discriminator A

            optimizer_PaD_A.zero_grad()
            prev_frame_real = torch.cat([real_fr1, real_fr1], dim=1)
            prev_frame_fake = torch.cat([fake_fr1.detach(), fake_fr1.detach()], dim=1)
            loss_DA = losses.standard_discriminator_loss(PaD_A(prev_frame_real, task = 'discriminator'), valid, PaD_A(prev_frame_fake, task = 'discriminator'), fake)

            loss_DA.backward()
            optimizer_PaD_A.step()

            # Training the Discriminator B
            prev_frame_real = torch.cat([real_fr2, real_fr2], dim=1)
            prev_frame_fake = torch.cat([fake_fr2.detach(), fake_fr2.detach()], dim=1)
            optimizer_PaD_B.zero_grad()

            loss_DB = losses.standard_discriminator_loss(PaD_B(prev_frame_real, task = 'discriminator'), valid,
                                                             PaD_B(prev_frame_fake, task = 'discriminator'), fake)

            loss_DB.backward()
            optimizer_PaD_B.step()

            # total discriminator loss (not backwarded! -> used only for tracking)
            loss_D = (loss_DA + loss_DB)/2



            loss_G_epoch += loss_G.item()
            loss_GAN_epoch += loss_GAN.item()
            loss_D_epoch += loss_D.item()
            loss_cycle_epoch += total_cycle_loss.item()
            loss_identity_epoch += total_identity_loss.item()
            num_batches += 1

        wandb.log({"training_loss_G": loss_G_epoch/num_batches,
                   "training_loss_GAN": loss_GAN/num_batches,
                   "training_loss_D": loss_D_epoch/num_batches,
                   "training_loss_cycle": loss_cycle_epoch/num_batches,
                   "training_loss_identity": loss_identity_epoch/num_batches,
                   })


        # step 7: testing loop
        total_testing_pose_loss = 0.0
        loss_testing_G_epoch = 0.0
        loss_testing_GAN_epoch = 0.0
        loss_testing_identity_epoch = 0.0
        loss_testing_D_epoch = 0.0
        loss_testing_cycle_epoch = 0.0
        num_batches = 0
        count = 0

        ATE_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ARE_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
        RTE_metric = metrics.RPE(metrics.PoseRelation.translation_part)
        RRE_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_deg)


        ground_truth_pose = []
        predictions = []
        ATE = []
        ARE = []
        RRE = []
        RTE = []


        print("[INFO]: evaluating the model...")
        print(testing_root_content)
        for i in range(len(testing_root_content)):
            testing_loader = datasetIO.endoslam_dataloader(testing_root_content, batch_size=1, num_worker=num_worker, i=i)

            G_AB.eval()
            G_BA.eval()
            PaD_A.eval()
            PaD_B.eval()

            gt_list = [np.eye(4)]
            pd_list = [np.eye(4)]

            with torch.no_grad():
                for batch, data in enumerate(testing_loader):
                    real_rgb1 = data["rgb1"]
                    real_rgb2 = data["rgb2"]
                    #real_dp1 = data["dp1"].unsqueeze(1)
                    #real_dp2 = data["dp2"].unsqueeze(1)
                    pose_fr1 = data["target"][0].to(DEVICE).float()
                    pose_fr2 = data["target"][1].to(DEVICE).float()
                    relative_pose = data["target"][2].to(DEVICE).float()

                    real_fr1 = real_rgb1.to(DEVICE)
                    real_fr2 = real_rgb2.to(DEVICE)
                    stacked_frame11 = torch.cat([real_fr1, real_fr1], dim=1)
                    stacked_frame22 = torch.cat([real_fr2, real_fr2], dim=1)

                    # Evaluate GAN & POSE

                    # Adversarial ground truths
                    disc_output = (PaD_shape.output_shape[0], 2 * PaD_shape.output_shape[1], 2 * PaD_shape.output_shape[2])
                    valid = torch.Tensor(np.ones((real_fr1.size(0), *disc_output))).to(DEVICE)  # requires_grad = False. Default.
                    fake = torch.Tensor(np.zeros((real_fr1.size(0), *disc_output))).to(DEVICE)  # requires_grad = False. Default.

                    # Estimate the pose
                    stacked_frame12 = torch.cat([real_fr1, real_fr2], dim=1)
                    stacked_frame21 = torch.cat([real_fr2, real_fr1], dim=1)
                    estimated_pose_AB_SE3 = G_AB(stacked_frame12, mode="pose")
                    estimated_pose_BA_SE3 = G_BA(stacked_frame21, mode="pose")

                    # get the absolute poses
                    pd_rel = estimated_pose_AB_SE3.squeeze().cpu().numpy()
                    gt_rel = relative_pose.squeeze().cpu().numpy()
                    pd_abs = pd_list[-1] @ pd_rel
                    gt_abs = gt_list[-1] @ gt_rel
                    # ensure the matrix is SO3 valid (needed for metric computation)
                    pd_abs[:3, :3] = poseOperator.ensure_so3_v2(pd_abs[:3, :3])
                    gt_abs[:3, :3] = poseOperator.ensure_so3_v2(gt_abs[:3, :3])
                    gt_list.append(gt_abs)
                    pd_list.append(pd_abs)

                    # Identity Loss
                    #identity_motion = torch.zeros(estimated_pose_AB_SE3.shape[0], estimated_pose_AB_SE3.shape[1]).to(DEVICE)
                    identity_motion = torch.eye(4).unsqueeze(0).expand(estimated_pose_AB_SE3.shape[0], -1, -1).to(
                        DEVICE)

                    if standard_identity:
                        # we compute the standard identity loss of the cyclegan
                        identity_fr1 = G_BA(stacked_frame11, identity_motion)
                        identity_fr2 = G_AB(stacked_frame22, identity_motion)
                        total_identity_loss = losses.standard_total_cycle_loss(identity_fr1, real_fr1, identity_fr2, real_fr2)

                    else:
                        # we compute our custom identity loss
                        identity_fr1 = G_BA(stacked_frame11, identity_motion)
                        identity_fr2 = G_AB(stacked_frame22, identity_motion)
                        identity_stacked_fr1 = torch.cat([identity_fr1, real_fr1],dim=1)
                        identity_stacked_fr2 = torch.cat([identity_fr2, real_fr2], dim=1)
                        identity_p1 = G_BA(identity_stacked_fr1, mode = "pose")
                        identity_p2 = G_AB(identity_stacked_fr2, mode = "pose")

                        total_identity_loss = losses.custom_total_identity_loss(identity_fr1, real_fr1, identity_p1,
                                                                                identity_motion, identity_fr2, real_fr2,
                                                                                identity_p2, identity_motion,
                                                                                weights_identity_loss)

                    # GAN loss
                    fake_fr2 = G_AB(stacked_frame11, estimated_pose_AB_SE3)
                    fake_fr1 = G_BA(stacked_frame22, estimated_pose_BA_SE3)
                    curr_frame_fake_2 = torch.cat([fake_fr2, fake_fr2], dim=1)
                    curr_frame_fake_1 = torch.cat([fake_fr1, fake_fr1], dim=1)
                    loss_GAN = losses.standard_total_GAN_loss(PaD_B(curr_frame_fake_2, task="discriminator"), valid,
                                                              PaD_A(curr_frame_fake_1, task="discriminator"), valid)


                    # Cycle loss
                    if standard_cycle:
                        # we compute the standard cycle loss of the cyclegan
                        recov_fr1 = G_BA(curr_frame_fake_2, estimated_pose_BA_SE3)
                        recov_fr2 = G_AB(curr_frame_fake_1, estimated_pose_AB_SE3)
                        total_cycle_loss = losses.standard_total_cycle_loss(recov_fr1, real_fr1, recov_fr2, real_fr2)
                    else:
                        # we compute our custom cycle loss
                        recov_fr1 = G_BA(curr_frame_fake_2, estimated_pose_BA_SE3)
                        recov_fr2 = G_AB(curr_frame_fake_1, estimated_pose_AB_SE3)
                        recov_fr12 = torch.cat([recov_fr1, recov_fr2], dim = 1)
                        recov_fr21 = torch.cat([recov_fr2, recov_fr1], dim = 1)
                        recov_P12 = G_BA(recov_fr12, mode = "pose")
                        recov_P21 = G_AB(recov_fr21, mode = "pose")
                        total_cycle_loss = losses.custom_total_cycle_loss(recov_fr1, real_fr1, recov_P12, estimated_pose_AB_SE3,
                                                                          recov_fr2, real_fr2, recov_P21, estimated_pose_BA_SE3,
                                                                          weights_cycle_loss)

                    scale_loss_AB = scale_consistency_loss(estimated_pose_AB_SE3)
                    scale_loss_BA = scale_consistency_loss(estimated_pose_BA_SE3)
                    total_scale_loss = (scale_loss_AB + scale_loss_BA)/2
                    loss_G = loss_GAN + (10.0 * total_cycle_loss) + (5.0 * total_identity_loss) + total_scale_loss
                    # Evaluate Discriminators
                    prev_frame_real = torch.cat([real_fr1, real_fr1], dim=1)
                    prev_frame_fake = torch.cat([fake_fr1.detach(), fake_fr1.detach()], dim=1)
                    loss_DA = losses.standard_discriminator_loss(PaD_A(prev_frame_real, task='discriminator'), valid,
                                                                 PaD_A(prev_frame_fake, task='discriminator'), fake)

                    loss_DB = losses.standard_discriminator_loss(PaD_B(prev_frame_real, task='discriminator'), valid,
                                                                 PaD_B(prev_frame_fake, task='discriminator'), fake)


                    # total discriminator loss (not backwarded! -> used only for tracking)
                    loss_D = (loss_DA + loss_DB) / 2

                    loss_testing_G_epoch += loss_G.item()
                    loss_testing_GAN_epoch += loss_GAN.item()
                    loss_testing_D_epoch += loss_D.item()
                    loss_testing_cycle_epoch += total_cycle_loss.item()
                    loss_testing_identity_epoch += total_identity_loss.item()

                    num_batches += 1


            # compute ATE, ARE, RRE, RTE
            # save trajectory in kitti format
            path_to_tmp_gt = "/mimer/NOBACKUP/groups/snic2022-5-277/gmanni/cyclepose/MPEM/tmp_test_pose/" + id + "_gt.txt"
            path_to_tmp_pd = "/mimer/NOBACKUP/groups/snic2022-5-277/gmanni/cyclepose/MPEM/tmp_test_pose/" + id + "_pd.txt"
            txtIO.save_poses_as_kitti(gt_list, path_to_tmp_gt)
            txtIO.save_poses_as_kitti(pd_list, path_to_tmp_pd)

            # now load trajectories with evo
            gt_traj = file_interface.read_kitti_poses_file(path_to_tmp_gt)
            pd_traj = file_interface.read_kitti_poses_file(path_to_tmp_pd)

            # align and correct the scale
            pd_traj.align_origin(gt_traj)
            pd_traj.align(gt_traj, correct_scale=True)

            data = (gt_traj, pd_traj)
            ATE_metric.process_data(data)
            ARE_metric.process_data(data)
            RTE_metric.process_data(data)
            RRE_metric.process_data(data)

            ate = ATE_metric.get_statistic(metrics.StatisticsType.rmse)
            are = ARE_metric.get_statistic(metrics.StatisticsType.rmse)
            rte = RTE_metric.get_statistic(metrics.StatisticsType.rmse)
            rre = RRE_metric.get_statistic(metrics.StatisticsType.rmse)



            ATE.append(ate)
            ARE.append(are)
            RRE.append(rre)
            RTE.append(rte)

        ate = 0.0
        are = 0.0
        rre = 0.0
        rte = 0.0
        num_dataset = len(ATE)
        for i in range(num_dataset):
            ate += ATE[i]
            are += ARE[i]
            rre += RRE[i]
            rte += RTE[i]

        
        # compute the other metrics
        wandb.log({"testing_loss_G": loss_testing_G_epoch / num_batches,
                   "testing_loss_GAN": loss_testing_GAN_epoch / num_batches,
                   "testing_loss_D": loss_testing_D_epoch / num_batches,
                   "testing_loss_cycle": loss_testing_cycle_epoch / num_batches,
                   "testing_loss_identity": loss_testing_identity_epoch / num_batches,
                   "ATE": ate/num_dataset,
                   "ARE": are/num_dataset,
                   "RRE": rre/num_dataset,
                   "RTE": rte/num_dataset,
                   })
        

        # Check if the current metrics are the best so far
        saving_path = path_to_the_model + "model.pth"
        avg_pose_metrics = (ate + are + rte + rre) / 4
        avg_best_pose_metrics = (best_metrics['ATE'] + best_metrics['ARE'] + best_metrics['RTE'] + best_metrics['RRE'])/4
        if avg_pose_metrics < avg_best_pose_metrics:
            best_metrics.update({'ATE': ate, 'ARE': are, 'RTE': rte, 'RRE': rre})
            print("[INFO]: saving the best models")
            saving_path_gab = path_to_the_model + id + "_model_gen_ab.pth"
            saving_path_gba = path_to_the_model + id + "_model_gen_ba.pth"
            saving_path_pada = path_to_the_model + id + "_model_PaD_A.pth"
            saving_path_padb = path_to_the_model + id + "_model_PaD_B.pth"
            training_var = {'epoch': epoch,
                            'iter_on_ucbm': i_folder,
                            'ate': ate,
                            'are': are,
                            'rre': rre,
                            'rte': rte,}
            # save generators
            modelIO.save_pose_model(saving_path_gab, G_AB, optimizer_G, training_var, best_model=True)
            modelIO.save_pose_model(saving_path_gba, G_BA, optimizer_G, training_var, best_model=True)
            # save pose and discriminator model
            modelIO.save_pose_model(saving_path_pada, PaD_A, optimizer_PaD_A, training_var, best_model=True)
            modelIO.save_pose_model(saving_path_padb, PaD_B, optimizer_PaD_B, training_var, best_model=True)
        else:
            # we save the model as a normal one:
            print("[INFO]: saving the models")
            saving_path_gab = path_to_the_model + id + "_gen_ab.pth"
            saving_path_gba = path_to_the_model + id + "_gen_ba.pth"
            saving_path_pada = path_to_the_model + id + "_PaD_A.pth"
            saving_path_padb = path_to_the_model + id + "_PaD_B.pth"
            training_var = {'epoch': epoch,
                            'iter_on_ucbm': i_folder,
                            'ate': ate,
                            'are': are,
                            'rre': rre,
                            'rte': rte, }
            # save generators
            modelIO.save_pose_model(saving_path_gab, G_AB, optimizer_G, training_var, best_model=False)
            modelIO.save_pose_model(saving_path_gba, G_BA, optimizer_G, training_var, best_model=False)
            # save pose and discriminator model
            modelIO.save_pose_model(saving_path_pada, PaD_A, optimizer_PaD_A, training_var, best_model=False)
            modelIO.save_pose_model(saving_path_padb, PaD_B, optimizer_PaD_B, training_var, best_model=False)






'''
training_dataset_path = "/home/gvide/Dataset/Odometry/UCBM_ODO/"
testing_dataset_path = "/home/gvide/Dataset/Odometry/Endo_mod/testing/"
path_to_model = "/home/gvide/PycharmProjects/SurgicalSlam/MPEM/Model/"
'''
# to logout remove key here /cephyr/users/soda/Alvis/.netrc





parser = argparse.ArgumentParser(description="Train a model with specified parameters")

parser.add_argument("--training_dataset_path", type=str, help="Path to the training dataset")
parser.add_argument("--testing_dataset_path", type=str, help="Path to the testing dataset")
parser.add_argument("--num_epoch", type=int, help="Number of epochs for training")
parser.add_argument("--batch_size", type=int, help="Batch size for training")
parser.add_argument("--path_to_the_model", type=str, help="Path to save the trained model")
parser.add_argument("--load_model", type=int, help="Flag to indicate whether to load a pre-trained model")

parser.add_argument("--num_worker", type=int, default=10, help="Number of workers (default: 10)")
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default: 0.0002)")
parser.add_argument("--input_shape", type=int, nargs=3, default=[6, 128, 128], help="Input shape as a list (default: [3, 256, 256])")
parser.add_argument("--standard_cycle", type=int, help="Standard flag (default: False)")
parser.add_argument("--standard_identity", type=int, help="Standard flag (default: False)")
parser.add_argument("--weigths_id_loss", nargs='*', type=float)
parser.add_argument("--weigths_cycle_loss", nargs='*', type=float)
parser.add_argument("--id", type=str)
parser.add_argument("--id_wandb_run", type=str)

args = parser.parse_args()

if not os.path.exists(args.path_to_the_model):
    os.mkdir(args.path_to_the_model)

train_model(
    args.training_dataset_path,
    args.testing_dataset_path,
    args.num_epoch,
    args.batch_size,
    args.path_to_the_model,
    args.load_model,
    num_worker=args.num_worker,
    lr=args.lr,
    input_shape=tuple(args.input_shape),
    standard_cycle=args.standard_cycle,
    standard_identity=args.standard_identity,
    weights_cycle_loss=args.weigths_cycle_loss,
    weights_identity_loss=args.weigths_id_loss,
    id=args.id,
    id_wandb_run=args.id_wandb_run
)










