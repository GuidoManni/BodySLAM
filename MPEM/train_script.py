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

# Stat lib
import wandb

# AI-lib
import torch
from torch.optim import Adam



# Internal Module
from architecture import MultiTaskModel, ConditionalGenerator
from training_utils import TrainingLoss
from UTILS.io_utils import ModelIO
from dataloader import DatasetsIO
from UTILS.geometry_utils import PoseOperator


datasetIO = DatasetsIO()
modelIO = ModelIO()
poseOperator = PoseOperator()

def check_value(value):
    assert value == 1 or value == 0, "value must be 0 or 1"
    if value == 1:
        return True
    elif value == 0:
        return False


def train_model(training_dataset_path, testing_dataset_path, num_epoch, batch_size,
                path_to_the_model, load_model, standard_cycle, standard_identity,
                num_worker = 10, lr = 0.0002, input_shape = (3, 256, 256), weights_cycle_loss = [0.5, 0.5, 0.5, 0.5],
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
            project="Pose Estimator",

            config={
                "learning_rate": 0.001,
                "architecture": "CycleGan",
                "epoch": 500,
                "standard": True,
                "weights": "[0.5, 0.5, 0.5, 0.5]"
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
            real_fr1 = data["rgb1"].to(DEVICE)
            real_fr2 = data["rgb2"].to(DEVICE)

            valid = torch.Tensor(np.ones((real_fr1.size(0), *PaD_shape.output_shape))).to(DEVICE)  # requires_grad = False. Default.
            fake = torch.Tensor(np.zeros((real_fr1.size(0), *PaD_shape.output_shape))).to(DEVICE)  # requires_grad = False. Default.

            # Training the Generator and Pose Network
            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()


            # Estimate the pose
            estimated_pose_AB_SE3, estimated_pose_AB = PaD_B(real_fr1, real_fr2)
            estimated_pose_BA_SE3, estimated_pose_BA = PaD_A(real_fr2, real_fr1)

            # Identity Loss
            identity_motion = torch.zeros(estimated_pose_AB.shape[0], estimated_pose_AB.shape[1]).to(DEVICE)

            if standard_identity:
                print("standard_id")
                # we compute the standard identity loss of the cyclegan
                identity_fr1 = G_BA(real_fr1, identity_motion)
                identity_fr2 = G_AB(real_fr2, identity_motion)
                total_identity_loss = losses.standard_total_cycle_loss(identity_fr1, real_fr1, identity_fr2, real_fr2)

            else:
                print("not standard_id")
                # we compute our custom identity loss
                identity_fr1 = G_BA(real_fr1, identity_motion)
                identity_fr2 = G_AB(real_fr2, identity_motion)
                _, identity_p1 = PaD_A(identity_fr1, real_fr1)
                _, identity_p2 = PaD_B(identity_fr2, real_fr2)

                total_identity_loss = losses.custom_total_identity_loss(identity_fr1, real_fr1, identity_p1, identity_motion, identity_fr2, real_fr2, identity_p2, identity_motion, weights_identity_loss)

            # GAN loss
            fake_fr2 = G_AB(real_fr1, estimated_pose_AB)
            fake_fr1 = G_BA(real_fr2, estimated_pose_BA)
            loss_GAN = losses.standard_total_GAN_loss(PaD_B(curr_frame = fake_fr2), valid, PaD_A(curr_frame = fake_fr1), valid)

            # Cycle loss
            if standard_cycle:
                print("standard_cycle")
                # we compute the standard cycle loss of the cyclegan
                recov_fr1 = G_BA(fake_fr2, estimated_pose_BA)
                recov_fr2 = G_AB(fake_fr1, estimated_pose_AB)
                total_cycle_loss = losses.standard_total_cycle_loss(recov_fr1, real_fr1, recov_fr2, real_fr2)
            else:
                print("not standard_cycle")
                # we compute our custom cycle loss
                recov_fr1 = G_BA(fake_fr2, estimated_pose_BA)
                recov_fr2 = G_AB(fake_fr1, estimated_pose_AB)
                _, recov_P12 = PaD_A(recov_fr1, recov_fr2)
                _, recov_P21 = PaD_B(recov_fr2, recov_fr1)
                total_cycle_loss = losses.custom_total_cycle_loss(recov_fr1, real_fr1, recov_P12, estimated_pose_AB, recov_fr2, real_fr2, recov_P21, estimated_pose_BA, weights_cycle_loss)

            loss_G = loss_GAN + (10.0 * total_cycle_loss) + (5.0 * total_identity_loss)
            loss_G.backward()
            optimizer_G.step()

            # Training the Discriminator A

            optimizer_PaD_A.zero_grad()

            loss_DA = losses.standard_discriminator_loss(PaD_A(prev_frame = real_fr1), valid, PaD_A(prev_frame = fake_fr1.detach()), fake)

            loss_DA.backward()
            optimizer_PaD_A.step()

            # Training the Discriminator B

            optimizer_PaD_B.zero_grad()

            loss_DB = losses.standard_discriminator_loss(PaD_B(curr_frame=real_fr2), valid,
                                                             PaD_A(curr_frame=fake_fr2.detach()), fake)

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
                    estimated_pose_AB_SE3, estimated_pose_AB = PaD_B(real_fr1, real_fr2)
                    estimated_pose_BA_SE3, estimated_pose_BA = PaD_A(real_fr2, real_fr1)


                    gt.append(target.squeeze().cpu().numpy())
                    pd.append(estimated_pose_AB_SE3.squeeze().cpu().numpy())

                    # Identity Loss
                    identity_motion = torch.zeros(estimated_pose_AB.shape[0], estimated_pose_AB.shape[1]).to(DEVICE)

                    if standard_identity:
                        # we compute the standard identity loss of the cyclegan
                        identity_fr1 = G_BA(real_fr1, identity_motion)
                        identity_fr2 = G_AB(real_fr2, identity_motion)
                        total_identity_loss = losses.standard_total_cycle_loss(identity_fr1, real_fr1, identity_fr2, real_fr2)

                    else:
                        # we compute our custom identity loss
                        identity_fr1 = G_BA(real_fr1, identity_motion)
                        identity_fr2 = G_AB(real_fr2, identity_motion)
                        _, identity_p1 = PaD_A(identity_fr1, real_fr1)
                        _, identity_p2 = PaD_B(identity_fr2, real_fr2)

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
                    if standard_cycle:
                        # we compute the standard cycle loss of the cyclegan
                        recov_fr1 = G_BA(fake_fr2, estimated_pose_BA)
                        recov_fr2 = G_AB(fake_fr1, estimated_pose_AB)
                        total_cycle_loss = losses.standard_total_cycle_loss(recov_fr1, real_fr1, recov_fr2, real_fr2)
                    else:
                        # we compute our custom cycle loss
                        recov_fr1 = G_BA(fake_fr2, estimated_pose_BA)
                        recov_fr2 = G_AB(fake_fr1, estimated_pose_AB)
                        _, recov_P12 = PaD_A(recov_fr1, recov_fr2)
                        _, recov_P21 = PaD_B(recov_fr2, recov_fr1)
                        total_cycle_loss = losses.custom_total_cycle_loss(recov_fr1, real_fr1, recov_P12, estimated_pose_AB,
                                                                          recov_fr2, real_fr2, recov_P21, estimated_pose_BA,
                                                                          weights_cycle_loss)

                    loss_G = loss_GAN + (10.0 * total_cycle_loss) + (5.0 * total_identity_loss)
                    # Evaluate Discriminators
                    loss_DA = losses.standard_discriminator_loss(PaD_A(prev_frame=real_fr1), valid,
                                                                 PaD_A(prev_frame=fake_fr1.detach()), fake)

                    loss_DB = losses.standard_discriminator_loss(PaD_B(curr_frame=real_fr2), valid,
                                                                 PaD_A(curr_frame=fake_fr2.detach()), fake)

                    # total discriminator loss (not backwarded! -> used only for tracking)
                    loss_D = (loss_DA + loss_DB) / 2

                    loss_testing_G_epoch += loss_G.item()
                    loss_testing_GAN_epoch += loss_GAN.item()
                    loss_testing_D_epoch += loss_D.item()
                    loss_testing_cycle_epoch += total_cycle_loss.item()
                    loss_testing_identity_epoch += total_identity_loss.item()

                    num_batches += 1

            # before computing ATE and ARE
            print(f"pd: {len(pd)}")
            print(f"gt: {len(gt)}")
            print(pd)
            absolute_ground_truth = poseOperator.integrate_relative_poses(gt)
            absolute_predictions = poseOperator.integrate_relative_poses(pd)

            # compute ATE and ARE
            #ate, are = losses.absolute_pose_error(gt, absolute_predictions)
            ate, are = losses.compute_ARE_and_ATE(absolute_ground_truth, absolute_predictions)

            # compute RTE and RRE
            #rre, rte = losses.relative_pose_error(relative_ground_truth, pd)
            rre, rte = losses.compute_RRE_and_RTE(gt, pd)

            ATE.append(ate)
            ARE.append(are)
            RRE.append(rre)
            RTE.append(rte)

        '''
        # before computing ATE and ARE
        absolute_predictions = poseOperator.integrate_relative_poses(predictions)

        relative_ground_truth = poseOperator.get_relative_poses(ground_truth_pose)



        # compute ATE and ARE
        ate, are = losses.absolute_pose_error(ground_truth_pose, absolute_predictions)

        # compute RTE and RRE
        rre, rte = losses.relative_pose_error(relative_ground_truth, predictions)
        '''
        ate = 0.0
        are = 0.0
        rre = 0.0
        rte = 0.0
        num_dataset = len(ATE)
        for i in range(num_dataset):
            ate += ATE[i]
            are += ARE[i]
            rre += RTE[i]
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
        if ate < best_metrics['ATE'] and are < best_metrics['ARE'] and rte < best_metrics['RTE'] and rre < \
                best_metrics['RRE']:
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
parser.add_argument("--input_shape", type=int, nargs=3, default=[3, 256, 256], help="Input shape as a list (default: [3, 256, 256])")
parser.add_argument("--standard_cycle", type=int, help="Standard flag (default: False)")
parser.add_argument("--standard_identity", type=int, help="Standard flag (default: False)")
parser.add_argument("--weigths_id_loss", nargs='*', type=float)
parser.add_argument("--weigths_cycle_loss", nargs='*', type=float)
parser.add_argument("--id", type=str)
parser.add_argument("--id_wandb_run", type=str)

args = parser.parse_args()

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










