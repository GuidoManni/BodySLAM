'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23

Description:
Provide helper function or classes for the training script
'''


# AI-lib
import torch
import torch.nn as nn

class TrainingLoss:
    def __init__(self):
        '''
        The init will automatically initialize all the loss function and pass it to the proper device
        '''
        # Losses employed during training
        self.standard_identity_loss = nn.L1Loss()
        self.standard_cycle_loss = nn.L1Loss()
        self.standard_GAN_loss = nn.MSELoss()
        self.standard_Discr_loss = nn.MSELoss()

    def standard_criterion_identity_loss(self, fake_of_the_real, real):
        '''
        This loss check:
        - If i put A into a generator that creates A with B, how much is the result is similar to the real A
        - If i put B into a generator that creates B with A, how much is the result is similar to the real B

        Parameters:
        - fake_of_the_real: it is the generated images passed to the generator that should reproduce it
        - real: it is the input of the generator

        Return:
        - standard_identity_loss
        '''

        standard_identity_loss = self.standard_identity_loss(fake_of_the_real, real)

        return standard_identity_loss

    def standard_total_identity_loss(self, identity_frA, real_frA, identity_frB, real_frB):
        '''
        Compute the standard total identity loss following this formula:

        standard_total_identity_loss = (loss_identity_A + loss_identity_B)/2

        Parameters:
        - identity_frA: the frame obtained from using GBA(real_A)
        - real_frA: the real frame A
        - identity_frB: the frame obtained from using GAB(real_B)
        - real_frB: the real frame B

        Return:
        - standard_total_identity_loss
        '''
        loss_identity_frA = self.standard_criterion_identity_loss(identity_frA, real_frA)
        loss_identity_frB = self.standard_criterion_identity_loss(identity_frB, real_frB)

        standard_total_identity_loss = (loss_identity_frA + loss_identity_frB) / 2

        return standard_total_identity_loss

    def custom_total_identity_loss(self, identity_frA, real_frA, identity_PA, real_identity_PA, identity_frB, real_frB, identity_PB, real_identity_PB, weights):
        '''
        Compute a modified total cycle loss where the identity constraint is also applied on the pose. The formula used is:

        custom_total_identity_loss = (weights[0] * loss_cycle_frA + weights[1] * loss_cycle_frB + weights[2] * loss_cycle_PA + weights[3] * loss_cycle_PB)

        Parameters:
        - identity_frA: the frame obtained from using GBA(real_A)
        - real_frA: the real frame A
        - identity_PA: Estimated_Pose(identity_frA, real_A)
        - real_identity_PA: Estimated_Pose(real_A, real_A)
        - identity_frB: the frame obtained from using GAB(real_B)
        - real_frB: the real frame B
        - identity_PB: Estimated_Pose(identity_frB, real_B)
        - real_identity_PB: Estimated_Pose(real_B, real_B)
        - weights: list of weights for the weigthed sum

        Return:
        - custom_total_cycle_loss
        '''
        loss_identity_frA = self.standard_criterion_identity_loss(identity_frA, real_frA)
        loss_identity_frB = self.standard_criterion_identity_loss(identity_frB, real_frB)
        loss_identity_PA = self.standard_criterion_identity_loss(identity_PA, real_identity_PA)
        loss_identity_PB = self.standard_criterion_identity_loss(identity_PB, real_identity_PB)

        custom_total_identity_loss = (weights[0] * loss_identity_frA + weights[1] * loss_identity_frB + weights[2] * loss_identity_PA + weights[3] * loss_identity_PB)

        return custom_total_identity_loss

    def standard_criterion_cycle_loss(self, recov, real):
        '''
        This loss check:
        - how much result_A_fake = GBA(fake_B) is similar to real_A
        - how much result_B_fake = GAB(fake_A) is similar to real_B

        Parameters:
        - recov: GBA(fake_B) or GAB(fake_A)
        - real: real_A or real_B

        Returns:
        - standard_cycle_loss
        '''

        standard_cycle_loss = self.standard_cycle_loss(recov, real)

        return standard_cycle_loss

    def standard_total_cycle_loss(self, recov_frA, real_frA, recov_frB, real_frB):
        '''
        Compute the standard total cycle loss following this formula:

        standard_total_cycle_loss = (loss_cycle_A + loss_cycle_B)/2

        Parameters:
        - recov_frA: the frame obtained from using GBA(fake_B)
        - recov_frB: the frame obtained from using GAB(fake_A)
        - real_frA: the real frame A
        - real_frB: the real frame B

        Returns
        - standard_total_cycle_loss
        '''

        loss_cycle_A = self.standard_criterion_cycle_loss(recov_frA, real_frA)
        loss_cycle_B = self.standard_criterion_cycle_loss(recov_frB, real_frB)

        standard_total_cycle_loss = (loss_cycle_A + loss_cycle_B) / 2

        return standard_total_cycle_loss

    def custom_total_cycle_loss(self, recov_frA, real_frA, recov_PA, real_PA, recov_frB, real_frB, recov_PB, real_PB, weights = [0.5, 0.5, 0.5, 0.5]):
        '''
        Compute a modified total cycle loss where the identity constraint is also applied on the pose. The formula used is:

        custom_total_cycle_loss = (weights[0] * loss_cycle_frA + weights[1] * loss_cycle_frB + weights[2] * loss_cycle_PA + weights[3] * loss_cycle_PB)

        Parameters:
        - recov_frA: the frame obtained from using GBA(fake_B)
        - real_frA: the real frame A
        - recov_PA: Estimated_Pose(recov_frA, real_B)
        - real_PA: Estimated_Pose(real_A, real_B)
        - recov_frB: the frame obtained from using GAB(fake_A)
        - real_frB: the real frame B
        - recov_PB: Estimated_Pose(recov_frB, real_A)
        - real_PB: Estimated_Pose(real_B, real_A)
        - weights: list of weights for the weigthed sum

        Return:
        - custom_total_cycle_loss
        '''

        loss_cycle_frA = self.standard_criterion_cycle_loss(recov_frA, real_frA)
        loss_cycle_frB = self.standard_criterion_cycle_loss(recov_frB, real_frB)
        loss_cycle_PA = self.standard_criterion_cycle_loss(recov_PA, real_PA)
        loss_cycle_PB = self.standard_criterion_cycle_loss(recov_PB, real_PB)

        custom_total_cycle_loss = (weights[0] * loss_cycle_frA + weights[1] * loss_cycle_frB + weights[2] * loss_cycle_PA + weights[3] * loss_cycle_PB)

        return custom_total_cycle_loss

    def standard_criterion_GAN_loss(self, generated, real):
        '''
        The classical adversarial loss

        Parameters:
        - generated: the result of the Generator
        - real: the real sample

        Return:
        - standard_GAN_loss
        '''

        standard_GAN_loss = self.standard_GAN_loss(generated, real)

        return standard_GAN_loss

    def standard_total_GAN_loss(self, real_frA, fake_frA, real_frB, fake_frB):
        '''
        Compute the standard total Adversarial Loss following this formula:

        standatd_total_identity_loss = (loss_GAN_AB + loss_GAN_BA)/2

        Parameters:
        - real_frA: the real frame
        - fake_frA: the generated frame
        - real_frB: the real frame
        - fake_frB: the generated frame

        Return:
        - standatd_total_identity_loss
        '''

        loss_GAN_AB = self.standard_criterion_GAN_loss(fake_frA, real_frA)
        loss_GAN_BA = self.standard_criterion_GAN_loss(fake_frB, real_frB)
        standatd_total_identity_loss = (loss_GAN_AB + loss_GAN_BA) / 2

        return standatd_total_identity_loss

    def standard_criterion_discriminator_loss(self, term1, term2):
        # TODO: commenta questa funzione perbene!
        loss = self.standard_Discr_loss(term1, term2)
        return loss

    def standard_discriminator_loss(self, term1, valid, term2, fake):
        # TODO: commenta questa funzione perbene!
        loss_real = self.standard_criterion_discriminator_loss(term1, valid)
        loss_fake = self.standard_criterion_discriminator_loss(term2, fake)

        loss_discr = (loss_real + loss_fake) / 2
        return loss_discr







