class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
        self.weight_r, self.weight_m, self.weight_l = 0, 0, 0
        self.weight_sum = 0

        self.reconstruction_loss = 0
        self.reconstruction_loss_sum = 0
        self.reconstruction_loss_avg = 0
        self.mask_loss = 0
        self.mask_loss_sum = 0
        self.mask_loss_avg = 0
        self.latent_loss = 0
        self.latent_loss_sum = 0
        self.latent_loss_avg = 0

    def update(self, loss, n=1):
        val = loss['total_loss'].detach().cpu().numpy()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        self.reconstruction_loss = loss['reconstruction_loss'].detach().cpu().numpy()
        self.reconstruction_loss_sum += self.reconstruction_loss * n
        self.mask_loss = loss['mask_loss'].detach().cpu().numpy()
        self.mask_loss_sum += self.mask_loss * n
        self.latent_loss = loss['latent_loss'].detach().cpu().numpy()
        self.latent_loss_sum += self.latent_loss * n
        self.reconstruction_loss_avg = self.reconstruction_loss_sum / self.count
        self.mask_loss_avg = self.mask_loss_sum / self.count
        self.latent_loss_avg = self.latent_loss_sum / self.count

        self.weight_r = loss['weight_r'].detach().cpu().numpy()
        self.weight_m = loss['weight_m'].detach().cpu().numpy()
        self.weight_l = loss['weight_l'].detach().cpu().numpy()
        self.weight_sum = self.weight_r + self.weight_m + self.weight_l
        self.weight_r_avg = self.weight_r / self.weight_sum
        self.weight_m_avg = self.weight_m / self.weight_sum
        self.weight_l_avg = self.weight_l / self.weight_sum
