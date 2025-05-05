import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse

def compare_models(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.equal(param1.data, param2.data):
            print(f"Mismatch found in: {name1}")
            return False
    return True

class BaseEncoder(nn.Module):
    def __init__(self, num_genes, hidden_size=128, dropout=0, CNN=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_genes, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.Mish(inplace=True),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.encoder(x)

class BaseDecoder(nn.Module):
    def __init__(self, num_genes, hidden_size=128, dropout=0, CNN=False):
        super().__init__()
        self.decoder = nn.Linear(hidden_size + num_genes, num_genes)

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self, num_genes, hidden_size=128, dropout=0, masked_weights=0.75, CNN=False, UWL=False, m=0.99):
        super().__init__()
        self.num_genes = num_genes
        self.hidden_size = hidden_size
        self.masked_weights = masked_weights
        self.UWL = UWL
        self.m = m

        self.encoderS = BaseEncoder(num_genes, hidden_size, dropout)
        self.encoderT = BaseEncoder(num_genes, hidden_size, dropout)
        self.encoderT.load_state_dict(self.encoderS.state_dict())

        self.fusionS = nn.Sequential(
            nn.Linear(hidden_size + num_genes, hidden_size + num_genes),
        )
        self.fusionT = nn.Sequential(
            nn.Linear(hidden_size + num_genes, hidden_size + num_genes),
        )
        self.fusionT.load_state_dict(self.fusionS.state_dict())

        for p in self.encoderT.parameters(): p.requires_grad = False
        for p in self.fusionT.parameters(): p.requires_grad = False

        self.maskPredictor = nn.Linear(hidden_size, num_genes)
        self.decoder = BaseDecoder(num_genes, hidden_size, dropout)

        self.weight_r = nn.Parameter(torch.zeros(1))
        self.weight_m = nn.Parameter(torch.zeros(1))
        self.weight_l = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def _update_teacher(self):
        for ps, pt in zip(self.encoderS.parameters(), self.encoderT.parameters()):
            pt.data = self.m * pt.data + (1 - self.m) * ps.data
        for ps, pt in zip(self.fusionS.parameters(), self.fusionT.parameters()):
            pt.data = self.m * pt.data + (1 - self.m) * ps.data

    def forward(self, x, comp_x, use_teacher=True):
        latent_S = self.encoderS(x)
        mask_pred = self.maskPredictor(latent_S)
        latent_S = self.fusionS(torch.cat([latent_S, mask_pred], dim=1))
        # latent_S = torch.cat([latent_S, mask_pred], dim=1)
        reconstruction = self.decoder(latent_S)

        if use_teacher:
            with torch.no_grad():
                latent_T = self.encoderT(comp_x)
                latent_T = self.fusionT(torch.cat([latent_T, 1 - mask_pred], dim=1))
                # latent_T = torch.cat([latent_T, 1 - mask_pred], dim=1)
        else:
            latent_T = latent_S.detach()

        return {
            "reconstruction": reconstruction,
            "mask_pred": mask_pred,
            "latent_S": latent_S,
            "latent_T": latent_T
        }

    def compute_loss(self, x, y, comp_x, mask, weight_r=0.15, weight_m=0.7, weight_l=0.15, use_teacher=True):
        outputs = self(x, comp_x, use_teacher=use_teacher)
        w_nums = mask * self.masked_weights + (1 - mask) * (1 - self.masked_weights)
        reconstruction_loss = (torch.mul(w_nums, mse(outputs["reconstruction"], y, reduction='none'))).mean()
        mask_loss = bce_logits(outputs["mask_pred"], mask)

        latent_S = F.normalize(outputs["latent_S"], dim=1)
        latent_T = F.normalize(outputs["latent_T"], dim=1)
        latent_loss = 1 - torch.cosine_similarity(latent_S, latent_T, dim=1).mean()

        if self.UWL:
            weight_r = torch.clamp(self.weight_r, min=-5, max=5)
            weight_m = torch.clamp(self.weight_m, min=-5, max=5)
            weight_l = torch.clamp(self.weight_l, min=-5, max=5)

            precision_l = torch.exp(-weight_l)
            precision_m = torch.exp(-weight_m)
            precision_r = torch.exp(-weight_r)

            loss = (
                precision_r * reconstruction_loss + precision_r / 2 +
                precision_m * mask_loss + precision_m / 2 +
                precision_l * latent_loss + precision_l / 2
            )
        else:
            loss = weight_r * reconstruction_loss + weight_m * mask_loss + weight_l * latent_loss

        return {
            "latent_S": latent_S,
            "latent_T": latent_T,
            "total_loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "latent_loss": latent_loss,
            "mask_loss": mask_loss,
            "weight_r": weight_r if not self.UWL else precision_r.item(),
            "weight_m": weight_m if not self.UWL else precision_m.item(),
            "weight_l": weight_l if not self.UWL else precision_l.item()
        }

    def inference(self, x):
        return self.encoderS(x)
