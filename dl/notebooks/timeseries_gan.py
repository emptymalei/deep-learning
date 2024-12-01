# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

# from .data import batch_generator
# from utils import extract_time, random_generator, NormMinMax
# from .model import Encoder, Recovery, Generator, Discriminator, Supervisor


# # Utilities


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0.0, 1, [T_mb[i], z_dim])
        temp[: T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


def NormMinMax(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val  # [3661, 24, 6]

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


# +
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Norm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)


class Encoder(nn.Module):
    """Embedding network between original feature space to latent space.

    Args:
      - input: input time-series features. (L, N, X) = (24, ?, 6)
      - h3: (num_layers, N, H). [3, ?, 24]

    Returns:
      - H: embeddings
    """

    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(
            input_size=opt.z_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer
        )
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        e_outputs, _ = self.rnn(input)
        H = self.fc(e_outputs)
        if sigmoid:
            H = self.sigmoid(H)
        return H


class Recovery(nn.Module):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - X_tilde: recovered data
    """

    def __init__(self, opt):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim, hidden_size=opt.z_dim, num_layers=opt.num_layer
        )

        self.fc = nn.Linear(opt.z_dim, opt.z_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        r_outputs, _ = self.rnn(input)
        X_tilde = self.fc(r_outputs)
        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)
        return X_tilde


class Generator(nn.Module):
    """Generator function: Generate time-series data in latent space.

    Args:
      - Z: random variables
      - T: input time information

    Returns:
      - E: generated embedding
    """

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(
            input_size=opt.z_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer
        )
        #   self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        g_outputs, _ = self.rnn(input)
        #  g_outputs = self.norm(g_outputs)
        E = self.fc(g_outputs)
        if sigmoid:
            E = self.sigmoid(E)
        return E


class Supervisor(nn.Module):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """

    def __init__(self, opt):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=opt.num_layer,
        )
        #  self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        s_outputs, _ = self.rnn(input)
        #  s_outputs = self.norm(s_outputs)
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S


class Discriminator(nn.Module):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=opt.num_layer,
        )
        #  self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        d_outputs, _ = self.rnn(input)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat


# +


class BaseModel:
    """Base Model for timegan"""

    def __init__(self, opt, ori_data):
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.ori_data, self.min_val, self.max_val = NormMinMax(ori_data)
        self.ori_time, self.max_seq_len = extract_time(self.ori_data)
        self.data_num, _, _ = np.asarray(ori_data).shape  # 3661; 24; 6
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, "train")
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, "test")
        self.device = torch.device("cuda:0" if self.opt.device != "cpu" else "cpu")

    def seed(self, seed_value):
        """Seed

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random

        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    def save_weights(self, epoch):
        """Save net weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, "train", "weights")
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        torch.save(
            {"epoch": epoch + 1, "state_dict": self.nete.state_dict()},
            "%s/netE.pth" % (weight_dir),
        )
        torch.save(
            {"epoch": epoch + 1, "state_dict": self.netr.state_dict()},
            "%s/netR.pth" % (weight_dir),
        )
        torch.save(
            {"epoch": epoch + 1, "state_dict": self.netg.state_dict()},
            "%s/netG.pth" % (weight_dir),
        )
        torch.save(
            {"epoch": epoch + 1, "state_dict": self.netd.state_dict()},
            "%s/netD.pth" % (weight_dir),
        )
        torch.save(
            {"epoch": epoch + 1, "state_dict": self.nets.state_dict()},
            "%s/netS.pth" % (weight_dir),
        )

    def train_one_iter_er(self):
        """Train the model for one epoch."""

        self.nete.train()
        self.netr.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

        # train encoder & decoder
        self.optimize_params_er()

    def train_one_iter_er_(self):
        """Train the model for one epoch."""

        self.nete.train()
        self.netr.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

        # train encoder & decoder
        self.optimize_params_er_()

    def train_one_iter_s(self):
        """Train the model for one epoch."""

        # self.nete.eval()
        self.nets.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

        # train superviser
        self.optimize_params_s()

    def train_one_iter_g(self):
        """Train the model for one epoch."""

        """self.netr.eval()
    self.nets.eval()
    self.netd.eval()"""
        self.netg.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
        self.Z = random_generator(
            self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len
        )

        # train superviser
        self.optimize_params_g()

    def train_one_iter_d(self):
        """Train the model for one epoch."""
        """self.nete.eval()
    self.netr.eval()
    self.nets.eval()
    self.netg.eval()"""
        self.netd.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
        self.Z = random_generator(
            self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len
        )

        # train superviser
        self.optimize_params_d()

    def train(self):
        """Train the model"""

        for iter in range(self.opt.iteration):
            # Train for one iter
            self.train_one_iter_er()

            print("Encoder training step: " + str(iter) + "/" + str(self.opt.iteration))

        for iter in range(self.opt.iteration):
            # Train for one iter
            self.train_one_iter_s()

            print(
                "Superviser training step: " + str(iter) + "/" + str(self.opt.iteration)
            )

        for iter in range(self.opt.iteration):
            # Train for one iter
            for kk in range(2):
                self.train_one_iter_g()
                self.train_one_iter_er_()

            self.train_one_iter_d()

            print(
                "Superviser training step: " + str(iter) + "/" + str(self.opt.iteration)
            )

        self.save_weights(self.opt.iteration)
        self.generated_data = self.generation(self.opt.batch_size)
        print("Finish Synthetic Data Generation")

    #  self.evaluation()

    """def evaluation(self):
    ## Performance metrics
    # Output initialization
    metric_results = dict()

    # 1. Discriminative Score
    discriminative_score = list()
    for _ in range(self.opt.metric_iteration):
      temp_disc = discriminative_score_metrics(self.ori_data, self.generated_data)
      discriminative_score.append(temp_disc)

    metric_results['discriminative'] = np.mean(discriminative_score)

    # 2. Predictive score
    predictive_score = list()
    for tt in range(self.opt.metric_iteration):
      temp_pred = predictive_score_metrics(self.ori_data, self.generated_data)
      predictive_score.append(temp_pred)

    metric_results['predictive'] = np.mean(predictive_score)

    # 3. Visualization (PCA and tSNE)
    visualization(self.ori_data, self.generated_data, 'pca')
    visualization(self.ori_data, self.generated_data, 'tsne')

    ## Print discriminative and predictive scores
    print(metric_results)
"""

    def generation(self, num_samples, mean=0.0, std=1.0):
        if num_samples == 0:
            return None, None
        ## Synthetic data generation
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        self.Z = random_generator(
            num_samples, self.opt.z_dim, self.T, self.max_seq_len  # , mean, std
        )
        self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
        self.E_hat = self.netg(self.Z)  # [?, 24, 24]
        self.H_hat = self.nets(self.E_hat)  # [?, 24, 24]
        generated_data_curr = (
            self.netr(self.H_hat).cpu().detach().numpy()
        )  # [?, 24, 24]

        generated_data = list()
        for i in range(num_samples):
            temp = generated_data_curr[i, : self.ori_time[i], :]
            generated_data.append(temp)

        # Renormalization
        generated_data = generated_data * self.max_val
        generated_data = generated_data + self.min_val
        return generated_data


# -


class TimeGAN(BaseModel):
    """TimeGAN Class"""

    @property
    def name(self):
        return "TimeGAN"

    def __init__(self, opt, ori_data):
        super(TimeGAN, self).__init__(opt, ori_data)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        # Create and initialize networks.
        self.nete = Encoder(self.opt).to(self.device)
        self.netr = Recovery(self.opt).to(self.device)
        self.netg = Generator(self.opt).to(self.device)
        self.netd = Discriminator(self.opt).to(self.device)
        self.nets = Supervisor(self.opt).to(self.device)

        if self.opt.resume != "":
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, "netG.pth"))[
                "epoch"
            ]
            self.nete.load_state_dict(
                torch.load(os.path.join(self.opt.resume, "netE.pth"))["state_dict"]
            )
            self.netr.load_state_dict(
                torch.load(os.path.join(self.opt.resume, "netR.pth"))["state_dict"]
            )
            self.netg.load_state_dict(
                torch.load(os.path.join(self.opt.resume, "netG.pth"))["state_dict"]
            )
            self.netd.load_state_dict(
                torch.load(os.path.join(self.opt.resume, "netD.pth"))["state_dict"]
            )
            self.nets.load_state_dict(
                torch.load(os.path.join(self.opt.resume, "netS.pth"))["state_dict"]
            )
            print("\tDone.\n")

        # loss
        self.l_mse = nn.MSELoss()
        self.l_r = nn.L1Loss()
        self.l_bce = nn.BCELoss()

        # Setup optimizer
        if self.opt.isTrain:
            self.nete.train()
            self.netr.train()
            self.netg.train()
            self.netd.train()
            self.nets.train()
            self.optimizer_e = optim.Adam(
                self.nete.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            self.optimizer_r = optim.Adam(
                self.netr.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            self.optimizer_g = optim.Adam(
                self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            self.optimizer_d = optim.Adam(
                self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            self.optimizer_s = optim.Adam(
                self.nets.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )

    def forward_e(self):
        """Forward propagate through netE"""
        self.H = self.nete(self.X)

    def forward_er(self):
        """Forward propagate through netR"""
        self.H = self.nete(self.X)
        self.X_tilde = self.netr(self.H)

    def forward_g(self):
        """Forward propagate through netG"""
        self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
        self.E_hat = self.netg(self.Z)

    def forward_dg(self):
        """Forward propagate through netD"""
        self.Y_fake = self.netd(self.H_hat)
        self.Y_fake_e = self.netd(self.E_hat)

    def forward_rg(self):
        """Forward propagate through netG"""
        self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
        """Forward propagate through netS"""
        self.H_supervise = self.nets(self.H)
        # print(self.H, self.H_supervise)

    def forward_sg(self):
        """Forward propagate through netS"""
        self.H_hat = self.nets(self.E_hat)

    def forward_d(self):
        """Forward propagate through netD"""
        self.Y_real = self.netd(self.H)
        self.Y_fake = self.netd(self.H_hat)
        self.Y_fake_e = self.netd(self.E_hat)

    def backward_er(self):
        """Backpropagate through netE"""
        self.err_er = self.l_mse(self.X_tilde, self.X)
        self.err_er.backward(retain_graph=True)
        print("Loss: ", self.err_er)

    def backward_er_(self):
        """Backpropagate through netE"""
        self.err_er_ = self.l_mse(self.X_tilde, self.X)
        self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])
        self.err_er = 10 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
        self.err_er.backward(retain_graph=True)

    #  print("Loss: ", self.err_er_, self.err_s)
    def backward_g(self):
        """Backpropagate through netG"""
        self.err_g_U = self.l_bce(self.Y_fake, torch.ones_like(self.Y_fake))

        self.err_g_U_e = self.l_bce(self.Y_fake_e, torch.ones_like(self.Y_fake_e))
        self.err_g_V1 = torch.mean(
            torch.abs(
                torch.sqrt(torch.std(self.X_hat, [0])[1] + 1e-6)
                - torch.sqrt(torch.std(self.X, [0])[1] + 1e-6)
            )
        )  # |a^2 - b^2|
        self.err_g_V2 = torch.mean(
            torch.abs((torch.mean(self.X_hat, [0])[0]) - (torch.mean(self.X, [0])[0]))
        )  # |a - b|
        self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])
        self.err_g = (
            self.err_g_U
            + self.err_g_U_e * self.opt.w_gamma
            + self.err_g_V1 * self.opt.w_g
            + self.err_g_V2 * self.opt.w_g
            + torch.sqrt(self.err_s)
        )
        self.err_g.backward(retain_graph=True)
        print("Loss G: ", self.err_g)

    def backward_s(self):
        """Backpropagate through netS"""
        self.err_s = self.l_mse(self.H[:, 1:, :], self.H_supervise[:, :-1, :])
        self.err_s.backward(retain_graph=True)
        print("Loss S: ", self.err_s)

    #   print(torch.autograd.grad(self.err_s, self.nets.parameters()))

    def backward_d(self):
        """Backpropagate through netD"""
        self.err_d_real = self.l_bce(self.Y_real, torch.ones_like(self.Y_real))
        self.err_d_fake = self.l_bce(self.Y_fake, torch.zeros_like(self.Y_fake))
        self.err_d_fake_e = self.l_bce(self.Y_fake_e, torch.zeros_like(self.Y_fake_e))
        self.err_d = (
            self.err_d_real + self.err_d_fake + self.err_d_fake_e * self.opt.w_gamma
        )
        if self.err_d > 0.15:
            self.err_d.backward(retain_graph=True)

    # print("Loss D: ", self.err_d)

    def optimize_params_er(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_er()

        # Backward-pass
        # nete & netr
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er()
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_er_(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_er()
        self.forward_s()
        # Backward-pass
        # nete & netr
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er_()
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_s(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_e()
        self.forward_s()

        # Backward-pass
        # nets
        self.optimizer_s.zero_grad()
        self.backward_s()
        self.optimizer_s.step()

    def optimize_params_g(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_e()
        self.forward_s()
        self.forward_g()
        self.forward_sg()
        self.forward_rg()
        self.forward_dg()

        # Backward-pass
        # nets
        self.optimizer_g.zero_grad()
        self.optimizer_s.zero_grad()
        self.backward_g()
        self.optimizer_g.step()
        self.optimizer_s.step()

    def optimize_params_d(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_e()
        self.forward_g()
        self.forward_sg()
        self.forward_d()
        self.forward_dg()

        # Backward-pass
        # nets
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


# +


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


# -

# # Train

import dataclasses


# +
@dataclasses.dataclass
class ModelParams:
    z_dim: int
    seq_len: int
    module: str
    hidden_dim: int
    num_layer: int
    iteration: int
    batch_size: int
    metric_iteration: int
    manualseed: int
    outf: str
    name: str
    device: str
    resume: str
    isTrain: bool = True
    lr: float = 0.0002
    beta1: float = 0.9
    w_gamma: float = 1
    w_es: float = 0.1
    w_e0: float = 10
    w_g: float = 100


model_params = {
    "z_dim": 6,
    "seq_len": 24,
    "module": "gru",
    "hidden_dim": 24,
    "num_layer": 3,
    "iteration": 100,
    "batch_size": 128,
    "metric_iteration": 10,
    "manualseed": 42,
    "outf": "tmp",
    "name": "timegan",
    "device": "cpu",
    "resume": "",
}
# -

data = sine_data_generation(3661, 24, 1)

# +

model = TimeGAN(ModelParams(**model_params), data)
# -

model.train()

model.generated_data.shape

model.generated_data

import matplotlib.pyplot as plt

data[0]

for i in range(len(data)):
    plt.plot(data[i].shape, ".")
