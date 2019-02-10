from pathlib import Path

import torch
import torch.optim as optim
import tqdm


class AbstractLearner:
    def __init__(self,
                 model,
                 criterion,
                 optimizer=None,
                 device=torch.device("cuda:0"),
                 folder=".",
                 is_notebook=True):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.folder = Path(folder)
        if not self.folder.is_dir():
            self.folder.mkdir()
        self.step = 0
        self.epoch = 0
        self.file_template = "checkpoint_{}_{}.pth"
        self.last_checkpoint = None
        self.opt = optimizer
        self.device = device
        self.tqdm_bar = tqdm.tqdm_notebook if is_notebook else tqdm.tqdm
        if self.opt is None:
            self.opt = optim.Adam(model.parameters(), lr=1e-4)

    def set_learning_rate(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def model_step(self, item, training):
        raise NotImplementedError

    def fit(self, data_loader, epochs=1, lr=None):
        self.model.train()
        if lr is not None:
            self.set_learning_rate(lr)

        loss_accum = 0.0
        for _ in range(epochs):
            loss_accum = 0.0
            start_step = self.step
            for i, item in enumerate(self.tqdm_bar(data_loader)):
                cur_loss, _ = self.model_step(item, training=True)
                loss_accum += cur_loss
                self.step += 1
            self.epoch += 1
            loss_accum /= (self.step - start_step)
        return loss_accum

    def develop(self, data_loader):
        self.model.eval()
        loss_accum = 0.0
        steps = 0
        for i, item in enumerate(self.tqdm_bar(data_loader)):
            cur_loss, logits = self.model_step(item, training=False)
            loss_accum += cur_loss
            steps += 1
        return loss_accum / steps

    def save_weights(self):
        path_to_save_model = self.folder / self.file_template.format(self.epoch, self.step)
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'state_dict': self.model.state_dict(),
        }, path_to_save_model)
        self.last_checkpoint = path_to_save_model

    def load_last_weights(self):
        self.load_weights(self.last_checkpoint)

    def load_weights(self, path_to_ckp, map_location=None):
        ckp = torch.load(path_to_ckp, map_location=map_location)
        self.model.load_state_dict(ckp["state_dict"])
        self.step = ckp["step"]
        self.epoch = ckp["epoch"]
