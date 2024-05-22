import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import numpy as np
import pandas as pd
import soundfile
from sklearn import metrics
from tqdm import tqdm

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt
        self.model = model
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.opt.epochs,
            pct_start=0.06,
            steps_per_epoch=self.len_trainloader,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        self.model.train()
        data, label = batch
        label = label.float().unsqueeze(1)
        mixed_data, label_a, label_b, lam = self.mixup(data, label)
        pred = self(mixed_data)
        loss = lam * self.criterion(pred, label_a) + (1 - lam) * self.criterion(
            pred, label_b
        )
        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(
        self,
        num_frames=200,
    ):
        self.model.eval()
        eval_list = f"./data/finvcup9th_1st_ds5/finvcup9th_1st_ds5_valid_data_fold_{self.opt.fold}.csv"

        outputs = torch.tensor([]).to(self.device)
        df_test = pd.read_csv(eval_list)
        label_list = df_test["label"].tolist()
        setfiles = df_test["wav_path"].tolist()
        loss, top1 = 0, 0
        for idx, file in tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(file)

            # Spliited utterance matrix
            max_audio = num_frames * 80
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = np.pad(audio, (0, shortage), "wrap")
            feats = []
            startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf) : int(asf) + max_audio])
            feats = np.stack(feats, axis=0).astype(np.float32)
            data_2 = torch.FloatTensor(feats).to(self.device)
            # Speaker embeddings
            with torch.no_grad():
                output = self(data_2)
                output = torch.mean(output, dim=0).view(1, -1)
            outputs = torch.cat((outputs, output), 0)
        acc, recall, prec, F1 = self.metrics_scores(
            outputs, torch.tensor(label_list).to(self.device)
        )
        self.log("valid_acc", acc)
        self.log("valid_recall", recall)
        self.log("valid_prec", prec)
        self.log("valid_F1", F1, prog_bar=True)

    def metrics_scores(self, output, target):
        # output = output.detach().cpu().numpy().argmax(axis=1)
        output = torch.sigmoid(output).detach().cpu().numpy()
        # 对 output 进行四舍五入
        output = np.round(output)[:, 0]
        target = target.detach().cpu().numpy()

        accuracy = metrics.accuracy_score(target, output)
        recall = metrics.recall_score(target, output)
        precision = metrics.precision_score(target, output)
        F1 = metrics.f1_score(target, output)

        return accuracy, recall, precision, F1

    def mixup(self, x, y, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0)).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
