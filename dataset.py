import torch
import random
from option import get_option
import glob, numpy, os, random, soundfile, torch
from scipy import signal
import pandas as pd

opt = get_option()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_list, num_frames=200):
        self.num_frames = num_frames
        df_train = pd.read_csv(train_list)
        self.data_list = df_train["wav_path"].tolist()
        self.data_label = df_train["label"].tolist()

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])
        # if random.random() > 0.5:
        #     audio_length = len(audio)
        #     speed_rate = random.uniform(0.8, 1.2)
        #     audio = numpy.interp(
        #         numpy.linspace(0, audio_length - 1, int(audio_length * speed_rate)),
        #         numpy.arange(audio_length),
        #         audio,
        #     )
        length = self.num_frames * 80
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), "wrap")
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame : start_frame + length]
        audio = numpy.stack([audio], axis=0)
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


def get_dataloader(opt):
    train_dataset = Dataset(
        f"./data/finvcup9th_1st_ds5/finvcup9th_1st_ds5_train_data_fold_{opt.fold}.csv"
    )
    # valid_dataset = Dataset("./data/finvcup9th_1st_ds5/finvcup9th_1st_ds5_valid_data.csv")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    # valid_dataloader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=False,
    #     num_workers=opt.num_workers,
    #     pin_memory=True,
    # )
    return train_dataloader  # , valid_dataloader


if __name__ == "__main__":
    train_dataloader = get_dataloader(opt)
    for i, (audio, label) in enumerate(train_dataloader):
        print(audio.shape, label.shape)
        break
    print("Done!")
