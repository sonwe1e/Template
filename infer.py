# -*- coding: utf-8 -*-

import argparse, glob, os, warnings, time
import soundfile
import torch
import math
import numpy as np
import tqdm
import numpy
import pandas as pd
import argparse


from models import resnet

CNN = resnet.CNN


class Inferencer(object):
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = CNN(nclass=1)
        loaded_state = torch.load(model_path)["state_dict"]
        for k in list(loaded_state.keys()):
            if k.startswith("model."):
                loaded_state[k[6:]] = loaded_state[k]
                del loaded_state[k]
        model.load_state_dict(loaded_state)
        model.eval().cuda(4)
        return model

    def eval_embedding(self, speech_path, num_frames=200, max_audio=48000):
        # 处理音频
        audio, _ = soundfile.read(speech_path)

        max_audio = num_frames * 80
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), "wrap")
        feats = []
        startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf) : int(asf) + max_audio])

        feats = np.stack(feats, axis=0).astype(np.float64)
        data = torch.FloatTensor(feats).cuda(4)

        # 推断
        with torch.no_grad():
            outputs = self.model.forward(data)
            outputs = torch.mean(outputs, dim=0).view(1, -1)

        # return outputs.detach().cpu().numpy().argmax(axis=1)[0]
        return numpy.round(torch.sigmoid(outputs).detach().cpu().numpy())[0][0]


def main(speech_list, infer, res_path):
    result = []
    for idx, file in tqdm.tqdm(enumerate(speech_list), total=len(speech_list)):
        output = infer.eval_embedding(file)
        result.append([os.path.basename(file), output])
    df_result = pd.DataFrame(result, columns=["speech_name", "pred_label"])
    df_result.to_csv(res_path, index=False, header=None)
    return df_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFake audio")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/songwei/Template/checkpoints/resnet50+mixup1.0+speedrate0.5_0.8_1.2/epoch=192-valid_F1=0.9856.ckpt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/home/songwei/IJCAI2024/data/finvcup9th_1st_ds5/test_filenames.csv",
        help="Path of test file, strictly same with the original file",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./submissions/submit.csv",
        help="Path of result",
    )
    args = parser.parse_args()
    print("loading model...")
    infer = Inferencer(args.model_path)
    df_test = pd.read_csv(args.test_path)

    print("model inferring...")
    main(df_test["wav_path"].tolist(), infer, res_path=args.save_path)
    print("done!")
