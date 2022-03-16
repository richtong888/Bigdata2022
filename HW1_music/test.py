import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchaudio import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import argparse
# import torchaudio
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math, random
import torchaudio
from IPython.display import Audio
import os
import torch.optim as optim
from torchaudio import datasets, transforms



sub = {"track": [],
        "score": []}

df = pd.DataFrame(sub)



parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0,1', type=str, help='设置使用哪些显卡')
parser.add_argument('--no_cuda', action='store_true', help='不适用GPU进行训练')
parser.add_argument('--filename', default = './music-regression/audios/clips')


def getData(mode):
    if mode == 'train':
        data = pd.read_csv("./music-regression/train.csv")
        audio_name = data.track
        score = data.score
        return np.squeeze(audio_name.values), np.squeeze(score.values)
    else:
        data = pd.read_csv("./music-regression/test.csv")
        audio_name = data.track
        return np.squeeze(audio_name.values)

class AudioUtil():
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      # Nothing to do
      return aud

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      resig = sig[:1, :]
    else:
      # Convert from mono to stereo by duplicating the first channel
      resig = torch.cat([sig, sig])

    return ((resig, sr))

  def resample(aud, newsr):
    sig, sr = aud

    if (sr == newsr):
      # Nothing to do
      return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
      retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
      resig = torch.cat([resig, retwo])

    return ((resig, newsr))

  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)

  def time_shift(aud, shift_limit):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec

class SoundDataset(Dataset):
    def __init__(self, audio_path, mode):
        self.audio_path = audio_path
        self.mode = mode
        if self.mode == "train":
            self.audio_name, self.score = getData(mode)
        else:
            self.audio_name = getData(mode)

        self.duration = 1000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.2
        
    def __getitem__(self, index):
        if self.mode == "train":
            single_audio_name = os.path.join(self.audio_path, self.audio_name[index])
            aud = AudioUtil.open(single_audio_name)
            reaud = AudioUtil.resample(aud, self.sr)
            rechan = AudioUtil.rechannel(reaud, self.channel)

            # dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
            # shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
            sgram = AudioUtil.spectro_gram(rechan, n_mels=64, n_fft=1024, hop_len=None)
            aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

            score = self.score[index]
            return aug_sgram, score
        else: 
            single_audio_name = os.path.join(self.audio_path, self.audio_name[index])
            aud = AudioUtil.open(single_audio_name)
            reaud = AudioUtil.resample(aud, self.sr)
            rechan = AudioUtil.rechannel(reaud, self.channel)

            # dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
            # shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
            sgram = AudioUtil.spectro_gram(rechan, n_mels=64, n_fft=1024, hop_len=None)
            aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
            return aug_sgram, single_audio_name[32:]

    def __len__(self):
        return len(self.audio_name)



class Net(nn.Module):

    def __init__(self):
      
        super(Net, self).__init__()
        conv_layers = []

        self.conv1 = nn.Conv2d(2,8,5,padding=2)
        self.conv2 = nn.Conv2d(8,16,3,padding=1)       
        self.conv3 = nn.Conv2d(16,32,3,padding=1)
        self.conv4 = nn.Conv2d(32,64,3,padding=1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv1.bias.data.zero_()
        self.conv2.bias.data.zero_()
        self.conv3.bias.data.zero_()
        self.conv4.bias.data.zero_()

        conv_layers += [self.conv1, self.relu1, self.bn1]
        conv_layers += [self.conv2, self.relu2, self.bn2]
        conv_layers += [self.conv3, self.relu3, self.bn3]
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=1)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


def train(model, train_loader, device):

    epochs = 200

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_loader)),
                                                epochs=epochs,
                                                anneal_strategy='linear')
    criterion = nn.MSELoss()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for i, (audio, score) in enumerate(train_loader):
            audio = audio.to(device,dtype = torch.float32)
            score = score.to(device, dtype = torch.float32)
        # audio = audio.unsqueeze(0)
            output = model(audio)
            score = score.view(-1,1)
            loss = criterion(output, score)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss = total_loss / len(train_loader)
        print(f'the epoch is : {epoch:>5d}  MSEloss : {total_loss:.6f}')


class Net(nn.Module):

    def __init__(self):
      
        super(Net, self).__init__()
        conv_layers = []

        self.conv1 = nn.Conv2d(2,8,5,padding=2)
        self.conv2 = nn.Conv2d(8,16,3,padding=1)       
        self.conv3 = nn.Conv2d(16,32,3,padding=1)
        self.conv4 = nn.Conv2d(32,64,3,padding=1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv1.bias.data.zero_()
        self.conv2.bias.data.zero_()
        self.conv3.bias.data.zero_()
        self.conv4.bias.data.zero_()

        conv_layers += [self.conv1, self.relu1, self.bn1]
        conv_layers += [self.conv2, self.relu2, self.bn2]
        conv_layers += [self.conv3, self.relu3, self.bn3]
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=1)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


if __name__== "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Net()
    model = model.to(device)

    train_dataset = SoundDataset("./music-regression/audios/clips", mode="train")
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    train(model=model, train_loader=train_loader, device=device)
    torch.save(model.state_dict(),"./audio2.pth")
# parser.add_argument('--filename', default='.',type = str,help = 'enter your filepath') 
args = parser.parse_args(args=['--device', '1',  '--no_cuda','--filename','./music-regression/audios/clips'])
print(args)

def evaluate(model, test_loader, device):
    count = 0
    with torch.set_grad_enabled(False):
        for batch_index, (audio, name) in enumerate(test_loader):
            audio = audio.to(device)
            predict = model(audio)
            for i in range(len(predict)):
                df.at[count, 'track'] = name[i]
                df.at[count, 'score'] = predict[i]
                count = count+1
            
                
            


# parser = argparse.ArgumentParser(description=__doc__)
# parser.add_argument('--filename', default="/content/drive/MyDrive/Colab_Notebooks/music-regression/audios/clips")
# args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset = SoundDataset("./music-regression/audios/clips", mode="test")
   
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True)
model = Net()
model.load_state_dict(torch.load("./audio2.pth"))
model=model.to(device)
evaluate(model, test_loader, device)
    # print(df)
df.to_csv("submission.csv", index=False)