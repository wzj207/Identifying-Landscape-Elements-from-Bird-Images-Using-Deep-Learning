from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os, random
import copy
import argparse

from utils import get_models
from PIL import Image
from torch import topk

def fix_seed(seed=518):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministric = True

def get_dataset(resize, mode, data_dir):
    if mode == 'train':
        data_transforms = transforms.Compose([
            transforms.Resize((int(1.25*resize), int(1.25*resize))),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    data_set = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    return data_set


def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    history_save_path = os.path.join(history_folder, 'history-two_fifth-{}-{}-{}.txt'.format(run_num, model_name, SEED))
    history = open(history_save_path, 'w+')
    since = time.time()
    print('Since Time:', since, file=history)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch, epoch_loss, epoch_acc))
            print('{},Epoch,{},Loss,{:.4f},Acc,{:.4f}'.format(phase, epoch, epoch_loss, epoch_acc), file=history)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #torch.save(model,best_models_folder + model_name + '-{}_{}_{}'.format(SEED, run_num, size)+'.mdl')
                #print('model - {} saved'.format(model_name))
                
        print()

    time_elapsed = time.time() - since
    print('End Time:', time.time(), file=history)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Training complete in: {:.0f}m_{:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=history)
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Acc: {:4f}'.format(best_acc), file=history)
    print('Best Training Epoch: {}'.format(best_epoch))
    print('Best Training Epoch: {}'.format(best_epoch), file=history)
    history.close()


    # load best model weights
    model.load_state_dict(best_model_wts)
    #torch.save(model, best_models_folder + model_name + '-{}_{}_{}'.format(SEED, run_num, size)+'.pkl')
    return model

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('run_number', type=int, help='Run number')
parser.add_argument('model_name', choices=get_models.use_models.keys(), help='model name')
parser.add_argument('-sd','--seed', type=int, default=111, help='random seed')
parser.add_argument('-eps','--epochs', type=int, default=25, help='loop number')
args = parser.parse_args()

run_num = args.run_number

SEED = args.seed
fix_seed(seed=SEED)
size = '2_5'
root = f'./images_fraction_{size}'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
#model_name = 'resnet50'
#model_name = 'resnet34'
model_name = args.model_name
epochs = args.epochs

best_models_folder = 'Best_models/2_5/'
#best_models_folder = 'best_models/three_fifths/'
#best_models_folder = 'best_models/two_fifths/'
#best_models_folder = 'best_models/one_fifth/'
history_folder = 'History/2_5/'
#history_folder = 'history/three_fifths/'
#history_folder = 'history/two_fifths/'
#history_folder = 'history/one_fifth/'
os.makedirs(best_models_folder, exist_ok=True)
os.makedirs(history_folder, exist_ok=True)
#
resize = 224 if model_name.lower() != 'inception_v3' else 299

#data_dir = os.path.join(root, mode)
#data_set = get_dataset(resize, mode, data_dir)
image_datasets = {x: get_dataset(resize, x, os.path.join(root, x)) for x in ['train', 'val', 'test']}
print(image_datasets.keys())
for k in image_datasets.keys():
    print(k,':',len(image_datasets[k]))
class_to_idx = image_datasets['train'].class_to_idx
idx_to_class = {class_to_idx[k]:k for k in class_to_idx.keys()}
print(class_to_idx)
print(idx_to_class)
dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=1)
            }
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


#print(dataset_sizes['train'])
#print(mode)
#print('# of {} set:'.format(mode), len(data_set))
#print('class_to_idx:', data_set.class_to_idx)

model_ft = get_models.use_models[model_name]

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)

#torch.save(model.state_dict(),best_models_folder + model_name + '-{}_{}'.format(SEED, run_num)+'.mdl')
model.eval()
test_folder = './images/test/'
test_imgs = []
wrongs = []
wrongs_columns = ('image_path', 'predicted_label', 'true_label')
test_result = []
test_result_columns = ['id', 'image_path', 'predicted_label', 'true_label', 'probs']
with torch.no_grad():
    for cat in os.listdir(test_folder):
        sub_folder = os.path.join(test_folder, cat)
        for i,fn in enumerate(os.listdir(sub_folder)):
            img_path = os.path.join(sub_folder, fn)
            test_imgs.append(img_path)
    #        print(i, img_path)
            im = Image.open(img_path).convert("RGB")
            tf = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
             ])
            tsr = tf(im).unsqueeze(0).to(device)
            outputs = model(tsr)
            probs = nn.functional.softmax(outputs, dim=1).squeeze().detach().cpu().numpy()
            p = idx_to_class[int(probs.argmax())] 
            t = img_path.split('/')[-2]
            wrong = (p!=t)
            if wrong:
                wrongs.append([img_path, p, t])
            #print(i, img_path, p, t, probs)
     #       print(i, img_path, p, t)
            test_result.append([i, img_path, p, t, probs])
res_df = pd.DataFrame(test_result, columns=test_result_columns)
res_df.to_csv(f'record_of_test/{model_name}_{run_num}_{size}_{SEED}_res.csv')
wrongs_df = pd.DataFrame(data=wrongs, columns=wrongs_columns)
wrongs_df.to_csv(f'record_of_test/{model_name}_{run_num}_{size}_{SEED}_wrongs.csv')
f = open('test_accs.csv', 'a')
print('# of Test Images:', len(test_imgs), file=f)
print('# of Wrongs:', len(wrongs), file=f)
print('Test Acc:', (1-len(wrongs)/len(test_imgs)))
print(f'{model_name}, {size}, {SEED}, {run_num},', (1-len(wrongs)/len(test_imgs)), file=f)
f.close()
print('=======================================================================')
print(wrongs_df)
