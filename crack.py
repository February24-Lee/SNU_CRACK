import torch
import torchvision

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from os import listdir
from PIL import Image

BATCH_SIZE = 16
LR = 0.0001
WARM_EPOCH = 10
MAX_EPOCH = 3000

trainset_path = '../Share_Data/crack_segmentation_dataset/train/'
testset_path = '../Share_Data/crack_segmentation_dataset/test/'


class my_module(pl.LightningModule):
    def __init__(self):
        super(my_module, self).__init__()
        self.my_model = models.segmentation.fcn_resnet50(num_classes=1)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.my_model(x)['out']
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y, y_hat)
        self.logger.experiment.log({'training_loss':loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y, y_hat)
        return loss
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.logger.experiment.log({'avg_val_loss' : avg_loss})
        return
    
    
    def configure_optimizers(self):
        my_opt = LARSWrapper(Adam(self.my_model.parameters(), lr=LR))
        my_sch = LinearWarmupCosineAnnealingLR(
            my_opt,
            warmup_epochs=WARM_EPOCH,
            max_epochs=MAX_EPOCH)
        rst_sch = {
            'scheduler' : my_sch,
            'interval': 'epoch',
            'frequency' : 1
        }
        return [my_opt], [rst_sch]

class my_dataset(Dataset):
    def __init__(self, data_path, folder_name_list = ['images', 'masks']):
        super(my_dataset, self).__init__()
        self.img_name_list = [img_f_name[:-4] for img_f_name in listdir(data_path+'images')]
        self.img_path = data_path + folder_name_list[0]+'/'
        self.mask_path = data_path + folder_name_list[1]+'/'
        self.img_transforms = transforms.Compose([
            transforms.ToTensor()])
        self.mask_transforms = transforms.Compose([
            transforms.ToTensor()])
    
    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self, idx):
        assert idx < len(self), 'wrong index'
        img_name = self.img_name_list[idx]
        return (self.img_transforms(Image.open(self.img_path+img_name+'.jpg')), 
               self.mask_transforms(Image.open(self.mask_path+img_name+'.jpg')))
    
    
class my_datamodule(pl.LightningDataModule):
    def __init__(self, trainset_path, testset_path):
        super(my_datamodule, self).__init__()
        self.trainset_path = trainset_path
        self.testset_path = testset_path
        
    def train_dataloader(self):
        train_dataset = my_dataset(self.trainset_path)
        return DataLoader(train_dataset, batch_size = BATCH_SIZE)
    
    def val_dataloader(self):
        val_dataset = my_dataset(self.testset_path)
        return DataLoader(val_dataset, batch_size=BATCH_SIZE)
        

class my_test_dataset(Dataset):
    def __init__(self, data_path, folder_name_list = ['images', 'masks']):
        super(my_test_dataset, self).__init__()
        self.img_name_list = [img_f_name[:-4] for img_f_name in listdir(data_path+'images')]
        self.img_path = data_path + folder_name_list[0]+'/'
        self.mask_path = data_path + folder_name_list[1]+'/'
        self.img_transforms = transforms.Compose([
            transforms.ToTensor()])
        self.mask_transforms = transforms.Compose([
            transforms.ToTensor()])
    
    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self, idx):
        assert idx < len(self), 'wrong index'
        img_name = self.img_name_list[idx]
        return (self.img_transforms(Image.open(self.img_path+img_name+'.jpg')), 
               self.mask_transforms(Image.open(self.mask_path+img_name+'.jpg')))
        
        
SAVE_DIR = 'logs/'
MODEL_NAME = 'CRACK'

checkpoint_callback_valid = pl.callbacks.ModelCheckpoint(monitor='avg_val_loss',
                                                        save_last = True,
                                                        save_top_k = 1,
                                                        mode='min')

tt_logger = pl.loggers.TestTubeLogger(
    save_dir=SAVE_DIR,
    name=MODEL_NAME,
    debug=False,
    create_git_tag=False)

crack_datamodule = my_datamodule(trainset_path=trainset_path, testset_path=testset_path)
crack_model = my_module()

runner = pl.Trainer(default_root_dir=f"{tt_logger.save_dir}",
                   logger=tt_logger,
                   checkpoint_callback = checkpoint_callback_valid,
                   gpus=1,
                   max_epochs=100)

runner.fit(crack_model, crack_datamodule)