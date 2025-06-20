import torch
import os
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler, ModelCheckpoint
from torchvision.models import resnet18, densenet121, mobilenet_v2
from torch.utils.data import DataLoader
from model import SimCLR_pl
from utility import Utility, CustomDataset, HParams


def reproducibility(config):
    seed = getattr(config, "seed", 42)  # Default to 42 if not set
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# A function to get images form folder
def get_data_from_path(folder_path, img_size, batch_size):
    transformer = Utility(img_size)
    dataset = CustomDataset(folder_path=folder_path, transform=transformer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, drop_last=True)


if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')

    available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    save_model_path = os.path.join(os.getcwd(), "saved_models/")
    print("Available GPUs: ", available_gpus)
    filename = "SimCLR_MobileNet-V2_Adam"
    resume_from_checkpoint = False
    train_config = HParams()

    reproducibility(train_config)
    save_name = filename + ".ckpt"

    model = SimCLR_pl(train_config, model=resnet18(weights=None), feat_dim=resnet18(weights=None).fc.in_features)

    data_loader = get_data_from_path("newset", 224, 20)

    accumulator = GradientAccumulationScheduler(scheduling={0: train_config.gradient_accumulation_steps})
    checkpoint_callback = ModelCheckpoint(filename=filename, dirpath=save_model_path, every_n_epochs=5, save_last=True, save_top_k=2, monitor='Contrastive loss_epoch', mode='min')

    if resume_from_checkpoint:
        trainer = Trainer(callbacks=[accumulator, checkpoint_callback], gpus=available_gpus, max_epochs=train_config.epochs, resume_from_checkpoint=train_config.checkpoint_path)
    else:
        trainer = Trainer(callbacks=[accumulator, checkpoint_callback], gpus=available_gpus, max_epochs=train_config.epochs)

    trainer.fit(model, data_loader)
    trainer.save_checkpoint(save_name)
