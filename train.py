import warnings

import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.regression import mean_squared_error
from tqdm import tqdm

from config import Config
from data import get_data
from models import *
from utils import *
import threading

warnings.filterwarnings('ignore')


def save_checkpoint_async(state, epoch, session, save_dir):
    def save():
        save_checkpoint(state, epoch, session, save_dir)
    threading.Thread(target=save).start()


def train():
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator(log_with='wandb', mixed_precision="fp16" if opt.OPTIM.MIXED_PRECISION else "no")
    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)
    device = accelerator.device

    config = {"dataset": opt.TRAINING.TRAIN_DIR}
    accelerator.init_trackers("shadow", config=config)

    train_dataset = get_data(opt.TRAINING.TRAIN_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'train', opt.TRAINING.ORI,
                             {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    val_dataset = get_data(opt.TRAINING.VAL_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})

    trainloader = DataLoader(train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                             pin_memory=True, persistent_workers=True, prefetch_factor=2)
    testloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)

    model = DeShadowNet()
    criterion_psnr = torch.nn.SmoothL1Loss()
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(".bias")], "weight_decay": opt.OPTIM.WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if n.endswith(".bias")], "weight_decay": 0},
    ]
    optimizer_b = optim.AdamW(param_groups, lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model, optimizer_b, scheduler_b = accelerator.prepare(model, optimizer_b, scheduler_b)

    best_rmse = float("inf")
    best_epoch = 1
    early_stopping_patience = 10
    epochs_without_improvement = 0

    for epoch in range(1, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()
        for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            inp, tar = data[0], data[1]

            optimizer_b.zero_grad()
            res = model(inp)
            train_loss = criterion_psnr(res, tar) + 0.3 * (1 - structural_similarity_index_measure(res, tar, data_range=1)) \
                         + 0.7 * criterion_lpips(res, tar)
            accelerator.backward(train_loss)
            optimizer_b.step()
        scheduler_b.step()

        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr, ssim, lpips, rmse = 0, 0, 0, 0
            for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                inp, tar = data[0], data[1]
                with torch.no_grad():
                    res = model(inp)
                res, tar = accelerator.gather((res, tar))

                psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
                ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
                lpips += criterion_lpips(res, tar).item()
                rmse += mean_squared_error(res * 255, tar * 255, squared=False).item()

            size = len(testloader)
            psnr /= size
            ssim /= size
            lpips /= size
            rmse /= size

            if rmse < best_rmse:
                best_rmse, best_epoch = rmse, epoch
                save_checkpoint_async({'state_dict': model.state_dict()}, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            accelerator.log({"PSNR": psnr, "SSIM": ssim, "RMSE": rmse, "LPIPS": lpips}, step=epoch)

            if accelerator.is_main_process:
                print(f"Epoch: {epoch}, RMSE: {rmse}, PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}, "
                      f"Best RMSE: {best_rmse}, Best Epoch: {best_epoch}")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    accelerator.end_training()

if __name__ == '__main__':
    train()