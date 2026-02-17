import os
# Solve multi-threading problem
# 解决多线程问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import csv
import random
import numpy as np
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
import collections
import datetime as dt
from datetime import datetime as dt2
import sys
import time
from torchvision.utils import save_image
import torch
# from loss.Loos_light import SegMultTaskLoss,SimilarityLoss, RefinementLoss
from loss.Loss import SegMultTaskLoss,SimilarityLoss, RefinementLoss
from models import Release,Release_lightly, Release_lightly_extra, Custom
from utils import sample_images, save_tensor_as_image, split_image, ImageBlocks ,get_transformer,maskToTensor, augment_batch_independent
from dataloader.seg_datasets import SegImageDataset
from dataloader.block_seg_datasets import BlockSegImageDataset
from config.config_utils import LoadConfig



# Work space
class RunNetworks():
    def __init__(self, config):
        self.config = config
        if config['run']['seed'] != False:
            seed = config['run']['seed']
            print(f'Seed is set to {seed}')
            self._set_random_seed(config['run']['seed'])
        # Device
        # 设置运算设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Work Type: train, evaluate or predict
        # 设置运行模式，分别有训练、评估、预测三种模式
        self.worktype = config['run']['type']

        # Model
        # 选择模型
        self._get_model()

        # Run ID
        # 运行ID
        self.run_id = self.model.get_name()+ '_' + dt2.now().strftime("%Y%m%d_%H%M%S")

        # The root path of this running
        # 该次运行的根路径，用于存储相关文件
        self.data_root_path = os.path.join(self.config['run']['data_path'],self.worktype, self.run_id)

        self.train_logs_fields = ['epoch', 'loss','L1','MSE','PSNR','ValidationMSE','ValidationPSNR']

        self.evaluate_logs_fields = ['idx', 'loss', 'L1','MSE','PSNR','SSIM','total_loss', 'total_L1','total_MSE','total_PSNR','total_SSIM','file_num']

    def work(self):
        # call function
        # 调用函数
        print("----------------------Start to work!----------------------")
        print("Run type:", self.worktype,"  Model:",self.model.get_name())
        pid = os.getpid()
        print(f'Run ID: {self.run_id}, Process ID: {pid}')
        if self.worktype == 'train':
            self.train()
        elif self.worktype == 'evaluate':
            self.evaluate()
        elif self.worktype == 'predict':
            self.predict()
        else:
            raise RuntimeError("Error: work type wrong!")

    def train(self):
        print("Epoch:", self.config['train']['epochs'], "  Batch size:",
              self.config['train']['batch_size'])
        # Info path
        # 创建相关路径
        save_models_dir = os.path.join(str(self.data_root_path), str('models'))
        save_samples_dir = os.path.join(str(self.data_root_path), str('samples'))
        # Log file
        # 创建日志文件
        save_logs_path = os.path.join(str(self.data_root_path), str('logs.csv'))
        # Init relative dir and file
        # 初始化训练的相关路径和日志文件
        self._init_data_path([save_models_dir, save_samples_dir],save_logs_path)

        # Use pretrained model
        # 如果需要使用预训练的模型，则加载预训练的模型
        if self.config['train']['pretrained']['use']:
            model_path = self.config['train']['pretrained']['model_path']
            self.model.load_state_dict(torch.load(model_path), strict=False)

        # Loss
        # 创建损失函数
        criterion = SegMultTaskLoss().to(self.device)

        # Train dataloader
        # 创建训练数据集，使用裁剪数据或否不使用裁剪数据
        # 具体来说，如果输入的图片是尺寸不一致的较大图片，那么必须使用裁剪
        # 如果输入的图片尺寸大小一致，可以选择性使用裁剪

        if self.config['run']['data_crop']['use'] :

            traindataset = DataLoader(
                BlockSegImageDataset(root=self.config['train']['dataset_path'], mode="train",tile_size=self.config['run']['data_crop'] ['size']),
                batch_size=self.config['train']['batch_size'],
                shuffle=True,
                drop_last=True,
                num_workers=self.config['train']['numberworks'])

            valdataset=BlockSegImageDataset(root=self.config['validation']['dataset_path'], mode="validation",tile_size=self.config['run']['data_crop'] ['size'])
            valdataloader = DataLoader(
                valdataset,
                batch_size=self.config['validation']['numberworks'],
                shuffle=True,
                num_workers=self.config['validation']['numberworks'])

        else:
            traindataset = DataLoader(
                SegImageDataset(root=self.config['train']['dataset_path'], mode="train"),
                batch_size=self.config['train']['batch_size'],
                drop_last=True,
                shuffle=True,
                num_workers=self.config['train']['numberworks'],
                pin_memory=True)
            valdataset=SegImageDataset(root=self.config['validation']['dataset_path'], mode="validation")
            valdataloader = DataLoader(
                valdataset,
                batch_size=self.config['validation']['batch_size'],
                shuffle=True,
                num_workers=self.config['validation']['numberworks'],
                pin_memory=True)

        # Optimizer
        # 创建优化器
        G_optimizer = optim.Adam(self.model.parameters(), lr=self.config['train']['learning_rate'], betas=(0.9, 0.999))

        # Calculate time
        # 计算时间
        recent_times = collections.deque(maxlen=len(traindataset)+1)
        start_time = time.time()
        pre_time = time.time()

        # Test file
        # 测试是否可以将模型和样例图片保存到目标路径

        sample_images(valdataset, self.model, os.path.join(save_samples_dir, str(0) + '.png'))
        torch.save(self.model.state_dict(), os.path.join(save_models_dir, str(0) + '.pth'))


        REFINEMENT = False
        #训练
        for epochs in range(1, self.config['train']['epochs'] + 1):
            if epochs ==  self.config['train']['mult_stage_loss']['epoch'] and self.config['train']['mult_stage_loss']['use']:
                criterion = RefinementLoss().to(self.device)
                print("change loss")
                REFINEMENT = True
            torch.cuda.empty_cache()
            epoch_loss = 0
            epoch_L1_loss = 0
            epoch_mse_loss = 0
            epoch_psnr = 0
            # 将模型切换到训练模式
            self.model.train()
            for idx, (imgs, gt, masks) in enumerate(traindataset):
                # Loading image, ground truth and mask
                # 从训练集中加载图像、标签和掩码
                imgs = imgs.to(self.device)
                gt = gt.to(self.device)
                masks = masks.to(self.device)

                # 图像增强技术，但是使用的时候，显存会发生变化，可能忽大忽小，确保显存够再使用
                if self.config['train']['aug']:
                    imgs, gt, masks = augment_batch_independent(imgs, gt, masks)
                self.model.zero_grad()

                # Get output from forward and calculate loss to train model
                # 获取输出并计算损失以训练模型
                if (self.model.return_num()) == 4:
                    x_o1, x_o2, x_o3, mm = self.model(imgs)
                    fake_images = x_o3
                else:
                    x_o1, x_o2, x_o3, fake_images, mm = self.model(imgs)

                if REFINEMENT:
                    G_loss = criterion(masks,fake_images,gt)
                else:
                    G_loss = criterion(masks, x_o1, x_o2, x_o3, fake_images, mm, gt, idx)

                target = masks.squeeze(1)
                u = torch.unique(target)

                #print("mm.shape:", mm.shape, "target dtype:", target.dtype, "target unique:", u[:20])

                if target.min() < 0 or target.max() >= mm.shape[1]:
                    raise RuntimeError(
                        f"target out of range: min={target.min().item()}, max={target.max().item()}, "
                        f"num_classes={mm.shape[1]}, unique={u.tolist()[:50]}"
                    )
                G_loss = G_loss.sum()
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()
                epoch_loss += G_loss.item()
                # print('[{}/{}] Generator Loss of epoch{} is {}'.format(k, len(dataloader), i, G_loss.item()))

                with torch.no_grad():
                    L1_loss, mse_loss, psnr = SimilarityLoss(fake_images, gt,SSIM=False)
                    L1_loss=L1_loss.item()
                    mse_loss=mse_loss.item()
                    psnr=psnr.item()
                    epoch_L1_loss += L1_loss
                    epoch_mse_loss += mse_loss
                    epoch_psnr += psnr

                    # Calculate batch time
                    # 计算本batch耗时
                    batch_time = time.time() - pre_time
                    recent_times.append(batch_time)

                    # Calculate statistics
                    # 计算统计信息
                    avg_batch_time = sum(recent_times) / len(recent_times)
                    elapsed_time = time.time() - start_time
                    formatted_elapsed = dt.timedelta(seconds=int(elapsed_time))

                    # Calculate remaining time
                    # 计算剩余时间
                    batches_done = (epochs - 1) * (len(traindataset) + 1) + idx + 1
                    total_batches = self.config['train']['epochs'] * (len(traindataset) + 1)
                    remaining_batches = total_batches - batches_done
                    eta_seconds = remaining_batches * avg_batch_time
                    formatted_eta = dt.timedelta(seconds=int(eta_seconds))

                # Print information
                # 更新进度显示
                sys.stdout.write(
                    f"\rEpoch: [{epochs}/{self.config['train']['epochs']}] "
                    f"Batch: [{idx + 1}/{len(traindataset)}] "
                    f"Epoch Avg Loss: {epoch_loss / (idx + 1):.4f} "
                    f"L1 Loss: {epoch_L1_loss / (idx + 1):.4f} "
                    f"MSE Loss: {epoch_mse_loss / (idx + 1):.4f} "
                    f"PSNR: {epoch_psnr / (idx + 1):.4f} "
                    f"Avg time/batch: {avg_batch_time:.3f}s "
                    f"Elapsed: {formatted_elapsed} "
                    f"ETA: {formatted_eta}  "
                )
                sys.stdout.flush()
                pre_time = time.time()


            val_start = time.time()
            val_mse, val_psnr = self.test_epoch(valdataloader)
            val_cost = time.time() - val_start  # 整个验证耗时
            recent_times.append(val_cost)
            pre_time = time.time()  # 下一 epoch 计时基点

            # Write logs into file
            # 更新日志文件
            with open(save_logs_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, self.train_logs_fields)
                writer.writerow({
                            'epoch': epochs,
                            'loss': epoch_loss / (len(traindataset)),
                            'L1': epoch_L1_loss / (len(traindataset)),
                            'MSE': epoch_mse_loss / (len(traindataset)),
                            'PSNR': epoch_psnr / (len(traindataset)),
                            'ValidationMSE': val_mse,
                            'ValidationPSNR': val_psnr,
                        })

            # Save sample images
            # 保存样例图片
            if (epochs % self.config['train']['sample_save_every'] == 0):
                sample_images(valdataset, self.model, os.path.join(save_samples_dir, str(epochs) + '.png'))

            # Save model
            # 保存模型
            if (epochs % self.config['train']['model_save_every'] == 0):
                 torch.save(self.model.state_dict(), os.path.join(save_models_dir, str(epochs) + '.pth'))

            print()

    @torch.no_grad()
    def test_epoch(self, testloader):
        """
        验证阶段：每个 batch 实时打印 (损失 + 计时)，格式与 train() 保持一致。
        返回：平均 MSE、平均 PSNR
        """
        # 切到 eval 模式
        print()
        self.model.eval()

        # 计时器
        recent_times = collections.deque(maxlen=len(testloader))
        start_time = time.time()
        pre_time = start_time

        epoch_L1_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_psnr = 0.0

        for idx, (imgs, gt, masks) in enumerate(testloader):
            # ---------------- 数据搬运 ----------------
            imgs = imgs.to(self.device)
            gt = gt.to(self.device)
            masks = masks.to(self.device)

            # ---------------- 前向推理 ----------------
            if self.model.return_num() == 4:
                _, _, fake_images, _ = self.model(imgs)  # x_o3 作为输出
            else:
                _, _, _, fake_images, _ = self.model(imgs)

            # ---------------- 计算指标 ----------------
            L1_loss, mse_loss, psnr = SimilarityLoss(fake_images, gt, SSIM=False)
            L1_loss = L1_loss.item()
            mse_loss = mse_loss.item()
            psnr = psnr.item()

            epoch_L1_loss += L1_loss
            epoch_mse_loss += mse_loss
            epoch_psnr += psnr

            # ---------------- 计时 ----------------
            batch_time = time.time() - pre_time
            recent_times.append(batch_time)
            avg_batch_time = sum(recent_times) / len(recent_times)
            elapsed_time = time.time() - start_time
            eta_seconds = (len(testloader) - idx - 1) * avg_batch_time

            # ---------------- 打印 ----------------
            sys.stdout.write(
                f"\rVal Batch: [{idx + 1}/{len(testloader)}] "
                f"L1: {epoch_L1_loss / (idx + 1):.4f} "
                f"MSE: {epoch_mse_loss / (idx + 1):.4f} "
                f"PSNR: {epoch_psnr / (idx + 1):.4f} "
                f"Avg time/batch: {avg_batch_time:.3f}s "
                f"Elapsed: {dt.timedelta(seconds=int(elapsed_time))} "
                f"ETA: {dt.timedelta(seconds=int(eta_seconds))}  "
            )
            sys.stdout.flush()
            pre_time = time.time()

        # 换行，避免覆盖下一条输出
        print()

        return epoch_mse_loss / len(testloader), epoch_psnr / len(testloader)

    def evaluate(self):

        # Log file
        # 创建日志文件
        save_logs_path = os.path.join(str(self.data_root_path), str('logs.csv'))

        # Init logs file
        # 初始化评估的日志文件
        self._init_data_path([self.data_root_path], save_logs_path)

        # 将模型切换到评估模式
        self.model.eval()

        # Use pretrained model
        model_path = self.config['evaluate']['model_path']
        self.model.load_state_dict(torch.load(model_path), strict=False)

        # Loss
        # 创建损失函数
        criterion = SegMultTaskLoss().to(self.device)


        # Train dataloader
        # 创建训练数据集，使用裁剪数据或否不使用裁剪数据
        if self.config['run']['data_crop']['use'] :
            evaldataset = DataLoader(
                BlockSegImageDataset(root=self.config['evaluate']['dataset_path'], mode="evaluate",tile_size=self.config['run']['data_crop'] ['size']),
                batch_size=1,
                shuffle=True,
                num_workers=self.config['evaluate']['numberworks'],
                pin_memory=True)

        else:
            evaldataset = DataLoader(
                SegImageDataset(root=self.config['evaluate']['dataset_path'], mode="evaluate"),
                batch_size=1,
                shuffle=True,
                num_workers=self.config['evaluate']['numberworks'],
                pin_memory=True)



        # Calculate time
        # 计算时间
        recent_times = collections.deque(maxlen=len(evaldataset))
        start_time = time.time()
        pre_time = time.time()

        epoch_loss = 0
        epoch_L1_loss = 0
        epoch_mse_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        logs = []  # 用于存储每个batch的记录
        start_time = time.time()
        for idx, (imgs, gt, masks) in enumerate(evaldataset):
            # 加载图像、标签和掩码
            imgs = imgs.to(self.device)
            gt = gt.to(self.device)
            masks = masks.to(self.device)

            with torch.no_grad():
                if self.model.return_num() == 4:
                    x_o1, x_o2, x_o3, mm = self.model(imgs)
                    fake_images = x_o3  # 为了保持和HTRNet的输出一致
                else:
                    x_o1, x_o2, x_o3, fake_images, mm = self.model(imgs)
                G_loss = criterion(masks, x_o1, x_o2, x_o3, fake_images, mm, gt, idx)
                epoch_loss += G_loss.item()
                L1_loss, mse_loss, psnr, ssim = SimilarityLoss(fake_images, gt)
                L1_loss = L1_loss.item()
                mse_loss = mse_loss.item()
                psnr = psnr.item()
                ssim = ssim.item()
                epoch_L1_loss += L1_loss
                epoch_mse_loss += mse_loss
                epoch_psnr += psnr
                epoch_ssim += ssim
                # 计算本batch耗时
                batch_time = time.time() - pre_time
                recent_times.append(batch_time)

                # 计算统计信息
                avg_batch_time = sum(recent_times) / len(recent_times)
                elapsed_time = time.time() - start_time
                formatted_elapsed = dt.timedelta(seconds=int(elapsed_time))

                # 计算剩余时间
                batches_done = idx + 1
                total_batches = len(evaldataset)
                remaining_batches = total_batches - batches_done
                eta_seconds = remaining_batches * avg_batch_time
                formatted_eta = dt.timedelta(seconds=int(eta_seconds))

            # 实时打印进度信息（可以保留或移除）
            sys.stdout.write(
                f"\rBatch: [{idx + 1}/{len(evaldataset)}] "
                f"Epoch Avg Loss: {epoch_loss / (idx + 1):.4f} "
                f"L1 Loss: {epoch_L1_loss / (idx + 1):.4f} "
                f"MSE Loss: {epoch_mse_loss / (idx + 1):.4f} "
                f"PSNR: {epoch_psnr / (idx + 1):.4f} "
                f"SSIM: {epoch_ssim / (idx + 1):.4f} "
                f"Avg time/batch: {avg_batch_time:.3f}s "
                f"Elapsed: {formatted_elapsed} "
                f"ETA: {formatted_eta}  "
            )
            sys.stdout.flush()

            # 将当前batch的idx和single_loss存储到logs对象中
            logs.append({
                'idx': idx,
                'loss': G_loss.item(),
                'L1': L1_loss,
                'MSE': mse_loss,
                'PSNR': psnr,
                'SSIM': ssim
            })

            pre_time = time.time()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"cost time: {elapsed_time:.6f} s")
        # 在循环结束后，将total_loss和file_num写入logs的第一行
        if logs:
            logs[0]['total_loss'] = str(epoch_loss)
            logs[0]['total_L1'] = str(epoch_L1_loss)
            logs[0]['total_MSE'] = str(epoch_mse_loss)
            logs[0]['total_PSNR'] = str(epoch_psnr)
            logs[0]['total_SSIM'] = str(epoch_ssim)
            logs[0]['file_num'] = str(len(evaldataset))

        # 一次性将所有记录写入CSV文件
        with open(save_logs_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.evaluate_logs_fields)
            writer.writeheader()
            writer.writerows(logs)

    def predict(self):

        self.model.eval()

        # Initialize data path
        # 初始化数据路径
        self._init_data_path([self.data_root_path],None)

        # Load model
        # 加载模型
        model_path = self.config['predict']['model_path']
        self.model.load_state_dict(torch.load(model_path), strict=False)

        # Load image
        # 加载图片
        image = Image.open(self.config['predict']['input_file_path'])
        transform = get_transformer()
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # 如果训练时使用的是裁剪模型，这里需要先将被预测的图片裁剪成小块，分别预测后再合并回整张图像
        if self.config['run']['data_crop']['use']:
            positions=split_image(image.size[0], image.size[1], self.config['run']['data_crop'] ['size'])
            x1 = ImageBlocks(image.size[0]//4, image.size[1]//4, split_image(image.size[0]//4, image.size[1]//4, self.config['run']['data_crop'] ['size']//4))
            x2 = ImageBlocks(image.size[0]//2, image.size[1]//2, split_image(image.size[0]//2, image.size[1]//2, self.config['run']['data_crop'] ['size']//2))
            x3 = ImageBlocks(image.size[0], image.size[1], positions)
            output = ImageBlocks(image.size[0], image.size[1], positions)
            mask = ImageBlocks(image.size[0], image.size[1], positions)



            with torch.no_grad():
                for pos in positions:
                    left, top, right, bottom = pos

                    tile = image.crop((left, top, right + 1, bottom + 1))
                    tile_tensor = transform(tile).unsqueeze(0).to(self.device)

                    if self.model.return_num()==4:
                        x1_b, x2_b, x3_b, mask_b = self.model(tile_tensor)
                        output_b = x3_b
                    elif self.model.return_num()==5:
                        x1_b, x2_b, x3_b, output_b, mask_b = self.model(tile_tensor)

                    # 将当前块的预测结果加入类中
                    x1.add_block([element // 4 for element in pos], x1_b)
                    x2.add_block([element // 2 for element in pos], x2_b)
                    x3.add_block(pos, x3_b)
                    output.add_block(pos, output_b)
                    mask_b = torch.argmax(mask_b,1).unsqueeze(0)
                    mask_b = maskToTensor(mask_b, self.device)
                    mask.add_block(pos, mask_b)

                save_tensor_as_image(x1.merge_blocks(), os.path.join(str(self.data_root_path), 'x1.png'))
                save_tensor_as_image(x2.merge_blocks(), os.path.join(str(self.data_root_path), 'x2.png'))
                save_tensor_as_image(x3.merge_blocks(), os.path.join(str(self.data_root_path), 'x3.png'))
                save_tensor_as_image(output.merge_blocks(), os.path.join(str(self.data_root_path), 'output.png'))
                # save_tensor_as_image(mask, os.path.join(str(self.data_root_path),'mask.png'))

                save_image(mask.merge_blocks(), os.path.join(str(self.data_root_path), 'mask.png'), nrow=1, normalize=True)

        else:
            print(image.size)
            # predict
            with torch.no_grad():
                x1, x2, x3, output, mask = self.model(image_tensor)
            print(x1.shape, x2.shape, x3.shape, output.shape, mask.shape)
            mask = torch.argmax(mask, 1).unsqueeze(0)
            mask = maskToTensor(mask, self.device)
            save_tensor_as_image(x1, os.path.join(str(self.data_root_path),'x1.png'))
            save_tensor_as_image(x2, os.path.join(str(self.data_root_path),'x2.png'))
            save_tensor_as_image(x3, os.path.join(str(self.data_root_path),'x3.png'))
            save_tensor_as_image(output, os.path.join(str(self.data_root_path),'output.png'))
            # save_tensor_as_image(mask, os.path.join(str(self.data_root_path),'mask.png'))

            save_image(mask, os.path.join(str(self.data_root_path),'mask.png'), nrow=1, normalize=True)


    def _get_model(self):
        self.model_name = self.config['run']['model']
        model = self.model_name
        if model == 'Release':
            self.model = Release.ResNet_UNet().to(self.device)
        elif model == 'Release_lightly':
            self.model = Release_lightly.ResNet_UNet().to(self.device)
        elif model =='Release_lightly_extra':
            self.model = Release_lightly_extra.ResNet_UNet().to(self.device)
        elif model == 'Custom':
            self.model = Custom.ResNet_UNet(base=self.config['custom']['base'],
                                            refinement=self.config['custom']['refinement'],
                                            ffp=self.config['custom']['ffp'],
                                            ppm=self.config['custom']['ppm'],
                                            down_sample=self.config['custom']['down_sample'],
                                            am=self.config['custom']['am']).to(self.device)
        else:
            print("Error: Model is not exist!")



    def _init_data_path(self,save_path,logs_file):
        work_type_path = os.path.join(self.config['run']['data_path'],self.worktype)

        if not os.path.exists(work_type_path):
            os.makedirs(work_type_path)

        for path in save_path:
            if not os.path.exists(path):
                os.makedirs(path)

        if logs_file is not None:
            if self.worktype == "train":
                with open(logs_file, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=self.train_logs_fields)
                    writer.writeheader()
            if self.worktype == "evaluate":
                with open(logs_file, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=self.evaluate_logs_fields)
                    writer.writeheader()

    def _set_random_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # Slower but deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    config = LoadConfig()
    network = RunNetworks(config)
    network.work()