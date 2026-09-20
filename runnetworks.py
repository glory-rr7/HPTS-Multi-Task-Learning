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
import torch.nn as nn
import torch.nn.functional as F
# from loss.Loos_light import SegMultTaskLoss,SimilarityLoss, RefinementLoss
from loss.Loss import (
    SegMultTaskLoss,
    SegMultTaskLoss3Tags,
    SimilarityLoss,
    RefinementLoss,
    RefinementLoss3Tags,
    DistillationLoss,
    deep_feature_l2_loss,
    deep_feature_at_loss,
)
from models import Release, Release_lightly, Release_lightly_extra, Custom, Custom3Tags
from models import ResNet, IEResNet, IEXResNet
from models.model_output import ensure_model_output
from utils import sample_images, save_tensor_as_image, split_image, ImageBlocks ,get_transformer,maskToTensor
from dataloader.seg_datasets import SegImageDataset
from dataloader.block_seg_datasets import BlockSegImageDataset
from dataloader.augmentation import STRATEGY_AUG_PROB
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

        # Work Type: train, train_distill, evaluate or predict
        # 设置运行模式
        self.worktype = config['run']['type']

        self.teacher_model = None
        self.student_model = None
        self.teacher_adapter = None
        self.teacher_hook_handle = None
        self.student_hook_handle = None
        self.teacher_deep_feature = None
        self.student_deep_feature = None
        self.deep_method = None
        self.teacher_adapter_trainable = False
        self.adapter_resume_path = None

        # Model
        # 选择模型
        if self.worktype == 'train_distill':
            self._get_distill_models()
            self.model = self.student_model
            self.run_id = (
                f"Distill_{self.teacher_model_name}_to_{self.student_model_name}_"
                f"{dt2.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            self._get_model()
            # Run ID
            # 运行ID
            self.run_id = self.model.get_name()+ '_' + dt2.now().strftime("%Y%m%d_%H%M%S")

        # The root path of this running
        # 该次运行的根路径，用于存储相关文件
        self.data_root_path = os.path.join(self.config['run']['data_path'],self.worktype, self.run_id)

        self.train_logs_fields = [
            'epoch',
            'loss',
            'TaskLoss',
            'StrategyLoss',
            'WeightedStrategyLoss',
            'MSRLoss',
            'RefinementLoss',
            'MaskLoss',
            'L1',
            'MSE',
            'PSNR',
            'ValidationMSE',
            'ValidationPSNR',
            'TrainInferenceTime',
            'ValidationInferenceTime',
        ]
        self.distill_logs_fields = [
            'epoch',
            'loss_total',
            'loss_task',
            'loss_deep',
            'loss_output',
            'loss_mm',
            'L1',
            'MSE',
            'PSNR',
            'ValidationMSE',
            'ValidationPSNR',
            'TrainInferenceTime',
            'ValidationInferenceTime',
        ]

        self.evaluate_logs_fields = ['idx', 'loss', 'L1','MSE','PSNR','SSIM','total_loss', 'total_L1','total_MSE','total_PSNR','total_SSIM','file_num']
        self.gradient_diagnostics_fields = [
            'epoch',
            'batch',
            'global_step',
            'layer',
            'cosine',
            'rebuild_norm',
            'segment_norm',
            'norm_ratio',
            'is_conflict',
            'parameter_tensors',
            'background_ratio',
            'handwriting_ratio',
            'print_ratio',
            'overlap_ratio',
        ]

    def work(self):
        # call function
        # 调用函数
        print("----------------------Start to work!----------------------")
        print("Run type:", self.worktype,"  Model:",self.model.get_name())
        if self.worktype == 'train_distill':
            print("Teacher:", self.teacher_model_name, " Student:", self.student_model_name)
        pid = os.getpid()
        print(f'Run ID: {self.run_id}, Process ID: {pid}')
        if self.worktype == 'train':
            self.train()
        elif self.worktype == 'train_distill':
            self.train_distill()
        elif self.worktype == 'evaluate':
            self.evaluate()
        elif self.worktype == 'predict':
            self.predict()
        else:
            raise RuntimeError("Error: work type wrong!")

    def train(self):
        train_cfg = self.config['train']
        train_strategy = self._get_train_strategy()
        self._validate_train_strategy(train_strategy)
        print("Epoch:", train_cfg['epochs'], "  Batch size:",
              train_cfg['batch_size'])
        print("Train strategy:", train_strategy)
        strategy_loss_weight = self._get_strategy_loss_weight(train_strategy)
        if train_strategy != 'none':
            print("Strategy loss weight:", strategy_loss_weight)
        train_roots = self._resolve_dataset_roots(
            train_cfg['dataset_path'], "train.dataset_path"
        )
        val_roots = self._resolve_dataset_roots(
            self.config['validation']['dataset_path'], "validation.dataset_path"
        )
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
        gradient_diagnostics_cfg = train_cfg.get('gradient_diagnostics', {})
        gradient_diagnostics_enabled = bool(gradient_diagnostics_cfg.get('enabled', False))
        gradient_diagnostics_interval = max(1, int(gradient_diagnostics_cfg.get('interval', 10)))
        gradient_diagnostics_path = os.path.join(
            str(self.data_root_path),
            str(gradient_diagnostics_cfg.get('filename', 'gradient_diagnostics.csv')),
        )
        if gradient_diagnostics_enabled:
            self._init_gradient_diagnostics(gradient_diagnostics_path)
            print(
                "Gradient diagnostics enabled:",
                gradient_diagnostics_path,
                f"(every {gradient_diagnostics_interval} batches)",
            )

        # Use pretrained model
        # 如果需要使用预训练的模型，则加载预训练的模型
        if train_cfg['pretrained']['use']:
            model_path = train_cfg['pretrained']['model_path']
            self._load_checkpoint_flexible(self.model, model_path, name="train.pretrained")

        # Loss
        # 创建损失函数
        criterion = self._build_task_loss(self.model_name).to(self.device)

        # Train dataloader
        # 创建训练数据集，使用裁剪数据或否不使用裁剪数据
        # 具体来说，如果输入的图片是尺寸不一致的较大图片，那么必须使用裁剪
        # 如果输入的图片尺寸大小一致，可以选择性使用裁剪

        dataset_use_aug = train_cfg['aug'] if train_strategy == 'none' else False
        dataset_use_strategy = train_strategy != 'none'
        train_loader_kwargs = self._build_loader_kwargs(
            num_workers=train_cfg['numberworks'],
            pin_memory=True,
            prefetch_factor=train_cfg.get('prefetch_factor'),
        )

        if self.config['run']['data_crop']['use'] :

            traindataset = DataLoader(
                BlockSegImageDataset(
                    root=train_roots,
                    mode="train",
                    tile_size=self.config['run']['data_crop'] ['size'],
                    use_aug=dataset_use_aug,
                    use_strategy=dataset_use_strategy,
                ),
                batch_size=train_cfg['batch_size'],
                shuffle=True,
                drop_last=True,
                **train_loader_kwargs)

            valdataset=BlockSegImageDataset(
                root=val_roots,
                mode="validation",
                tile_size=self.config['run']['data_crop'] ['size'],
                use_aug=False,
            )
            valdataloader = DataLoader(
                valdataset,
                batch_size=self.config['validation']['batch_size'],
                shuffle=True,
                num_workers=self.config['validation']['numberworks'],
                pin_memory=True)

        else:
            traindataset = DataLoader(
                SegImageDataset(
                    root=train_roots,
                    mode="train",
                    use_aug=dataset_use_aug,
                    use_strategy=dataset_use_strategy,
                ),
                batch_size=train_cfg['batch_size'],
                drop_last=True,
                shuffle=True,
                **train_loader_kwargs)
            valdataset=SegImageDataset(
                root=val_roots,
                mode="validation",
                use_aug=False,
            )
            valdataloader = DataLoader(
                valdataset,
                batch_size=self.config['validation']['batch_size'],
                shuffle=True,
                num_workers=self.config['validation']['numberworks'],
                pin_memory=True)

        # Optimizer
        # 创建优化器
        G_optimizer = optim.Adam(self.model.parameters(), lr=train_cfg['learning_rate'], betas=(0.9, 0.999))
        lr_schedule_map = self._parse_lr_schedule_map(train_cfg)

        # Calculate time
        # 计算时间
        recent_times = collections.deque(maxlen=len(traindataset)+1)
        start_time = time.time()
        pre_time = time.time()

        # Test file
        # 测试是否可以将模型和样例图片保存到目标路径

        sample_images(valdataset, self.model, os.path.join(save_samples_dir, str(0) + '.png'))
        torch.save(self.model.state_dict(), os.path.join(save_models_dir, str(0) + '.pth'))

        #训练
        for epochs in range(1, train_cfg['epochs'] + 1):
            self._apply_lr_schedule(G_optimizer, epochs, lr_schedule_map)
            if epochs ==  train_cfg['mult_stage_loss']['epoch'] and train_cfg['mult_stage_loss']['use']:
                criterion = self._build_refinement_loss(self.model_name).to(self.device)
                print("change loss")
            torch.cuda.empty_cache()
            epoch_loss = 0
            epoch_task_loss = 0
            epoch_strategy_loss = 0
            epoch_weighted_strategy_loss = 0
            epoch_msr_loss = 0
            epoch_refinement_loss = 0
            epoch_mask_loss = 0
            epoch_L1_loss = 0
            epoch_mse_loss = 0
            epoch_psnr = 0
            epoch_inference_time = 0.0
            # 将模型切换到训练模式
            self.model.train()
            for idx, batch in enumerate(traindataset):
                if train_strategy == 'none':
                    imgs, gt, masks = batch
                    # Loading image, ground truth and mask
                    # 从训练集中加载图像、标签和掩码
                    imgs = imgs.to(self.device)
                    gt = gt.to(self.device)
                    masks = masks.to(self.device)
                else:
                    # Dataset 已在 worker 中生成两个视角，直接解包
                    ori_imgs, ori_gt, ori_masks, aug_imgs, aug_gt, aug_masks = batch
                    pair_imgs  = torch.cat([ori_imgs,  aug_imgs],  dim=0).to(self.device)
                    pair_gt    = torch.cat([ori_gt,    aug_gt],    dim=0).to(self.device)
                    pair_masks = torch.cat([ori_masks, aug_masks], dim=0).to(self.device)

                self.model.zero_grad()

                if train_strategy == 'none':
                    # Get output from forward and calculate loss to train model
                    # 获取输出并计算损失以训练模型
                    infer_start = self._start_inference_timer()
                    outputs = self._forward_model(self.model, imgs)
                    epoch_inference_time += self._stop_inference_timer(infer_start)

                    task_masks = masks
                    task_gt = gt
                    task_batch = self._make_task_batch(task_masks, task_gt)
                    task_loss_dict = criterion(task_batch, outputs)
                    task_loss = task_loss_dict["total"]
                    strategy_loss = torch.tensor(0.0, device=self.device)
                    weighted_strategy_loss = torch.tensor(0.0, device=self.device)
                    G_loss = task_loss
                else:
                    select_idx = self._sample_strategy_supervision_indices(ori_imgs.shape[0], pair_imgs.device)
                    infer_start = self._start_inference_timer()
                    if hasattr(self.model, 'forward_strategy'):
                        pair_outputs = ensure_model_output(self.model.forward_strategy(pair_imgs, select_idx))
                        outputs = pair_outputs
                    else:
                        pair_outputs = self._forward_model(self.model, pair_imgs)
                        outputs = self._select_model_output(pair_outputs, select_idx)
                    epoch_inference_time += self._stop_inference_timer(infer_start)
                    task_masks = self._select_batch(pair_masks, select_idx)
                    task_gt = self._select_batch(pair_gt, select_idx)

                    task_batch = self._make_task_batch(task_masks, task_gt)
                    task_loss_dict = criterion(task_batch, outputs)
                    task_loss = task_loss_dict["total"]
                    strategy_loss = self._compute_strategy_loss(train_strategy, ori_imgs.shape[0], pair_outputs)
                    weighted_strategy_loss = strategy_loss_weight * strategy_loss
                    G_loss = task_loss + weighted_strategy_loss

                fake_images = outputs["output"]
                mm = outputs["mask_logits"]
                G_loss = G_loss.sum()
                if gradient_diagnostics_enabled and idx % gradient_diagnostics_interval == 0:
                    self._record_gradient_diagnostics(
                        loss_dict=task_loss_dict,
                        masks=task_masks,
                        epoch=epochs,
                        batch_idx=idx,
                        global_step=(epochs - 1) * len(traindataset) + idx,
                        save_path=gradient_diagnostics_path,
                    )
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()
                epoch_loss += G_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_strategy_loss += strategy_loss.item()
                epoch_weighted_strategy_loss += weighted_strategy_loss.item()
                epoch_msr_loss += self._loss_item(task_loss_dict, "multiscale")
                epoch_refinement_loss += self._loss_item(task_loss_dict, "refinement")
                epoch_mask_loss += self._loss_item(task_loss_dict, "mask")
                # print('[{}/{}] Generator Loss of epoch{} is {}'.format(k, len(dataloader), i, G_loss.item()))

                with torch.no_grad():
                    L1_loss, mse_loss, psnr = SimilarityLoss(fake_images, task_gt,SSIM=False)
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
                    f"\rEpoch: [{epochs}/{train_cfg['epochs']}] "
                    f"Batch: [{idx + 1}/{len(traindataset)}] "
                    f"Epoch Avg Loss: {epoch_loss / (idx + 1):.4f} "
                    f"Task: {epoch_task_loss / (idx + 1):.4f} "
                    f"Strategy: {epoch_strategy_loss / (idx + 1):.4f} "
                    f"WStrategy: {epoch_weighted_strategy_loss / (idx + 1):.4f} "
                    f"MSR: {epoch_msr_loss / (idx + 1):.4f} "
                    f"Refine: {epoch_refinement_loss / (idx + 1):.4f} "
                    f"Mask: {epoch_mask_loss / (idx + 1):.4f} "
                    f"L1 Loss: {epoch_L1_loss / (idx + 1):.4f} "
                    f"MSE Loss: {epoch_mse_loss / (idx + 1):.4f} "
                    f"PSNR: {epoch_psnr / (idx + 1):.4f} "
                    f"Avg time/batch: {avg_batch_time:.3f}s "
                    f"Elapsed: {formatted_elapsed} "
                    f"ETA: {formatted_eta} "
                    f"LR: {G_optimizer.param_groups[0]['lr']:.6g}  "
                )
                sys.stdout.flush()
                pre_time = time.time()


            val_start = time.time()
            val_mse, val_psnr, val_inference_time = self.test_epoch(valdataloader)
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
                            'TaskLoss': epoch_task_loss / (len(traindataset)),
                            'StrategyLoss': epoch_strategy_loss / (len(traindataset)),
                            'WeightedStrategyLoss': epoch_weighted_strategy_loss / (len(traindataset)),
                            'MSRLoss': epoch_msr_loss / (len(traindataset)),
                            'RefinementLoss': epoch_refinement_loss / (len(traindataset)),
                            'MaskLoss': epoch_mask_loss / (len(traindataset)),
                            'L1': epoch_L1_loss / (len(traindataset)),
                            'MSE': epoch_mse_loss / (len(traindataset)),
                            'PSNR': epoch_psnr / (len(traindataset)),
                            'ValidationMSE': val_mse,
                            'ValidationPSNR': val_psnr,
                            'TrainInferenceTime': epoch_inference_time,
                            'ValidationInferenceTime': val_inference_time,
                        })

            # Save sample images
            # 保存样例图片
            if (epochs % train_cfg['sample_save_every'] == 0):
                sample_images(valdataset, self.model, os.path.join(save_samples_dir, str(epochs) + '.png'))

            # Save model
            # 保存模型
            if (epochs % train_cfg['model_save_every'] == 0):
                 torch.save(self.model.state_dict(), os.path.join(save_models_dir, str(epochs) + '.pth'))

            print()

    def train_distill(self):
        train_cfg = self.config['train']
        distill_cfg = self.config['distill']
        loss_weights = distill_cfg['loss_weights']
        print("Distill Epoch:", train_cfg['epochs'], "  Batch size:", train_cfg['batch_size'])
        train_roots = self._resolve_dataset_roots(
            train_cfg['dataset_path'], "train.dataset_path"
        )
        val_roots = self._resolve_dataset_roots(
            self.config['validation']['dataset_path'], "validation.dataset_path"
        )

        save_models_dir = os.path.join(str(self.data_root_path), str('models'))
        save_samples_dir = os.path.join(str(self.data_root_path), str('samples'))
        save_logs_path = os.path.join(str(self.data_root_path), str('logs.csv'))
        self._init_data_path([save_models_dir, save_samples_dir], save_logs_path)

        criterion_task = self._build_task_loss(self.student_model_name).to(self.device)
        criterion_distill = DistillationLoss().to(self.device)

        if self.config['run']['data_crop']['use']:
            traindataset = DataLoader(
                BlockSegImageDataset(
                    root=train_roots,
                    mode="train",
                    tile_size=self.config['run']['data_crop']['size'],
                    use_aug=train_cfg['aug'],
                ),
                batch_size=train_cfg['batch_size'],
                shuffle=True,
                drop_last=True,
                num_workers=train_cfg['numberworks']
            )
            valdataset = BlockSegImageDataset(
                root=val_roots,
                mode="validation",
                tile_size=self.config['run']['data_crop']['size'],
                use_aug=False,
            )
            valdataloader = DataLoader(
                valdataset,
                batch_size=self.config['validation']['batch_size'],
                shuffle=True,
                num_workers=self.config['validation']['numberworks']
            )
        else:
            traindataset = DataLoader(
                SegImageDataset(
                    root=train_roots,
                    mode="train",
                    use_aug=train_cfg['aug'],
                ),
                batch_size=train_cfg['batch_size'],
                drop_last=True,
                shuffle=True,
                num_workers=train_cfg['numberworks'],
                pin_memory=True
            )
            valdataset = SegImageDataset(
                root=val_roots,
                mode="validation",
                use_aug=False,
            )
            valdataloader = DataLoader(
                valdataset,
                batch_size=self.config['validation']['batch_size'],
                shuffle=True,
                num_workers=self.config['validation']['numberworks'],
                pin_memory=True
            )

        optimizer_params = list(self.student_model.parameters())
        if self.teacher_adapter is not None and self.teacher_adapter_trainable:
            optimizer_params += list(self.teacher_adapter.parameters())

        optimizer = optim.Adam(
            optimizer_params,
            lr=train_cfg['learning_rate'],
            betas=(0.9, 0.999)
        )
        lr_schedule_map = self._parse_lr_schedule_map(train_cfg)

        recent_times = collections.deque(maxlen=len(traindataset) + 1)
        start_time = time.time()
        pre_time = time.time()

        sample_images(valdataset, self.student_model, os.path.join(save_samples_dir, str(0) + '.png'))
        torch.save(self.student_model.state_dict(), os.path.join(save_models_dir, str(0) + '.pth'))
        if self.deep_method == 'l2' and self.teacher_adapter is not None:
            torch.save(self.teacher_adapter.state_dict(), os.path.join(save_models_dir, str(0) + '_adapter.pth'))

        for epochs in range(1, train_cfg['epochs'] + 1):
            self._apply_lr_schedule(optimizer, epochs, lr_schedule_map)
            torch.cuda.empty_cache()
            epoch_loss_total = 0.0
            epoch_loss_task = 0.0
            epoch_loss_deep = 0.0
            epoch_loss_output = 0.0
            epoch_loss_mm = 0.0
            epoch_L1_loss = 0.0
            epoch_mse_loss = 0.0
            epoch_psnr = 0.0
            epoch_inference_time = 0.0

            self.teacher_model.eval()
            self.student_model.train()
            if self.teacher_adapter is not None and self.teacher_adapter_trainable:
                self.teacher_adapter.train()

            for idx, (imgs, gt, masks) in enumerate(traindataset):
                imgs = imgs.to(self.device)
                gt = gt.to(self.device)
                masks = masks.to(self.device)

                self.teacher_deep_feature = None
                self.student_deep_feature = None

                with torch.no_grad():
                    infer_start = self._start_inference_timer()
                    teacher_outputs = self._forward_model(self.teacher_model, imgs)
                    epoch_inference_time += self._stop_inference_timer(infer_start)

                infer_start = self._start_inference_timer()
                student_outputs = self._forward_model(self.student_model, imgs)
                epoch_inference_time += self._stop_inference_timer(infer_start)
                student_output = student_outputs["output"]

                if self.teacher_deep_feature is None or self.student_deep_feature is None:
                    raise RuntimeError("Error: Deep feature hook failed to capture outputs.")

                task_loss_dict = criterion_task(self._make_task_batch(masks, gt), student_outputs)
                task_loss = task_loss_dict["total"]
                distill_loss_dict = criterion_distill(student_outputs, teacher_outputs)
                output_loss = distill_loss_dict["output"]
                mm_loss = distill_loss_dict["mask"]
                deep_loss = self._compute_deep_loss(
                    student_deep=self.student_deep_feature,
                    teacher_deep=self.teacher_deep_feature,
                    optimizer=optimizer
                )

                total_loss = (
                    loss_weights['task'] * task_loss +
                    loss_weights['deep'] * deep_loss +
                    loss_weights['output'] * output_loss +
                    loss_weights['mm'] * mm_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss_total += total_loss.item()
                epoch_loss_task += task_loss.item()
                epoch_loss_deep += deep_loss.item()
                epoch_loss_output += output_loss.item()
                epoch_loss_mm += mm_loss.item()

                with torch.no_grad():
                    L1_loss, mse_loss, psnr = SimilarityLoss(student_output, gt, SSIM=False)
                    epoch_L1_loss += L1_loss.item()
                    epoch_mse_loss += mse_loss.item()
                    epoch_psnr += psnr.item()

                    batch_time = time.time() - pre_time
                    recent_times.append(batch_time)
                    avg_batch_time = sum(recent_times) / len(recent_times)
                    elapsed_time = time.time() - start_time
                    formatted_elapsed = dt.timedelta(seconds=int(elapsed_time))

                    batches_done = (epochs - 1) * (len(traindataset) + 1) + idx + 1
                    total_batches = train_cfg['epochs'] * (len(traindataset) + 1)
                    remaining_batches = total_batches - batches_done
                    eta_seconds = remaining_batches * avg_batch_time
                    formatted_eta = dt.timedelta(seconds=int(eta_seconds))

                sys.stdout.write(
                    f"\rEpoch: [{epochs}/{train_cfg['epochs']}] "
                    f"Batch: [{idx + 1}/{len(traindataset)}] "
                    f"LossTotal: {epoch_loss_total / (idx + 1):.4f} "
                    f"Task: {epoch_loss_task / (idx + 1):.4f} "
                    f"Deep: {epoch_loss_deep / (idx + 1):.4f} "
                    f"Output: {epoch_loss_output / (idx + 1):.4f} "
                    f"MM: {epoch_loss_mm / (idx + 1):.4f} "
                    f"L1: {epoch_L1_loss / (idx + 1):.4f} "
                    f"MSE: {epoch_mse_loss / (idx + 1):.4f} "
                    f"PSNR: {epoch_psnr / (idx + 1):.4f} "
                    f"Avg time/batch: {avg_batch_time:.3f}s "
                    f"Elapsed: {formatted_elapsed} "
                    f"ETA: {formatted_eta} "
                    f"LR: {optimizer.param_groups[0]['lr']:.6g}  "
                )
                sys.stdout.flush()
                pre_time = time.time()

            self.model = self.student_model
            val_start = time.time()
            val_mse, val_psnr, val_inference_time = self.test_epoch(valdataloader)
            val_cost = time.time() - val_start
            recent_times.append(val_cost)
            pre_time = time.time()

            with open(save_logs_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, self.distill_logs_fields)
                writer.writerow({
                    'epoch': epochs,
                    'loss_total': epoch_loss_total / len(traindataset),
                    'loss_task': epoch_loss_task / len(traindataset),
                    'loss_deep': epoch_loss_deep / len(traindataset),
                    'loss_output': epoch_loss_output / len(traindataset),
                    'loss_mm': epoch_loss_mm / len(traindataset),
                    'L1': epoch_L1_loss / len(traindataset),
                    'MSE': epoch_mse_loss / len(traindataset),
                    'PSNR': epoch_psnr / len(traindataset),
                    'ValidationMSE': val_mse,
                    'ValidationPSNR': val_psnr,
                    'TrainInferenceTime': epoch_inference_time,
                    'ValidationInferenceTime': val_inference_time,
                })

            if (epochs % train_cfg['sample_save_every'] == 0):
                sample_images(valdataset, self.student_model, os.path.join(save_samples_dir, str(epochs) + '.png'))

            if (epochs % train_cfg['model_save_every'] == 0):
                torch.save(self.student_model.state_dict(), os.path.join(save_models_dir, str(epochs) + '.pth'))
                if self.deep_method == 'l2' and self.teacher_adapter is not None:
                    torch.save(
                        self.teacher_adapter.state_dict(),
                        os.path.join(save_models_dir, str(epochs) + '_adapter.pth')
                    )

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
        epoch_inference_time = 0.0

        for idx, (imgs, gt, masks) in enumerate(testloader):
            # ---------------- 数据搬运 ----------------
            imgs = imgs.to(self.device)
            gt = gt.to(self.device)
            masks = masks.to(self.device)

            # ---------------- 前向推理 ----------------
            infer_start = self._start_inference_timer()
            outputs = self._forward_model(self.model, imgs)
            epoch_inference_time += self._stop_inference_timer(infer_start)
            fake_images = outputs["output"]

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

        return epoch_mse_loss / len(testloader), epoch_psnr / len(testloader), epoch_inference_time

    def evaluate(self):

        # Log file
        # 创建日志文件
        save_logs_path = os.path.join(str(self.data_root_path), str('logs.csv'))
        eval_roots = self._resolve_dataset_roots(
            self.config['evaluate']['dataset_path'], "evaluate.dataset_path"
        )

        # Init logs file
        # 初始化评估的日志文件
        self._init_data_path([self.data_root_path], save_logs_path)

        # 将模型切换到评估模式
        self.model.eval()

        # Use pretrained model
        model_path = self.config['evaluate']['model_path']
        self._load_checkpoint_flexible(self.model, model_path, name="evaluate.model")

        # Loss
        # 创建损失函数
        criterion = self._build_task_loss(self.model_name).to(self.device)


        # Train dataloader
        # 创建训练数据集，使用裁剪数据或否不使用裁剪数据
        if self.config['run']['data_crop']['use'] :
            evaldataset = DataLoader(
                BlockSegImageDataset(
                    root=eval_roots,
                    mode="evaluate",
                    tile_size=self.config['run']['data_crop'] ['size'],
                    use_aug=False,
                ),
                batch_size=1,
                shuffle=True,
                num_workers=self.config['evaluate']['numberworks'],
                pin_memory=True)

        else:
            evaldataset = DataLoader(
                SegImageDataset(
                    root=eval_roots,
                    mode="evaluate",
                    use_aug=False,
                ),
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
                outputs = self._forward_model(self.model, imgs)
                fake_images = outputs["output"]
                loss_dict = criterion(self._make_task_batch(masks, gt), outputs)
                G_loss = loss_dict["total"]
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
        self._load_checkpoint_flexible(self.model, model_path, name="predict.model")

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

                    outputs_b = self._forward_model(self.model, tile_tensor)
                    x1_b = outputs_b["x1"]
                    x2_b = outputs_b["x2"]
                    x3_b = outputs_b["x3"]
                    output_b = outputs_b["output"]
                    mask_b = outputs_b["mask_logits"]

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
                outputs = self._forward_model(self.model, image_tensor)
                x1 = outputs["x1"]
                x2 = outputs["x2"]
                x3 = outputs["x3"]
                output = outputs["output"]
                mask = outputs["mask_logits"]
            print(x1.shape, x2.shape, x3.shape, output.shape, mask.shape)
            mask = torch.argmax(mask, 1).unsqueeze(0)
            mask = maskToTensor(mask, self.device)
            save_tensor_as_image(x1, os.path.join(str(self.data_root_path),'x1.png'))
            save_tensor_as_image(x2, os.path.join(str(self.data_root_path),'x2.png'))
            save_tensor_as_image(x3, os.path.join(str(self.data_root_path),'x3.png'))
            save_tensor_as_image(output, os.path.join(str(self.data_root_path),'output.png'))
            # save_tensor_as_image(mask, os.path.join(str(self.data_root_path),'mask.png'))

            save_image(mask, os.path.join(str(self.data_root_path),'mask.png'), nrow=1, normalize=True)


    def _build_model(self, model_name, custom_cfg=None):
        if model_name == 'Release':
            return Release.ResNet_UNet().to(self.device)
        if model_name == 'Release_lightly':
            return Release_lightly.ResNet_UNet().to(self.device)
        if model_name == 'Release_lightly_extra':
            return Release_lightly_extra.ResNet_UNet().to(self.device)
        if model_name == 'Custom':
            cfg = custom_cfg if custom_cfg is not None else self.config['custom']
            return Custom.ResNet_UNet(
                base=cfg['base'],
                refinement=cfg['refinement'],
                ffp=cfg['ffp'],
                ppm=cfg['ppm'],
                down_sample=cfg['down_sample'],
                am=cfg['am'],
                enable_srpdg_attention=self._get_train_strategy() == 'srpdg',
            ).to(self.device)
        if model_name == 'Custom3Tags':
            cfg = custom_cfg if custom_cfg is not None else self.config['custom']
            return Custom3Tags.ResNet_UNet(
                base=cfg['base'],
                refinement=cfg['refinement'],
                ffp=cfg['ffp'],
                ppm=cfg['ppm'],
                down_sample=cfg['down_sample'],
                am=cfg['am'],
                enable_srpdg_attention=self._get_train_strategy() == 'srpdg',
            ).to(self.device)
        if model_name == 'ResNet':
            return ResNet.ResNet_UNet().to(self.device)
        if model_name == 'IEResNet':
            return IEResNet.ResNet_UNet().to(self.device)
        if model_name == 'IEXResNet':
            return IEXResNet.ResNet_UNet().to(self.device)
        raise RuntimeError(f"Error: Model `{model_name}` is not exist!")

    def _get_train_strategy(self):
        if getattr(self, 'worktype', None) != 'train':
            return 'none'
        train_cfg = self.config.get('train', {})
        strategy_cfg = train_cfg.get('strategy', 'none')
        if isinstance(strategy_cfg, dict):
            strategy = strategy_cfg.get('name', 'none')
        else:
            strategy = strategy_cfg
        if strategy is None:
            strategy = 'none'
        return str(strategy).lower()

    def _validate_train_strategy(self, strategy):
        if strategy not in ('none', 'srpdg', 'coral'):
            raise RuntimeError("Error: train.strategy must be one of `none`, `srpdg`, or `coral`.")
        if strategy in ('srpdg', 'coral') and self.model_name not in ('Custom', 'Custom3Tags'):
            raise RuntimeError(f"Error: train.strategy `{strategy}` only supports Custom and Custom3Tags.")

    def _sample_strategy_supervision_indices(self, batch_size, device):
        # 全部 batch_size 个样本参与任务损失
        # 每个样本以 STRATEGY_AUG_PROB 概率选 aug 视角，与普通训练的增强概率对齐
        # pair batch 布局：[ori_0..ori_{n-1}, aug_0..aug_{n-1}]
        base = torch.arange(batch_size, device=device)
        use_aug = torch.rand(batch_size, device=device) < STRATEGY_AUG_PROB
        return base + use_aug.long() * batch_size

    def _select_batch(self, tensor, indices):
        return tensor.index_select(0, indices)

    def _forward_model(self, model, imgs):
        return ensure_model_output(model(imgs))

    def _make_task_batch(self, masks, gt):
        return {
            "mask": masks,
            "rebuild": gt,
        }

    def _select_model_output(self, outputs, indices):
        selected = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                selected[key] = self._select_batch(value, indices)
            elif isinstance(value, dict):
                selected[key] = {
                    sub_key: self._select_batch(sub_value, indices)
                    if isinstance(sub_value, torch.Tensor) else sub_value
                    for sub_key, sub_value in value.items()
                }
            else:
                selected[key] = value
        return selected

    def _loss_item(self, loss_dict, key):
        value = loss_dict.get(key)
        if value is None:
            return 0.0
        if isinstance(value, torch.Tensor):
            return value.detach().item()
        return float(value)

    def _init_gradient_diagnostics(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv.DictWriter(csv_file, fieldnames=self.gradient_diagnostics_fields).writeheader()

    @staticmethod
    def _gradient_parameter_group(parameter_name):
        """Map shared model parameters to coarse architectural regions."""
        name = parameter_name[7:] if parameter_name.startswith('module.') else parameter_name
        root = name.split('.', 1)[0]
        if root in ('inc', 'stem', 'down1', 'layer1'):
            return 'encoder_shallow'
        if root in ('down2', 'down3', 'layer2', 'layer3'):
            return 'encoder_middle'
        if root in ('down4', 'layer4', 'ppm'):
            return 'encoder_deep'
        if root in ('up1', 'up2'):
            return 'shared_decoder'
        return None

    @staticmethod
    def _gradient_statistics(gradient_pairs, eps=1e-12):
        dot = None
        rebuild_squared_norm = None
        segment_squared_norm = None
        parameter_tensors = 0
        for rebuild_gradient, segment_gradient in gradient_pairs:
            if rebuild_gradient is None or segment_gradient is None:
                continue
            pair_dot = torch.sum(rebuild_gradient * segment_gradient).double()
            pair_rebuild_norm = torch.sum(rebuild_gradient * rebuild_gradient).double()
            pair_segment_norm = torch.sum(segment_gradient * segment_gradient).double()
            dot = pair_dot if dot is None else dot + pair_dot
            rebuild_squared_norm = (
                pair_rebuild_norm
                if rebuild_squared_norm is None
                else rebuild_squared_norm + pair_rebuild_norm
            )
            segment_squared_norm = (
                pair_segment_norm
                if segment_squared_norm is None
                else segment_squared_norm + pair_segment_norm
            )
            parameter_tensors += 1

        if parameter_tensors == 0:
            return {
                'cosine': float('nan'),
                'rebuild_norm': 0.0,
                'segment_norm': 0.0,
                'norm_ratio': 0.0,
                'is_conflict': '',
                'parameter_tensors': 0,
            }

        dot = dot.item()
        rebuild_squared_norm = rebuild_squared_norm.item()
        segment_squared_norm = segment_squared_norm.item()
        rebuild_norm = rebuild_squared_norm ** 0.5
        segment_norm = segment_squared_norm ** 0.5
        denominator = rebuild_norm * segment_norm
        cosine = dot / denominator if denominator > eps else float('nan')
        norm_ratio = rebuild_norm / max(segment_norm, eps)
        return {
            'cosine': cosine,
            'rebuild_norm': rebuild_norm,
            'segment_norm': segment_norm,
            'norm_ratio': norm_ratio,
            'is_conflict': int(cosine < 0.0) if cosine == cosine else '',
            'parameter_tensors': parameter_tensors,
        }

    def _record_gradient_diagnostics(
        self,
        *,
        loss_dict,
        masks,
        epoch,
        batch_idx,
        global_step,
        save_path,
    ):
        reconstruction_terms = [
            loss_dict.get('multiscale'),
            loss_dict.get('refinement'),
        ]
        reconstruction_terms = [
            value for value in reconstruction_terms if isinstance(value, torch.Tensor)
        ]
        segmentation_loss = loss_dict.get('mask')
        if not reconstruction_terms or not isinstance(segmentation_loss, torch.Tensor):
            return

        reconstruction_loss = sum(value.sum() for value in reconstruction_terms)
        segmentation_loss = segmentation_loss.sum()
        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and self._gradient_parameter_group(name) is not None
        ]
        if not named_parameters:
            return

        parameters = [parameter for _, parameter in named_parameters]
        rebuild_gradients = torch.autograd.grad(
            reconstruction_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        segment_gradients = torch.autograd.grad(
            segmentation_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

        grouped_pairs = collections.defaultdict(list)
        all_pairs = []
        for (name, _), rebuild_gradient, segment_gradient in zip(
            named_parameters,
            rebuild_gradients,
            segment_gradients,
        ):
            pair = (rebuild_gradient, segment_gradient)
            grouped_pairs[self._gradient_parameter_group(name)].append(pair)
            all_pairs.append(pair)
        grouped_pairs['shared_all'] = all_pairs

        flat_masks = masks.detach()
        total_pixels = max(int(flat_masks.numel()), 1)
        class_ratios = {
            'background_ratio': int((flat_masks == 0).sum().item()) / total_pixels,
            'handwriting_ratio': int((flat_masks == 1).sum().item()) / total_pixels,
            'print_ratio': int((flat_masks == 2).sum().item()) / total_pixels,
            'overlap_ratio': int((flat_masks == 3).sum().item()) / total_pixels,
        }

        rows = []
        group_order = (
            'encoder_shallow',
            'encoder_middle',
            'encoder_deep',
            'shared_decoder',
            'shared_all',
        )
        for group_name in group_order:
            gradient_pairs = grouped_pairs.get(group_name, [])
            statistics = self._gradient_statistics(gradient_pairs)
            if statistics['parameter_tensors'] == 0:
                continue
            rows.append({
                'epoch': epoch,
                'batch': batch_idx,
                'global_step': global_step,
                'layer': group_name,
                **statistics,
                **class_ratios,
            })

        if rows:
            with open(save_path, 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.gradient_diagnostics_fields)
                writer.writerows(rows)

    def _build_loader_kwargs(self, num_workers, pin_memory=True, prefetch_factor=None):
        kwargs = {
            "num_workers": int(num_workers),
            "pin_memory": pin_memory,
        }
        if kwargs["num_workers"] > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        return kwargs

    def _compute_strategy_loss(self, strategy, batch_size, outputs=None):
        features = {}
        if isinstance(outputs, dict):
            features = outputs.get("features", {}) or {}
        if not features:
            features = getattr(self.model, 'strategy_features', {})
        if strategy == 'srpdg':
            loss = 0.0
            for key in ('attn_down2', 'attn_down3'):
                attn = features.get(key)
                if attn is None:
                    raise RuntimeError(f"Error: missing `{key}` for srpdg strategy.")
                loss = loss + F.mse_loss(attn[:batch_size], attn[batch_size:])
            return loss
        if strategy == 'coral':
            loss = 0.0
            for key in ('down2', 'down3'):
                feature = features.get(key)
                if feature is None:
                    raise RuntimeError(f"Error: missing `{key}` for coral strategy.")
                loss = loss + self._deep_coral_loss(feature[:batch_size], feature[batch_size:])
            return loss
        return torch.tensor(0.0, device=self.device)

    def _get_strategy_loss_weight(self, strategy):
        if strategy == 'srpdg':
            return float(self.config.get('srpdg', {}).get('lambda', 1.0))
        return 1.0

    def _deep_coral_loss(self, source, target):
        source = F.adaptive_avg_pool2d(source, output_size=1).flatten(1)
        target = F.adaptive_avg_pool2d(target, output_size=1).flatten(1)
        source = source - source.mean(dim=0, keepdim=True)
        target = target - target.mean(dim=0, keepdim=True)
        denom = max(source.shape[0] - 1, 1)
        source_cov = source.t().matmul(source) / denom
        target_cov = target.t().matmul(target) / denom
        channels = source.shape[1]
        return ((source_cov - target_cov) ** 2).sum() / (4.0 * channels * channels)

    def _build_task_loss(self, model_name):
        if model_name == 'Custom3Tags':
            return SegMultTaskLoss3Tags()
        return SegMultTaskLoss()

    def _build_refinement_loss(self, model_name):
        if model_name == 'Custom3Tags':
            return RefinementLoss3Tags()
        return RefinementLoss()

    def _mask_target_for_model(self, masks, mm):
        if mm.shape[1] == 3:
            return torch.where(masks == 3, torch.ones_like(masks), masks)
        return masks

    def _get_model(self):
        self.model_name = self.config['run']['model']
        self.model = self._build_model(self.model_name)

    def _resolve_deep_module(self, model, model_name):
        tap = self.config['distill'].get('deep_feature_tap', 'ppm_post')
        if tap != 'ppm_post':
            raise RuntimeError("Error: only `ppm_post` deep_feature_tap is supported.")

        # Decision-fixed behavior for current project.
        if model_name == 'Release':
            return model.ppm
        if model_name in ('Release_lightly', 'Release_lightly_extra', 'Custom'):
            return model.down4
        if model_name in ('ResNet', 'IEResNet', 'IEXResNet'):
            return model.layer4

        # Fallback for unknown model names.
        if hasattr(model, 'ppm'):
            return model.ppm
        if hasattr(model, 'down4'):
            return model.down4
        raise RuntimeError(f"Error: cannot resolve deep feature module for model `{model_name}`.")

    def _register_deep_hooks(self):
        teacher_module = self._resolve_deep_module(self.teacher_model, self.teacher_model_name)
        student_module = self._resolve_deep_module(self.student_model, self.student_model_name)

        def teacher_hook(_, __, output):
            self.teacher_deep_feature = output

        def student_hook(_, __, output):
            self.student_deep_feature = output

        self.teacher_hook_handle = teacher_module.register_forward_hook(teacher_hook)
        self.student_hook_handle = student_module.register_forward_hook(student_hook)

    def _get_distill_models(self):
        if 'distill' not in self.config:
            raise RuntimeError("Error: `distill` config is required for train_distill mode.")

        distill_cfg = self.config['distill']
        self.teacher_model_name = distill_cfg['teacher_model']
        self.student_model_name = distill_cfg['student_model']
        if self.teacher_model_name == 'Custom3Tags' or self.student_model_name == 'Custom3Tags':
            raise RuntimeError("Error: `Custom3Tags` does not support train_distill yet.")
        self.deep_method = distill_cfg.get('deep_method', 'l2').lower()
        if self.deep_method not in ('l2', 'at'):
            raise RuntimeError("Error: distill.deep_method must be `l2` or `at`.")

        teacher_custom_cfg = None
        student_custom_cfg = None
        if self.teacher_model_name == 'Custom':
            teacher_custom_cfg = distill_cfg.get('teacher_custom', None)
            if teacher_custom_cfg is None:
                raise RuntimeError("Error: `distill.teacher_custom` is required when teacher_model is Custom.")
        if self.student_model_name == 'Custom':
            student_custom_cfg = distill_cfg.get('student_custom', None)
            if student_custom_cfg is None:
                raise RuntimeError("Error: `distill.student_custom` is required when student_model is Custom.")

        self.teacher_model = self._build_model(self.teacher_model_name, custom_cfg=teacher_custom_cfg)
        self.student_model = self._build_model(self.student_model_name, custom_cfg=student_custom_cfg)

        teacher_model_path = distill_cfg['teacher_model_path']
        self._load_checkpoint_flexible(self.teacher_model, teacher_model_path, name="distill.teacher")
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        student_pretrained_cfg = distill_cfg.get('student_pretrained', {'use': False})
        if student_pretrained_cfg.get('use', False):
            student_model_path = student_pretrained_cfg['model_path']
            self._load_checkpoint_flexible(self.student_model, student_model_path, name="distill.student_pretrained")
            # Legacy behavior: try to resume sidecar adapter checkpoint.
            self.adapter_resume_path = os.path.splitext(student_model_path)[0] + '_adapter.pth'
            if not os.path.exists(self.adapter_resume_path):
                self.adapter_resume_path = None

        adapter_cfg = distill_cfg.get('adapter', {}).get('teacher_conv1x1', {})
        self.teacher_adapter_trainable = adapter_cfg.get('trainable', True)
        # Explicit adapter checkpoint has higher priority than legacy sidecar auto-detection.
        adapter_resume_cfg = adapter_cfg.get('resume', {})
        if isinstance(adapter_resume_cfg, dict) and adapter_resume_cfg.get('use', False):
            explicit_adapter_path = str(adapter_resume_cfg.get('path', '')).strip()
            if not explicit_adapter_path:
                raise RuntimeError(
                    "Error: `distill.adapter.teacher_conv1x1.resume.path` is required when resume.use is True."
                )
            if not os.path.exists(explicit_adapter_path):
                raise RuntimeError(f"Error: adapter checkpoint not found: {explicit_adapter_path}")
            self.adapter_resume_path = explicit_adapter_path
        else:
            # Compatible with a flat field: adapter.teacher_conv1x1.resume_path
            explicit_adapter_path = str(adapter_cfg.get('resume_path', '')).strip()
            if explicit_adapter_path:
                if not os.path.exists(explicit_adapter_path):
                    raise RuntimeError(f"Error: adapter checkpoint not found: {explicit_adapter_path}")
                self.adapter_resume_path = explicit_adapter_path

        self._register_deep_hooks()

    def _ensure_teacher_adapter(self, teacher_deep, student_deep, optimizer):
        if self.teacher_adapter is not None:
            return

        in_channels = teacher_deep.shape[1]
        out_channels = student_deep.shape[1]
        self.teacher_adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(self.device)

        if self.adapter_resume_path is not None and os.path.exists(self.adapter_resume_path):
            self._load_checkpoint_flexible(self.teacher_adapter, self.adapter_resume_path, name="distill.teacher_adapter")
            print(f"Loaded adapter checkpoint: {self.adapter_resume_path}")

        if not self.teacher_adapter_trainable:
            self.teacher_adapter.eval()
            for p in self.teacher_adapter.parameters():
                p.requires_grad = False
        else:
            optimizer.add_param_group({'params': self.teacher_adapter.parameters()})

    def _compute_deep_loss(self, student_deep, teacher_deep, optimizer):
        if self.deep_method == 'at':
            if teacher_deep.shape[2:] != student_deep.shape[2:]:
                teacher_deep = torch.nn.functional.interpolate(
                    teacher_deep,
                    size=student_deep.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            return deep_feature_at_loss(student_deep, teacher_deep)

        if self.deep_method == 'l2':
            self._ensure_teacher_adapter(teacher_deep, student_deep, optimizer)
            teacher_mapped = self.teacher_adapter(teacher_deep.detach())
            if teacher_mapped.shape[2:] != student_deep.shape[2:]:
                teacher_mapped = torch.nn.functional.interpolate(
                    teacher_mapped,
                    size=student_deep.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            return deep_feature_l2_loss(student_deep, teacher_mapped)

        raise RuntimeError(f"Error: unknown deep_method `{self.deep_method}`.")

    def _parse_lr_schedule_map(self, train_cfg):
        lr_schedule_cfg = train_cfg.get('lr_schedule', {})
        if not isinstance(lr_schedule_cfg, dict) or not lr_schedule_cfg.get('use', False):
            return {}

        steps = lr_schedule_cfg.get('steps', [])
        if not isinstance(steps, list):
            raise RuntimeError("Error: `train.lr_schedule.steps` must be a list.")

        schedule_map = {}
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                raise RuntimeError(f"Error: `train.lr_schedule.steps[{idx}]` must be a dict.")

            epoch = step.get('epoch', None)
            lr = step.get('lr', None)
            if epoch is None or lr is None:
                raise RuntimeError(
                    f"Error: `train.lr_schedule.steps[{idx}]` must include both `epoch` and `lr`."
                )

            try:
                epoch = int(epoch)
                lr = float(lr)
            except (TypeError, ValueError):
                raise RuntimeError(
                    f"Error: invalid schedule item at index {idx}, epoch={epoch}, lr={lr}."
                )

            if epoch < 1:
                raise RuntimeError(f"Error: `train.lr_schedule.steps[{idx}].epoch` must be >= 1.")
            if lr <= 0:
                raise RuntimeError(f"Error: `train.lr_schedule.steps[{idx}].lr` must be > 0.")
            if epoch in schedule_map:
                raise RuntimeError(f"Error: duplicated lr schedule epoch: {epoch}.")

            schedule_map[epoch] = lr

        if schedule_map:
            schedule_text = ", ".join(
                [f"epoch {ep}->{value:.6g}" for ep, value in sorted(schedule_map.items())]
            )
            print(f"LR schedule enabled: {schedule_text}")
        else:
            print("LR schedule enabled but no step provided; keep constant learning rate.")
        return schedule_map

    def _apply_lr_schedule(self, optimizer, epoch, schedule_map):
        if epoch not in schedule_map:
            return

        new_lr = schedule_map[epoch]
        old_lrs = sorted({float(group['lr']) for group in optimizer.param_groups})
        for group in optimizer.param_groups:
            group['lr'] = new_lr

        old_lr_text = "/".join([f"{value:.6g}" for value in old_lrs]) if old_lrs else "N/A"
        print(f"\n[LR] epoch {epoch}: {old_lr_text} -> {new_lr:.6g}")


    def _load_checkpoint_flexible(self, model, model_path, name="model"):
        """
        Load only compatible checkpoint tensors.

        A tensor is compatible when its key exists in the current model and its
        shape is identical. All other current model tensors keep their initial
        random values.
        """
        model_path = os.fspath(model_path)
        if not model_path.strip():
            raise RuntimeError(f"Error: checkpoint path is empty for `{name}`.")
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Error: checkpoint not found for `{name}`: {model_path}")

        print(f"[Checkpoint:{name}] Loading flexible weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        checkpoint_state = self._extract_state_dict_from_checkpoint(checkpoint, model_path)
        checkpoint_state = self._normalize_checkpoint_keys(checkpoint_state)
        model_state = model.state_dict()

        loadable_state = {}
        unexpected_keys = []
        shape_mismatch_keys = []
        non_tensor_keys = []

        for key, value in checkpoint_state.items():
            if key not in model_state:
                unexpected_keys.append(key)
                continue

            target_value = model_state[key]
            if not hasattr(value, "shape") or not hasattr(target_value, "shape"):
                non_tensor_keys.append(key)
                continue

            source_shape = tuple(value.shape)
            target_shape = tuple(target_value.shape)
            if source_shape != target_shape:
                shape_mismatch_keys.append((key, source_shape, target_shape))
                continue

            loadable_state[key] = value

        model.load_state_dict(loadable_state, strict=False)
        missing_keys = [key for key in model_state.keys() if key not in loadable_state]

        print(
            f"[Checkpoint:{name}] loaded {len(loadable_state)}/{len(model_state)} current keys "
            f"from {len(checkpoint_state)} checkpoint keys."
        )
        self._print_checkpoint_key_preview(
            name,
            "missing or randomly initialized keys",
            missing_keys,
        )
        self._print_checkpoint_key_preview(
            name,
            "extra checkpoint keys ignored",
            unexpected_keys,
        )
        self._print_checkpoint_key_preview(
            name,
            "shape-mismatched keys ignored",
            shape_mismatch_keys,
            formatter=lambda item: f"{item[0]}: checkpoint{item[1]} -> model{item[2]}",
        )
        self._print_checkpoint_key_preview(
            name,
            "non-tensor entries ignored",
            non_tensor_keys,
        )

        if not loadable_state:
            print(
                f"[Checkpoint:{name}] Warning: no compatible weights were loaded; "
                "the model remains randomly initialized."
            )

        return {
            "loaded": list(loadable_state.keys()),
            "missing": missing_keys,
            "unexpected": unexpected_keys,
            "shape_mismatch": shape_mismatch_keys,
            "non_tensor": non_tensor_keys,
        }

    def _extract_state_dict_from_checkpoint(self, checkpoint, model_path):
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f"Error: unsupported checkpoint format at {model_path}: {type(checkpoint)}"
            )

        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        return checkpoint

    def _normalize_checkpoint_keys(self, state_dict):
        normalized = {}
        for key, value in state_dict.items():
            if not isinstance(key, str):
                continue
            new_key = key[7:] if key.startswith("module.") else key
            normalized[new_key] = value
        return normalized

    def _print_checkpoint_key_preview(self, name, title, items, formatter=str, limit=10):
        if not items:
            return

        print(f"[Checkpoint:{name}] {title}: {len(items)}")
        for item in items[:limit]:
            print(f"  - {formatter(item)}")
        if len(items) > limit:
            print(f"  ... {len(items) - limit} more")


    def _resolve_dataset_roots(self, dataset_path, cfg_key):
        if isinstance(dataset_path, (str, os.PathLike)):
            roots = [os.fspath(dataset_path)]
        elif isinstance(dataset_path, (list, tuple)):
            roots = [
                os.fspath(path) for path in dataset_path
                if isinstance(path, (str, os.PathLike)) and str(path).strip()
            ]
        else:
            raise RuntimeError(
                f"Error: `{cfg_key}` should be a path string or a yaml list of path strings."
            )

        if not roots:
            raise RuntimeError(f"Error: `{cfg_key}` is empty.")

        missing_paths = [path for path in roots if not os.path.isdir(path)]
        if missing_paths:
            raise RuntimeError(f"Error: dataset path(s) not found for `{cfg_key}`: {missing_paths}")

        print(f"{cfg_key} roots ({len(roots)}): {roots}")
        return roots


    def _sync_device_for_timing(self):
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _start_inference_timer(self):
        self._sync_device_for_timing()
        return time.perf_counter()

    def _stop_inference_timer(self, start_time):
        self._sync_device_for_timing()
        return time.perf_counter() - start_time


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
            if self.worktype == "train_distill":
                with open(logs_file, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=self.distill_logs_fields)
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
