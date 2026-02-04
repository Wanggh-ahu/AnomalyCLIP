import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
from mamba_module import MambaAdapter  # [新增] 导入我们写的模块

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    # 加载 CLIP 模型
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    
    # [新增] --- 初始化 Mamba Adapter ---
    # 自动获取 CLIP 的特征维度
    # 注意: AnomalyCLIP_lib 可能封装了 visual，如果报错请尝试 model.visual.output_dim
    # ViT-L/14 输出通常是 768
    if hasattr(model, 'visual'):
        visual_dim = model.visual.output_dim
    else:
        visual_dim = 768 # Fallback for ViT-L/14
        
    # 实例化并移到 GPU
    visual_adapter = MambaAdapter(d_model=visual_dim).to(device)
    # [新增结束] ------------------------

    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    ##########################################################################################
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)
    ##########################################################################################
    
    # [修改] 优化器定义：同时包含 PromptLearner 和 MambaAdapter
    optimizer = torch.optim.Adam([
        {'params': prompt_learner.parameters(), 'lr': args.learning_rate},
        {'params': visual_adapter.parameters(), 'lr': 1e-4} # [新增] Mamba 学习率
    ], lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    lam = 4
    
    model.eval() # CLIP 始终 Eval
    prompt_learner.train()
    visual_adapter.train() # [新增] Mamba 需要 Train 模式

    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        visual_adapter.train() # 确保每个 epoch 开始都设为 train

        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            # --- 修改 Forward 流程以支持 Mamba 训练 ---
            
            # 1. 获取 CLIP 原始特征 (保持无梯度，因为 CLIP 锁死了)
            with torch.no_grad():
                # image_features: [B, Dim] (Global)
                # patch_features: List of [B, L, Dim] (Multi-scale)
                image_features, patch_features_list = model.encode_image(image, args.features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 2. [新增] Mamba 增强 Patch Features
            # 我们需要让梯度流过 visual_adapter，所以这里不加 no_grad
            # 对 patch_features_list 中的每一层特征都过一遍 Mamba 
            # (注意：patch_features 是一个 list，里面是 tensor)
            
            enhanced_patch_features = []
            for patch_feat in patch_features_list:
                # Mamba 需要 [Batch, Seq_Len, Dim]
                # patch_feat 已经是这个形状了 (除了第一维可能是 Class Token, 具体看 AnomalyCLIP 实现)
                # 通常 patch_feat: [B, 1+H*W, C] 或 [B, H*W, C]
                
                # 如果是 Tensor 才处理
                if isinstance(patch_feat, torch.Tensor):
                    # 确保 requires_grad (虽然输入来自 no_grad，但经过 adapter 后会有 grad)
                    feat_in = patch_feat.detach() # 断开与 CLIP 的计算图
                    feat_in.requires_grad = True  
                    
                    # 过 Mamba
                    feat_out = visual_adapter(feat_in)
                    enhanced_patch_features.append(feat_out)
                else:
                    enhanced_patch_features.append(patch_feat)
            
            # 使用增强后的特征列表替代原始列表
            patch_features = enhanced_patch_features

            # 3. Prompt Learner (带梯度)
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            
            # 4. 计算 Image Loss
            # 注意：这里的 text_probs 涉及 image_features (无梯度) 和 text_features (有梯度)
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...]/0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().to(device))
            image_loss_list.append(image_loss.item())
            
            # 5. 计算 Similarity Map 和 Mask Loss
            similarity_map_list = []
            
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    # 归一化
                    patch_feature = patch_feature / patch_feature.norm(dim = -1, keepdim = True)
                    
                    # 计算相似度
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    
                    # 生成 Map
                    # 注意: patch_feature[:, 1:, :] 假设第0个是 cls token
                    # 如果 AnomalyCLIP 改过，可能不需要 slicing
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            loss = lam * loss
            
            optimizer.zero_grad()
            (loss + image_loss).backward()
            optimizer.step()
            
            loss_list.append(loss.item())
            
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            # [修改] 保存 prompt_learner 和 visual_adapter
            torch.save({
                "prompt_learner": prompt_learner.state_dict(),
                "visual_adapter": visual_adapter.state_dict() # [新增]
            }, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')


    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
