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

import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform

# [新增]
from mamba_module import MambaAdapter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    # 加载 CLIP 模型
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    
    # [新增] --- 初始化 Mamba Adapter ---
    # 确保维度匹配
    if hasattr(model, 'visual'):
        visual_dim = model.visual.output_dim
    else:
        visual_dim = 768 # Fallback
    
    visual_adapter = MambaAdapter(d_model=visual_dim).to(device)
    # 设为 Eval 模式
    visual_adapter.eval()
    # [新增结束] ------------------------

    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list


    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    
    # [修改] 加载权重
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # 加载 Prompt Learner
    if "prompt_learner" in checkpoint:
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    else:
        # 兼容旧版权重，如果 checkpoint 本身就是 state_dict
        try:
            prompt_learner.load_state_dict(checkpoint)
        except:
            print("Warning: Could not load prompt_learner weights.")

    # [新增] 加载 Mamba Adapter 权重
    if "visual_adapter" in checkpoint:
        visual_adapter.load_state_dict(checkpoint["visual_adapter"])
        print("Success: Loaded Mamba visual_adapter weights.")
    else:
        print("Warning: 'visual_adapter' not found in checkpoint. Using random initialization (Performance will be low).")

    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)


    model.to(device)
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            # 1. 获取 CLIP 原始特征
            image_features, patch_features_list = model.encode_image(image, features_list, DPAM_layer = 20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 2. [新增] Mamba 增强 Patch Features
            # 推理阶段直接在这里调用，因为不需要梯度，所以放在 no_grad 里也没问题
            enhanced_patch_features = []
            for patch_feat in patch_features_list:
                if isinstance(patch_feat, torch.Tensor):
                    # 输入: [B, L, D]
                    feat_out = visual_adapter(patch_feat)
                    enhanced_patch_features.append(feat_out)
                else:
                    enhanced_patch_features.append(patch_feat)
            
            # 使用增强后的列表
            patch_features = enhanced_patch_features

            # ... 之后的逻辑保持不变 ...
            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                    # The following code is equivalent. 
                    # anomaly_map = similarity_map[...,1] 
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            
            anomaly_map = anomaly_map.sum(dim = 0)
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            # visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    for obj in obj_list:
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        table_ls.append(table)

    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                       ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default=".autodl-fs/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/epoch_15.pth', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
