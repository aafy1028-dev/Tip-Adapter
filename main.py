# Tip-Adapter 主程序：few-shot 图像分类
# 支持 CLIP、SigLIP2、SigLIP2+DINOv3 三种编码器

import os          # 用于路径操作、检查文件是否存在
import random      # 用于设置随机种子，保证可复现性
import argparse    # 解析命令行参数
import yaml        # 解析 YAML 配置文件
from tqdm import tqdm  # 进度条显示

import torch       # PyTorch 深度学习框架
import torch.nn.functional as F  # 常用函数如 cross_entropy
import torch.nn as nn            # 神经网络模块，如 Linear
import torchvision.transforms as transforms  # 图像变换（Resize、Crop、Normalize 等）

from datasets import build_dataset        # 根据配置构建数据集
from datasets.utils import build_data_loader  # 构建 DataLoader
from utils import *  # cls_acc, clip_classifier, dual_encoder_classifier, build_cache_model, pre_load_features, search_hp
import pdb

def get_arguments():
    """
    解析命令行参数。
    Returns:
        args: 解析后的参数对象，包含 config 等属性
    """
    parser = argparse.ArgumentParser()  # 创建参数解析器
    # dest='config' 表示解析结果存到 args.config
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()  # 解析命令行，如 --config configs/food101.yaml
    return args


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, text_weights):
    """
    运行 Tip-Adapter（无训练版本）：在 val 上搜索超参，在 test 上评估。

    Args:
        cfg: 配置字典，包含 init_beta、init_alpha、search_scale、search_step 等
        cache_keys: 形状 (C, NK)，few-shot 样本特征矩阵，C=特征维度，NK=样本数
        cache_values: 形状 (NK, N)，few-shot 样本的 one-hot 标签矩阵
        val_features: 形状 (N_val, C)，验证集图像特征
        val_labels: 形状 (N_val,)，验证集真实标签
        test_features: 形状 (N_test, C)，测试集图像特征
        test_labels: 形状 (N_test,)，测试集真实标签
        text_weights: 形状 (C, N)，文本分类器权重，N=类别数
    """
    print("\n-------- Searching hyperparameters on the val set. --------")
    # pdb.set_trace()
    # Zero-shot：仅用文本权重分类，公式 logits = 100 * features @ text_weights
    zero_shot_logits = 100. * val_features @ text_weights
    acc = cls_acc(zero_shot_logits, val_labels)  # 计算 top-1 准确率
    print("\n**** Zero-shot val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter：affinity = 特征与 cache 的相似度，cache_logits 通过 soft retrieval 得到
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    affinity = val_features @ cache_keys  # (N_val, NK)，与每个 few-shot 样本的相似度
    # exp(-beta + beta*affinity) 将相似度转为权重，再与 cache_values 相乘得到各类 logits
    phi_affinity = ((-1) * (beta - beta * affinity)).exp()
    cache_logits = phi_affinity @ cache_values
    tip_logits = zero_shot_logits + cache_logits * alpha  # 融合 zero-shot 与 cache
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # 在 val 上搜索最优 beta、alpha
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, text_weights)

    print("\n-------- Evaluating on the test set. --------")
    # 用最优超参在 test 上评估
    zero_shot_logits = 100. * test_features @ text_weights
    acc = cls_acc(zero_shot_logits, test_labels)
    print("\n**** Zero-shot test accuracy: {:.2f}. ****\n".format(acc))

    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    tip_logits = zero_shot_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

def main():
    """
    主函数：加载配置、构建数据集与编码器、构建 cache、提取特征、运行 Tip-Adapter 与 Tip-Adapter-F。
    """
    args = get_arguments()  # 解析命令行
    assert os.path.exists(args.config)  # 检查配置文件是否存在

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)  # 加载 YAML 配置
    # 为未显式指定的配置项设置默认值
    cfg.setdefault('use_dual_encoder', False)   # 是否使用 SigLIP2+DINOv3 双编码器
    cfg.setdefault('use_siglip_only', False)   # 是否仅使用 SigLIP2
    cfg.setdefault('siglip_model', 'google/siglip2-base-patch16-224')
    cfg.setdefault('dinov3_model', 'XMFY111/dinov3-vits16')

    # 根据 encoder 类型设置 cache 子目录，避免不同 encoder 的 cache 互相覆盖
    cache_dir = os.path.join('./caches', cfg['dataset'])
    if cfg['use_dual_encoder']:
        cache_dir = cache_dir + '_dual'
    elif cfg['use_siglip_only']:
        cache_dir = cache_dir + '_siglip'
    os.makedirs(cache_dir, exist_ok=True)  # 确保 cache 目录存在
    cfg['cache_dir'] = cache_dir  # 写入配置供后续使用

    print("\nRunning configs.")
    print(cfg, "\n")

    use_siglip = cfg['use_dual_encoder'] or cfg['use_siglip_only']  # 是否使用 SigLIP 系列
    random.seed(1)       # 固定 Python 随机种子
    torch.manual_seed(1) # 固定 PyTorch 随机种子

    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])  # 构建 few-shot 数据集

    if use_siglip:
        from models.dual_encoder import DualEncoder
        # SigLIP-only 时 dinov3_model=None，双编码器时使用配置中的 dinov3 模型
        dinov3_model = None if cfg['use_siglip_only'] else cfg.get('dinov3_model', 'MFY111/dinov3-vits16')
        encoder = DualEncoder(
            siglip_model_name=cfg['siglip_model'],
            dinov3_model_name=dinov3_model,
        ).cuda().eval()
        # pdb.set_trace()
        # SigLIP2 使用 [0,1] 图像，不做 CLIP 风格的 normalize
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    else:
        import clip
        encoder, preprocess = clip.load(cfg['backbone'])  # 加载 CLIP
        encoder.eval()
        # CLIP 需要特定 mean/std 归一化
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    # 验证集 DataLoader：无 shuffle，用于超参搜索
    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    # 测试集 DataLoader
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    # 训练集（用于构建 cache）：shuffle=False，保证 augment 时可复现
    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_transform, is_train=True, shuffle=False)

    print("\nGetting textual features as classifier.")
    if use_siglip:
        text_weights = dual_encoder_classifier(dataset.classnames, dataset.template, encoder)
    else:
        text_weights = clip_classifier(dataset.classnames, dataset.template, encoder)

    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, encoder, train_loader_cache)

    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", encoder, val_loader)

    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", encoder, test_loader)
    # pdb.set_trace()
    run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, text_weights)


if __name__ == '__main__':
    main()
