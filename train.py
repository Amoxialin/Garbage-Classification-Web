import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import torch.multiprocessing as mp

# 中文字体配置（Windows系统）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑更适配您的深色界面
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ---------------------- 系统级配置 ----------------------
mp.freeze_support()  # Windows多进程必须调用
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 路径配置 ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本目录
DATA_ROOT = os.path.join(BASE_DIR, "garbage_classification")  # 规范路径
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")  # 模型保存路径

# ---------------------- 模型参数配置 ----------------------
NUM_CLASSES = 12  # 对应12个分类文件夹
BATCH_SIZE = 64
EPOCHS = 10
IMG_SIZE = 227  # AlexNet输入尺寸

# ---------------------- 数据预处理 ----------------------

train_transform = transforms.Compose([  # 训练集数据增强变换组合
    transforms.Resize(IMG_SIZE + 50),  # 调整图像尺寸（为后续裁剪留出空间）
    transforms.RandomRotation(30),  # 随机旋转（-30度到+30度范围）
    transforms.RandomPerspective(0.2),  # 随机透视变换（20%的失真强度）
    transforms.RandomCrop(IMG_SIZE),  # 随机裁剪到目标尺寸（最终输出尺寸）
    transforms.RandomHorizontalFlip(),  # 50%概率水平翻转（常规增强）
    transforms.RandomVerticalFlip(),  # 50%概率垂直翻转
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),  # 颜色抖动（亮度/对比度/饱和度各30%调整幅度，色调20%调整）
    transforms.RandomGrayscale(p=0.1),  # 10%概率转为灰度图（增强颜色鲁棒性）
    transforms.ToTensor(),  # 将PIL图像转为Tensor格式（HWC -> CHW）并归一化到[0,1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化（ImageNet数据集统计值）
])
test_transform = transforms.Compose([  # 测试集/验证集数据预处理
    transforms.Resize(IMG_SIZE),  # 直接缩放到目标尺寸
    transforms.CenterCrop(IMG_SIZE),  # 中心裁剪保证输入一致性（无随机性）
    transforms.ToTensor(),  # 格式转换
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 与训练集相同的标准化参数
])


# ---------------------- 增强型数据集加载 ----------------------
class SafeImageFolder(datasets.ImageFolder):
    """带损坏文件过滤的数据集加载"""

    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except:
                print(f"⚠️ 跳过损坏文件: {self.samples[index][0]}")
                index = (index + 1) % len(self.samples)


def create_loaders():
    # 完整数据集加载
    full_dataset = SafeImageFolder(DATA_ROOT, transform=train_transform)

    # 分层抽样
    labels = [label for _, label in full_dataset]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(sss_val.split(np.zeros(len(train_val_idx)), [labels[i] for i in train_val_idx]))

    # 创建子数据集
    train_dataset = Subset(full_dataset, [train_val_idx[i] for i in train_idx])
    val_dataset = Subset(full_dataset, [train_val_idx[i] for i in val_idx])
    test_dataset = Subset(full_dataset, test_idx)
    test_dataset.dataset.transform = test_transform

    # Windows系统优化加载配置
    num_workers = 0 if os.name == 'nt' else 4
    return (
        DataLoader(train_dataset, BATCH_SIZE, True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_dataset, BATCH_SIZE, num_workers=num_workers, pin_memory=True),
        DataLoader(test_dataset, BATCH_SIZE, num_workers=num_workers, pin_memory=True),
        full_dataset.classes
    )


# ---------------------- 改进型AlexNet模型 ----------------------
class AdvancedAlexNet(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()  # 特征提取网络（卷积部分）
        self.features = nn.Sequential(  # 首个卷积块：大感受野捕捉基础特征
            nn.Conv2d(3, 96, 11, 4),  # 输入通道3，输出96，11x11核，步长4
            nn.ReLU(inplace=True),  # 原址激活节省内存
            nn.LocalResponseNorm(2),  # 通道间归一化，增强抑制效果
            nn.MaxPool2d(3, 2),  # 3x3池化，步长2（缩减特征图尺寸）
            # 第二卷积块：中等感受野特征组合
            nn.Conv2d(96, 256, 5, padding=2),  # 保持特征图尺寸不变
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(2),
            nn.MaxPool2d(3, 2),
            # 深层特征抽象块（连续3个3x3卷积）
            nn.Conv2d(256, 384, 3, padding=1),  # 小卷积核组合等效大感受野
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),  # 通道数加倍提升表达能力
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),  # 最后压缩通道数
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 最终池化得到高维抽象特征
        )
        self.classifier = nn.Sequential(  # 分类网络（全连接部分）
            nn.Dropout(0.5),  # 强正则化防止过拟合
            nn.Linear(256 * 6 * 6, 4096),  # 展平后的特征维度计算需匹配输入尺寸
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 再次正则化
            nn.Linear(4096, 4096),  # 双4096维隐藏层保持强大拟合能力
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # 输出层适配指定类别数
        )

    def forward(self, x):  # 特征提取流程
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平特征张量（保留batch维度）
        x = self.classifier(x)  # 分类决策流程
        return x


# ---------------------- 训练流程 ----------------------
def main():
    # 初始化组件
    train_loader, val_loader, test_loader, class_names = create_loaders()
    model = AdvancedAlexNet(NUM_CLASSES).to(DEVICE)

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # 学习率为0.0001
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    # 训练循环
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = correct / len(val_loader.dataset)
        scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"🏆 保存最佳模型 | 准确率: {val_acc:.2%}")

        print(f"Epoch {epoch + 1:02d} | "
              f"训练损失: {train_loss / len(train_loader):.4f} | "
              f"验证损失: {val_loss / len(val_loader):.4f} | "
              f"验证准确率: {val_acc:.2%}")

    # 保存模型时添加结构哈希
    import hashlib
    struct_hash = hashlib.md5(str(model.state_dict().keys()).encode()).hexdigest()[:6]
    torch.save(model.state_dict(), f"model_v1_{struct_hash}.pth")

    # 最终评估
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            all_preds.extend(model(images).argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    # 生成报告
    print("\n📊 分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(all_labels, all_preds),
                annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("混淆矩阵")
    plt.savefig("confusion_matrix.png")


if __name__ == '__main__':
    main()
