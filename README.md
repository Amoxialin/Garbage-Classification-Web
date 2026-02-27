# ♻️ End-to-End Garbage Classification Web System (端到端智能垃圾分类识别系统)

## 📖 项目简介
本项目是一个集深度学习模型训练、轻量级后端推理与前端交互于一体的城市垃圾分类 AI Web 应用。系统支持对 12 类常见垃圾（如电池、玻璃、纸板等）进行高精度自动化识别，并提供完整的端到端闭环体验。

## ✨ 核心工程亮点
* **轻量级推理后端：** 基于 `Flask` 框架搭建推理服务，设计 RESTful 风格接口，高效处理前端传递的图像预处理及 Top-3 预测结果回传，确保系统高可用性。
* **模型重构与防过拟合：** 基于 `PyTorch` 框架构建改进版 AlexNet 架构，引入 Dropout (0.5) 机制与 AdamW 优化器。通过这些策略有效抑制了模型过拟合，在验证集上实现了 79% 的整体准确率。
* **工业级数据增强流水线：** 针对真实场景下光照与拍摄角度的复杂性，构建了包含色彩抖动 (Color Jitter)、随机透视变换 (RandomPerspective) 及随机灰度化等策略的自动化数据增强流水线，大幅提升了模型的泛化能力。

## 🛠️ 技术栈
* **深度学习：** Python 3.x, PyTorch, Torchvision
* **后端工程开发：** Flask, RESTful API
* **前端交互：** HTML5, CSS3

## 🚀 快速开始 (Quick Start)
1. 克隆本项目到本地环境：
   `git clone https://github.com/你的GitHub用户名/Garbage-Classification-Web.git`
2. 启动 Flask 后端服务：
   `python app.py`
3. 使用浏览器访问 `http://127.0.0.1:5000` 即可体验。

> **⚠️ 提示：** 由于 GitHub 存储限制，本仓库暂不包含训练好的 `.pth` 模型权重文件。如需完整体验推理流程，请先运行 `train.py` 自行训练，或联系作者获取预训练权重。