from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from PIL import Image
import json
from datetime import datetime
import time

app = Flask(__name__)



# 加载类别映射json文件
with open('classes.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)  # 类别索引到名称映射

# 初始化模型和设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from train import AdvancedAlexNet, IMG_SIZE  # 确保train.py里有这个类和变量

model = AdvancedAlexNet(num_classes=len(class_names))
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 图像预处理，与训练test_transform保持一致
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route('/')  # 定义根路由，处理GET请求（访问网站首页）
def index():
    return render_template('index.html')  # 渲染并返回首页模板

# 定义预测路由，仅接受POST请求（用于接收图片上传）
@app.route('/predict', methods=['POST'])
def predict():
    # 上传目录，必须在static内部才可以直接通过url访问
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # 检查请求中是否包含文件参数（前端表单需设置enctype="multipart/form-data"）
    if 'file' not in request.files:
        return render_template('error.html', message="上传文件为空！")
    file = request.files['file']     # 获取上传的文件对象
    if file.filename == '': # 检查文件名是否为空（用户未选择文件直接提交的情况）
        return render_template('error.html', message="未选择文件！")
    filename = secure_filename(file.filename)    # 安全处理文件名（移除危险字符，防止路径注入攻击）
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) # 构建完整保存路径（需预先配置app.config['UPLOAD_FOLDER']）
    file.save(filepath)       # 保存文件到服务器指定目录
    print("图片保存路径：", filepath)

    try:     # 使用PIL加载图像文件并强制转换为RGB格式（处理灰度图/RGBA等特殊情况）
        image = Image.open(filepath).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(DEVICE) # 应用预处理转换流程 -> 添加批次维度 -> 转移到指定设备（GPU/CPU）

        with torch.no_grad():   # 禁用梯度计算（推理模式，节省显存并加速）
            outputs = model(img_tensor)       # 执行模型推理（前向传播）
            probs, indices = torch.topk(torch.softmax(outputs, dim=1), 3)   # 计算概率分布并取前3个预测结果（dim=1表示在类别维度操作）
            confidence_float = probs[0, 0].item() * 100     # 最高置信度转为百分比数值
            pred_class = indices[0, 0].item()           # 获取类别索引

            class_label = class_names[str(pred_class)]      # 映射类别名称
            confidence = confidence_float        # 保留原始精度用于后续计算

            top_preds = []       # 构建Top3预测结果列表（从第二项开始遍历
            for i in range(1, len(probs[0])):       # 遍历第2、3名预测结果
                idx = indices[0, i].item()       # 获取当前索引
                prob = probs[0, i].item() * 100     # 转换置信度百分比
                top_preds.append({
                    "class_name": class_names[str(idx)],
                    "confidence": f"{prob:.2f}"     # 格式化保留两位小数
                })

        # 识别完及时删除上传图片，节省空间
        # os.remove(filepath)

        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return render_template('result.html',
                               pred_class=class_label,
                               confidence=f"{confidence:.2f}%",
                               confidence_float=confidence,
                               filename=filename,
                               datetime=dt,
                               top_preds=top_preds)

    except Exception as e:
        return render_template('error.html', message=f"识别发生错误：{str(e)}")


# 这个路由其实不用，因为静态文件已经被flask自动映射，但保留也无妨：
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))


if __name__ == '__main__':
    app.run(debug=True)
