import matplotlib.pyplot as plt

epochs = list(range(1, 11))
train_loss = [1.7940, 1.2665, 1.0439, 0.9362, 0.8107, 0.7445, 0.6463, 0.5813, 0.5341, 0.4653]
val_loss = [1.4065, 1.0805, 0.9552, 0.9323, 0.8271, 0.8576, 0.8014, 0.7432, 0.7199, 0.6817]
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'o-', label='训练损失', color='blue')
plt.plot(epochs, val_loss, 's-', label='验证损失', color='orange')
plt.title('训练损失与验证损失随Epoch变化曲线')
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('损失值')
plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.png')  # 保存图像
plt.show()
