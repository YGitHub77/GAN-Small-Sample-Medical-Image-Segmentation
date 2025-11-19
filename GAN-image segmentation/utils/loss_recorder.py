import matplotlib.pyplot as plt

class TrainingPlotter:
    def __init__(self, save_dir="./"):
        self.train_losses_g = []
        self.train_losses_d = []
        self.train_dices = []
        self.val_dices = []
        self.val_ious = []
        self.val_recalls = []
        self.save_dir = save_dir

    def update(self, g_loss, d_loss, train_dice, val_dice, val_iou, val_recall):
        """每个epoch结束后调用，更新数据"""
        self.train_losses_g.append(g_loss)
        self.train_losses_d.append(d_loss)
        self.train_dices.append(train_dice)
        self.val_dices.append(val_dice)
        self.val_ious.append(val_iou)
        self.val_recalls.append(val_recall)

    def save_curves(self):
        """训练结束后调用，保存曲线图"""
        # 损失曲线
        plt.figure(figsize=(10,4))
        plt.plot(self.train_losses_g, label='Train G_loss')
        plt.plot(self.train_losses_d, label='Train D_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses')
        plt.savefig(f"{self.save_dir}/loss_curve.png")
        plt.close()

        # 指标曲线
        plt.figure(figsize=(10,4))
        plt.plot(self.train_dices, label='Train Dice')
        plt.plot(self.val_dices, label='Val Dice')
        plt.plot(self.val_ious, label='Val IoU')
        plt.plot(self.val_recalls, label='Val Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Training & Validation Metrics')
        plt.savefig(f"{self.save_dir}/metrics_curve.png")
        plt.close()
