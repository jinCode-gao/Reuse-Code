import cv2
import numpy as np
import torch
import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')

# 加载预训练的模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# 设置模型为评估模式
model.eval()
# 将模型移动到GPU上，如果有的话
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# 读取bmp格式的图像
image = cv2.imread("/media/x303-07/新加卷/data/打标的dcm数据/ROI网络数据/image-ROI网络小数据集测试-增强图/020-ZSSYX-01580-CYZH-202209130909-双波段-R-D.bmp")

# 将图像从BGR格式转换为RGB格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image=cv2.resize(image,(128,102))
# # 将图像从numpy数组转换为torch.Tensor，并且缩放到0-1范围
image = torch.from_numpy(image).float() #torch
# # 将图像从HWC格式转换为CHW格式
image = image.permute(2, 0, 1)


# 禁用梯度计算
with torch.no_grad():
    # 将图像传入模型中，得到输出字典，包含'boxes'和'labels'两个键
    output = model([image])[0]

# 获取预测的框和标签
boxes = output["boxes"]
labels = output["labels"]

# 定义一个颜色列表用于不同类别的框
colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

boxes = torch.tensor(boxes, dtype=torch.int)
image = torch.tensor(image, dtype=torch.uint8)

# 在图像上绘制矩形框
image_boxes = torchvision.utils.draw_bounding_boxes(image, boxes)
image_boxes = image_boxes.numpy()
image_boxes=image_boxes.transpose(1,2,0)


cv2.imshow("Image with boxes", image_boxes) # ndarray(102,128,3)
cv2.waitKey(0)
cv2.destroyAllWindows()
