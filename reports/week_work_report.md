# 车牌检测与识别项目周工作报告

## 一、本周工作目标

本周围绕“整车照片自动识别车牌号”这一目标，对原有 CRNN + CTC 车牌识别项目进行了扩展。原项目已经能够完成“裁切后的车牌图片 -> 车牌号码”的识别，但不能直接处理整车照片。本周新增 YOLO 车牌检测模块，将系统流程扩展为：

```text
整车照片 -> YOLO 检测车牌位置 -> 裁切车牌区域 -> CRNN 识别车牌号 -> 输出结果图
```

## 二、本周完成的主要工作

### 1. 整理 CCPD 数据并构建 YOLO 检测数据集

基于已有的 `data/ccpd/labels.csv`，新增了 YOLO 数据集准备脚本 `src/prepare_yolo_plate_dataset.py`。该脚本读取 CCPD 标签中的 `filename`、`split`、`bbox` 字段，将 CCPD 的车牌矩形框转换为 YOLO 所需格式：

```text
class_id x_center y_center width height
```

其中坐标均按照原图宽高归一化。脚本只使用 `train` 和 `val` 数据，不使用 `test` 数据，避免测试集泄漏到检测器训练中。

当前 CCPD 数据划分情况如下：

| 数据集 | 样本数 |
|---|---:|
| train | 246370 |
| val | 52827 |
| test | 52780 |

YOLO 数据集输出目录包括：

```text
data/yolo_plate/images/train
data/yolo_plate/images/val
data/yolo_plate/labels/train
data/yolo_plate/labels/val
data/yolo_plate/plate.yaml
```

为了节省磁盘空间，图片默认使用软链接方式组织；如果软链接失败，可以自动改为复制，也可以通过 `--copy-images` 强制复制图片。

### 2. 新增 YOLO 车牌检测与裁切模块

新增 `src/detect_plate_yolo.py`，用于加载训练好的 YOLO 检测器 `models/plate_detector.pt`，对整车照片进行车牌检测。检测时如果出现多个候选框，默认选择置信度最高的车牌框。

该模块会输出：

- 检测到的 bbox；
- 检测置信度；
- 裁切后的车牌图片；
- 带检测框的可视化图片。

检测可视化结果保存到：

```text
results/detections/
```

### 3. 串联 YOLO 检测与 CRNN 识别

完善 `src/detect_and_recognize.py`，实现完整推理流程：

```text
加载 YOLO 检测器
-> 检测整车图中的车牌 bbox
-> 裁切车牌区域
-> 加载已有 CRNN 识别模型
-> 复用 predict.py 中的 build_transform 和 predict_tensor
-> 输出预测车牌号
-> 在原图上绘制 bbox 和识别结果
```

该部分没有改动原有 `train.py`、`evaluate.py`、`predict.py`，因此不会破坏原有 CRNN 识别流程。

### 4. 完成自摄图片测试流程

本周已将多张自摄车辆图片放入 `data/self_photos/raw/`，并生成了对应的检测、裁切和识别结果。结果文件位于：

```text
results/detections/
```

其中包括：

- `*_detected.jpg`：仅包含 YOLO 检测框；
- `*_plate_crop.jpg`：YOLO 裁切出的车牌区域；
- `*_result.jpg`：原图上叠加检测框和 CRNN 识别结果。

## 三、模型训练与性能结果

### 1. CRNN 识别模型表现

CRNN + CTC 模型在 CCPD 数据集上训练 20 个 epoch。训练后期 loss 明显下降，验证集准确率趋于稳定。

![CRNN 训练曲线](../results/work_report_figures/crnn_training_summary.png)

最终验证集结果：

| 指标 | 数值 |
|---|---:|
| val loss | 0.0456 |
| val 字符准确率 | 99.00% |
| val 整牌准确率 | 94.79% |
| val normalized edit distance | 0.0100 |

在 CCPD test split 上重新统计结果：

| 指标 | 数值 |
|---|---:|
| 测试样本数 | 52780 |
| 字符准确率 | 98.94% |
| 整牌准确率 | 94.43% |
| 平均编辑距离 | 0.0743 |
| 空预测数量 | 0 |

这说明识别模型已经具备较好的单车牌图识别能力。对于多数裁切质量较好的车牌图，模型可以稳定输出完整车牌号。

### 2. YOLO 检测模型表现

本周进行了不同规模数据集上的 YOLO 检测器训练，包括 5k 和 20k 规模实验。主要检测器为 `plate_detector_20k-3`，训练 50 个 epoch，使用 `data/yolo_plate_20k/plate.yaml`。

![YOLO 训练曲线](../results/work_report_figures/yolo_training_summary.png)

主要 YOLO 结果如下：

| 训练配置 | epoch | precision | recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| plate_detector_5k | 10 | 99.88% | 100.00% | 99.49% | 74.95% |
| plate_detector_20k-3 | 48 最佳 | 99.88% | 99.89% | 99.46% | 79.34% |

从结果看，YOLO 对“是否能找到车牌”的能力已经较强，precision、recall 和 mAP50 都接近 99% 以上。mAP50-95 相对低一些，说明在更严格 IoU 阈值下，bbox 精细定位仍有提升空间。

### 3. 检测 + 识别整体效果

当前完整系统的主要性能可以概括如下：

![整体性能摘要](../results/work_report_figures/pipeline_performance_summary.png)

检测模块负责把整车照片变成车牌裁切图，识别模块负责将裁切图转成车牌号。两部分串联后，系统已经从“只能识别裁切车牌图”升级为“可以处理整车照片”。

## 四、当前系统流程

当前推荐运行流程如下：

```bash
cd /Users/nemo7/Documents/车牌号识别/license-plate-recognition-week3
```

生成 YOLO 数据集：

```bash
python src/prepare_yolo_plate_dataset.py \
  --labels data/ccpd/labels.csv \
  --output data/yolo_plate
```

训练 YOLO 检测器：

```bash
yolo detect train \
  data=data/yolo_plate/plate.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640
```

单独检测并裁切：

```bash
python src/detect_plate_yolo.py \
  --image data/self_photos/raw/car1.jpg \
  --detector models/plate_detector.pt \
  --output data/self_photos/cropped/car1_plate.jpg \
  --conf 0.25
```

完整检测 + 识别：

```bash
python src/detect_and_recognize.py \
  --image data/self_photos/raw/car1.jpg \
  --detector models/plate_detector.pt \
  --recognizer models/license_plate_crnn_best.pth \
  --output results/detections/car1_result.jpg \
  --conf 0.25 \
  --device auto
```

## 五、本周遇到的问题与解决方式

### 1. 原项目只能识别裁切车牌

原有 `predict.py` 只能处理裁切好的车牌图。对于整车照片，模型无法自动知道车牌在哪里。本周通过新增 YOLO 检测器解决了这个问题。

### 2. CCPD 标注格式与 YOLO 格式不同

CCPD 的 bbox 格式是：

```text
left&top_right&bottom
```

而 YOLO 需要：

```text
class_id x_center y_center width height
```

本周新增转换脚本，完成 bbox 解析、归一化和标签文件生成。

### 3. 大规模图片复制占用磁盘

完整 CCPD 图片数量较大，直接复制会占用大量空间。本周采用软链接方式构建 YOLO 数据集，在不复制原图的情况下完成训练数据组织。

### 4. 检测框精度仍有提升空间

YOLO 的 mAP50 很高，但 mAP50-95 仍低于 mAP50，说明检测器已经能稳定找到车牌，但框的位置精细程度还可以继续优化。

## 六、当前不足

1. YOLO 检测器主要基于 CCPD 数据训练，自摄照片的光照、角度、清晰度与 CCPD 仍可能存在差异。
2. 检测后只是矩形裁切，没有做透视矫正，倾斜车牌可能影响 CRNN 识别。
3. 当前系统默认选择置信度最高的一个车牌框，暂未支持一张图中多车牌识别。
4. 识别模型仍基于固定字符表，暂未覆盖新能源车牌等更复杂场景。
5. 目前主要评估了检测器和识别器的独立指标，端到端整车图准确率还需要更系统地统计。

## 七、下周计划

1. 增加端到端测试集，对“整车图 -> 最终车牌号”的准确率进行统计。
2. 对 YOLO 检测后的车牌区域增加透视矫正，提升倾斜车牌识别效果。
3. 支持多车牌检测与逐个识别，输出多条结果。
4. 对自摄照片建立小规模人工标注集，评估真实场景下的检测和识别准确率。
5. 整理错误案例，分析是检测框偏移、车牌模糊、裁切不完整还是 CRNN 字符混淆导致。

## 八、本周总结

本周完成了车牌识别项目从“裁切车牌识别”到“整车照片检测 + 识别”的关键升级。YOLO 检测模块已经可以从整车图中定位车牌，CRNN 识别模块继续负责车牌字符识别，两者串联后形成了完整的车牌识别流程。

从实验结果看，YOLO 检测器在验证集上 mAP50 达到 99.46%，CRNN 在 CCPD test split 上整牌准确率达到 94.43%、字符准确率达到 98.94%。当前系统已经具备较完整的工程流程，下一步重点是提升真实自摄场景下的端到端稳定性。
