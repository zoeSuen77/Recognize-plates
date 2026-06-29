# 中国车牌号识别项目

本项目已经从临时合成数据流程切换为真实 CCPD 数据集流程。临时 `/private/tmp/week3_synthetic_ccpd` 合成图片只适合 debug 代码流程，不用于正式训练、验证或评估。

当前模型目标是：

```text
裁剪后的中国车牌图片 -> 完整车牌号码
```

CCPD 原图文件名自带车牌位置和字符编码。本项目默认不复制大批量图片，而是在 `data/ccpd/labels.csv` 中记录原图路径、车牌号、数据集划分和裁剪坐标；训练加载时会按文件名坐标在线裁剪车牌区域。

## 目录结构

```text
license-plate-recognition-week3/
├── data/
│   ├── ccpd/
│   │   ├── raw/              # 用户自行下载并放入的原始 CCPD 图片
│   │   ├── train/            # 预留目录，不复制图片时可为空
│   │   ├── val/              # 预留目录，不复制图片时可为空
│   │   ├── test/             # 预留目录，不复制图片时可为空
│   │   └── labels.csv        # prepare_ccpd.py 生成的标签文件
│   └── self_photos/
│       ├── raw/              # 自己拍摄的真实车牌照片
│       ├── cropped/          # 裁剪后的车牌区域，可选
│       └── labels.csv        # 自己拍摄图片的真实车牌号标签，可选
├── src/
│   ├── prepare_ccpd.py       # 解析 CCPD 文件名并生成标签
│   ├── load_data.py          # 加载真实 CCPD 数据集
│   ├── train.py              # 训练模型
│   ├── evaluate.py           # 评估模型
│   ├── predict.py            # 单张图片或 CCPD test 图片预测
│   ├── utils.py
│   ├── plate_chars.py
│   ├── model_crnn.py
│   └── metrics.py
├── models/
├── results/
└── README.md
```

## 数据准备

请自行下载 CCPD 数据集。由于完整 `CCPD2019.tar.xz` 约 12.26GB，下载和解压都不适合放在本机小磁盘上。本项目推荐把完整数据放到外接硬盘：

```text
/Volumes/ksid/ccpd_data/
```

项目内路径保持为：

```text
data/ccpd/raw/
```

当前推荐通过软链接连接两者：

```bash
mkdir -p /Volumes/ksid/ccpd_data
mkdir -p data/ccpd
ln -s /Volumes/ksid/ccpd_data data/ccpd/raw
```

这样代码仍然读取 `data/ccpd/raw/`，但真实数据实际存储在外接硬盘。可以是扁平目录，也可以保留 CCPD 子目录结构，脚本会递归遍历图片。如果外接硬盘没有连接，训练和评估会找不到数据。

OpenXLab 下载完整 CCPD：

```bash
conda activate digit
pip install -U openxlab
openxlab login
openxlab dataset info --dataset-repo OpenDataLab/CCPD
openxlab dataset download \
  --dataset-repo OpenDataLab/CCPD \
  --source-path /raw/CCPD2019.tar.xz \
  --target-path /Volumes/ksid/ccpd_data
```

解压到外接硬盘：

```bash
tar -xJf /Volumes/ksid/ccpd_data/CCPD2019.tar.xz -C /Volumes/ksid/ccpd_data/
```

OpenDataLab 的 `/sample/` 数据只用于流程验证，不用于正式训练。

生成标签文件：

```bash
python src/prepare_ccpd.py --raw-dir data/ccpd/raw --output data/ccpd/labels.csv
```

快速小样本检查真实 CCPD：

```bash
python src/prepare_ccpd.py --raw-dir data/ccpd/raw --output data/ccpd/labels.csv --max-samples 1000
python src/load_data.py --labels data/ccpd/labels.csv --split train --batch-size 4
```

`labels.csv` 至少包含：

```csv
filename,plate_number,split
data/ccpd/raw/xxx.jpg,皖A12345,train
data/ccpd/raw/yyy.jpg,沪B88888,val
```

实际文件还会包含 `bbox,points`，用于训练时在线裁剪车牌区域。

## 训练

```bash
python src/train.py
```

常用参数：

```bash
python src/train.py --labels data/ccpd/labels.csv --epochs 20 --batch-size 32 --learning-rate 0.001
```

如果 `data/ccpd/labels.csv` 不存在，训练脚本会停止并提示先运行：

```bash
python src/prepare_ccpd.py --raw-dir data/ccpd/raw --output data/ccpd/labels.csv
```

不会自动退回合成数据，避免误以为真实训练已经成功。

输出：

```text
models/license_plate_crnn_best.pth
results/ccpd_train_history.csv
results/ccpd_train_history.json
results/ccpd_loss_curve.png
results/ccpd_accuracy_curve.png
```

## 评估

```bash
python src/evaluate.py
```

评估使用 `test` split，输出字符级准确率、整牌准确率、平均编辑距离、归一化编辑距离和空预测数量。

预测明细保存到：

```text
results/ccpd_predictions.csv
```

格式：

```csv
filename,true_plate,pred_plate,is_correct
xxx.jpg,皖A12345,皖A12345,1
yyy.jpg,沪B88888,沪B8888,0
```

## 预测

从 CCPD test split 随机预测一张：

```bash
python src/predict.py
```

预测自己拍摄或自己裁剪的图片：

```bash
python src/predict.py --image data/self_photos/raw/car001.jpg
```

注意：当前识别模型只负责“车牌区域 -> 车牌号码”。如果 `data/self_photos/raw/car001.jpg` 是整车图片，需要先检测或手动裁剪出车牌区域，再进行识别。可以把裁剪后的车牌放到：

```text
data/self_photos/cropped/
```

再运行：

```bash
python src/predict.py --image data/self_photos/cropped/car001_plate.jpg
```

## YOLO 车牌检测

当前 CRNN 识别模型仍然只负责：

```text
裁剪后的车牌图片 -> 完整车牌号码
```

如果要处理整车照片，需要先训练或提供一个 YOLO 车牌检测器。YOLO 依赖不在基础环境里时，请安装：

```bash
pip install ultralytics
```

从 CCPD 的 `labels.csv` 生成 YOLO 检测数据集：

```bash
python src/prepare_yolo_plate_dataset.py \
  --labels data/ccpd/labels.csv \
  --output data/yolo_plate
```

小样本调试：

```bash
python src/prepare_yolo_plate_dataset.py \
  --labels data/ccpd/labels.csv \
  --output data/yolo_plate \
  --limit 1000
```

默认会尽量给图片创建软链接，节省磁盘空间。如果当前文件系统不支持软链接，会自动复制图片。也可以强制复制：

```bash
python src/prepare_yolo_plate_dataset.py --copy-images
```

生成后的结构：

```text
data/yolo_plate/
├── images/train
├── images/val
├── labels/train
├── labels/val
└── plate.yaml
```

训练 YOLO 检测器示例：

```bash
yolo detect train data=data/yolo_plate/plate.yaml model=yolov8n.pt epochs=50 imgsz=640
```

训练完成后，把最佳权重放到：

```text
models/plate_detector.pt
```

单独检测并裁切车牌：

```bash
python src/detect_plate_yolo.py \
  --image data/self_photos/raw/car001.jpg \
  --detector models/plate_detector.pt \
  --output data/self_photos/cropped/car001_plate.jpg \
  --conf 0.25
```

整车图检测 + CRNN 识别：

```bash
python src/detect_and_recognize.py \
  --image data/self_photos/raw/car001.jpg \
  --detector models/plate_detector.pt \
  --recognizer models/license_plate_crnn_best.pth \
  --output results/detections/car001_result.jpg \
  --conf 0.25 \
  --device auto
```

检测框和识别结果图会保存到：

```text
results/detections/
```

## 推荐运行顺序

```bash
cd /Users/nemo7/Documents/车牌号识别/license-plate-recognition-week3
conda activate digit
python src/prepare_ccpd.py --raw-dir data/ccpd/raw --output data/ccpd/labels.csv
python src/train.py
python src/evaluate.py
python src/predict.py --image data/self_photos/cropped/car001_plate.jpg
```

---

## 纯 YOLO 车牌字符检测方案（YOLO + YOLO）

除了已有的 **YOLO 检测车牌 → CRNN 识别字符** 方案外，本项目支持另一种纯 YOLO 方案：

```
整车图片 → YOLO 检测车牌 → 裁剪车牌 → YOLO 检测每个字符 → 按 x_center 排序 → 拼接输出完整车牌号
```

### 新增文件说明

| 文件 | 作用 |
|---|---|
| `src/prepare_yolo_char_dataset.py` | 从 CCPD labels.csv 生成 YOLO 字符检测数据集（按字符数均分车牌宽度生成伪字符框） |
| `src/train_yolo_char.py` | 训练 YOLOv8n 字符检测模型 |
| `src/recognize_plate_yolo.py` | 整车照 → YOLO 检测车牌 → 裁剪 → YOLO 检测字符 → 排序 → 输出 |
| `src/evaluate_yolo_char.py` | 在 CCPD test split 上评估 YOLO 字符识别准确率 |

### 运行步骤

```bash
# 1. 生成字符检测数据集（完整 CCPD 约 35 万张，外接盘需要插着）
/opt/anaconda3/envs/digit/bin/python src/prepare_yolo_char_dataset.py \
  --labels data/ccpd/labels.csv \
  --output data/yolo_chars

# 小样本调试（--limit 限制 train+val 总数，test 不受限）
/opt/anaconda3/envs/digit/bin/python src/prepare_yolo_char_dataset.py \
  --labels data/ccpd/labels.csv \
  --output data/yolo_chars \
  --limit 5000

# 2. 首次训练字符检测模型（M2 Mac 建议 imgsz=160）
/opt/anaconda3/envs/digit/bin/python src/train_yolo_char.py \
  --data data/yolo_chars/chars.yaml \
  --model yolov8n.pt \
  --epochs 20 \
  --imgsz 160 \
  --batch 8 \
  --device cpu \
  --save-period 1 \
  --exist-ok

# 2b. 中断后继续训练（自动恢复 last.pt）
/opt/anaconda3/envs/digit/bin/python src/train_yolo_char.py \
  --resume

# 2c. 从指定 checkpoint 恢复
/opt/anaconda3/envs/digit/bin/python src/train_yolo_char.py \
  --resume \
  --checkpoint runs/detect/yolo_chars/weights/last.pt

# 3. 单张整车图识别
/opt/anaconda3/envs/digit/bin/python src/recognize_plate_yolo.py \
  --image data/self_photos/raw/皖E.66290.jpg \
  --plate-detector models/plate_detector.pt \
  --char-detector models/char_detector.pt

# 4. 评估 YOLO 字符识别在 CCPD test 上的准确率
/opt/anaconda3/envs/digit/bin/python src/evaluate_yolo_char.py \
  --labels data/ccpd/labels.csv \
  --char-detector models/char_detector.pt \
  --conf 0.25 \
  --use-plate-rules

# 5. 分析 YOLO 字符识别错误类型
/opt/anaconda3/envs/digit/bin/python src/analyze_yolo_char_errors.py \
  --predictions results/yolo_char_predictions.csv
```

### 断点续训说明

`train_yolo_char.py` 支持完整的中断恢复机制：

| 功能 | 参数 |
|---|---|
| 中断时 | 自动捕获 Ctrl+C，提示恢复命令，保留 `last.pt` |
| 恢复训练 | `--resume` 默认从 `runs/detect/yolo_chars/weights/last.pt` 恢复 |
| 指定 checkpoint | `--resume --checkpoint <路径>` |
| 检查 checkpoint 存在 | `ls runs/detect/yolo_chars/weights/last.pt` |

训练完成后 `best.pt` 自动复制到 `models/char_detector.pt`。

### 完整训练参数说明（train_yolo_char.py）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data` | `data/yolo_chars/chars.yaml` | 数据集配置 |
| `--model` | `yolov8n.pt` | 预训练模型（首次训练用） |
| `--epochs` | `50` | 训练轮数 |
| `--imgsz` | `640` | 输入图片尺寸 |
| `--batch` | `8` | 批次大小 |
| `--device` | `cpu` | 训练设备 |
| `--project` | `runs/detect` | YOLO 项目目录 |
| `--name` | `yolo_chars` | 运行名称 |
| `--lr0` | `0.01` | 初始学习率 |
| `--patience` | `0` | 早停（0=禁用） |
| `--save-period` | `1` | 每 N 轮保存一次 checkpoint |
| `--exist-ok` | `False` | 允许覆盖已有 run 目录 |
| `--resume` | `False` | 从 checkpoint 恢复训练 |
| `--checkpoint` | `runs/detect/yolo_chars/weights/last.pt` | 指定 checkpoint 路径（配合 --resume） |

### M2 Mac（16GB）配置说明

| 项目 | 建议值 | 原因 |
|---|---|---|
| `--imgsz` | `160` | 裁剪后的车牌图通常只有 200~400 像素宽，160 足够，CPU 训练快 4~8 倍 |
| `--batch` | `8` | 16GB 内存 + CPU 训练的安全批次 |
| `--device` | `cpu` | M2 Mac 的 MPS 后端对 YOLOv8 支持不稳定，建议 CPU |
| `--epochs` | `20` | 伪字符框训练不如真实 bbox 收敛快，20 epoch 兼顾质量与时间 |
| 全量训练（246k 样本） | 约 30 分钟/epoch（imgsz=160） | 20 epoch ≈ 10 小时 |
| 小样本（5k 样本） | 约 40 秒/epoch（imgsz=160） | 20 epoch ≈ 13 分钟 |

### YOLO 字符检测重新训练建议

当前 YOLO 字符检测数据来自伪字符框：`prepare_yolo_char_dataset.py` 会按整牌宽度均分每个字符框。这个标签不是真实字符边界，遇到倾斜、字符宽度差异、车牌裁剪偏移时，容易造成字符漏检、多检或相邻字符混淆。建议保留原始 `runs/detect/yolo_chars`，用新的 run name 做对比实验。

实验 1：`yolov8n + imgsz224 + epochs80`

```bash
python src/train_yolo_char.py \
  --data data/yolo_chars/chars.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 224 \
  --batch 8 \
  --save-period 1 \
  --exist-ok \
  --name yolo_chars_n224_e80
```

实验 2：`yolov8s + imgsz224 + epochs80`

```bash
python src/train_yolo_char.py \
  --data data/yolo_chars/chars.yaml \
  --model yolov8s.pt \
  --epochs 80 \
  --imgsz 224 \
  --batch 4 \
  --save-period 1 \
  --exist-ok \
  --name yolo_chars_s224_e80
```

评估命令：

```bash
python src/evaluate_yolo_char.py \
  --labels data/ccpd/labels.csv \
  --plate-model models/plate_detector.pt \
  --char-model runs/detect/yolo_chars_n224_e80/weights/best.pt \
  --split test \
  --conf 0.15 \
  --use-plate-rules

python src/evaluate_yolo_char.py \
  --labels data/ccpd/labels.csv \
  --plate-model models/plate_detector.pt \
  --char-model runs/detect/yolo_chars_s224_e80/weights/best.pt \
  --split test \
  --conf 0.15 \
  --use-plate-rules
```

对比后处理规则时，可以追加 `--no-use-plate-rules`，并保存或重命名 `results/yolo_char_predictions.csv`、`results/yolo_char_error_summary.txt` 后再跑下一组实验。

### 两种方案对比

| 维度 | YOLOv8 + CRNN | YOLOv8 + YOLOv8 字符检测 |
|---|---|---|
| **识别方式** | CRNN（CNN+BiLSTM+CTC）端到端识别 | YOLO 检测 + 字符框排序拼接 |
| **训练数据集** | 需要字符级别的序列标注（CTCLoss 隐式对齐） | 需要字符级 bbox 标注（伪字符框可替代） |
| **模型数量** | 2 个（YOLO 检测器 + CRNN 识别器） | 2 个（YOLO 车牌检测器 + YOLO 字符检测器） |
| **字符集** | 全覆盖（省份+字母+数字，65类 + blank） | 全覆盖（省份+字母+数字，65类 YOLO） |
| **字符顺序** | CTC 时序解码自动处理 | 依赖检测框 x_center 排序 |
| **抗倾斜/形变** | CRNN 的 CNN+RNN 对形变有容忍度 | 依赖检测框准确性，均分伪框需矫正 |
| **端到端能力** | 强，一步输出 | 弱，依赖检测+排序后处理 |
| **可解释性** | 低，CTC 解码不易调试 | 高，每个字符的检测框和置信度可见 |
| **泛化到非 CCPD** | CRNN 对非 皖A 车牌泛化不足 | 字符检测对单字符泛化可能更好（当前未验证） |

### 已知限制

1. **伪字符框**：`prepare_yolo_char_dataset.py` 按车牌宽度均分生成字符框，不精确对应真实字符边界。需要 YOLO 在训练中自行学习更精确的定位。
2. **字符类数量**：65 个 YOLO 类别（31 省份 + 24 字母 + 10 数字）较多，需要足够数据量。
3. **车牌图片尺寸**：裁剪后的车牌图较小（通常 200-300 像素宽），YOLO 训练时会 resize 到 imgsz。
4. **数据依赖**：生成训练数据需要外接盘（读取原始 CCPD 图片），但训练和推理不依赖外接盘。

## 预测为空的常见原因

- 训练轮数太少，CTC 模型还没有学到有效对齐。
- 训练数据太少，或只用了 `--max-samples` 小样本。
- 输入是整车图片而不是裁剪后的车牌区域。
- CCPD 文件名解析错误，标签和图片不匹配。
- 图像尺寸、颜色模式或字符表与训练 checkpoint 不一致。
- 模型保存的是早期 debug 合成数据训练出的权重，需要用真实 CCPD 重新训练。
# Recognize-plates
