# YOLO 字符识别优化方案工作总结

生成日期：2026-07-06

## 一、当前问题概述

当前项目中，YOLO 字符识别方案已经可以完成以下流程：

```text
整车图片
-> YOLO 检测车牌位置
-> 裁剪车牌区域
-> YOLO 检测单个字符
-> 按字符框 x_center 排序
-> 拼接输出车牌号
```

但从现有实现看，YOLO 字符识别效果仍弱于 CRNN + CTC 整牌识别。主要原因不是 YOLO 模型完全不可用，而是字符检测数据和后处理规则还比较粗糙。

目前影响效果的核心问题有三点：

1. 车牌裁剪主要依赖矩形 bbox，未充分利用 CCPD 中的四点标注，倾斜车牌没有做透视矫正。
2. 字符框标签不是人工标注，而是将裁剪车牌宽度平均切分得到的伪标签。
3. 后处理虽然已经加入基础车牌规则，但还没有系统区分普通蓝牌、新能源车牌和特殊车牌类型。

因此，本阶段优化重点应放在“数据生成质量”和“车牌规则后处理”两方面。

## 二、优化方向一：四点标注与透视矫正

### 1. 当前实现

当前字符检测数据集由 `src/prepare_yolo_char_dataset.py` 生成。脚本读取 `data/ccpd/labels.csv`，使用 `bbox` 和 `points` 字段裁剪车牌区域：

```python
plate_img = crop_from_ccpd_fields(original, row.get("bbox", ""), row.get("points", ""))
```

虽然数据中包含 `points` 四点信息，但当前 YOLO 字符检测训练仍主要基于裁剪后的车牌矩形图。对于倾斜、旋转、拍摄角度较大的车牌，矩形裁剪会保留明显透视变形，导致字符框位置不稳定。

### 2. 修改建议

建议新增车牌几何处理模块：

```text
src/plate_geometry.py
```

该文件建议实现以下函数：

```python
parse_ccpd_points(points_text)
order_plate_points(points)
warp_plate_perspective(image, points)
```

其中：

- `parse_ccpd_points()` 负责解析 CCPD 标签中的四点字符串。
- `order_plate_points()` 将四点统一排序为左上、右上、右下、左下。
- `warp_plate_perspective()` 使用四点透视变换，将倾斜车牌矫正为规则矩形。

然后在 `src/prepare_yolo_char_dataset.py` 中优先使用四点透视矫正。如果 `points` 不可用，再回退到原来的 bbox 裁剪。

推荐流程：

```text
读取原图
-> 解析 points
-> 四点排序
-> 透视矫正为正面车牌图
-> 保存到 YOLO 字符训练集
```

### 3. 预期收益

透视矫正后，字符在图像中的排列会更接近水平直线，字符高度和宽度也会更稳定。即使仍然使用伪字符框，等分标签也会比原来更接近真实字符位置。

这一步是三项优化中优先级最高的一项。

## 三、优化方向二：字符框伪标签改进

### 1. 当前实现

当前 `src/prepare_yolo_char_dataset.py` 中的字符框生成方式是按车牌宽度平均切分：

```python
char_w = pw / n
for i, ch in enumerate(plate_number):
    x_center = (i + 0.5) * char_w
    y_center = ph / 2.0
    box_w = char_w
    box_h = ph
```

这种方法实现简单，但问题明显：

- 每个字符被认为宽度完全相同；
- 字符框高度直接等于整张车牌高度；
- 没有考虑左右边距；
- 没有考虑省份简称、字母、数字宽度差异；
- 对新能源 8 位车牌和特殊车牌不够友好。

### 2. 修改建议

建议在 `src/prepare_yolo_char_dataset.py` 中将“等分字符框”拆成独立函数，例如：

```python
generate_pseudo_char_boxes(plate_number, plate_width, plate_height, plate_type)
```

第一阶段可以仍然使用规则伪标签，但要比平均切分更精细：

- 增加左右边距，避免字符框贴到车牌边界；
- 字符框高度不要覆盖整张车牌，可以只覆盖中间字符区域；
- 普通 7 位车牌和新能源 8 位车牌使用不同布局；
- 省份简称位置可以略宽，后续字符位置保持稳定间距；
- 对明显不符合车牌规则的样本跳过或记录。

第二阶段可以加入图像处理辅助：

```text
透视矫正车牌图
-> 灰度化
-> 二值化或边缘检测
-> 垂直投影
-> 找字符候选区域
-> 结合车牌字符串顺序生成伪框
```

这一步可以逐步实现，不需要一次完成。建议先完成透视矫正后的规则伪框，再考虑垂直投影。

### 3. 预期收益

字符框标签质量提升后，YOLO 学到的定位能力会更稳定，推理时多检、漏检和框位置漂移都会减少。相比单纯调训练参数，改进伪标签通常收益更大。

## 四、优化方向三：车牌类型规则

### 1. 当前实现

当前 `src/recognize_plate_yolo.py` 中已经有基础位置规则：

```python
def _allowed_at_position(char: str, position: int) -> bool:
    from plate_chars import DIGITS, LETTERS, PROVINCES

    if position == 0:
        return char in PROVINCES
    if position == 1:
        return char in LETTERS
    return char in LETTERS or char in DIGITS
```

该规则可以处理大多数普通车牌，但还不够细：

- 没有明确区分 7 位普通车牌和 8 位新能源车牌；
- 没有处理警、学、挂、港、澳等特殊字符；
- 没有把车牌类型用于候选框数量筛选；
- 没有在候选组合评分中体现不同车牌类型的优先级。

### 2. 修改建议

建议新增车牌规则模块：

```text
src/plate_rules.py
```

该文件建议实现：

```python
infer_plate_type(plate_text)
expected_plate_lengths(plate_type)
allowed_chars_for_position(plate_type, position)
score_plate_candidate(candidate, plate_type)
```

车牌类型可以先支持以下几类：

| 类型 | 长度 | 说明 |
|---|---:|---|
| 普通车牌 | 7 | 常见蓝牌、小型汽车牌 |
| 新能源车牌 | 8 | 新能源小型车、大型车 |
| 特殊车牌 | 7 或 8 | 警、学、挂、港、澳等 |

然后在 `src/recognize_plate_yolo.py` 中替换现有 `_allowed_at_position()`，将候选字符组合评分逻辑改为：

```text
候选字符框
-> 按位置排序
-> 尝试不同车牌类型
-> 检查长度是否符合
-> 检查每一位字符是否合法
-> 综合 YOLO 置信度、字符间距、车牌规则评分
-> 选择最高分结果
```

### 3. 预期收益

车牌类型规则可以减少明显不合法的输出，尤其是在 YOLO 字符检测出现多检或误检时，可以帮助后处理选择更合理的字符组合。

## 五、训练脚本调整建议

`src/train_yolo_char.py` 当前主要负责调用 Ultralytics YOLO 训练接口，整体不需要大改。建议只做轻量增强：

1. 增加训练增强参数，例如 HSV、平移、缩放、轻微旋转。
2. 保持 `imgsz` 不宜过大，字符检测图像是裁剪车牌，小图训练更关注局部细节。
3. 使用新的数据集输出目录，例如：

```text
data/yolo_chars_warped
```

对应训练命令可以是：

```bash
python src/prepare_yolo_char_dataset.py \
  --labels data/ccpd/labels.csv \
  --output data/yolo_chars_warped

python src/train_yolo_char.py \
  --data data/yolo_chars_warped/chars.yaml \
  --name yolo_chars_warped
```

## 六、推荐修改文件清单

| 文件 | 修改内容 | 优先级 |
|---|---|---:|
| `src/prepare_yolo_char_dataset.py` | 接入四点透视矫正、改进字符框伪标签、增加数据增强输出 | 高 |
| `src/plate_geometry.py` | 新增四点解析、排序、透视变换等几何工具函数 | 高 |
| `src/plate_rules.py` | 新增车牌类型判断、位置合法性和候选评分规则 | 中 |
| `src/recognize_plate_yolo.py` | 接入车牌类型规则，优化候选字符框筛选和排序 | 中 |
| `src/train_yolo_char.py` | 轻量增加训练增强参数，支持新数据目录训练 | 低 |
| `src/evaluate_yolo_char.py` | 增加按车牌类型统计准确率，便于对比优化效果 | 中 |

## 七、推荐实施顺序

### 第一步：先做透视矫正

新增 `src/plate_geometry.py`，并在 `src/prepare_yolo_char_dataset.py` 中优先使用四点透视矫正生成车牌图。

这是最值得优先做的部分，因为它直接改善训练输入质量。

### 第二步：改进伪字符框

在透视矫正后的车牌图上，替换原来的简单等分逻辑，加入左右边距、字符高度比例和不同车牌长度布局。

这一阶段仍然不需要人工字符框标注，工程成本较低。

### 第三步：加入车牌类型规则

新增 `src/plate_rules.py`，将普通车牌、新能源车牌和特殊车牌规则接入 `src/recognize_plate_yolo.py` 的候选框后处理。

这一步主要提升推理阶段的容错能力。

### 第四步：重新训练与评估

使用新生成的数据集重新训练 YOLO 字符检测器，并在 `src/evaluate_yolo_char.py` 中对比：

- 字符准确率；
- 整牌准确率；
- 普通车牌和新能源车牌分类型准确率；
- 多检、漏检、错检样本数量。

## 八、总结

本次 YOLO 字符识别优化的核心不是单纯更换模型，而是提高字符检测数据质量和后处理规则质量。

当前最关键的问题是字符框标签由平均切分生成，且裁剪车牌没有充分利用四点透视矫正。建议优先完成“透视矫正 + 更合理伪字符框”，再加入车牌类型规则。这样可以在不依赖人工字符框标注的前提下，明显提升 YOLO 字符检测方案的稳定性。

最终目标是让纯 YOLO 字符识别方案从“可运行的对照实验”提升为“可解释、可视化、具备进一步优化空间的字符级识别方案”。在实际系统中，短期仍建议保留 YOLO 车牌检测 + CRNN 整牌识别作为主流程，YOLO 字符识别作为辅助分析和后续增强方向。
