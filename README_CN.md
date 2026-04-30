# Hybrid Vision — AMI 2026

**融合 RGB 帧与事件相机数据的无人机检测系统。**

> AMI 2026 课程项目 · FRED 数据集 · PyTorch · Docker

---

## 目录

1. [项目概述](#项目概述)
2. [数据集](#数据集)
3. [快速开始](#快速开始)
4. [Phase 1 — 混合检测模型](#phase-1--混合检测模型)
5. [Phase 2 — 事件重建](#phase-2--事件重建)
6. [Phase 3 — 前端界面](#phase-3--前端界面)
7. [项目结构](#项目结构)
8. [环境要求](#环境要求)

---

## 项目概述

传统 RGB 相机在运动模糊、高动态范围、低光照等复杂场景下性能急剧下降。事件相机以微秒级分辨率异步记录每个像素的亮度变化，有效弥补了上述不足。本项目探索**混合感知**：融合两种传感器的互补优势，实现更鲁棒的无人机检测。

三个阶段逐步构建一个完整的 Docker 化演示系统：

| 阶段 | 目标 | 方法 |
|------|------|------|
| Phase 1 | 从融合的 RGB + 事件数据中检测无人机 | 双流 CNN + 特征融合 |
| Phase 2 | 从原始事件数据重建强度帧 | E2VID · FireNet（直接用预训练权重） |
| Phase 3 | 交互式 Web 演示 | FastAPI 后端 · HTML/JS 前端 |

---

## 数据集

**FRED — Florence RGB-Event Drone Dataset**  
主页：<https://miccunifi.github.io/FRED/>

| 属性 | 内容 |
|------|------|
| 传感器 | RGB 相机 + DAVIS 事件相机（同步） |
| 内容 | 户外无人机飞行，多样背景 |
| 格式 | RGB 帧（JPEG）· 事件流（x, y, t, p）· YOLO 标注 |
| 划分 | 官方 train / test |

下载后的目录结构：

```
data/FRED/
├── train/
│   └── sequences/
│       ├── seq_001/
│       │   ├── rgb/          # 000001.jpg …
│       │   ├── events/       # events.npy  (x y t p)
│       │   └── labels/       # 000001.txt  (YOLO 格式)
│       └── …
└── test/
    └── sequences/
        └── …
```

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/Dark-Fantasy-K/AMI-KK.git
cd AMI-KK

# 将 FRED 数据放置到 data/FRED/（见上方目录结构）

# 一键启动全部服务（API + 前端）
docker-compose up --build
```

在浏览器中打开 <http://localhost>。

---

## Phase 1 — 混合检测模型

### 架构

```
RGB 帧  (3 通道)  ──►  RGB 编码器  ──►  RGB 特征   ─┐
                                                      ├─► 融合 ─► FPN ─► 检测头
事件帧 (1 通道)  ──►  事件编码器  ─►  事件特征   ─┘
```

支持两种融合模式（通过 `configs/config.yaml` 配置）：

| 模式 | 说明 |
|------|------|
| `early` | 将 RGB 与事件帧拼接为 4 通道张量，送入共享骨干网络 |
| `late` | 两路独立 ResNet18 编码器，在 FPN 各层级融合特征图 |

检测头采用 3 尺度 YOLO 风格（步长 /8、/16、/32）。  
输出：边界框 + 置信度 + 类别（`drone`）。

### 训练

```bash
# 下载 FRED，在 configs/config.yaml 中配置路径，然后：
python models/detection/hybrid/train.py \
    --config configs/config.yaml

# Checkpoint 保存路径见配置文件（默认：/data/checkpoints/）
```

### 核心文件

| 文件 | 作用 |
|------|------|
| [models/detection/hybrid/model.py](models/detection/hybrid/model.py) | `HybridDetector`：骨干网络、FPN、YOLO 检测头 |
| [preprocessing/dataset.py](preprocessing/dataset.py) | `FREDDataset`：加载 RGB + 事件 → 4 通道张量 |
| [preprocessing/event_processor.py](preprocessing/event_processor.py) | 事件流 → voxel grid / 事件帧 |
| [configs/config.yaml](configs/config.yaml) | 全部超参数 |

---

## Phase 2 — 事件重建

### 动机

Phase 2 回答另一个问题：*如果我们完全跳过融合，只用事件数据重建出普通的灰度帧，再送入标准 RGB 检测器，效果如何？*

这为评估事件相机的独立贡献建立了一条基线，无需任何专用融合架构。两种重建网络均以**纯推理模式**运行，不需要重新训练。

---

### 方法一 — E2VID

**论文**：Rebecq et al., *"Events-to-Video: Bringing Modern Computer Vision to Event Cameras"*, CVPR 2019 / TPAMI 2021  
**代码库**：<https://github.com/uzh-rpg/rpg_e2vid>

#### 网络结构

```
Voxel Grid         循环 UNet
(5 × H × W)  ──►  head（Conv）
                    │
                   enc[0]  ConvLSTM  stride=2  →  H/2 ,  64 ch
                   enc[1]  ConvLSTM  stride=2  →  H/4 , 128 ch
                   enc[2]  ConvLSTM  stride=2  →  H/8 , 256 ch
                    │
                  ResBlock × 2（瓶颈层）
                    │
                   dec[0]  ×2 上采样  →  H/4 , 128 ch  ⊕ 跳跃连接 enc[1]
                   dec[1]  ×2 上采样  →  H/2 ,  64 ch  ⊕ 跳跃连接 enc[0]
                   dec[2]  ×2 上采样  →  H   ,  32 ch  ⊕ 跳跃连接 head
                    │
                   pred（1×1 Conv）+ Sigmoid
                    │
               重建帧（H × W，float32）
```

**关键设计**：
- 输入为 **voxel grid**（5 个时间 bin，双线性插值），而非简单事件帧——保留了帧间时序结构。
- **循环隐藏状态**（ConvLSTM）在连续帧间传递上下文，对平滑重建慢速运动内容至关重要。
- 每个编码器层的**加法跳跃连接**防止步长卷积丢失空间信息。
- 我们的实现（[models/reconstruction/e2vid/model.py](models/reconstruction/e2vid/model.py)）与官方 rpg_e2vid 参数命名完全对齐，**官方预训练权重可直接加载**，无需任何 key 映射。

#### 使用方法

```bash
# 1. 下载预训练权重
bash models/reconstruction/e2vid/download_weights.sh /data/weights

# 2. 对全部 FRED test 序列进行重建
python models/reconstruction/e2vid/reconstruct.py \
    --fred_root /data/FRED \
    --weights   /data/weights/e2vid.pth \
    --output    /data/reconstructed/e2vid \
    --num_bins  5 \
    --device    cuda
```

输出路径：`/data/reconstructed/e2vid/seq_XXX/000001.png …`

---

### 方法二 — FireNet

**论文**：Scheerlinck et al., *"Fast Image Reconstruction with an Event Camera"*, WACV 2020  
**代码库**：<https://github.com/cedric-scheerlinck/rpg_e2vid>（firenet 分支）

#### 网络结构

```
Voxel Grid         FireNet
(5 × H × W)  ──►  head（Conv → BN → ReLU）
                    │
                   G1  ConvLSTM（16 ch，原始分辨率）
                   G2  ConvLSTM
                   G3  ConvLSTM
                    │
                   pred（1×1 Conv）+ Sigmoid
                    │
               重建帧（H × W，float32）
```

**与 E2VID 的对比**：

| 维度 | E2VID | FireNet |
|------|-------|---------|
| 架构 | UNet（编码器-解码器） | 线性链 |
| 参数量 | ~5 M | ~0.1 M |
| 下采样 | 3× stride-2 卷积 | 无 |
| 空间上下文 | 多尺度（FPN） | 单尺度 |
| 速度 | 较慢 | **更快** |
| 重建质量（典型） | 较高 | 适合快速运动 |

FireNet 以较低的重建质量换取更快的推理速度——适用于延迟比细节更重要的场景。

**Checkpoint 兼容说明**：`load_firenet()` 自动处理两种 checkpoint 格式：
- 分离的 `ih` / `hh` 卷积（标准 FireNet 格式）
- 合并的 `Gates` 卷积（部分变体）——加载时自动拆分

#### 使用方法

```bash
# 1. 下载预训练权重
bash models/reconstruction/method2/download_weights.sh /data/weights

# 2. 重建
python models/reconstruction/method2/reconstruct.py \
    --fred_root /data/FRED \
    --weights   /data/weights/firenet.pth \
    --output    /data/reconstructed/firenet \
    --device    cuda
```

---

### 评估

重建完成后，运行三路对比：

```bash
python evaluation/compare.py \
    --fred_root      /data/FRED \
    --e2vid_frames   /data/reconstructed/e2vid \
    --firenet_frames /data/reconstructed/firenet \
    --hybrid_checkpoint /data/checkpoints/best.pt   # 可选
```

示例输出：

```
==============================================================
  FRED Detection Comparison
==============================================================
Method                               mAP@0.5     mAP@0.5:0.95    Time (s)
-----------------------------------  ----------  --------------  ----------
E2VID → YOLOv8 (COCO)               0.XXXX      0.XXXX          XX.X
FireNet → YOLOv8 (COCO)             0.XXXX      0.XXXX          XX.X
Hybrid（RGB + Event，Phase 1）       0.XXXX      0.XXXX          XX.X
==============================================================
```

重建帧上的检测使用 **YOLOv8n**（COCO 预训练，自动下载）作为冻结的基线检测器。E2VID 与 FireNet 行的 mAP 差异纯粹反映了重建质量，不涉及任何检测器微调。

### Phase 2 文件导航

```
models/reconstruction/
├── utils.py                          # 公共工具：事件加载、voxel grid、帧保存
├── e2vid/
│   ├── model.py                      # E2VID 网络（checkpoint 兼容）
│   ├── reconstruct.py                # FRED test 重建脚本
│   └── download_weights.sh           # gdown 从 Google Drive 下载
└── method2/                          # FireNet
    ├── model.py                      # FireNet 网络
    ├── reconstruct.py                # 与 E2VID 接口相同
    └── download_weights.sh

evaluation/
├── detect_on_frames.py               # YOLOv8 推理 + mAP 计算
└── compare.py                        # 打印三路对比表
```

---

## Phase 3 — 前端界面

Phase 3 将完整 pipeline 封装为浏览器可访问的演示：
- 上传 RGB 帧和原始事件文件（`.npy` / `.h5`）
- 选择检测模式：*混合模式*（Phase 1 模型）或 *重建 + 检测*（Phase 2）
- 检测结果以边界框叠加的形式实时展示

### 服务组成

| 服务 | 镜像 | 端口 | 作用 |
|------|------|------|------|
| `api` | `webapp/backend/Dockerfile` | 8000 | FastAPI 推理服务 |
| `frontend` | `webapp/frontend/Dockerfile` | 80 | nginx 静态 HTML/JS |

```bash
docker-compose up --build
# 前端界面 → http://localhost
# API 文档  → http://localhost:8000/docs
```

---

## 项目结构

```
AMI-KK/
├── configs/
│   └── config.yaml                   # 全部超参数
├── data/                             # 挂载 FRED 数据
├── evaluation/
│   ├── compare.py                    # 三路 mAP 对比表
│   ├── detect_on_frames.py           # YOLOv8 + mAP（重建帧）
│   └── metrics.py                   # mAP 工具函数
├── models/
│   ├── detection/hybrid/
│   │   ├── model.py                  # HybridDetector
│   │   ├── train.py                  # 训练循环
│   │   └── predict.py               # 推理
│   └── reconstruction/
│       ├── utils.py                  # 公共事件/帧工具
│       ├── e2vid/
│       │   ├── model.py              # E2VID（Phase 2，方法一）
│       │   ├── reconstruct.py
│       │   └── download_weights.sh
│       └── method2/
│           ├── model.py              # FireNet（Phase 2，方法二）
│           ├── reconstruct.py
│           └── download_weights.sh
├── preprocessing/
│   ├── dataset.py                    # FREDDataset（PyTorch）
│   └── event_processor.py           # 事件流 → voxel grid / 事件帧
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   └── download_data.sh
├── webapp/
│   ├── backend/
│   │   ├── app.py                    # FastAPI
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── frontend/
│       ├── index.html
│       ├── nginx.conf
│       └── Dockerfile
└── docker-compose.yml
```

---

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+（可选，支持 CPU 回退） |
| Docker + Compose | 24+ |

**关键 Python 依赖**：

```
torch torchvision
ultralytics          # YOLOv8（Phase 2 检测基线）
fastapi uvicorn python-multipart
opencv-python-headless
numpy scipy h5py pyyaml
gdown                # checkpoint 下载
```

---

## 许可证

MIT — 见 `LICENSE`。

---

*AMI 2026 · Hybrid Vision Project*
