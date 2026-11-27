# OmniEEG Challenge 2026 复赛 · 全部十四项任务

每个公开任务对应一个分类器和一个 checkpoint。使用 CBraMod 的任务共用同一份公开预训练权重。推理根据任务代号选择模型和对应的标签顺序，输入为单个 EEG 窗口及其通道掩码。一个窗口的预测独立于同批的其他窗口。开发阶段比较的其他候选和训练种子不参与该窗口的预测。

## 离线推理

在提交包根目录加载 `predict.py`。评测接口调用 `load_model(model_dir, task_info, device)`，其中 `model_dir` 可以是含 `model_config.json` 的根目录或其 `checkpoints` 子目录。`predict(model, batch)` 返回 CPU float32 张量，形状为批大小乘类别数，各列依照 `task_info["label_values"]`。输入以发布版通道顺序和原始采样率传入。通道掩码缺省时，全部输入通道视为有效。模型仅依赖包内文件以及 Python、PyTorch、NumPy。

所有拟合后的特征变换参数、空间滤波器、模板和分类权重均在训练后固定。各模型仅接收信号和通道有效性。用于划分和重加权的公开被试标识只在训练端使用。

## 可复现训练

运行环境为 Linux、Python 3.10、PyTorch 2.4.0+cu121、NumPy 2.2.6，训练另用 SciPy 1.15.3、scikit-learn 1.7.2、pylance 14.0.3、pyarrow 25.0.1，树模型适配用 LightGBM 4.6.0。端到端适配器在支持 BF16 的 CUDA 设备上训练。本地开发设备为 RTX A6000。官方评测材料标注公开评测包 1.0.0-rc8。评测镜像的 tag 或 digest 尚待以主办方分发文件核对。

在包含复赛下游数据的环境中，从解压后的提交包目录执行以下命令。将示例路径换为自己的发布目录和可写本地目录。可用不同 GPU 并行运行不同任务。

```bash
CUDA_VISIBLE_DEVICES=0 python training/train.py \
  --task motor_01 \
  --downstream-dir /path/to/downstream \
  --work-dir /path/to/work \
  --output /path/to/retrained/motor_01.pt
```

`downstream` 下应有 `public-release/public_alias_20261118_001` 以及 `public_release_metadata/public_alias_20261118_001/public_train_window_metadata`。读取程序按 Lance 行顺序解码信号，逐条与官方窗口元数据核对标签、样本 ID、被试 ID。重跑同一任务使用相同数据版本、环境与空的输出文件路径。首次运行会缓存数组和确定性特征。运行十四个任务时依次将 `--task` 和输出文件名替换成 `model_config.json` 的十四个任务名。种子和每个任务锁定的算法、训练轮数列在 `training/train_config.yaml`。不同设备或软件版本可能造成浮点舍入差异。

各任务在按被试隔离的开发折上确定训练轮数。在原留出被试上分别用三个种子评估，并用另外三次开发被试划分复核候选。历史留出成绩参与了最终取舍。交付权重使用种子 431 在各任务的全部公开训练窗口上拟合。用 `--partition development --seed 431` 可单独复训开发集模型，训练范围记录于对应的 splits.json。

## 数据与外部权重

下游训练使用比赛公开发布的十四项数据。决赛 OpenNeuro 语料没有用于复赛训练。训练过程使用 CBraMod 官方公开参数和源代码。使用基座的任务以公开参数初始化。SSVEP 的模板与空间投影由本任务的公开训练被试拟合，公开 CBraMod 对当前信号的编码也进入同一个分类器。

- 公开模型代码采用 MIT 许可。固定版本为 `b9e961003214326972c567eff390e75b0287e32a`，包内附有 `cbra/LICENSE`。适配时只调整了打包所需的相对导入。[作者代码](https://github.com/wjq-learning/CBraMod/tree/b9e961003214326972c567eff390e75b0287e32a)
- 公开权重采用 Apache-2.0 许可。固定版本为 `500543c7e30bda1b22bfd51a49301b238dee21fd`。包内 `assets/cbramod-pretrained.pth` 的 SHA-256 为 `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`。[作者公开权重](https://huggingface.co/weighting666/CBraMod/tree/500543c7e30bda1b22bfd51a49301b238dee21fd)
- 复赛数据和评测接口以赛事发布版本为准。[复赛材料说明](https://github.com/xiaobri/OminiEEG/blob/main/%E5%A4%8D%E8%B5%9B%E6%9D%90%E6%96%99%E8%AF%B4%E6%98%8E.md)

在组委会指定的训练或复现环境中，可按上述命令将各任务输出写到独立目录，并按相应任务名称放入 `checkpoints`。替换任何文件后，按当次官方评测包的示例核对 manifest 字段，在该包目录执行 `refresh_submission_hashes.py` 和 `validate_submission.py`，最终 ZIP 解压后须再次运行预检。系统镜像、队伍资格与实际上传状态以赛事平台为准。

## 各任务提交配置

| 任务 ID | 单模型架构 | 训练轮数 |
|---|---|---:|
| disease_01 | geometry | 1 |
| disease_02 | neural | 6 |
| emotion_01 | montage | 22 |
| emotion_02 | neural | 11 |
| motor_01 | montage | 16 |
| motor_02 | motor_aligned | 16 |
| reliability_01 | montage | 24 |
| reliability_02 | spectrum | 6 |
| sleep_01 | spectrum | 22 |
| sleep_02 | spectrum | 25 |
| ssvep_01 | public_latency | 3 |
| ssvep_02 | public_target | 11 |
| stimulus_01 | spectrum | 7 |
| stimulus_02 | montage | 13 |

SSVEP 任务从源被试拟合响应模板、空间投影及谐波参照。训练分类器时按被试交叉拟合这些特征，结合冻结的公开 CBraMod 编码后，用一个联合训练的分类头对各刺激类别评分。每个测试窗口独立推理。
public_target 在源被试间按刺激类别估计空间投影，并把投影后的波形相关性作为同一个分类头的输入特征。
延迟校正使用训练被试估计相对响应延迟，部署时每个类别的模板固定。

对原始波形做端到端微调的模型采用 float32 和固定大小的内部批次。末批补齐后只返回真实输入行的输出。

## 在主办方共享 GPU 池重建整包

解压完整附件后，在 submission 目录执行以下命令。程序使用配置中固定的种子和训练轮数重新训练 14 项任务，将新权重和重新计算的 manifest 写入一个新 ZIP。工作目录保留每项任务的训练日志和权重哈希，重试时可以续用此前完成的任务。正式提交的算力任务 ID 以共享平台作业记录为准。

```bash
python training/rebuild_all.py --downstream-dir /path/to/downstream --work-dir /path/to/work --output /path/to/OmniEEG-official.zip
```

拿到官方评测包后，请按其说明对新 ZIP 解压出的 submission 运行 validate_submission.py。
