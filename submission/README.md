# OmniEEG Challenge 2026 复赛 · 全部十四项任务

每个公开任务对应一个分类器和一个 checkpoint。模型共用 CBraMod 公开预训练权重。推理根据任务代号选择模型和对应的标签顺序，输入为单个 EEG 窗口及其通道掩码。一个窗口的预测独立于同批的其他窗口。对采用双分支的任务，两个分支在同一窗口上给出分数，开发集估计的固定系数用于合并分数。每项任务的两个分支随同一个 checkpoint 加载。

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

模型选择依据开发被试交叉验证的平衡准确率。所有折使用统一轮数，以折均值减去 0.15 倍折间标准差选模型。另留独立被试检验泛化。在所有选择锁定后，三次独立拟合用 431、1907、8849 三个训练种子评估稳定性。要复现这一步，在命令中加入 `--partition development --seed 431`，并在生成的 `cache/<task>/splits.json` 指定的预留被试上评测。最终 checkpoint 统一使用种子 431，并在该任务全部公开训练窗口上重新拟合。

## 数据与外部权重

下游训练使用比赛公开发布的十四项数据。决赛 OpenNeuro 语料没有用于复赛训练。训练过程使用 CBraMod 官方公开参数和源代码。所有任务适配参数从该公开基座或随机初始化的任务头开始训练。

- 公开模型代码采用 MIT 许可。固定版本为 `b9e961003214326972c567eff390e75b0287e32a`，包内附有 `cbra/LICENSE`。适配时只调整了打包所需的相对导入。[作者代码](https://github.com/wjq-learning/CBraMod/tree/b9e961003214326972c567eff390e75b0287e32a)
- 公开权重采用 Apache-2.0 许可。固定版本为 `500543c7e30bda1b22bfd51a49301b238dee21fd`。包内 `assets/cbramod-pretrained.pth` 的 SHA-256 为 `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`。[作者公开权重](https://huggingface.co/weighting666/CBraMod/tree/500543c7e30bda1b22bfd51a49301b238dee21fd)
- 复赛数据和评测接口以赛事发布版本为准。[复赛材料说明](https://github.com/xiaobri/OminiEEG/blob/main/%E5%A4%8D%E8%B5%9B%E6%9D%90%E6%96%99%E8%AF%B4%E6%98%8E.md)

在组委会指定的训练或复现环境中，可按上述命令将各任务输出写到独立目录，并按相应任务名称放入 `checkpoints`。替换任何文件后，按当次官方评测包的示例核对 manifest 字段，在该包目录执行 `refresh_submission_hashes.py` 和 `validate_submission.py`，最终 ZIP 解压后须再次运行预检。系统镜像、队伍资格与实际上传状态以赛事平台为准。

## 每个任务的最终配置

| 任务 | 模型 | 训练轮数或树迭代数 |
|---|---|---:|
| disease_01 | fusion | 见 v3 配置 |
| disease_02 | neural | 6 |
| emotion_01 | fusion | 见 v3 配置 |
| emotion_02 | fusion | 见 v3 配置 |
| motor_01 | fusion | 见 v3 配置 |
| motor_02 | fusion | 见 v3 配置 |
| reliability_01 | fusion | 见 v3 配置 |
| reliability_02 | spectrum | 6 |
| sleep_01 | fusion | 见 v3 配置 |
| sleep_02 | fusion | 见 v3 配置 |
| ssvep_01 | fusion | 见 v3 配置 |
| ssvep_02 | fusion | 见 v3 配置 |
| stimulus_01 | spectrum | 7 |
| stimulus_02 | montage | 13 |

解析求解的分类头以 1 表示一次拟合。

- **cbra_mlp**　CBraMod 第 4、12 层的池化表征与单窗频谱特征接入一个两层分类头。
- **delay**　学习延时嵌入下的空间判别滤波器，提取时域及谐波匹配特征，与 CBraMod 表征共用一个分类头。
- **geometry**　用源训练类协方差拟合白化与空间投影，对滤波器组对数协方差与 CBraMod 表征拟合一个岭分类头。
- **montage**　由全部输入通道学习空间投影，编码重叠的时间片，微调公共输入编码器和最后四个 Transformer 模块。
- **neural**　从原始波形学习时域卷积和分频带空间投影，与冻结的 CBraMod 表征一起输入一个任务头。
- **source**　在延时和谐波特征上加入按源被试拟合的谐波响应子空间，与 CBraMod 表征共用一个分类头。
- **spectrum**　短时频谱及双侧电极功率差经卷积，与冻结的 CBraMod 表征一起输入一个任务头。
- **motor_aligned**　源训练被试等权估计通道中心和收缩协方差。训练时固定空间滤波参数，微调 CBraMod 输入层和十二层瓶颈适配器。评测时使用 checkpoint 中的固定矩阵和通道中心，每窗独立推理。

motor_02 采用后续锁定的筛选协议。在原开发池的 6 名被试上逐人留出，以种子 431 和 1907 分别比较两种配置，每次训练 16 轮。两个种子均有 6 人改善。原留出集的 2 人在升级方案锁定后用于评估。

## v3 最终推理与训练配置

下表列出升级任务的两条分支、各自的训练轮数和第二条分支的固定系数。解析求解使用 1 表示一次拟合。

| 任务 ID | 第一分支（轮数） | 第二分支（轮数） | 系数 |
|---|---|---|---:|
| disease_01 | geometry（1） | montage（21） | 0.28797409 |
| emotion_01 | montage（22） | full（17） | 1.0174635 |
| emotion_02 | cbra_mlp（3） | neural（11） | 1.317925 |
| motor_01 | montage（16） | neural（18） | 0.75233914 |
| motor_02 | motor_aligned（16） | geometry（1） | 6.1304987 |
| reliability_01 | montage（24） | cbra_mlp（11） | 0.35273684 |
| sleep_01 | spectrum（22） | neural（21） | 1.1189885 |
| sleep_02 | spectrum（25） | cbra_linear（5） | 0.49600996 |
| ssvep_01 | source（3） | canonical（4） | 0.94215423 |
| ssvep_02 | delay（4） | canonical（2） | 0.95975037 |

双分支的训练入口会按配置分别从公开训练数据拟合两个分支，然后写入带任务 ID 的单个 checkpoint。训练动作及训练轮数详见 training/train_config.yaml。canonical 使用源训练刺激频率以及诱发平均波形生成参考子空间，在每个输入窗口内计算典型相关。

## v4.1 推理

emotion_01 的两条分支使用 float32，以每批 8 个窗口计算。尾批补齐后只返回原始输入的预测。其余任务沿用各自配置。
