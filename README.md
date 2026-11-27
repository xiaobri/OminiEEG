# OmniEEG Challenge 2026 复赛

[下载 v5.1 完整模型包（104,189,930 字节）](https://github.com/xiaobri/OminiEEG/releases/download/v5.1/OmniEEG-semi-14.zip)

v5.1 是本地验证候选版本。它为七个分赛道的 14 个任务各提供一个模型。每个模型使用公开 CBraMod 权重，对每个 EEG 窗口独立计算一组类别分数。Release 附件包含训练和推理代码、14 个任务 checkpoint、公共权重及完整性清单。

模型包的 SHA-256 为 `49a889fbc1cc5d5c21823963053a8302f25eb138dcd6c38c5a688504edfaac0f`。

## 本地验证

在相同留出被试上，14 项平均平衡准确率由单模型基线的 **62.42%** 提高到 **62.97%**。SSVEP 的短窗任务从 **35.99%** 提高到 **38.83%**，长窗任务从 **89.17%** 提高到 **93.93%**。每项报告三个训练种子的平均成绩。历史留出成绩参与过版本选择。官方按七个分赛道的名次累计积分。[逐任务成绩与复训结果](validation-v5.1.md)

## 在比赛环境重建权重

赛事手册要求复赛使用主办方共享 GPU 池。此附件的开发设备为 RTX A6000，其官方资源池归属尚待确认。在官方实例解压附件，进入 `submission/` 目录执行以下命令，将路径替换为该实例上的公开下游数据目录、工作目录和新 ZIP 的位置。[官方参赛手册](https://challenge.omni-intel.cn/handbook.html)

```bash
python training/rebuild_all.py \
  --downstream-dir /path/to/downstream \
  --work-dir /path/to/work \
  --output /path/to/OmniEEG-official.zip
```

脚本按任务配置重训全部权重，更新 manifest 的 SHA-256，并在工作目录记录训练日志和权重哈希。正式提交时使用对应共享算力作业生成的包和平台任务 ID。

- [模型接口、环境及外部权重来源](submission/README.md)
- [任务 ID、checkpoint 与哈希清单](submission/manifest.json)
- [训练配置](submission/training/train_config.yaml)
