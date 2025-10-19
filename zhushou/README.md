# Green Energy Blockchain MVP

> 可信的绿色能源数据共享平台 - 结合区块链数据溯源与气候AI分析

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目概述

这是一个展示区块链数据溯源能力的MVP系统，专注于气候和绿色能源数据的可信共享。核心设计理念：

- **区块链溯源**: 在本地PoA以太坊链上记录数据指纹（哈希值+元数据）
- **IPFS存储**: 大文件存储在IPFS上，通过内容标识符(CID)访问
- **可验证性**: 任何人都可以通过CID下载数据并验证哈希值
- **气候AI**: 基于结构化指标和政策文本的轻量级机器学习分析

## 技术栈

- **区块链**: Hardhat (Node 18+), Solidity 0.8.20, ethers.js
- **存储**: IPFS (本地节点)
- **数据分析**: Python 3.10+, pandas, scikit-learn, matplotlib
- **智能合约**: ClimateDataRegistry.sol

## 快速开始

### 前置要求

```bash
# 必需软件
Node.js >= 18
Python >= 3.10
IPFS daemon (可选，用于完整流程)
```

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd zhushou

# 2. 安装依赖
npm install
pip install -r requirements.txt

# 3. 运行简化版（无需IPFS和区块链）
scripts\run_all.bat
```

### 完整流程（含区块链）

```bash
# 终端1: 启动Hardhat本地节点
npm run node

# 终端2: 部署合约并运行pipeline
npm run deploy
python etl/03_ipfs_add.py
python etl/04_register_onchain.py

# 验证数据
python scripts/verify_from_chain.py --sample 3
```

## 项目结构

```
├── contracts/              # Solidity智能合约
│   └── ClimateDataRegistry.sol
├── etl/                    # ETL数据处理pipeline
│   ├── 01_fetch.py        # 获取原始数据
│   ├── 02_clean_and_hash.py  # 清洗并计算哈希
│   ├── 03_ipfs_add.py     # 添加到IPFS
│   └── 04_register_onchain.py  # 区块链注册
├── analysis/               # 数据分析脚本
│   ├── 10_baseline_models.py    # 机器学习模型
│   ├── 11_policy_text_features.py  # 政策文本特征提取
│   └── 12_merge_and_plots.py    # 可视化和报告
├── scripts/
│   ├── deploy.js          # 合约部署脚本
│   ├── verify_from_chain.py  # 验证CLI
│   └── run_all.bat        # 一键运行脚本
├── data_sample/           # 示例数据
│   ├── indicators.csv     # 气候指标数据
│   └── policies/          # 政策PDF文档
├── docs/                  # 文档和报告
│   ├── report.html        # 分析报告
│   └── figures/           # 图表
└── test/                  # 测试文件
```

## 核心功能

### 1. 数据溯源

```python
# 每个文件都会被计算SHA-256哈希值
hash = sha256(file_content)

# 上传到IPFS获取CID
cid = ipfs.add(file)

# 在区块链上注册元数据
contract.addRecord(hash, source_url, license, cid)
```

### 2. 独立验证

```bash
# 任何人都可以验证数据完整性
python scripts/verify_from_chain.py

# 流程:
# 1. 从区块链读取 hash + CID
# 2. 从IPFS下载文件
# 3. 重新计算哈希值
# 4. 对比验证 ✓
```

### 3. 气候数据分析

- **预测模型**: ElasticNet + RandomForest 预测可再生能源占比
- **特征工程**: 滞后特征、年度变化、国家编码
- **政策分析**: 关键词提取、政策强度计算
- **可视化**: 趋势图、特征重要性、滞后相关分析

## 示例数据

项目包含3个国家（德国、印度、巴西）2019-2023年的演示数据：

- **指标**: 可再生能源占比、人均CO₂排放、装机容量
- **政策文档**: 3个示例政策PDF（演示用）

⚠️ **注意**: 这是演示数据，实际应用请使用真实来源的数据。

## 输出文件

- `output/artifacts_index.csv` - 所有文件的哈希和CID索引
- `output/tx_log.csv` - 区块链交易记录
- `output/model_results.json` - 模型性能指标
- `docs/report.html` - 完整分析报告（含图表）
- `contracts/contract-address.json` - 部署的合约地址

## 独立验证指南

任何人都可以验证数据来源和完整性：

**需要的信息**:
- 合约地址 (从 `contracts/contract-address.json`)
- IPFS CID (从区块链记录中读取)
- 原始文件的哈希值 (从区块链记录中读取)

**验证步骤**:
```bash
# 1. 读取区块链记录
python scripts/verify_from_chain.py --sample 5

# 2. 系统会自动:
#    - 连接区块链获取元数据
#    - 从IPFS下载文件
#    - 重新计算哈希
#    - 对比验证

# 3. 查看验证结果表格
```

## 运行测试

```bash
# 运行基础测试
pytest test/test_basic.py -v

# 预期输出: 所有测试通过 ✓
```

## 许可证

- **代码**: MIT License
- **数据**: 请参考 `data_sample/README.md` 中的数据来源许可
- **依赖**: 各开源库保留各自许可证

详见 `NOTICE` 文件。

## 设计原则

1. **本地优先**: 无需云服务，完全本地运行
2. **可重现性**: 固定随机种子，确定性哈希
3. **透明性**: 所有操作有日志和交易记录
4. **简单性**: MVP规模，保持代码清晰
5. **可验证性**: 任何人都能独立验证数据

## 注意事项

⚠️ **这是一个演示MVP项目，不适用于生产环境**

- 使用本地测试网络，非真实区块链
- 演示数据，非真实气候数据
- 分析结果仅显示相关性，不表示因果关系
- 需要进一步的安全审计才能用于生产

## 常见问题

**Q: IPFS连接失败怎么办？**
A: 简化版本会自动回退到本地验证，不影响核心功能展示。

**Q: Hardhat节点启动失败？**
A: 确保端口8545未被占用，或修改`hardhat.config.js`中的端口配置。

**Q: 如何添加自己的数据？**
A: 将数据放入`data_sample/`目录，运行ETL pipeline即可。

## 进一步阅读

- `docs/README_onepager.md` - 可重现性检查清单
- `data_sample/README.md` - 数据来源和许可
- `CHANGELOG.md` - 开发历程

## 贡献

欢迎提Issue和PR！

---

**Built with ❤️ for transparent climate data**

