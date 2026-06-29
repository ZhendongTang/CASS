# ClinTime-XML 初版结果

## 完全配对的100例结果

HistGBDT、Flat、PCES、ClinTime-XML均评估于每库相同的固定哈希100例。

| 数据集 | 系统 | AUROC | AUPRC | Brier | ECE |
|---|---|---:|---:|---:|---:|
| MIMIC | HistGBDT | 0.8331 | 0.5186 | 0.0967 | 0.0674 |
| MIMIC | Flat | 0.6275 | 0.3670 | 0.2269 | 0.2815 |
| MIMIC | PCES×3 | 0.6341 | 0.3281 | 0.2145 | 0.2766 |
| MIMIC | ClinTime-XML×3 | 0.6819 | 0.3554 | 0.2185 | 0.2953 |
| MIMIC | Canonical XML×1 | 0.6466 | 0.3161 | 0.2420 | 0.3175 |
| eICU | HistGBDT | 0.7894 | 0.5389 | 0.0544 | 0.0482 |
| eICU | Flat | 0.7147 | 0.3068 | 0.1142 | 0.1917 |
| eICU | PCES×3 | 0.7575 | 0.3751 | 0.1057 | 0.1814 |
| eICU | ClinTime-XML×3 | 0.6793 | 0.3836 | 0.1530 | 0.2677 |
| eICU | Canonical XML×1 | 0.6270 | 0.3199 | 0.1622 | 0.2714 |
| NWICU | HistGBDT | 0.8563 | 0.4892 | 0.0932 | 0.0701 |
| NWICU | Flat | 0.7157 | 0.3680 | 0.1282 | 0.1509 |
| NWICU | PCES×3 | 0.7193 | 0.3530 | 0.1263 | 0.1698 |
| NWICU | ClinTime-XML×3 | 0.8152 | 0.4612 | 0.1219 | 0.2071 |
| NWICU | Canonical XML×1 | 0.7555 | 0.3792 | 0.1333 | 0.1797 |

ClinTime-XML×3相对Flat的AUROC在MIMIC提高0.0544、NWICU提高0.0995，但eICU下降0.0353；AUPRC在eICU和NWICU提高，在MIMIC下降。完全配对的HistGBDT仍在三个数据库全面领先，说明XML缩小了部分差距但不能替代有监督学习。

## 顺序稳定性

| 数据集 | 系统 | 概率标准差 | 平均最大差 | 类标翻转率 |
|---|---|---:|---:|---:|
| MIMIC | PCES / ClinTime-XML | 0.0232 / 0.0269 | 0.0496 / 0.0570 | 6% / 10% |
| eICU | PCES / ClinTime-XML | 0.0194 / 0.0305 | 0.0414 / 0.0646 | 2% / 8% |
| NWICU | PCES / ClinTime-XML | 0.0179 / 0.0199 | 0.0381 / 0.0423 | 3% / 4% |

显式写出 `semantics="unordered"` 并不能让通用LLM自动获得集合不变性。Canonical XML对原始输入行序字节级不变，但单视图性能低于三视图均值。因此表示主张为partial，纯prompt顺序鲁棒主张不成立。

## 成本与完整性

- XML排列实验：900/900调用成功；Canonical实验：300/300调用成功。
- XML prompt tokens合计2,114,256；Canonical合计704,753。
- XML证据ID有效率：MIMIC 94.67%，eICU 88.33%，NWICU 97.33%。
- 原始指标与响应：`refine-logs/remote-results/clintime-xml/`。

## 研究决策

- 保留ClinTime-XML的原子局部绑定和跨库可审计接口。
- 不宣称XML格式本身解决顺序鲁棒性。
- 下一阶段主创新收敛为 **ClinTime-XML + Permutation-Contrastive Tuning**：真实标签BCE加同患者多排列对称Bernoulli KL；推断只使用Canonical XML。
- 暂停扩大DeepSeek零样本调用，先实现可训练开源模型。

