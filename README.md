# House Prices Kaggle 项目

本项目用于训练一个回归模型，解决 Kaggle 的 “House Prices - Advanced Regression Techniques” 竞赛问题，并生成如下格式的提交文件：

```csv
Id,SalePrice
```

该竞赛的预测目标是 `SalePrice`，评价指标是预测房价与真实房价取对数后的 RMSLE。

## 数据

默认的数据目录为：

```text
E:\project\house_price
```

该目录下应包含 Kaggle 提供的以下文件：

* `train.csv`
* `test.csv`
* `sample_submission.csv`
* `data_description.txt`

## 环境配置

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 训练模型并生成提交文件

```powershell
python src\train.py
```

输出文件会被保存到 `outputs/` 目录下：

* `submission.csv`：Kaggle 提交文件
* `cv_results.csv`：候选模型的交叉验证结果
* `model.joblib`：训练完成后保存的最终模型流水线

也可以通过命令行参数覆盖默认路径：

```powershell
python src\train.py --data-dir E:\project\house_price --output-dir outputs
```

## 探索性可视化

从训练数据中生成图表：

```powershell
python src\visualize.py
```

生成的图片会保存到 `docs/images/` 目录下，用于在建模前对训练集进行概括分析。

### 历史房价趋势

下面的折线图按 `YrSold` 和 `MoSold` 对 `SalePrice` 进行聚合。柱状图展示了每个月售出的房屋数量，这有助于区分真实的价格变化和由于某些月份样本量过小而造成的波动。

![历史房价趋势](docs/images/historical_price_trend.png)

### 房价分布

目标变量呈现右偏分布，因此训练流水线会对 `SalePrice` 使用 `log1p` 进行建模，并在预测后将结果转换回实际的美元价格。

![房价分布](docs/images/sale_price_distribution.png)

### 房屋质量与面积关系

整体质量和地上居住面积是该数据集中能够明显影响房价的两个重要因素。

![整体质量与房价关系](docs/images/quality_price_boxplot.png)

![地上居住面积与房价关系](docs/images/living_area_sale_price.png)
