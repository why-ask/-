# House Prices Kaggle 项目

本项目用于训练一个回归模型，解决 Kaggle 的 “House Prices - Advanced Regression Techniques” 竞赛问题，并生成如下格式的提交文件：

```csv
Id,SalePrice
```

该竞赛的预测目标是 `SalePrice`，评价指标是预测房价与真实房价取对数后的 RMSLE。

## 数据
 Kaggle 提供的以下文件：

* `train.csv`
* `test.csv`
* `sample_submission.csv`
* `data_description.txt`

## 模型训练

 程序首先读取训练集和测试集，然后通过自定义的 HouseFeatureEngineer 类构造总面积、总浴室数、出售时房龄、是否翻新、是否有车库、是否有地下室等与房价密切相关的新特征。之后，代码使用 ColumnTransformer 对数值特征和类别特征分别进行处理：数值特征采用中位数填充和标准化，类别特征采用缺失值填充和独热编码。模型部分使用 TransformedTargetRegressor 对房价进行对数变换，使目标变量分布更加平滑，并使用 RMSLE 作为评价指标。程序分别训练 Ridge、GradientBoosting、ExtraTrees 和 RandomForest 四种模型，通过 K 折交叉验证选择平均 RMSLE 最低的模型，最终使用完整训练集重新训练最佳模型。

