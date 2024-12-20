
# 股票涨停预测模型 XGBOOST

## 概述
本项目旨在通过分析2023年9月至2024年9月期间所有涨停股票的数据，构建一个基于日频数据的预测模型，以预测下一个交易日的涨跌情况。我们不考虑时间因素，专注于日频数据，以期捕捉市场动态。

## 功能亮点
- **机器学习模型**：运用多种机器学习模型预测股票涨跌，其中XGBoost模型在预测大幅上涨时的准确率高达75%。
- **SHAP模型解释**：通过SHAP模型对XGBoost进行解释，以可视化方式展示影响预测的关键因素。
- **Streamlit部署**：项目已部署于Streamlit服务器，欢迎访问[点我](https://xgboost-leixydyapprszn7cwnmxhdb.streamlit.app/)。

## 技术栈
- **编程语言**：Python
- **机器学习库**：scikit-learn, XGBoost
- **解释性工具**：SHAP
- **部署平台**：Streamlit

