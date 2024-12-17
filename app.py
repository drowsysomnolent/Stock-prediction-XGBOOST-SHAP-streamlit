import streamlit_shap as st_shap  # 导入 Streamlit-SHAP 库，用于可视化 SHAP 值
import streamlit as st  # type: ignore # 导入 Streamlit 库，用于创建 Web 应用
import numpy as np  # 导入 Numpy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os  # 导入 os 库，用于处理文件路径 
import shap  # 导入 SHAP 库，用于解释模型
import matplotlib.pyplot as plt  # 导入 Matplotlib 库，用于绘制图表
import matplotlib
import plotly.io as pio
import io


def save_plot_to_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  # 关闭当前图，释放资源
    return buf

X = pd.read_csv('X_output.csv')  # 读取数据集
sample_x = X.sample(frac = 0.1,random_state=42)
# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'xgboost_model.pkl')
print(model_path)
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)  # 使用 pickle 加载模型文件
 
 
st.set_page_config(
    page_title="XGBOOST Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        body {
            font-family: "Helvetica Neue", Arial, sans-serif;
            background-color: #f8f8f8;
            color: #333333;
        }
        .stButton button {
            background-color: #007aff;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #005ecb;
            color: white;
        }
        .prediction-box {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #007aff;
            background-color: #e6f0ff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stSlider .st-de {
            color: #007aff !important;
        }
        .stRadio [role=radiogroup] label {
            color: #007aff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

 
st.title("📈 XGBOOST Prediction App")


st.sidebar.subheader("⚙️ Input Features")
is_one = st.sidebar.slider("Is One", min_value=0, max_value=1, value=0)  # 二元特征
consecutive_limit_up = st.sidebar.slider("Consecutive Limit Up", min_value=0, max_value=10, value=0)
i_change_ratio = st.sidebar.slider("I Change Ratio", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
billboard_weight = st.sidebar.slider("Billboard Weight", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)
large_alpha = st.sidebar.slider("Large Alpha", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
circulated_market_value_discrete = st.sidebar.slider("Circulated Market Value Discrete", min_value=0, max_value=3, step=1)
turnover = st.sidebar.slider("Turnover", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
class_i = st.sidebar.selectbox("Class i", options=list(range(5)))
    
# 创建输入数据框，将输入的特征整理为 DataFrame 格式
input_data = pd.DataFrame({
    'is_one': [is_one],
    'ConsecutiveLimitUp': [consecutive_limit_up],
    'IChangeRatio': [i_change_ratio],
    'billboard_weight': [billboard_weight],
    'large_alpha': [large_alpha],
    'CirculatedMarketValue_Discrete': [circulated_market_value_discrete],
    'Turnover': [turnover],
    'IndcdZX_C39': [0],  # 可以根据需要设定默认值
    'IndcdZX_C37': [0]   # 可以根据需要设定默认值
})

columns_to_standardize = ['ITurnover', 'IChangeRatio', 'Turnover', ]
 
 

st.subheader("🔍 Prediction")
prediction = model.predict(input_data)  # 使用加载的模型进行预测

prediction_texts = {
    0: "大跌",
    1: "小跌",
    2: "平牌",
    3: "小涨",
    4: "大涨"
}
predicted_value = prediction_texts.get(prediction[0], "未知") 
st.markdown(
    f"<div class='prediction-box'>预测结果: {predicted_value}</div>",
    unsafe_allow_html=True,
)


    # 计算 SHAP 值
explainer = shap.TreeExplainer(model)  # 或者使用 shap.TreeExplainer(model) 来计算树模型的 SHAP 值
shap_values = explainer(input_data)
 
    # 提取单个样本的 SHAP 值和期望值
sample_shap_values = shap_values[0]  # 提取第一个样本的 SHAP 值
expected_value = explainer.expected_value[0]  # 获取对应输出的期望值
 
    # 创建 Explanation 对象
explanation = shap.Explanation(
        values=sample_shap_values[:, 0],  # 选择特定输出的 SHAP 值
        base_values=expected_value,
        data=input_data.iloc[0].values,
        feature_names=input_data.columns.tolist()
    )


# 绘制各个 SHAP 图并缓存
# Waterfall Plot
fig1 = plt.figure()
shap.plots.waterfall(shap_values[0, :, class_i], show=False)
buf1 = save_plot_to_buffer(fig1)

# Bar Plot
fig2 = plt.figure()
shap.plots.bar(shap_values[0, :, class_i], show=False)
buf2 = save_plot_to_buffer(fig2)

shap_values = explainer(sample_x)

# Summary Plot
fig3 = plt.figure()
shap.summary_plot(shap_values[:, :, class_i], show=False)
buf3 = save_plot_to_buffer(fig3)

# Heatmap Plot
fig4 = plt.figure()
shap.plots.heatmap(shap_values[:, :, class_i], show=False)
buf4 = save_plot_to_buffer(fig4)

st.title("🔍 SHAP Visualization for Class {}".format(class_i))
plot_option = st.selectbox(
    "Select a SHAP Visualization:",
    ["Waterfall Plot", "Bar Plot", "Summary Plot", "Heatmap Plot"]
)

# 根据选择显示对应的图
if plot_option == "Waterfall Plot":
    st.image(buf1, caption="Waterfall Plot", use_column_width=True)
elif plot_option == "Bar Plot":
    st.image(buf2, caption="Bar Plot", use_column_width=True)
elif plot_option == "Summary Plot":
    st.image(buf3, caption="Summary Plot", use_column_width=True)
elif plot_option == "Heatmap Plot":
    st.image(buf4, caption="Heatmap Plot", use_column_width=True)

st.markdown("---")
st.info("Adjust input features to observe how predictions change.")


# 添加分隔线
st.markdown("---")

# 添加字段解释
st.subheader("📋 Feature Description")
st.markdown(
    """
    | Feature Name                 | Description                                             |
    |------------------------------|---------------------------------------------------------|
    | **Is One**                   | Binary feature indicating if high price = Loe price     |
    | **Consecutive Limit Up**     | Number of consecutive upward limits.                    |
    | **I Change Ratio**           | Change ratio of a specific metric, ranging [-1, 1].     |
    | **Billboard Weight**         | Weight or importance of a specific billboard.           |
    | **Large Alpha**              | Alpha value representing a scaling factor.              |
    | **Circulated Market Value**  | Discrete value indicating market value classification.  |
    | **Turnover**                 | Turnover of a specific stock, ranging [0, 100].         |
    | **Class_i**                  | Class label of the prediction, ranging [0, 4].          |
    """,
    unsafe_allow_html=True
)
 

