import shap
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

# 加载 XGBoost 模型（.pkl 格式）
with open('C:/Users/yy06a/model双结局/xgb_full.pkl', 'rb') as f:
    xgb_full = pickle.load(f)  # 加载 pkl 模型

# 设置页面标题
st.title(
    'An Explainable XGB Model for Predicting Massive Intraoperative Blood Loss and Transfusion in Thoracic Spinal Stenosis Surgery')

# 用户输入特征
st.sidebar.header('Features input')


def user_input_features():
    experienced_surgeon = st.sidebar.selectbox('Experienced surgeon', options=['Yes', 'No'])
    txa_used = st.sidebar.selectbox('Use of TXA', options=['Yes', 'No'])

    # 允许输入小数
    levels_of_pd = st.sidebar.number_input('Levels of PD', min_value=1, max_value=15, value=1, step=1)
    cd = st.sidebar.selectbox('CD', options=['Yes', 'No'])

    # 允许输入小数
    BMI = st.sidebar.number_input('BMI', min_value=1.00, max_value=100.00, value=1.00, step=0.01)
    PT = st.sidebar.number_input('PT', min_value=1.0, max_value=100.0, value=1.0, step=0.1)
    APTT = st.sidebar.number_input('APTT', min_value=1.0, max_value=100.0, value=1.0, step=0.1)
    PLT = st.sidebar.number_input('PLT', min_value=1, max_value=1000, value=1, step=1)
    pre_Hb = st.sidebar.number_input('Pre-HB', min_value=1, max_value=1000, value=1, step=1)
    experienced_surgeon = 1 if experienced_surgeon == 'Yes' else 0
    txa_used = 1 if txa_used == 'Yes' else 0
    cd = 1 if cd == 'Yes' else 0

    data = {
        'TXA used': [txa_used],
        'levels of PD': [levels_of_pd],
        'experienced surgeon': [experienced_surgeon],
        'CD': [cd],
        'PLT': [PLT],
        'BMI': [BMI],
        'PT': [PT],
        'APTT': [APTT],
        'pre-Hb':[pre_Hb]

    }
    features = pd.DataFrame(data)
    return features


df = user_input_features()

# 获取 XGBoost 模型训练时的特征名称顺序
model_feature_names = xgb_full.get_booster().feature_names

# 将 df 的列顺序调整为模型的训练顺序
df = df[model_feature_names]

# 显示用户输入的特征
st.write('Features input:')
st.write(df)

# 添加按钮并执行预测
if st.button('Predict'):
    try:
        # 获取预测概率（使用 output_margin=True 得到原始分数，再转换为概率）
        raw_pred = xgb_full.predict(df, output_margin=True)  # 获取原始分数
        prediction_proba = 1 / (1 + np.exp(-raw_pred))  # 使用 sigmoid 转换为概率

        # 根据最佳 cutoff 值 0.22 得到最终预测结果
        cutoff = 0.509
        prediction = (prediction_proba >= cutoff).astype(int)

        # 输出预测结果
        risk_message = 'High risk of massive intraoperative blood loss' if prediction[
                                                                               0] == 1 else 'low risk of massive intraoperative blood loss'
        st.write(f'Predict: {risk_message}')
        st.write('Predicted possibility:')
        st.write(prediction_proba)

        # 使用 SHAP TreeExplainer 来计算 SHAP 值
        explainer = shap.TreeExplainer(xgb_full)
        shap_values = explainer.shap_values(df)  # 获取 SHAP 值

        # SHAP 对于二分类问题返回的是单一数组，不需要区分类别
        shap_values_class = shap_values  # 直接使用 shap_values

        # 选择需要解释的样本（在这里我们假设用户输入的是第一个样本）
        indx = 0  # 使用第一个样本进行解释

        # 初始化 js 环境
        shap.initjs()

        # 保存 SHAP force plot 为 HTML 文件
        shap_html = shap.force_plot(explainer.expected_value, shap_values_class[indx, :], df.iloc[indx, :],
                                    link='logit', feature_names=model_feature_names)
        shap.save_html("shap_force_plot.html", shap_html)

        # 使用 Streamlit 组件加载 HTML 文件
        with open("shap_force_plot.html", "r", encoding="utf-8") as f:  # 指定编码为 'utf-8'
            st.components.v1.html(f.read(), height=600)

    except Exception as e:
        st.write(f'预测时出错: {e}')
