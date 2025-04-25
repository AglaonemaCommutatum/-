import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from main import FresnelBiprismSim
matplotlib.use('Agg')  # 使用非交互式后端

# 导入你的 FresnelBiprismSim 类或函数
# 假设你的原始代码如下


# Streamlit 界面
st.title('菲涅尔双棱镜干涉仿真')

# 侧边栏添加参数控制
st.sidebar.header('参数设置')

# 波长设置（以纳米为单位）
wavelength_nm = st.sidebar.slider('波长 (nm)', 380, 780, 500)
wavelength = wavelength_nm * 1e-9  # 转换为米

# 距离设置
distance = st.sidebar.slider('观察屏距离 (m)', 0.1, 2.0, 1.0)

# 双棱镜角度设置
angle_mrad = st.sidebar.slider('双棱镜角度 (毫弧度)', 0.1, 5.0, 1.0)
angle = angle_mrad * 1e-3  # 转换为弧度

# 创建仿真对象并运行仿真
sim = FresnelBiprismSim(wavelength, distance, angle)

# 添加运行按钮
if st.sidebar.button('运行仿真'):
    with st.spinner('计算中...'):
        fig = sim.simulate()
        st.pyplot(fig)
    st.success('仿真完成!')

# 添加干涉原理说明
st.header('菲涅尔双棱镜干涉原理')
st.write("""
菲涅尔双棱镜是一种能够产生光波干涉的光学装置。它由两个小角度的棱镜组成，
当单色光通过双棱镜时，光线会被分成两束，就像来自两个相干光源一样。
这两束光在观察屏上相遇时会产生干涉条纹。

干涉条纹的间距 Δx 由以下关系式给出：
Δx = λ/(2α)
其中：
- λ 是光的波长
- α 是棱镜的角度
""")

# 显示公式的图片
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Young%27s_experiment_with_double_slit.svg/500px-Young%27s_experiment_with_double_slit.svg.png", 
         caption="双缝干涉示意图（类似于双棱镜干涉）")

# 添加参考资料
st.header('参考资料')
st.markdown("""
- [菲涅尔双棱镜介绍](https://en.wikipedia.org/wiki/Fresnel_biprism)
- [波动光学基本原理](https://optics.synopsys.com/kbase/wave-optics/)
""")