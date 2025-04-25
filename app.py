import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
import traceback
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.font_manager import FontProperties


# 设置中文字体
def configure_chinese_fonts():
    """配置中文字体支持"""
    # 尝试多种可能的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
    font_found = False

    # 检查字体是否可用
    for font_name in chinese_fonts:
        try:
            font_prop = FontProperties(fname=mpl.font_manager.findfont(font_name))
            if font_prop.get_name() != 'DejaVu Sans':  # 不是默认字体
                mpl.rcParams['font.family'] = font_prop.get_family()
                mpl.rcParams['font.sans-serif'] = [font_name] + mpl.rcParams['font.sans-serif']
                mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                st.toast(f"使用中文字体: {font_name}", icon="✅")
                font_found = True
                break
        except Exception:
            continue

    # 如果没有找到理想字体，使用备选方案
    if not font_found:
        st.warning("未找到中文字体，图表中的中文可能显示为乱码。请安装中文字体。")

        # 尝试使用基本配置
        mpl.rcParams['font.sans-serif'] = ['DejaVu Sans'] + mpl.rcParams['font.sans-serif']
        mpl.rcParams['axes.unicode_minus'] = False

        # 在Linux上，可以尝试直接指定字体文件
        try:
            font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
            if os.path.exists(font_path):
                mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] + mpl.rcParams['font.sans-serif']
                st.toast("使用 WenQuanYi Micro Hei 字体", icon="✅")
        except Exception:
            pass


# 设置页面配置
st.set_page_config(
    page_title="菲涅尔双棱镜干涉仿真 V2.9.5",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置中文字体
configure_chinese_fonts()

# 设置全局变量，用于在会话期间保持状态
if 'params' not in st.session_state:
    st.session_state.params = {}
    st.session_state.results = None


# 定义菲涅尔双棱镜仿真类的核心计算逻辑
class FresnelBiprismSimCore:
    # 定义常量
    Y_DISPLAY_RANGE = 2.0  # 垂直范围 (mm)
    Y_PIXELS = 200  # 垂直像素数
    PLOT_XLIM = (-6, 6)  # 水平范围 (mm)
    NUM_X_POINTS = 1000  # 屏幕上计算强度的点数

    def __init__(self):
        # 输入参数配置
        self.input_params_config = {
            'x1': {'val': 10.0, 'label': "光源狭缝位置 x₁ (cm)"},
            'x2': {'val': 40.0, 'label': "凸透镜位置 x₂ (cm)"},
            'x3': {'val': 110.0, 'label': "测微目镜位置 x₃ (cm)"},
            'P1': {'val': 0.00, 'label': "所成实像 S₁' 位置 P₁ (mm)"},
            'P2': {'val': 0.80, 'label': "所成实像 S₂' 位置 P₂ (mm)"},
            'x4': {'val': 5.00, 'label': "第 0 条读数 x₄ (mm)"},
            'x5': {'val': 15.94, 'label': "第 10 条读数 x₅ (mm)"},
            'slit_width': {'val': 0.05, 'label': "光源狭缝宽度 b (mm)"},
        }
        self.default_params = {key: config['val'] for key, config in self.input_params_config.items()}
        self.save_file = "fresnel_params.json"

        # 输出参数标签
        self.output_params_labels = {
            'u_cm': "物距 u (cm):",
            'v_cm': "像距 v (cm):",
            'D_cm': "屏间距 D (cm):",
            'd_prime_mm': "实像间距 d' (mm):",
            'd_mm': "虚光源距 d (mm):",
            'delta_x_mm': "条纹间距 Δx (mm):",
            'wavelength_nm': "计算波长 λ (nm):"
        }
        # 输出布局
        self.output_layout = [
            ['u_cm', 'd_mm'],
            ['v_cm', 'delta_x_mm'],
            ['D_cm', 'wavelength_nm'],
            ['d_prime_mm', None]
        ]

    def _load_params(self):
        """加载参数文件"""
        if not os.path.exists(self.save_file):
            print("未找到参数文件，使用默认参数。")
            return self.default_params.copy()

        try:
            with open(self.save_file, 'r', encoding='utf-8') as f:
                loaded_params = json.load(f)

            valid_params = {}
            valid_keys_loaded = 0
            invalid_keys = []

            for key, value in loaded_params.items():
                if key in self.default_params:
                    try:
                        float_val = float(value)
                        valid_params[key] = float_val
                        valid_keys_loaded += 1
                    except (ValueError, TypeError):
                        invalid_keys.append((key, value))
                        valid_params[key] = self.default_params[key]

            if valid_keys_loaded > 0:
                st.toast(f"成功加载 {valid_keys_loaded} 个有效参数", icon="✅")
            if invalid_keys:
                st.toast(f"发现 {len(invalid_keys)} 个无效参数，已使用默认值", icon="⚠️")

            if valid_keys_loaded == 0 and not invalid_keys:
                valid_params = self.default_params.copy()

            return valid_params

        except (json.JSONDecodeError, IOError, TypeError) as e:
            st.toast(f"加载参数文件错误: {e}，使用默认参数", icon="❌")
            return self.default_params.copy()

    def _save_params(self, params):
        """保存当前参数到文件"""
        try:
            params_to_save = {}
            for key in self.input_params_config:
                try:
                    params_to_save[key] = float(params[key])
                except (ValueError, TypeError, KeyError):
                    params_to_save[key] = self.default_params[key]

            with open(self.save_file, 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, indent=4, ensure_ascii=False)
            st.toast(f"参数已保存到 {self.save_file}", icon="💾")
        except IOError as e:
            st.toast(f"保存参数错误: {e}", icon="❌")
        except Exception as e:
            st.toast(f"保存参数时发生意外错误: {e}", icon="❌")

    def calculate_physics(self, params):
        """根据当前参数计算物理量"""
        try:
            p = params
            d_p = self.default_params
            x1_cm = float(p.get('x1', d_p['x1']))
            x2_cm = float(p.get('x2', d_p['x2']))
            x3_cm = float(p.get('x3', d_p['x3']))
            P1_mm = float(p.get('P1', d_p['P1']))
            P2_mm = float(p.get('P2', d_p['P2']))
            x4_mm = float(p.get('x4', d_p['x4']))
            x5_mm = float(p.get('x5', d_p['x5']))
            b_slit_mm = float(p.get('slit_width', d_p['slit_width']))

            # 参数验证
            error_msgs = []
            if not (x1_cm < x2_cm < x3_cm):
                error_msgs.append(f"位置错误: 必须满足 x₁ ({x1_cm:.1f}) < x₂ ({x2_cm:.1f}) < x₃ ({x3_cm:.1f}) cm。")
            if abs(P1_mm - P2_mm) < 1e-9:
                error_msgs.append(f"实像位置 P₁ ({P1_mm:.3f}) 和 P₂ ({P2_mm:.3f}) mm 不能相同。")
            if x5_mm <= x4_mm:
                error_msgs.append(f"读数 x₅ ({x5_mm:.3f}) mm 必须大于 x₄ ({x4_mm:.3f}) mm。")
            if b_slit_mm <= 0:
                error_msgs.append(f"光源狭缝宽度 b ({b_slit_mm:.3f}) mm 必须为正数。")

            if error_msgs:
                for msg in error_msgs:
                    st.error(msg)
                return None

            # 计算物理量
            u_cm = x2_cm - x1_cm
            v_cm = x3_cm - x2_cm
            d_prime_mm = abs(P2_mm - P1_mm)
            u_mm = u_cm * 10.0
            v_mm = v_cm * 10.0
            D_mm = (x3_cm - x1_cm) * 10.0

            if abs(v_mm) < 1e-9:
                st.error("计算得到的像距 v 接近零 (v ≈ 0)，无法计算虚光源间距 d。检查 x₂ 和 x₃。")
                return None
            if abs(D_mm) < 1e-9:
                st.error("屏间距 D (x₃-x₁) 计算为零或接近零 (D ≈ 0)，无法计算波长 λ。检查 x₁ 和 x₃。")
                return None

            d_mm_calc = d_prime_mm * (u_mm / v_mm)
            if abs(d_mm_calc) < 1e-9:
                st.warning(f"计算得到的虚光源间距 d = {d_mm_calc:.3e} mm 非常小，可能无明显干涉。")

            delta_x_mm_calc = abs(x5_mm - x4_mm) / 10.0
            if delta_x_mm_calc <= 1e-9:
                st.error("计算得到的条纹间距 Δx 非正数或过小。检查 x₄ 和 x₅。")
                return None

            wavelength_mm = (d_mm_calc * delta_x_mm_calc) / D_mm
            wavelength_nm_calc = wavelength_mm * 1e6

            current_date = datetime.now().strftime("%Y-%m-%d")
            if not (1 < wavelength_nm_calc < 10000):
                st.warning(
                    f"({current_date}) 警告: 计算得到的波长 λ = {wavelength_nm_calc:.2f} nm 超出常规范围。请仔细检查所有输入参数。")
            elif not (380 <= wavelength_nm_calc <= 780):
                st.info(
                    f"({current_date}) 提示: 计算波长 λ = {wavelength_nm_calc:.2f} nm 不在典型可见光范围 (380-780 nm)。")

            results = {
                'u_cm': u_cm,
                'v_cm': v_cm,
                'D_cm': D_mm / 10.0,
                'd_prime_mm': d_prime_mm,
                'd_mm': d_mm_calc,
                'delta_x_mm': delta_x_mm_calc,
                'wavelength_nm': wavelength_nm_calc,
                'D_mm_plot': D_mm,
                'd_mm_plot': d_mm_calc,
                'b_slit_mm_plot': b_slit_mm,
                'lambda_mm_plot': wavelength_mm,
            }
            return results

        except ValueError as e:
            st.error(f"计算错误 (输入值或逻辑): {e}")
            return None
        except ZeroDivisionError as e:
            st.error(f"计算错误 (除以零): {e}")
            return None
        except Exception as e:
            st.error(f"发生意外的计算错误: {type(e).__name__} - {e}")
            return None

    def generate_plot(self, results):
        """根据计算结果生成图形"""
        # 创建新图形
        plt.rcParams['font.sans-serif'] = mpl.rcParams['font.sans-serif']  # 确保使用配置的中文字体
        plt.rcParams['axes.unicode_minus'] = mpl.rcParams['axes.unicode_minus']

        if results is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlim(*self.PLOT_XLIM)
            ax.set_ylim(-self.Y_DISPLAY_RANGE, self.Y_DISPLAY_RANGE)
            ax.set_xlabel("屏幕位置 x (mm)", fontsize=12)
            ax.set_ylabel("")
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            ax.set_title("模拟干涉条纹图样 (参数无效或计算错误)", fontsize=14)
            ax.text(0.5, 0.5, '参数无效或计算错误\n请检查输入',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray', wrap=True)
            return fig

        try:
            lambda_mm = results['lambda_mm_plot']
            d_mm = results['d_mm_plot']
            D_mm = results['D_mm_plot']
            b_slit_mm = results['b_slit_mm_plot']

            if abs(D_mm) < 1e-9 or abs(lambda_mm) < 1e-15:
                st.error("绘图需要非零的屏距 D 和波长 λ。")
                return None

            fig, ax = plt.subplots(figsize=(10, 5))

            x_screen = np.linspace(self.PLOT_XLIM[0], self.PLOT_XLIM[1], self.NUM_X_POINTS)
            lambda_D = lambda_mm * D_mm
            if abs(lambda_D) < 1e-15:
                lambda_D = 1e-15

            common_factor = (np.pi * x_screen) / lambda_D

            # 计算衍射项
            if abs(b_slit_mm) < 1e-9:
                diffraction_term = np.ones_like(x_screen)
            else:
                alpha = b_slit_mm * common_factor
                with np.errstate(divide='ignore', invalid='ignore'):
                    diffraction_term = np.sinc(alpha / np.pi) ** 2
                diffraction_term = np.nan_to_num(diffraction_term, nan=0.0)

            # 计算干涉项
            if abs(d_mm) < 1e-9:
                interference_term = 0.5
            else:
                beta = d_mm * common_factor
                interference_term = np.cos(beta) ** 2

            # 计算总强度
            intensity_1d = diffraction_term * interference_term
            intensity_1d = np.maximum(intensity_1d, 0)
            intensity_2d = np.repeat(intensity_1d[np.newaxis, :], self.Y_PIXELS, axis=0)

            # 绘制干涉图样
            extent = [self.PLOT_XLIM[0], self.PLOT_XLIM[1], -self.Y_DISPLAY_RANGE, self.Y_DISPLAY_RANGE]
            cmap = 'gray'
            vmin_val = 0.0
            vmax_val = np.max(intensity_2d)
            if vmax_val <= vmin_val + 1e-9:
                vmax_val = vmin_val + 1e-9

            norm = mcolors.Normalize(vmin=vmin_val, vmax=vmax_val)
            im = ax.imshow(intensity_2d, cmap=cmap, norm=norm,
                           origin='lower', extent=extent,
                           aspect='auto', interpolation='bilinear')

            # 设置图形属性
            title_text = f"模拟干涉图样 (计算值: λ = {results['wavelength_nm']:.2f} nm)"
            ax.set_title(title_text, fontsize=14)
            ax.set_xlabel("屏幕位置 x (mm)", fontsize=12)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('归一化光强', fontsize=11)

            return fig

        except Exception as e:
            st.error(f"绘图时发生错误: {type(e).__name__} - {e}")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, f'绘图错误:\n{type(e).__name__}\n请检查计算结果和参数',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10, color='red', wrap=True)
            return fig


# 初始化模拟器
@st.cache_resource
def get_simulator():
    return FresnelBiprismSimCore()


sim = get_simulator()

# 设置标题和页面布局
st.title("菲涅尔双棱镜干涉仿真 V2.9.5")

# 创建两列布局
col1, col2 = st.columns([2, 1])

# 输入参数部分 (在侧边栏)
with st.sidebar:
    st.header("输入参数")

    # 如果session_state中没有参数，尝试加载
    if not st.session_state.params:
        st.session_state.params = sim._load_params()

    # 创建输入控件
    params = {}
    for key, config in sim.input_params_config.items():
        # 根据不同参数类型设置不同的小数位数展示
        step = 0.1 if key in ['x1', 'x2', 'x3'] else 0.001
        format_str = "%.1f" if key in ['x1', 'x2', 'x3'] else "%.3f"

        # 创建滑动条并初始化为当前值
        current_value = st.session_state.params.get(key, config['val'])

        # 为不同参数设置合适的范围
        if key == 'x1':
            min_val, max_val = 0.0, 50.0
        elif key == 'x2':
            min_val, max_val = st.session_state.params.get('x1', 10.0) + 1, 100.0
        elif key == 'x3':
            min_val, max_val = st.session_state.params.get('x2', 40.0) + 1, 200.0
        elif key == 'slit_width':
            min_val, max_val = 0.001, 0.5
        elif key in ['P1', 'P2']:
            min_val, max_val = -5.0, 5.0
        else:  # x4, x5
            min_val, max_val = 0.0, 50.0

        params[key] = st.number_input(
            config['label'],
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(current_value),
            step=step,
            format=format_str
        )

    # 按钮区域
    st.header("操作")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("计算并绘图", type="primary"):
            # 运行计算
            st.session_state.params = params
            st.session_state.results = sim.calculate_physics(params)
            # 保存参数
            sim._save_params(params)

    with col_btn2:
        if st.button("重置参数"):
            st.session_state.params = sim.default_params.copy()
            # 更新UI需要重新加载页面
            st.rerun()

# 在主区域显示图形
with col1:
    # 如果有结果，就显示图形
    if st.session_state.results is not None:
        fig = sim.generate_plot(st.session_state.results)
        st.pyplot(fig)
    else:
        # 显示空白或默认图形
        fig = sim.generate_plot(None)
        st.pyplot(fig)

# 在侧面显示计算结果
with col2:
    st.header("计算结果")

    if st.session_state.results:
        results = st.session_state.results

        # 使用网格布局展示计算结果
        for row in sim.output_layout:
            cols = st.columns(len(row))
            for i, key in enumerate(row):
                if key is not None and key in results:
                    with cols[i]:
                        label = sim.output_params_labels[key]
                        # 根据不同参数设置不同小数位数
                        if key in ['u_cm', 'v_cm', 'D_cm']:
                            value = f"{results[key]:.1f}"  # 1位小数
                        elif key == 'wavelength_nm':
                            value = f"{results[key]:.2f}"  # 2位小数
                        else:
                            value = f"{results[key]:.3f}"  # 3位小数
                        st.text_input(label, value, disabled=True)
    else:
        st.info("请设置参数并点击「计算并绘图」按钮")

# 添加干涉原理说明
# with st.expander("菲涅尔双棱镜干涉原理"):
#     st.write("""
#     菲涅尔双棱镜是一种能够产生光波干涉的光学装置。它由两个小角度的棱镜组成，
#     当单色光通过双棱镜时，光线会被分成两束，就像来自两个相干光源一样。
#     这两束光在观察屏上相遇时会产生干涉条纹。
#
#     干涉条纹的间距 Δx 由以下关系式给出：
#     Δx = λ/(2α)
#     其中：
#     - λ 是光的波长
#     - α 是棱镜的角度
#     """)
#
#     st.image(
#         "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Young%27s_experiment_with_double_slit.svg/500px-Young%27s_experiment_with_double_slit.svg.png",
#         caption="双缝干涉示意图（类似于双棱镜干涉）")

# 添加页脚信息
st.markdown("---")
st.caption("菲涅尔双棱镜干涉仿真 V2.9.5 | Streamlit版本")