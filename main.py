# -*- coding: utf-8 -*-
"""
菲涅耳双棱镜干涉实验仿真 (Matplotlib GUI)
功能：使用 Matplotlib Widgets 实现参数交互、计算结果展示和 2D 干涉条纹模拟
版本：2.9.5 (调整输入/输出小数位数)
"""

import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib.colors as mcolors
import traceback # Import traceback for better error logging
from datetime import datetime # Import datetime

# --- Matplotlib 全局字体和样式设置 ---
try:
    # 优先尝试常见的无衬线中文字体
    plt.rcParams['font.family'] = ['sans-serif']
    # SimHei 和 Microsoft YaHei 是 Windows 上常用的字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'] # 添加更多备选和通用后备
    plt.rcParams['axes.unicode_minus'] = False # 正确显示负号
    print("Matplotlib 中文字体设置成功 (尝试 SimHei/Microsoft YaHei/Arial Unicode MS)。")
except Exception as e:
    print(f"警告：设置 Matplotlib 中文字体时出错: {e}")
    print("绘图中的中文可能无法正常显示。请确保已安装 SimHei 或 Microsoft YaHei 等支持中文的字体。")

# 设置深色主题
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1e1e1e'
plt.rcParams['axes.facecolor'] = '#2a2a2a'
plt.rcParams['text.color'] = 'white'
plt.rcParams['xtick.color'] = 'lightgray'
plt.rcParams['ytick.color'] = 'lightgray'
plt.rcParams['axes.labelcolor'] = 'lightgray'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['legend.facecolor'] = '#3a3a3a'
plt.rcParams['legend.edgecolor'] = 'gray'
plt.rcParams['legend.labelcolor'] = 'white'

class FresnelBiprismSim:
    # OPT: Define static constants at class level
    Y_DISPLAY_RANGE = 2.0   # Vertical extent of the displayed pattern (mm)
    Y_PIXELS = 200          # Number of vertical pixels for the pattern
    PLOT_XLIM = (-6, 6)     # Horizontal range of the screen display (mm)
    NUM_X_POINTS = 1000     # Number of points to calculate intensity across the screen

    def __init__(self):
        # --- 输入参数及其默认值 ---
        self.input_params_config = {
            'x1': {'val': 10.0, 'label': r"光源狭缝位置 $x_1$ (cm)"}, # Use raw string for LaTeX
            'x2': {'val': 40.0, 'label': r"凸透镜位置 $x_2$ (cm)"},
            'x3': {'val': 110.0, 'label': r"测微目镜位置 $x_3$ (cm)"},
            'P1': {'val': 0.00, 'label': r"所成实像 $S_1'$ 位置 $P_1$ (mm)"},
            'P2': {'val': 0.80, 'label': r"所成实像 $S_2'$ 位置 $P_2$ (mm)"},
            'x4': {'val': 5.00, 'label': r"第 0 条读数 $x_4$ (mm)"},
            'x5': {'val': 15.94, 'label': r"第 10 条读数 $x_5$ (mm)"},
            'slit_width': {'val': 0.05, 'label': r"光源狭缝宽度 $b$ (mm)"},
        }
        self.default_params = {key: config['val'] for key, config in self.input_params_config.items()}
        self.params = self.default_params.copy()

        # --- 保存/加载功能 ---
        self.save_file = "fresnel_params.json"
        self._load_params()

        # --- 输出参数及其标签 ---
        self.output_params_labels = {
            'u_cm': "物距 u (cm):",
            'v_cm': "像距 v (cm):",
            'D_cm': "屏间距 D (cm):",
            'd_prime_mm': "实像间距 d' (mm):",
            'd_mm': "虚光源距 d (mm):",
            'delta_x_mm': r"条纹间距 $\Delta x$ (mm):", # Use raw string for LaTeX
            'wavelength_nm': r"计算波长 $\lambda$ (nm):" # Use raw string for LaTeX
        }
        # Layout for the 4x2 output grid
        self.output_layout = [
            ['u_cm', 'd_mm'],
            ['v_cm', 'delta_x_mm'],
            ['D_cm', 'wavelength_nm'],
            ['d_prime_mm', None] # Last row has only one item
        ]

        # 存储控件
        self.textboxes = {}
        self.output_axes = {}
        self.output_texts = {}
        self.colorbar = None # Store colorbar object

        # 创建图形
        self.fig = plt.figure(figsize=(13, 7))
        window_title = "菲涅耳双棱镜干涉仿真 V2.9.5 (调整输入/输出小数位数)"
        # Use backend-agnostic way to set title if possible, otherwise keep original
        try:
            self.fig.canvas.manager.set_window_title(window_title)
        except AttributeError:
            try:
                self.fig.canvas.set_window_title(window_title) # Fallback for some backends
            except Exception:
                 pass # Ignore if setting title fails

        # --- 定义主绘图区和 Colorbar 区域 ---
        plot_left = 0.05
        plot_bottom = 0.08
        plot_width = 0.50
        plot_height = 0.84
        cbar_left = plot_left + plot_width + 0.01
        cbar_bottom = plot_bottom
        cbar_width = 0.015
        cbar_height = plot_height

        self.ax_plot = self.fig.add_axes([plot_left, plot_bottom, plot_width, plot_height])
        self.cax = self.fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        self.cax.set_visible(False)

        # 设置固定的主绘图区属性
        self.ax_plot.set_xlim(*self.PLOT_XLIM)
        self.ax_plot.set_ylim(-self.Y_DISPLAY_RANGE, self.Y_DISPLAY_RANGE)
        self.ax_plot.set_aspect('auto', adjustable='box')
        self.ax_plot.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        # --- 连接关闭事件以保存参数 ---
        self.fig.canvas.mpl_connect('close_event', self._save_params)

        # --- 创建输入、输出模块和按钮 ---
        self.create_input_module()
        self.create_output_module()
        self.create_buttons()

        # 初始绘图
        self.run_calculation_and_plot(None)

        plt.show()

    # --- 新增: 辅助方法用于格式化输入框显示 ---
    def _format_input_display(self, key, value):
        """根据参数key决定输入框显示的小数位数"""
        try:
            num_val = float(value)
            if key in ['x1', 'x2', 'x3']:
                return f"{num_val:.1f}" # x1, x2, x3 保留1位小数
            else:
                return f"{num_val:.3f}" # 其他输入参数保留3位小数
        except (ValueError, TypeError):
             # 如果值无效，尝试返回默认值的格式化字符串或空字符串
             default_val = self.default_params.get(key, 0)
             try:
                 if key in ['x1', 'x2', 'x3']:
                     return f"{float(default_val):.1f}"
                 else:
                     return f"{float(default_val):.3f}"
             except:
                 return "---" # Fallback

    def _load_params(self):
        """加载参数文件"""
        if not os.path.exists(self.save_file):
            print("未找到参数文件，使用默认参数。")
            return
        try:
            with open(self.save_file, 'r', encoding='utf-8') as f:
                loaded_params = json.load(f)
            print(f"从 {self.save_file} 加载参数...")
            valid_keys_loaded = 0
            invalid_keys = []
            for key, value in loaded_params.items():
                if key in self.default_params:
                    try:
                        float_val = float(value)
                        self.params[key] = float_val
                        valid_keys_loaded += 1
                    except (ValueError, TypeError):
                        invalid_keys.append((key, value))
                        self.params[key] = self.default_params[key] # Use default if invalid
            if valid_keys_loaded > 0:
                 print(f"成功加载 {valid_keys_loaded} 个有效参数。")
            if invalid_keys:
                print("警告：以下参数值无效，已使用默认值:")
                for key, value in invalid_keys:
                    print(f"  - '{key}': '{value}' (使用默认: {self.default_params[key]})")
            if valid_keys_loaded == 0 and not invalid_keys:
                 print("参数文件为空或不包含有效参数，将使用默认值。")
                 self.params = self.default_params.copy()

        except (json.JSONDecodeError, IOError, TypeError) as e:
            print(f"加载参数文件 {self.save_file} 时出错: {e}。将使用默认参数。")
            self.params = self.default_params.copy()

    def _save_params(self, event=None):
        """保存当前参数到文件"""
        try:
            params_to_save = {}
            for key in self.input_params_config:
                try:
                    params_to_save[key] = float(self.params[key])
                except (ValueError, TypeError, KeyError):
                    print(f"警告: 保存时参数 '{key}' 值无效或丢失，将使用默认值 {self.default_params[key]} 进行保存。")
                    params_to_save[key] = self.default_params[key]

            with open(self.save_file, 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, indent=4, ensure_ascii=False)
            print(f"参数已保存到 {self.save_file}")
        except IOError as e:
            print(f"保存参数到 {self.save_file} 时出错: {e}")
        except Exception as e:
             print(f"准备保存参数时发生意外错误: {e}")


    def create_input_module(self):
        """创建输入参数模块 (TextBox)"""
        module_left = 0.70
        module_width = 0.26
        label_width_ratio = 0.55
        textbox_width_ratio = 0.4
        row_height = 0.040
        v_spacing = 0.010
        start_top = 0.94

        ax_title = self.fig.add_axes([module_left, start_top, module_width, 0.03], facecolor='none')
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, "输入参数", color='white', fontsize=14, weight='bold', ha='center', va='center')
        current_top = start_top - 0.08

        input_keys = list(self.input_params_config.keys())
        num_inputs = len(input_keys)

        for i, key in enumerate(input_keys):
            config = self.input_params_config[key]
            top = current_top - i * (row_height + v_spacing)

            ax_label = self.fig.add_axes([module_left, top, module_width * label_width_ratio, row_height], facecolor='none')
            ax_label.axis('off')
            ax_label.text(0.95, 0.5, config['label'], color='lightgray', fontsize=11,
                          ha='right', va='center', usetex=False)

            ax_text = self.fig.add_axes([module_left + module_width * label_width_ratio + 0.01, top,
                                         module_width * textbox_width_ratio, row_height])
            try:
                initial_val = float(self.params[key])
            except (ValueError, TypeError, KeyError):
                initial_val = self.default_params[key]
                self.params[key] = initial_val # Correct internal state if loaded value was bad

            # --- 使用辅助方法格式化初始显示的文本 ---
            initial_val_str = self._format_input_display(key, initial_val)

            textbox = TextBox(ax_text, "", initial=initial_val_str,
                              color='#3a3a3a', hovercolor='#4a4a4a', label_pad=0.01)
            textbox.label.set_visible(False)
            textbox.text_disp.set_color('white')
            textbox.text_disp.set_fontsize(11)
            textbox.on_submit(lambda text, k=key: self.textbox_update(text, k))
            self.textboxes[key] = textbox

        self._input_module_bottom = current_top - (num_inputs - 1) * (row_height + v_spacing) - row_height

    def create_output_module(self):
        """创建计算结果显示模块 (4x2 矩阵布局)"""
        module_left = 0.70
        module_width = 0.26
        num_cols = 2
        num_rows = len(self.output_layout)
        col_gap = 0.1
        col_width = (module_width - col_gap * (num_cols - 1)) / num_cols
        label_width_ratio = 0.55
        value_width_ratio = 0.40
        h_spacing = 0.01
        row_height = 0.035
        v_spacing = 0.01
        start_top = self._input_module_bottom - 0.07

        ax_title = self.fig.add_axes([module_left, start_top, module_width, 0.03], facecolor='none')
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, "计算结果", color='white', fontsize=14, weight='bold', ha='center', va='center')
        current_top_row_start = start_top - 0.06

        for r in range(num_rows):
            for c in range(num_cols):
                key = self.output_layout[r][c] if c < len(self.output_layout[r]) else None
                if key is None or key not in self.output_params_labels:
                    continue

                label_text = self.output_params_labels[key]
                cell_left = module_left + c * (col_width + col_gap)
                top = current_top_row_start - r * (row_height + v_spacing)

                ax_label = self.fig.add_axes([cell_left, top, col_width * label_width_ratio, row_height], facecolor='none')
                ax_label.axis('off')
                ax_label.text(0.95, 0.5, label_text, color='lightgray', fontsize=11, ha='right', va='center', usetex=False)

                value_left = cell_left + col_width * label_width_ratio + h_spacing
                value_width = col_width * value_width_ratio
                ax_value = self.fig.add_axes([value_left, top, value_width, row_height], facecolor='none')
                ax_value.axis('off')
                value_text_obj = ax_value.text(0.05, 0.5, "--", color='cyan', fontsize=11, weight='bold', ha='left', va='center')

                self.output_axes[key] = ax_value
                self.output_texts[key] = value_text_obj


    def create_buttons(self):
        """创建计算和重置按钮"""
        button_y_pos = 0.02
        button_height = 0.04
        button_width = 0.12
        gap = 0.02
        calc_button_left = 0.70
        reset_button_left = calc_button_left + button_width + gap

        ax_calc = self.fig.add_axes([calc_button_left, button_y_pos, button_width, button_height])
        self.button_calc = Button(ax_calc, "计算并绘图", color='dodgerblue', hovercolor='deepskyblue')
        self.button_calc.label.set_color('white')
        self.button_calc.label.set_fontsize(11)
        self.button_calc.on_clicked(self.run_calculation_and_plot)

        ax_reset = self.fig.add_axes([reset_button_left, button_y_pos, button_width, button_height])
        self.button_reset = Button(ax_reset, "重置参数", color='gray', hovercolor='dimgray')
        self.button_reset.label.set_color('white')
        self.button_reset.label.set_fontsize(11)
        self.button_reset.on_clicked(self.reset_parameters)

    def textbox_update(self, text, key):
        """文本框输入时更新参数值 (on_submit)"""
        try:
            value = float(text)
            if key == 'slit_width' and value <= 0:
                 raise ValueError("光源狭缝宽度 b 必须为正数。")
            if key in ['x1', 'x2', 'x3'] and value < 0:
                 raise ValueError("位置参数 (x1, x2, x3) 不能为负数。")

            self.params[key] = value
            # --- 输入框确认后，使用辅助方法更新显示 ---
            display_text = self._format_input_display(key, value)
            self.textboxes[key].set_val(display_text)
            print(f"参数 {key} 更新为: {value}") # 控制台可以显示原始精度

        except ValueError as e:
            current_val = self.params.get(key, self.default_params[key])
            # --- 无效输入时，使用辅助方法恢复显示 ---
            display_text = self._format_input_display(key, current_val)
            self.textboxes[key].set_val(display_text)
            print(f"无效输入 for {key} ('{text}'): {e}. 恢复为: {current_val:.3f}")


    def reset_parameters(self, event):
        """重置参数为默认值并更新UI"""
        self.params = self.default_params.copy()
        for key, textbox in self.textboxes.items():
             try:
                 # --- 重置时，使用辅助方法更新显示 ---
                 display_text = self._format_input_display(key, self.params[key])
                 textbox.set_val(display_text)
             except KeyError:
                 print(f"警告: 重置时未找到键 '{key}' 对应的文本框。")
        print("参数已重置为默认值。")
        self.clear_plot_and_results()

    def run_calculation_and_plot(self, event):
        """执行计算并更新绘图和结果显示"""
        print("开始计算和绘图...")
        results = self._calculate_physics()
        self.update_output_display(results)
        self.update_plot(results)

    def _calculate_physics(self):
        """根据当前参数计算物理量，包含验证"""
        try:
            p = self.params
            d_p = self.default_params
            x1_cm = float(p.get('x1', d_p['x1']))
            x2_cm = float(p.get('x2', d_p['x2']))
            x3_cm = float(p.get('x3', d_p['x3']))
            P1_mm = float(p.get('P1', d_p['P1']))
            P2_mm = float(p.get('P2', d_p['P2']))
            x4_mm = float(p.get('x4', d_p['x4']))
            x5_mm = float(p.get('x5', d_p['x5']))
            b_slit_mm = float(p.get('slit_width', d_p['slit_width']))

            if not (x1_cm < x2_cm < x3_cm):
                raise ValueError(f"位置错误: 必须满足 x₁ ({x1_cm:.1f}) < x₂ ({x2_cm:.1f}) < x₃ ({x3_cm:.1f}) cm。") # Error msg format reflects input format
            if abs(P1_mm - P2_mm) < 1e-9 :
                raise ValueError(f"实像位置 P₁ ({P1_mm:.3f}) 和 P₂ ({P2_mm:.3f}) mm 不能相同。")
            if x5_mm <= x4_mm:
                raise ValueError(f"读数 x₅ ({x5_mm:.3f}) mm 必须大于 x₄ ({x4_mm:.3f}) mm。")
            if b_slit_mm <= 0:
                raise ValueError(f"光源狭缝宽度 b ({b_slit_mm:.3f}) mm 必须为正数。")

            u_cm = x2_cm - x1_cm
            v_cm = x3_cm - x2_cm
            d_prime_mm = abs(P2_mm - P1_mm)
            u_mm = u_cm * 10.0
            v_mm = v_cm * 10.0
            D_mm = (x3_cm - x1_cm) * 10.0

            if abs(v_mm) < 1e-9:
                raise ZeroDivisionError("计算得到的像距 v 接近零 (v ≈ 0)，无法计算虚光源间距 d。检查 x₂ 和 x₃。")
            if abs(D_mm) < 1e-9:
                raise ZeroDivisionError("屏间距 D (x₃-x₁) 计算为零或接近零 (D ≈ 0)，无法计算波长 λ。检查 x₁ 和 x₃。")

            d_mm_calc = d_prime_mm * (u_mm / v_mm)
            if abs(d_mm_calc) < 1e-9:
                print(f"警告: 计算得到的虚光源间距 d = {d_mm_calc:.3e} mm 非常小，可能无明显干涉。")

            delta_x_mm_calc = abs(x5_mm - x4_mm) / 10.0
            if delta_x_mm_calc <= 1e-9:
                 raise ValueError("计算得到的条纹间距 Δx 非正数或过小。检查 x₄ 和 x₅。")

            wavelength_mm = (d_mm_calc * delta_x_mm_calc) / D_mm
            wavelength_nm_calc = wavelength_mm * 1e6

            current_date = datetime.now().strftime("%Y-%m-%d") # Get current date string
            if not (1 < wavelength_nm_calc < 10000):
                print(f"({current_date}) 警告: 计算得到的波长 λ = {wavelength_nm_calc:.2f} nm 超出常规范围。请仔细检查所有输入参数。")
            elif not (380 <= wavelength_nm_calc <= 780):
                print(f"({current_date}) 提示: 计算波长 λ = {wavelength_nm_calc:.2f} nm 不在典型可见光范围 (380-780 nm)。")

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
            print(f"计算错误 (输入值或逻辑): {e}")
            return None
        except ZeroDivisionError as e:
            print(f"计算错误 (除以零): {e}")
            return None
        except Exception as e:
            print(f"发生意外的计算错误: {type(e).__name__} - {e}")
            traceback.print_exc()
            return None

    def update_output_display(self, results):
        """更新计算结果显示区域 (根据要求调整小数位数)"""
        if results:
            # --- 按要求设置不同参数的显示小数位数 ---
            self.output_texts['u_cm'].set_text(f"{results.get('u_cm', 0):.1f}")     # u: 1位小数
            self.output_texts['v_cm'].set_text(f"{results.get('v_cm', 0):.1f}")     # v: 1位小数
            self.output_texts['D_cm'].set_text(f"{results.get('D_cm', 0):.1f}")     # D: 1位小数

            # d', d, Δx 保持3位小数
            if 'd_prime_mm' in self.output_texts:
                 self.output_texts['d_prime_mm'].set_text(f"{results.get('d_prime_mm', 0):.3f}")
            if 'd_mm' in self.output_texts:
                 self.output_texts['d_mm'].set_text(f"{results.get('d_mm', 0):.3f}")
            if 'delta_x_mm' in self.output_texts:
                 self.output_texts['delta_x_mm'].set_text(f"{results.get('delta_x_mm', 0):.3f}")

            # wavelength: 2位小数
            if 'wavelength_nm' in self.output_texts:
                 self.output_texts['wavelength_nm'].set_text(f"{results.get('wavelength_nm', 0):.2f}")
        else:
            for key in self.output_texts:
                 try:
                     self.output_texts[key].set_text("--")
                 except KeyError:
                      print(f"警告: 更新输出显示时未找到键 '{key}' 对应的文本对象。")

        try:
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"在 update_output_display 中重绘画布时出错: {e}")


    def update_plot(self, results):
        """根据计算结果更新 2D 干涉图样，使用固定的 ax_plot 和 cax"""
        self.ax_plot.cla()
        self.cax.cla()
        self.cax.set_visible(False)

        self.ax_plot.set_xlim(*self.PLOT_XLIM)
        self.ax_plot.set_ylim(-self.Y_DISPLAY_RANGE, self.Y_DISPLAY_RANGE)
        self.ax_plot.set_xlabel("屏幕位置 x (mm)")
        self.ax_plot.set_ylabel("")
        self.ax_plot.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        if results is None:
            self.ax_plot.set_title("模拟干涉条纹图样 (参数无效或计算错误)")
            self.ax_plot.text(0.5, 0.5, '参数无效或计算错误\n请检查输入或控制台输出',
                              horizontalalignment='center', verticalalignment='center',
                              transform=self.ax_plot.transAxes, fontsize=12, color='gray', wrap=True)
            try:
                self.fig.canvas.draw_idle()
            except Exception as e:
                 print(f"在 update_plot (错误处理) 中重绘画布时出错: {e}")
            return

        try:
            lambda_mm = results['lambda_mm_plot']
            d_mm = results['d_mm_plot']
            D_mm = results['D_mm_plot']
            b_slit_mm = results['b_slit_mm_plot']

            if abs(D_mm) < 1e-9 or abs(lambda_mm) < 1e-15:
                 raise ValueError("绘图需要非零的屏距 D 和波长 λ。")

            x_screen = np.linspace(self.PLOT_XLIM[0], self.PLOT_XLIM[1], self.NUM_X_POINTS)
            lambda_D = lambda_mm * D_mm
            if abs(lambda_D) < 1e-15:
                 lambda_D = 1e-15

            common_factor = (np.pi * x_screen) / lambda_D

            if abs(b_slit_mm) < 1e-9:
                diffraction_term = np.ones_like(x_screen)
            else:
                alpha = b_slit_mm * common_factor
                with np.errstate(divide='ignore', invalid='ignore'):
                    diffraction_term = np.sinc(alpha / np.pi)**2
                diffraction_term = np.nan_to_num(diffraction_term, nan=0.0)

            if abs(d_mm) < 1e-9:
                interference_term = 0.5
            else:
                beta = d_mm * common_factor
                interference_term = np.cos(beta)**2

            intensity_1d = diffraction_term * interference_term
            intensity_1d = np.maximum(intensity_1d, 0)
            intensity_2d = np.repeat(intensity_1d[np.newaxis, :], self.Y_PIXELS, axis=0)

            extent = [self.PLOT_XLIM[0], self.PLOT_XLIM[1], -self.Y_DISPLAY_RANGE, self.Y_DISPLAY_RANGE]
            cmap = 'gray'
            vmin_val = 0.0
            vmax_val = np.max(intensity_2d)
            if vmax_val <= vmin_val + 1e-9:
                vmax_val = vmin_val + 1e-9

            norm = mcolors.Normalize(vmin=vmin_val, vmax=vmax_val)
            im = self.ax_plot.imshow(intensity_2d, cmap=cmap, norm=norm,
                                     origin='lower', extent=extent,
                                     aspect='auto', interpolation='bilinear')

            # --- 更新标题中的波长格式为2位小数 ---
            title_text = f"模拟干涉图样 (计算值: λ = {results['wavelength_nm']:.2f} nm)"
            self.ax_plot.set_title(title_text)

            self.colorbar = self.fig.colorbar(im, cax=self.cax, label='归一化光强')
            self.cax.set_visible(True)

        except Exception as e:
            print(f"绘图时发生错误: {type(e).__name__} - {e}")
            traceback.print_exc()
            self.ax_plot.set_title("模拟干涉条纹图样 (绘图错误)")
            self.ax_plot.text(0.5, 0.5, f'绘图错误:\n{type(e).__name__}\n请检查计算结果和参数\n或查看控制台输出。',
                              horizontalalignment='center', verticalalignment='center',
                              transform=self.ax_plot.transAxes, fontsize=10, color='red', wrap=True)
            self.cax.set_visible(False)

        try:
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"在 update_plot (完成) 中重绘画布时出错: {e}")

    def clear_plot_and_results(self):
        """清除绘图区和结果显示区"""
        self.update_plot(None)
        self.update_output_display(None)

# --- Main Execution ---
if __name__ == '__main__':
    try:
        sim = FresnelBiprismSim()
    except Exception as main_exception:
        print("\n--- 程序初始化或执行期间发生严重错误 ---")
        traceback.print_exc()
        print("-----------------------------------------------------")
        input("发生错误，按 Enter 键退出...")
        sys.exit(1)