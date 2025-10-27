import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
from tools.dir_setting import Dir


class InteractiveBinarizer:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("交互式区域二值化工具")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)

        # 加载图像
        self.image_path = image_path
        self.img_array = np.fromfile(image_path, dtype=np.uint8)
        self.original_image = cv2.imdecode(self.img_array, cv2.IMREAD_COLOR)
        
        # 添加图像加载验证
        if self.original_image is None:
            messagebox.showerror("错误", f"无法加载图像: {image_path}")
            self.root.destroy()
            return
        
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # 加载erased文件夹中的同名PNG图像
        image_dir = os.path.dirname(self.image_path)
        parent_dir = os.path.dirname(image_dir)
        erased_dir = os.path.join(parent_dir, 'erased')
        filename = os.path.basename(self.image_path)
        name, ext = os.path.splitext(filename)
        erased_filename = f"{name}.png"
        self.erased_image_path = os.path.join(erased_dir, erased_filename)

        try:
            self.erased_img_array = np.fromfile(self.erased_image_path, dtype=np.uint8)
            self.erased_original_image = cv2.imdecode(self.erased_img_array, cv2.IMREAD_COLOR)
            if self.erased_original_image is None:
                raise FileNotFoundError(f"无法解码erased图像: {self.erased_image_path}")
            self.erased_gray_image = cv2.cvtColor(self.erased_original_image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            messagebox.showwarning("警告", f"无法加载erased图像: {str(e)}")
            self.erased_gray_image = None

        # 验证图像尺寸
        h, w = self.gray_image.shape[:2]
        if h <= 0 or w <= 0:
            messagebox.showerror("错误", "图像尺寸无效")
            self.root.destroy()
            return
        
        self.display_image = self.original_image.copy()

        # 初始化十字交叉中心点 (默认图像中心)
        h, w = self.gray_image.shape[:2]
        self.center = (w//2, h//2)  # 单个十字交叉中心点
        self.selected = False
        self.thresholds = [100, 130, 100, 130]  # 四个区域的阈值
        self.regions = []
        
        # 添加缩放功能相关属性
        self.zoom_factor = 1.0  # 默认缩放因子
        self.min_zoom = 0.5     # 最小缩放比例
        self.max_zoom = 4.0     # 最大缩放比例
        
        # 创建UI
        self.create_widgets()
        self.update_regions()
        self.draw_regions()

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 图像显示区域
        self.canvas_frame = ttk.LabelFrame(main_frame, text="图像区域划分", padding="10")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        
        # 添加鼠标滚轮事件绑定（支持Windows和Linux系统）
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows系统
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)   # Linux系统滚轮上滚
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)   # Linux系统滚轮下滚
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="阈值设置", padding="10")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # 阈值输入框
        self.threshold_entries = []
        region_names = ["左上区域", "右上区域", "左下区域", "右下区域"]
        for i, name in enumerate(region_names):
            frame = ttk.Frame(control_frame)
            frame.pack(fill=tk.X, pady=5)

            ttk.Label(frame, text=f"{name}阈值:").pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=5)
            entry.insert(0, str(self.thresholds[i]))
            entry.pack(side=tk.RIGHT)
            self.threshold_entries.append(entry)

        # 按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=20)

        ttk.Button(button_frame, text="应用二值化", command=self.apply_binarization).pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="重置区域", command=self.reset_regions).pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="保存结果", command=self.save_result).pack(fill=tk.X, pady=5)
    
    def on_mouse_wheel(self, event):
        # Handle mouse wheel events for zooming
        # Determine zoom direction (Windows/Linux compatibility)
        if event.num == 4 or event.delta > 0:
            self.zoom_factor *= 1.1
        elif event.num == 5 or event.delta < 0:
            self.zoom_factor /= 1.1
        
        # Limit zoom range
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor))
        self.update_canvas_image()
    def on_click(self, event):
        # 获取画布和图像尺寸信息
        h, w = self.display_image.shape[:2]
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        base_scale = min(canvas_w/w, canvas_h/h) if canvas_w > 0 and canvas_h > 0 else 1
        scale = base_scale * self.zoom_factor  # 应用缩放因子
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 计算图像在画布上的偏移量（居中显示）
        margin_x = (canvas_w - new_w) // 2
        margin_y = (canvas_h - new_h) // 2
        
        # 检查点击是否在图像范围内
        if (margin_x <= event.x <= margin_x + new_w and
            margin_y <= event.y <= margin_y + new_h):
            # 将画布坐标转换为原始图像坐标
            img_x = int((event.x - margin_x) / scale)
            img_y = int((event.y - margin_y) / scale)
            
            # 检查是否点击到十字中心点
            center_x, center_y = self.center
            distance = ((img_x - center_x)**2 + (img_y - center_y)** 2)**0.5
            self.selected = distance < 15  # 15像素范围内视为选中

    def on_drag(self, event):
        # 拖动十字中心点
        if self.selected:
            # 确保display_image是numpy数组
            if not hasattr(self.display_image, 'shape'):
                self.display_image = np.array(self.display_image)  # 补充转换
            h, w = self.display_image.shape[:2]
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            base_scale = min(canvas_w/w, canvas_h/h) if canvas_w > 0 and canvas_h > 0 else 1
            scale = base_scale * self.zoom_factor  # 应用缩放因子
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 计算图像在画布上的偏移量
            margin_x = (canvas_w - new_w) // 2
            margin_y = (canvas_h - new_h) // 2
            
            # 检查拖动是否在图像范围内
            if (margin_x <= event.x <= margin_x + new_w and
                margin_y <= event.y <= margin_y + new_h):
                # 将画布坐标转换为原始图像坐标
                img_x = int((event.x - margin_x) / scale)
                img_y = int((event.y - margin_y) / scale)
                
                # 更新中心点
                self.center = (img_x, img_y)
                self.update_regions()
                self.draw_regions()

    def update_regions(self):
        # 根据十字中心点计算四个区域
        h, w = self.gray_image.shape[:2]
        center_x, center_y = self.center

        # 四个区域坐标 (x, y, width, height)
        self.regions = [
            (0, 0, center_x, center_y),                # 左上区域
            (center_x, 0, w-center_x, center_y),        # 右上区域
            (0, center_y, center_x, h-center_y),        # 左下区域
            (center_x, center_y, w-center_x, h-center_y) # 右下区域
        ]

    def draw_regions(self):
        # 在图像上绘制十字交叉线和区域
        # 将PIL Image转换为numpy数组（假设self.original_image是PIL Image）
        self.display_image = np.array(self.original_image.copy())  # 添加numpy转换
        center_x, center_y = self.center
        h, w = self.display_image.shape[:2]  # 现在可以安全获取shape

        # 绘制十字交叉线（cv2需要numpy数组）
        cv2.line(self.display_image, (center_x, 0), (center_x, h), (0, 255, 255), 2)
        cv2.line(self.display_image, (0, center_y), (w, center_y), (0, 255, 255), 2)

        # 绘制中心点
        cv2.circle(self.display_image, (center_x, center_y), 10, (0, 0, 255), -1)
        cv2.circle(self.display_image, (center_x, center_y), 12, (255, 255, 255), 2)

        # 绘制区域边界
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # 红、绿、蓝、黄
        for (x, y, w, h), color in zip(self.regions, colors):
            cv2.rectangle(self.display_image, (x, y), (x+w, y+h), color, 2)

        # 更新画布
        self.update_canvas_image()

    def update_canvas_image(self):
        # 将OpenCV图像转换为Tkinter可用格式
        h, w = self.display_image.shape[:2]
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()

        # 保持纵横比缩放 - 结合缩放因子
        base_scale = min(canvas_w/w, canvas_h/h) if canvas_w > 0 and canvas_h > 0 else 1
        scale = base_scale * self.zoom_factor  # 应用缩放因子
        # 确保尺寸至少为1像素
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(self.display_image, (new_w, new_h))
        bgr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(bgr)
        self.tk_img = ImageTk.PhotoImage(image=pil_img)

        # 显示图像
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=self.tk_img, anchor=tk.CENTER)

    def apply_binarization(self):
        # 获取阈值（原有逻辑不变）
        try:
            thresholds = [int(entry.get()) for entry in self.threshold_entries]
            for t in thresholds:
                if not (0 <= t <= 255):
                    raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "阈值必须是0-255之间的整数")
            return

        # 应用二值化（保持原有逻辑）
        result = np.zeros_like(self.gray_image)
        for (x, y, w, h), threshold in zip(self.regions, thresholds):
            region = self.gray_image[y:y+h, x:x+w]
            _, binary = cv2.threshold(region, threshold, 255, cv2.THRESH_BINARY)
            result[y:y+h, x:x+w] = binary
        self.result_image = result

        # 移除覆盖主窗口的代码（关键修改）
        # 原代码：result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        # 原代码：self.display_image = result_bgr
        # 原代码：self.update_canvas_image()

        # 处理erased图像（原有逻辑不变）
        if self.erased_gray_image is not None:
            self.erased_result_image = np.zeros_like(self.erased_gray_image)
            for (x, y, w, h), threshold in zip(self.regions, thresholds):
                region = self.erased_gray_image[y:y+h, x:x+w]
                _, binary = cv2.threshold(region, threshold, 255, cv2.THRESH_BINARY)
                self.erased_result_image[y:y+h, x:x+w] = binary
        else:
            self.erased_result_image = None

        # 显示结果到预览窗口（原有逻辑优化）
        self.create_result_window(result)  # 确保预览窗口显示二值化结果

    def create_result_window(self, result):
        """创建带滚动条和缩放功能的结果预览窗口"""
        # 创建顶层窗口
        self.result_window = tk.Toplevel()
        self.result_window.title("二值化结果预览")
        self.result_window.geometry("800x600")  # 固定窗口大小

        # 创建画布和滚动条
        self.result_canvas = tk.Canvas(self.result_window, bg="white")
        h_scroll = ttk.Scrollbar(self.result_window, orient="horizontal", command=self.result_canvas.xview)
        v_scroll = ttk.Scrollbar(self.result_window, orient="vertical", command=self.result_canvas.yview)

        # 布局组件
        self.result_canvas.pack(side="left", fill="both", expand=True)
        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right", fill="y")

        # 配置画布滚动
        self.result_canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        # 转换OpenCV图像为PIL格式（保持原有逻辑）
        pil_img = Image.fromarray(result)
        self.preview_image = pil_img  # 保存预览图像到新变量

        # 保存图像尺寸（关键修改：使用预览图像尺寸）
        self.preview_width, self.preview_height = pil_img.size  # 正确获取二值化结果尺寸

        # 计算缩放比例以适应窗口（关键修改：使用预览图像尺寸）
        canvas_width, canvas_height = 800, 600
        self.scale_factor = min(canvas_width/self.preview_width, canvas_height/self.preview_height)  # 基于预览尺寸计算

        # 缩放图像以适应初始窗口（关键修改：使用预览尺寸）
        new_width = int(self.preview_width * self.scale_factor)
        new_height = int(self.preview_height * self.scale_factor)
        resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        self.result_tk_img = ImageTk.PhotoImage(resized_img)

        # 计算图像在画布中的居中位置（保持原有逻辑）
        x_center = (800 - new_width) // 2
        y_center = (600 - new_height) // 2

        # 显示图像并设置滚动范围（关键修改：使用预览图像实际尺寸）
        self.result_canvas.create_image(x_center, y_center, anchor="nw", image=self.result_tk_img)
        self.result_canvas.configure(scrollregion=(0, 0, max(800, self.preview_width), max(600, self.preview_height)))
        
        # 初始化当前缩放级别
        self.current_scale = self.scale_factor

        # 绑定鼠标滚轮事件（可选缩放功能）
        self.result_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.result_canvas.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        self.result_canvas.bind("<Button-4>", self._on_mousewheel)  # Linux support
        self.result_canvas.bind("<Button-5>", self._on_mousewheel)  # Linux support
        
        # 绑定鼠标拖动事件
        self.result_canvas.bind("<ButtonPress-1>", self._on_button_press)
        self.result_canvas.bind("<B1-Motion>", self._on_move_press)
        
        self.result_canvas.image = self.result_tk_img  # 保持引用防止被垃圾回收

    def _on_mousewheel(self, event):
        # 垂直滚动
        if event.num == 4 or event.delta > 0:
            self.result_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.result_canvas.yview_scroll(1, "units")
    
    def _on_ctrl_wheel(self, event):
        # 强制使用窗口中心坐标
        canvas_width = self.result_canvas.winfo_width()
        canvas_height = self.result_canvas.winfo_height()
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        scale_multiplier = 1.1 if event.delta > 0 else 0.9
        self.current_scale = max(0.1, min(self.current_scale * scale_multiplier, 5.0))
        
        self._resize_image(center_x, center_y, scale_multiplier)
    
    def _resize_image(self, center_x, center_y, scale_multiplier):
        # 计算新尺寸
        new_width = int(self.img_width * self.current_scale)
        new_height = int(self.img_height * self.current_scale)
        
        # 获取画布实际尺寸
        canvas_width = self.result_canvas.winfo_width()
        canvas_height = self.result_canvas.winfo_height()
        
        # 计算图像中心点坐标
        img_center_x = new_width / 2
        img_center_y = new_height / 2
        
        # 设置自适应滚动区域
        scrollregion = (
            min(0, (canvas_width - new_width) // 2),
            min(0, (canvas_height - new_height) // 2),
            max(new_width, canvas_width),
            max(new_height, canvas_height)
        )
        self.result_canvas.configure(scrollregion=scrollregion)
        
        # 更新图像显示
        resized_img = self.original_image.resize((new_width, new_height), Image.LANCZOS)
        self.result_tk_img = ImageTk.PhotoImage(resized_img)
        self.result_canvas.delete("all")
        self.result_canvas.create_image(0, 0, anchor="nw", image=self.result_tk_img)
        
        # 计算精确居中位置
        if new_width > canvas_width:
            x_ratio = (img_center_x - canvas_width/2) / new_width
        else:
            x_ratio = (canvas_width/2 - img_center_x) / canvas_width
        
        if new_height > canvas_height:
            y_ratio = (img_center_y - canvas_height/2) / new_height
        else:
            y_ratio = (canvas_height/2 - img_center_y) / canvas_height
        
        # 应用视图位置
        self.result_canvas.xview_moveto(max(0, min(x_ratio, 1)))
        self.result_canvas.yview_moveto(max(0, min(y_ratio, 1)))
        
        # 保持图像引用
        self.result_canvas.image = self.result_tk_img
    
    def _on_button_press(self, event):
        # 记录鼠标按下位置
        self.result_canvas.scan_mark(event.x, event.y)
    
    def _on_move_press(self, event):
        # 实现拖动功能
        self.result_canvas.scan_dragto(event.x, event.y, gain=1)

    def save_result(self):
        if not hasattr(self, 'result_image') or self.result_image is None:
            messagebox.showwarning("警告", "请先应用二值化并确保处理成功")
            return

        try:
            dir = Dir()
            # 检查Dir类是否包含所需目录属性
            if not hasattr(dir, 'images_binary_dir'):
                raise AttributeError("Dir类缺少images_binary_dir属性")
            if not hasattr(dir, 'erased_binary_dir'):
                raise AttributeError("Dir类缺少erased_binary_dir属性")

            # 主二值化结果保存到image_binary文件夹
            main_output_dir = dir.images_binary_dir  # 无需硬编码绝对路径
            # 擦除结果保存到erased_binary文件夹
            erased_output_dir = dir.erased_binary_dir

            # 验证目录存在性（exist_ok=True兼容未创建的情况）
            os.makedirs(main_output_dir, exist_ok=True)
            os.makedirs(erased_output_dir, exist_ok=True)

            filename = os.path.basename(self.image_path)
            name, ext = os.path.splitext(filename)
            # 统一使用原图像名称.png作为文件名
            common_name = f"{name}.png"

            # 主二值化结果路径（image_binary/原名称.png）
            main_output_path = os.path.join(main_output_dir, common_name)

            # 尝试保存主二值化结果
            success = cv2.imwrite(main_output_path, self.result_image)
            if not success:
                # 回退使用PIL处理中文路径
                from PIL import Image
                img = Image.fromarray(self.result_image)
                img.save(main_output_path)

            # 保存擦除结果（若存在，路径为erased_binary/原名称.png）
            if self.erased_result_image is not None:
                erased_output_path = os.path.join(erased_output_dir, common_name)
                success_erased = cv2.imwrite(erased_output_path, self.erased_result_image)
                if not success_erased:
                    from PIL import Image
                    img_erased = Image.fromarray(self.erased_result_image)
                    img_erased.save(erased_output_path)

            # 更新成功提示信息
            success_msg = f"主结果已保存至:\n{main_output_path}"
            if self.erased_result_image is not None:
                success_msg += f"\n擦除结果已保存至:\n{erased_output_path}"
            messagebox.showinfo("成功", success_msg)

        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存图像: {str(e)}")
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存图像: {str(e)}\n请检查路径权限或输出目录设置")

    def reset_regions(self):
        # 重置中心点
        h, w = self.gray_image.shape[:2]
        self.centers = [
            (w//4, h//4),
            (3*w//4, h//4),
            (w//4, 3*h//4),
            (3*w//4, 3*h//4)
        ]
        self.update_regions()
        self.draw_regions()


if __name__ == '__main__':
    dir = Dir()
    dir.validate_paths()

    # 获取图像路径 (这里可以修改为命令行参数或文件选择对话框)
    image_filename = '10.jpg'  # 图像文件名
    image_path = os.path.join(dir.images_dir, image_filename)

    if not os.path.exists(image_path):
        messagebox.showerror("文件不存在", f"图像文件不存在:\n{image_path}")
    else:
        root = tk.Tk()
        app = InteractiveBinarizer(root, image_path)
        root.mainloop()