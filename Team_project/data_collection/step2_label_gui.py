#-----------------------------------------------------------------------------#
#                     Step 2 - 交互式轨迹标注 GUI                              #
#                                                                             #
#  跑完 Step 1 后执行本脚本：                                                  #
#    python data_collection/step2_label_gui.py                                #
#                                                                             #
#  操作方法：                                                                  #
#    1. 脚本自动读取 dataset/raw/coords.csv，画出QBot行驶轨迹                   #
#    2. 右侧面板选择标签类型（crossroad / corner / straight）                   #
#    3. 在轨迹图上按住左键拖拽，画出圆形区域                                     #
#    4. 重复直到标注完所有关键区域                                               #
#    5. 点击"保存区域定义"，区域保存到 dataset/raw/regions.json                 #
#    6. Step 3 读取这个 JSON 批量打标签                                         #
#-----------------------------------------------------------------------------#

import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # 支持交互；如果报错换成 'Qt5Agg'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons
from matplotlib.patches import Circle

# ── 路径配置 ──────────────────────────────────────────────────────────────────
COORDS_CSV  = os.path.join("dataset", "raw", "coords.csv")
REGIONS_JSON = os.path.join("dataset", "raw", "regions.json")

# ── 颜色配置 ──────────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "crossroad": "#FF4444",
    "corner":    "#FFAA00",
    "straight":  "#44BB44",
}
DEFAULT_RADIUS = 0.12   # 默认圆形区域半径（米）

# =============================================================================

def load_coords(csv_path):
    """读取 coords.csv，返回 (image_names, xs, ys) 三个列表"""
    names, xs, ys = [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                names.append(row["image_name"])
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
            except (ValueError, KeyError):
                pass
    return names, np.array(xs), np.array(ys)


class RegionLabeler:
    def __init__(self):
        # ---- Load data ----
        if not os.path.isfile(COORDS_CSV):
            raise FileNotFoundError(
                f"Cannot find {COORDS_CSV}. Please run Step 1 first.")
        self.names, self.xs, self.ys = load_coords(COORDS_CSV)
        print(f"[Step2] Loaded {len(self.names)} coordinate records")

        # ---- Load existing regions (resume support) ----
        if os.path.isfile(REGIONS_JSON):
            with open(REGIONS_JSON, "r") as f:
                self.regions = json.load(f)
            print(f"[Step2] Loaded {len(self.regions)} existing regions")
        else:
            self.regions = []   # list of {"label": str, "cx": float, "cy": float, "r": float}

        # ---- State ----
        self.current_label  = "crossroad"
        self.drag_start     = None      # (x, y) mouse press position
        self.preview_circle = None      # preview circle object
        self.radius         = DEFAULT_RADIUS

        self._build_ui()

    # ── UI Build ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.fig = plt.figure(figsize=(13, 8), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("Step 2 - Region Labeling Tool")

        # Main trajectory axis
        self.ax = self.fig.add_axes([0.05, 0.12, 0.65, 0.82])
        self.ax.set_facecolor("#0d0d1a")
        self.ax.set_title("QBot Trajectory  |  Click and drag to draw circular regions",
                          color="white", fontsize=11, pad=10)
        self.ax.tick_params(colors="gray")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#444")

        # Draw trajectory scatter
        self.ax.scatter(self.xs, self.ys, s=2, c="#4488ff",
                        alpha=0.5, zorder=2, label="trajectory")
        # Start / end markers
        if len(self.xs) > 0:
            self.ax.plot(self.xs[0],  self.ys[0],  "g^", ms=10, label="start", zorder=5)
            self.ax.plot(self.xs[-1], self.ys[-1], "rs", ms=10, label="end",   zorder=5)
        self.ax.legend(loc="upper right", facecolor="#222", labelcolor="white",
                       fontsize=8, framealpha=0.7)
        self.ax.set_xlabel("X (m)", color="gray")
        self.ax.set_ylabel("Y (m)", color="gray")
        self.ax.set_aspect("equal")
        self.ax.grid(True, color="#222244", linewidth=0.5)

        # Redraw existing regions
        for r in self.regions:
            self._draw_region(r, permanent=True)

        # ---- Right-side controls ----
        # Label selector
        ax_radio = self.fig.add_axes([0.73, 0.60, 0.24, 0.28],
                                     facecolor="#111122")
        self.radio = RadioButtons(
            ax_radio,
            labels=["crossroad", "corner", "straight"],
            activecolor="white"
        )
        for label in self.radio.labels:
            label.set_color(LABEL_COLORS[label.get_text()])
            label.set_fontsize(11)
        self.radio.on_clicked(self._on_label_change)
        ax_radio.set_title("Label Type", color="white", fontsize=10, pad=6)

        # Instructions
        self.ax_info = self.fig.add_axes([0.73, 0.46, 0.24, 0.12],
                                         facecolor="#111122")
        self.ax_info.axis("off")
        self._update_info_text()

        # Undo button
        ax_undo = self.fig.add_axes([0.73, 0.34, 0.11, 0.07])
        self.btn_undo = Button(ax_undo, "Undo", color="#332222", hovercolor="#553333")
        self.btn_undo.label.set_color("white")
        self.btn_undo.on_clicked(self._on_undo)

        # Clear button
        ax_clear = self.fig.add_axes([0.86, 0.34, 0.11, 0.07])
        self.btn_clear = Button(ax_clear, "Clear", color="#332222", hovercolor="#553333")
        self.btn_clear.label.set_color("white")
        self.btn_clear.on_clicked(self._on_clear)

        # Save button
        ax_save = self.fig.add_axes([0.73, 0.22, 0.24, 0.09])
        self.btn_save = Button(ax_save, "Save Regions", color="#113322", hovercolor="#225533")
        self.btn_save.label.set_color("#88ffbb")
        self.btn_save.label.set_fontsize(11)
        self.btn_save.on_clicked(self._on_save)

        # Stats panel
        self.ax_stat = self.fig.add_axes([0.73, 0.05, 0.24, 0.15],
                                         facecolor="#111122")
        self.ax_stat.axis("off")
        self._update_stat_text()

        # ---- 鼠标事件 ----
        self.fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_drag)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("scroll_event",         self._on_scroll)

        plt.show()

    # ── 鼠标事件 ────────────────────────────────────────────────────────────────
    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self.drag_start = (event.xdata, event.ydata)

    def _on_drag(self, event):
        if self.drag_start is None or event.inaxes != self.ax:
            return
        cx, cy = self.drag_start
        r = np.hypot(event.xdata - cx, event.ydata - cy)
        r = max(r, 0.02)
        if self.preview_circle:
            self.preview_circle.remove()
        color = LABEL_COLORS[self.current_label]
        self.preview_circle = Circle(
            (cx, cy), r,
            fill=True, alpha=0.25, color=color,
            linestyle="--", linewidth=1.5, zorder=6
        )
        self.ax.add_patch(self.preview_circle)
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self.drag_start is None or event.inaxes != self.ax or event.button != 1:
            self.drag_start = None
            return
        cx, cy = self.drag_start
        r = np.hypot(event.xdata - cx, event.ydata - cy)
        r = max(r, 0.02)
        self.drag_start = None
        if self.preview_circle:
            self.preview_circle.remove()
            self.preview_circle = None

        region = {"label": self.current_label, "cx": cx, "cy": cy, "r": r}
        self.regions.append(region)
        self._draw_region(region, permanent=True)
        self._update_stat_text()
        self.fig.canvas.draw_idle()
        print(f"[Step2] 新增区域: {region}")

    def _on_scroll(self, event):
        """滚轮缩放视图"""
        if event.inaxes != self.ax:
            return
        scale = 0.9 if event.button == "up" else 1.1
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xm, ym = event.xdata, event.ydata
        self.ax.set_xlim([xm + (x - xm)*scale for x in xlim])
        self.ax.set_ylim([ym + (y - ym)*scale for y in ylim])
        self.fig.canvas.draw_idle()

    # ── 绘图辅助 ────────────────────────────────────────────────────────────────
    def _draw_region(self, region, permanent=False):
        color = LABEL_COLORS[region["label"]]
        circle = Circle(
            (region["cx"], region["cy"]), region["r"],
            fill=True, alpha=0.30, color=color,
            linewidth=2, zorder=4
        )
        self.ax.add_patch(circle)
        self.ax.text(
            region["cx"], region["cy"],
            region["label"],
            color="white", fontsize=7,
            ha="center", va="center", zorder=7,
            fontweight="bold"
        )

    # ── 按钮回调 ────────────────────────────────────────────────────────────────
    def _on_label_change(self, label):
        self.current_label = label

    def _on_undo(self, event):
        if not self.regions:
            return
        self.regions.pop()
        self._refresh_all_regions()
        self._update_stat_text()
        print("[Step2] Removed last region")

    def _on_clear(self, event):
        self.regions.clear()
        self._refresh_all_regions()
        self._update_stat_text()
        print("[Step2] Cleared all regions")

    def _on_save(self, event):
        os.makedirs(os.path.dirname(REGIONS_JSON), exist_ok=True)
        with open(REGIONS_JSON, "w") as f:
            json.dump(self.regions, f, indent=2)
        print(f"[Step2] Regions saved to {REGIONS_JSON}  ({len(self.regions)} regions)")
        self.btn_save.color = "#336633"
        self.fig.canvas.draw_idle()

    # ── Refresh / Stats ─────────────────────────────────────────────────────────
    def _refresh_all_regions(self):
        for patch in list(self.ax.patches):
            patch.remove()
        for txt in list(self.ax.texts):
            txt.remove()
        for region in self.regions:
            self._draw_region(region, permanent=True)
        self.fig.canvas.draw_idle()

    def _update_stat_text(self):
        self.ax_stat.clear()
        self.ax_stat.axis("off")
        counts = {k: 0 for k in LABEL_COLORS}
        for r in self.regions:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        lines = ["Region stats:"]
        for label, cnt in counts.items():
            lines.append(f"  {label}: {cnt}")
        lines.append(f"\nTotal: {len(self.regions)} regions")
        self.ax_stat.text(0.05, 0.95, "\n".join(lines),
                          transform=self.ax_stat.transAxes,
                          color="white", fontsize=9,
                          va="top", family="monospace")
        self.fig.canvas.draw_idle()

    def _update_info_text(self):
        self.ax_info.clear()
        self.ax_info.axis("off")
        msg = ("Controls:\n"
               " Drag left btn -> draw region\n"
               " Scroll wheel  -> zoom\n"
               " Undo          -> remove last\n")
        self.ax_info.text(0.05, 0.95, msg,
                          transform=self.ax_info.transAxes,
                          color="#aaaaaa", fontsize=8,
                          va="top", family="monospace")


# =============================================================================
if __name__ == "__main__":
    labeler = RegionLabeler()
