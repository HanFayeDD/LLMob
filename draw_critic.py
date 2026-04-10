import seaborn as sns
sns.set_theme(style="darkgrid", font="Times New Roman")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('fakedata.csv')

# ── 图1: API 调用次数纵向柱状图 (原有逻辑) ────────────────────────────
g = sns.catplot(
    data=df, kind="bar",
    x="Model", y="API Invocation Count(Real)", hue="Mode",
    palette="dark", alpha=.6,
)

for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontsize=13, padding=2)

g.savefig('critic1.png', dpi=1000)
plt.close('all')

# ── 图2: 发散纵向柱状图 (↑成功  ↓失败) ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

models = df['Model'].unique()
modes = df['Mode'].unique()
n_models = len(models)
n_modes = len(modes)
bar_width = 0.35
x_positions = np.arange(n_models)

palette = sns.color_palette("dark", n_colors=n_modes)

for i, mode in enumerate(modes):
    subset = df[df['Mode'] == mode]
    x_pos = x_positions + i * bar_width

    # 成功 (上方, 正值)
    success_vals = [row['Success Traj Cnt'] for _, row in subset.iterrows()]
    bars_s = ax.bar(x_pos, success_vals, width=bar_width,
                    color=palette[i], alpha=0.6, label=f'{mode}')
    # 失败 (下方, 负值)
    fail_vals = [-row['Fail Traj Cnt'] for _, row in subset.iterrows()]
    bars_f = ax.bar(x_pos, fail_vals, width=bar_width,
                    color=palette[i], alpha=0.6)

    # 数值标签
    for bar, val in zip(bars_s, success_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha='center', va='bottom', fontsize=11,
                fontweight='bold', color='black')
    for bar, val in zip(bars_f, fail_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() - 2,
            str(abs(val)), ha='center', va='top', fontsize=11,
            fontweight='bold', color='black'
        )

# x 轴  
ax.set_xticks(x_positions + bar_width * (n_modes - 1) / 2)
ax.set_xticklabels(models, fontsize=12)

# y 轴: 上方保留成功范围，下方限制到 ~200
top_lim = ax.get_ylim()[1] * 1.10
ax.set_ylim(-200, top_lim)

ticks = ax.get_yticks()
ax.set_yticklabels([str(int(abs(t))) for t in ticks])
ax.set_ylabel('Trajectory Count', fontsize=13)

# 中线
ax.axhline(0, color='black', linewidth=0.8)

# 上下标题
ax.text(ax.get_xlim()[1] + 0.05, top_lim * 0.5, '↑ Success',
        ha='left', va='center', fontsize=13, fontweight='bold')
ax.text(ax.get_xlim()[1] + 0.05, -100, '↓ Fail',
        ha='left', va='center', fontsize=13, fontweight='bold')

# 图例放到下方空白区域（失败柱子短，下方空间大）
ax.legend(title='Mode', fontsize=11, title_fontsize=12,
          loc='lower right')
fig.tight_layout()
fig.savefig('critic_success_fail.png', dpi=1000)
plt.close('all')