import pandas as pd
import pydeck as pdk

# 读取数据
df = pd.read_csv('LLMobData.csv')

# 创建 HeatmapLayer
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    df,
    get_position=["longitude", "latitude"],
    opacity=0.8,
    radius=500,        # 增大半径，让热力分布更广泛
    intensity=0.3,     # 降低强度，减少高密度区域的峰值
    threshold=0.005,     # 降低阈值，让更多低密度区域也能显示
    color_range=[
        [252, 146, 114],  # 粉红
        [251, 106, 74],   # 橙红
        [222, 45, 38],    # 红色
        [165, 15, 21],    # 深红（高密度）
    ],
)

# 计算视图状态
view_state = pdk.data_utils.compute_view(df.loc[:, ["longitude", "latitude"]])
view_state.pitch = 0
view_state.zoom = 10

# 创建 Deck
r = pdk.Deck(
    layers=[heatmap_layer],
    initial_view_state=view_state,
    map_style=pdk.map_styles.LIGHT,
    tooltip={"text": "位置: ({longitude}, {latitude})"},
)

# 导出为 HTML
r.to_html("heatmap_layer.html")
print("热力图已生成: heatmap_layer.html")
