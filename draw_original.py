import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
sns.set_theme(style="darkgrid", font="Times New Roman")

df = pd.read_csv('LLMobData.csv')
df.describe()

df["lat3"] = df["latitude"].round(3)
df["lat2"] = df["latitude"].round(2)
df["lng3"] = df["longitude"].round(3)
df["lng2"] = df["longitude"].round(2)


layer = pdk.Layer(
    "HexagonLayer",
    df.loc[:, :],
    get_position=["lng2", "lat2"],
    auto_highlight=True,
    elevation_scale=20,
    pickable=True,
    elevation_range=[0, 2000],
    extruded=True,
    coverage=1,
)

COLOR_BREWER_BLUE_SCALE = [
    [240, 249, 232],
    [204, 235, 197],
    [168, 221, 181],
    [123, 204, 196],
    [67, 162, 202],
    [8, 104, 172],
]

print(df.head(5))
view_state = pdk.data_utils.compute_view(df.loc[:, ["longitude", "latitude"]])
view_state.pitch = 55
r = pdk.Deck(layers=[layer], initial_view_state=view_state,
             map_style=pdk.map_styles.LIGHT)   
r.to_html("hexagon_layer.html")

