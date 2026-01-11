## TODO
- add record done 
- change critic prompt way done
- 命令行设置对比实验（默认参数）done
- init role prompt done
- llm as judge done
- 绘制概率分布
- 分箱参数
- DARD和STVD的数字化很奇怪

## 对比试验
- 有无critic done
- 推荐地点算法
- 保存pattern中间结果，保证实验可复现性质 done

## eval
### SD
![alt text](<SD Distribute (deplot).png>)
### SI
![alt text](<SI Distribute (deplot).png>)
### DARD
![alt text](<DARD Distribute (deplot).png>)
### STVD
![alt text](<STVD Distribute (deplot).png>)

```bash
git clone https://github.com/Wangjw6/LLMob.git
cd LLMob
conda env create -f environment.yml
conda activate llm
# Run the LLMob agent to generate 2019 data then evaluate, mode 0 for learning based retrieval, 1 for evolving based retrieval
python generate.py --dataset 2019 --mode 1 
python evaluate.py --dataset 2019 --mode 1 

# Run the LLMob agent to generate 2021 data then evaluate, mode 0 for learning based retrieval, 1 for evolving based retrieval
python generate.py --dataset 2021 --mode 1 
python evaluate.py --dataset 2021 --mode 1 

# Run the LLMob agent to generate 2021 data based on 2019 data then evaluate, mode 0 for learning based retrieval, 1 for evolving based retrieval
python generate.py --dataset 20192021 --mode 1 
python evaluate.py --dataset 20192021 --mode 1 
```

## now draw
- DARD v2
- STVD v2 v3

## 实验结果
### v1
#### gemini
- critic
  llm_e: SD: 0.0348, SI: 0.0415, DARD: 0.2282, STVD: 0.5021
- no critic 
  llm_e: SD: 0.0399, SI: 0.0647, DARD: 0.3043, STVD: 0.5509
#### gpt
- critic 
  llm_e: SD: 0.0304, SI: 0.0451, DARD: 0.2750, STVD: 0.5474
- no critic 
  llm_e: SD: 0.0234, SI: 0.0621, DARD: 0.3136, STVD: 0.5696
#### deepseek
- critic
  llm_e: SD: 0.0287, SI: 0.0446, DARD: 0.2633, STVD: 0.5328
- no critic 
  llm_e: SD: 0.0416, SI: 0.0609, DARD: 0.2885, STVD: 0.5598

### v2
#### gemini
- critic
  llm_e: SD: 0.0348, SI: 0.0415, DARD: 0.2282, STVD: 0.1688
- no critic 
  llm_e: SD: 0.0399, SI: 0.0647, DARD: 0.3043, STVD: 0.2390
#### gpt
- critic 
  llm_e: SD: 0.0304, SI: 0.0451, DARD: 0.2750, STVD: 0.2141
- no critic 
  llm_e: SD: 0.0234, SI: 0.0621, DARD: 0.3136, STVD: 0.2544

#### deepseek
- critic
  llm_e: SD: 0.0287, SI: 0.0446, DARD: 0.2633, STVD: 0.2071
- no critic 
  llm_e: SD: 0.0416, SI: 0.0609, DARD: 0.2885, STVD: 0.2345


## JS散度Q1
我有从两个不同的一维概率分布中抽样得到的样本点。样本点个数不同，怎样计算JS散度
*Thinking...*

> **Examining the User's Input**
> 
> I'm starting to deconstruct the user's request. The core seems to involve comparing two datasets, each with a varying number of samples. This immediately suggests a need to account for potential statistical biases introduced by the unequal sample sizes. I will need to consider metrics or methods robust to this difference.
> 
> 
> **Calculating JS Divergence**
> 
> I'm now zeroing in on the Jensen-Shannon Divergence calculation, a critical step. My focus is on how to accurately compute this metric. Specifically, I'm thinking about the most efficient method for handling the probability distributions derived from the sample points. I will need to consider the context of Chinese language instructions.
> 
> 
计算两个不同样本集（样本数量不同）之间的 Jensen-Shannon (JS) 散度是一个常见的统计问题。由于 JS 散度是基于概率分布（而非原始样本）定义的，因此核心挑战在于**如何从离散的样本点估计出概率分布**。

以下是完整的推理步骤和计算方法：

### 1. 理解 JS 散度的定义

JS 散度是 KL 散度（Kullback-Leibler Divergence）的对称平滑版本。对于两个概率分布 $P$ 和 $Q$，JS 散度定义为：

$$
JSD(P || Q) = \frac{1}{2} D_{KL}(P || M) + \frac{1}{2} D_{KL}(Q || M)
$$

其中：
*   $M = \frac{1}{2}(P + Q)$ 是平均分布。
*   $D_{KL}(P || Q) = \sum P(x) \log \left( \frac{P(x)}{Q(x)} \right)$ （对于离散情况）。

**难点：** 你手头只有样本点 $X = \{x_1, ..., x_n\}$ 和 $Y = \{y_1, ..., y_m\}$，而不是概率分布 $P$ 和 $Q$。

### 2. 解决方案：从样本估计分布

要计算 JS 散度，必须先将样本转换为在**相同支撑集（Support）**上的概率向量。主要有两种方法：

#### 方法一：直方图离散化 (Binning / Histogram Method) —— 最常用且稳健

这是最直接的方法。你需要定义一组统一的“桶”（Bins），然后统计两个样本落在每个桶里的频率。

**步骤：**

1.  **确定范围**：找到两个样本集 $X$ 和 $Y$ 的全局最小值和最大值。
    *   $Min = \min(\min(X), \min(Y))$
    *   $Max = \max(\max(X), \max(Y))$
2.  **划分桶 (Bins)**：在这个范围内划分 $k$ 个桶（例如 50 个或 100 个）。桶的划分必须对两个样本集完全一致。
3.  **计算频率**：
    *   统计 $X$ 中落入每个桶的数量，除以 $X$ 的总数，得到概率向量 $P = [p_1, ..., p_k]$。
    *   统计 $Y$ 中落入每个桶的数量，除以 $Y$ 的总数，得到概率向量 $Q = [q_1, ..., q_k]$。
4.  **处理零概率**：如果某个桶的概率为 0，计算 KL 散度时会出现 $\log(0)$ 错误。通常需要加一个极小的平滑值（如 $\epsilon = 1e^{-10}$）并重新归一化。
5.  **计算公式**：代入 JS 散度公式计算。

#### 方法二：核密度估计 (Kernel Density Estimation, KDE) —— 适用于连续且平滑的数据

如果你认为数据来自连续分布，使用 KDE 可以得到更平滑的概率密度函数 (PDF)。

**步骤：**

1.  分别对 $X$ 和 $Y$ 进行核密度估计，得到函数 $f_X(t)$ 和 $f_Y(t)$。
2.  在统一的网格点（Grid Points）上对这两个函数进行采样，得到离散的概率向量 $P$ 和 $Q$。
3.  代入 JS 散度公式。

---

### 3. Python 代码实现

下面提供一个完整的 Python 实现，包含**直方图法**（推荐）和**KDE法**。你可以直接运行此代码。

```python
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def compute_js_divergence_histogram(sample_p, sample_q, num_bins=50):
    """
    使用直方图法计算两个不同大小样本集的 JS 散度
    """
    # 1. 确定统一的边界
    min_val = min(np.min(sample_p), np.min(sample_q))
    max_val = max(np.max(sample_p), np.max(sample_q))
    
    # 2. 生成统一的桶 (Bins)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    # 3. 计算直方图（频率）
    p_hist, _ = np.histogram(sample_p, bins=bins, density=True)
    q_hist, _ = np.histogram(sample_q, bins=bins, density=True)
    
    # 注意：np.histogram 的 density=True 返回的是密度，乘以 bin 宽度才是概率
    # 但由于 JS 散度比较的是相对比例，且 bin 宽度一致，直接归一化成和为 1 的向量即可
    p_prob = p_hist / np.sum(p_hist)
    q_prob = q_hist / np.sum(q_hist)
    
    # 4. 处理零概率 (平滑处理，避免 log(0))
    # 实际上 scipy.spatial.distance.jensenshannon 会自动处理 log(0) 为无穷大，
    # 但为了数值稳定性，通常加上一个极小值 epsilon
    epsilon = 1e-10
    p_prob = p_prob + epsilon
    q_prob = q_prob + epsilon
    
    # 重新归一化
    p_prob = p_prob / np.sum(p_prob)
    q_prob = q_prob / np.sum(q_prob)
    
    # 5. 计算 JS 散度 (base=2 时单位为 bit，默认 base=e 单位为 nat)
    # scipy 返回的是 JS 距离 (JS 散度的平方根)，所以需要平方
    js_dist = jensenshannon(p_prob, q_prob, base=2)
    js_divergence = js_dist ** 2
    
    return js_divergence

def compute_js_divergence_kde(sample_p, sample_q, num_points=1000):
    """
    使用核密度估计 (KDE) 法计算 JS 散度
    """
    # 1. 训练 KDE 模型
    kde_p = gaussian_kde(sample_p)
    kde_q = gaussian_kde(sample_q)
    
    # 2. 定义统一的评估范围
    min_val = min(np.min(sample_p), np.min(sample_q))
    max_val = max(np.max(sample_p), np.max(sample_q))
    # 稍微扩展范围以覆盖尾部
    range_width = max_val - min_val
    x_grid = np.linspace(min_val - 0.1*range_width, max_val + 0.1*range_width, num_points)
    
    # 3. 在网格点上评估 PDF
    p_pdf = kde_p(x_grid)
    q_pdf = kde_q(x_grid)
    
    # 4. 归一化为概率向量
    p_prob = p_pdf / np.sum(p_pdf)
    q_prob = q_pdf / np.sum(q_pdf)
    
    # 5. 计算 JS 散度
    js_dist = jensenshannon(p_prob, q_prob, base=2)
    js_divergence = js_dist ** 2
    
    return js_divergence

# --- 示例用法 ---

# 生成两个不同大小的样本
np.random.seed(42)
# 样本 P: 正态分布 N(0, 1), 1000 个点
sample_P = np.random.normal(0, 1, 1000)
# 样本 Q: 正态分布 N(1, 1.5), 500 个点 (数量不同)
sample_Q = np.random.normal(1, 1.5, 500)

# 计算
js_hist = compute_js_divergence_histogram(sample_P, sample_Q)
js_kde = compute_js_divergence_kde(sample_P, sample_Q)

print(f"样本 P 数量: {len(sample_P)}")
print(f"样本 Q 数量: {len(sample_Q)}")
print("-" * 30)
print(f"JS 散度 (直方图法): {js_hist:.6f}")
print(f"JS 散度 (KDE 法):   {js_kde:.6f}")

# 理论参考：JS 散度范围是 [0, 1] (当 base=2 时)
```

### 4. 结果解释与注意事项

1.  **样本数量不同完全没问题**：
    *   在上述代码中，我们从未直接比较 `sample_P[i]` 和 `sample_Q[i]`。
    *   我们比较的是它们生成的**概率分布**（即直方图的高度或 KDE 曲线的形状）。只要归一化正确（和为 1），原始样本有多少个点都不影响计算流程。

2.  **直方图的 Bin 数量选择**：
    *   Bin 太少：分布过于粗糙，丢失细节，JS 散度可能偏小。
    *   Bin 太多：对于样本量少的数据，会出现很多空的 Bin，导致噪声过大，JS 散度可能虚高。
    *   **经验法则**：可以使用平方根法则（$\sqrt{N}$）或 Sturges 公式来确定 Bin 的数量。通常 50-100 个 Bin 足以应对大多数情况。

3.  **Scipy 的 `jensenshannon` 函数**：
    *   注意 `scipy.spatial.distance.jensenshannon` 返回的是 **JS 距离 (JS Distance)**。
    *   JS 距离 = $\sqrt{\text{JS Divergence}}$。
    *   如果你需要标准的 JS 散度，**记得对结果求平方**（如代码所示）。

4.  **Base 的选择**：
    *   通常使用 `base=2`，此时 JS 散度的单位是比特 (bit)，且上界为 1。
    *   如果使用 `base=e` (默认)，单位是纳特 (nat)。

### 5. 可视化辅助工具

为了更好地理解这两个分布的差异，我为你生成一个 React 组件，用于可视化这两个不同样本集的直方图以及它们重叠的部分。

```jsx type=react
import React, { useState, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

const JSDivergenceVisualizer = () => {
  // Config
  const [sampleSize1, setSampleSize1] = useState(1000);
  const [sampleSize2, setSampleSize2] = useState(500);
  const [mean1, setMean1] = useState(0);
  const [std1, setStd1] = useState(1);
  const [mean2, setMean2] = useState(2);
  const [std2, setStd2] = useState(1.5);
  const [binsCount, setBinsCount] = useState(40);

  // Helper: Gaussian Random Generator (Box-Muller transform)
  const generateGaussian = (mean, std) => {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return mean + std * Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
  };

  // Generate Data
  const data = useMemo(() => {
    const samples1 = Array.from({ length: sampleSize1 }, () => generateGaussian(mean1, std1));
    const samples2 = Array.from({ length: sampleSize2 }, () => generateGaussian(mean2, std2));

    // Determine range
    const allSamples = [...samples1, ...samples2];
    const min = Math.min(...allSamples);
    const max = Math.max(...allSamples);
    const range = max - min;
    const binWidth = range / binsCount;

    // Create Histogram Bins
    const histogramData = [];
    for (let i = 0; i < binsCount; i++) {
      const binStart = min + i * binWidth;
      const binEnd = binStart + binWidth;
      const mid = (binStart + binEnd) / 2;
      
      // Count frequencies
      const count1 = samples1.filter(v => v >= binStart && v < binEnd).length;
      const count2 = samples2.filter(v => v >= binStart && v < binEnd).length;

      // Normalize to Probability
      const prob1 = count1 / sampleSize1;
      const prob2 = count2 / sampleSize2;

      histogramData.push({
        bin: mid.toFixed(2),
        prob1: prob1,
        prob2: prob2,
        // For visualization of overlap
        overlap: Math.min(prob1, prob2)
      });
    }

    // Calculate JS Divergence (Approximate)
    let jsDivergence = 0;
    const p = histogramData.map(d => d.prob1);
    const q = histogramData.map(d => d.prob2);
    
    // KL Divergence Helper
    const kl_divergence = (p_dist, q_dist) => {
        let sum = 0;
        for(let i=0; i<p_dist.length; i++) {
            if (p_dist[i] > 0 && q_dist[i] > 0) {
                sum += p_dist[i] * Math.log2(p_dist[i] / q_dist[i]);
            }
        }
        return sum;
    };

    // M distribution
    const m = p.map((val, i) => 0.5 * (val + q[i]));
    
    const kl_pm = kl_divergence(p, m);
    const kl_qm = kl_divergence(q, m);
    
    jsDivergence = 0.5 * kl_pm + 0.5 * kl_qm;

    return { histogramData, jsDivergence };
  }, [sampleSize1, sampleSize2, mean1, std1, mean2, std2, binsCount]);

  return (
    <div className="p-6 max-w-4xl mx-auto bg-white rounded-xl shadow-md space-y-6">
      <div className="border-b pb-4">
        <h2 className="text-2xl font-bold text-gray-800">JS 散度直观演示</h2>
        <p className="text-gray-600 mt-2">
          即使样本数量不同（N={sampleSize1}, M={sampleSize2}），我们通过将样本转化为“概率密度”（直方图高度归一化），
          即可计算它们之间的分布差异。
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Controls for Distribution 1 */}
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
          <h3 className="font-semibold text-blue-800 mb-3">分布 P (蓝色)</h3>
          <div className="space-y-3">
            <div>
              <label className="text-sm text-gray-600 block">样本数量 (N)</label>
              <input type="range" min="100" max="2000" step="100" value={sampleSize1} onChange={(e) => setSampleSize1(Number(e.target.value))} className="w-full" />
              <span className="text-xs font-mono">{sampleSize1}</span>
            </div>
            <div>
              <label className="text-sm text-gray-600 block">均值 (Mean)</label>
              <input type="range" min="-5" max="5" step="0.5" value={mean1} onChange={(e) => setMean1(Number(e.target.value))} className="w-full" />
              <span className="text-xs font-mono">{mean1}</span>
            </div>
          </div>
        </div>

        {/* Controls for Distribution 2 */}
        <div className="bg-green-50 p-4 rounded-lg border border-green-100">
          <h3 className="font-semibold text-green-800 mb-3">分布 Q (绿色)</h3>
          <div className="space-y-3">
            <div>
              <label className="text-sm text-gray-600 block">样本数量 (M)</label>
              <input type="range" min="100" max="2000" step="100" value={sampleSize2} onChange={(e) => setSampleSize2(Number(e.target.value))} className="w-full" />
              <span className="text-xs font-mono">{sampleSize2}</span>
            </div>
            <div>
              <label className="text-sm text-gray-600 block">均值 (Mean)</label>
              <input type="range" min="-5" max="5" step="0.5" value={mean2} onChange={(e) => setMean2(Number(e.target.value))} className="w-full" />
              <span className="text-xs font-mono">{mean2}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Result Display */}
      <div className="text-center py-4 bg-gray-50 rounded-lg border border-gray-200">
        <span className="text-gray-500 text-sm uppercase tracking-wide">计算结果 (Base 2)</span>
        <div className="text-3xl font-bold text-indigo-600 mt-1">
          JS Divergence ≈ {data.jsDivergence.toFixed(4)}
        </div>
        <div className="text-xs text-gray-400 mt-1">
          (0 = 完全相同, 1 = 完全不重叠)
        </div>
      </div>

      {/* Chart */}
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data.histogramData}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            barCategoryGap={0} // Remove gap to look like histogram
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="bin" tick={{fontSize: 10}} interval={4} label={{ value: 'Value', position: 'insideBottom', offset: -5 }} />
            <YAxis label={{ value: 'Probability', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend wrapperStyle={{ paddingTop: '10px' }}/>
            {/* We use opacity to show overlap */}
            <Bar dataKey="prob1" name="P Probability" fill="#3b82f6" fillOpacity={0.6} />
            <Bar dataKey="prob2" name="Q Probability" fill="#22c55e" fillOpacity={0.6} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default JSDivergenceVisualizer;
```

##JS散度Q2
*Thinking...*

> **Evaluating User Needs**
> 
> I've zeroed in on the user's core need: calculating the Jensen-Shannon divergence. The data source is from different 2D datasets, which I must factor into the approach.
> 
> 
> **Understanding Divergence Parameters**
> 
> My focus has shifted. I'm now dissecting the JS divergence formula, ensuring I account for differing sample sizes and multi-dimensional data. Chinese language is an interesting constraint, but not core to the math itself.
> 
> 
计算两个不同样本大小的二维（或多维）数据集之间的 Jensen-Shannon (JS) 散度是一个常见的问题，特别是在生成对抗网络（GANs）评估或分布差异分析中。

由于你拥有的是**离散的样本点**而不是连续的概率密度函数（PDF），直接积分是不可行的。你需要先将这些样本点转换为概率分布。

以下是处理样本数不同、维度为二维或多维的 JS 散度计算的完整步骤和方法。

### 核心思路

JS 散度是基于 KL 散度（Kullback-Leibler Divergence）的对称平滑版本。
公式如下：
$$ JS(P || Q) = \frac{1}{2} KL(P || M) + \frac{1}{2} KL(Q || M) $$
其中 $M = \frac{1}{2}(P + Q)$ 是平均分布。

**难点在于：** 你的数据是两个点集 $X = \{x_1, ..., x_n\}$ 和 $Y = \{y_1, ..., y_m\}$，且 $n \neq m$。

### 方法一：直方图离散化 (Binning / Histogramming) —— 最常用

这是最直接的方法。将多维空间划分为网格（Bins），计算落入每个网格的样本比例，从而得到离散概率分布向量。

**步骤：**

1.  **确定范围 (Range):** 找到两个数据集中所有维度的最大值和最小值，确定一个包围两个数据集的公共边界。
2.  **划分网格 (Binning):** 在该范围内划分 $K \times K$ (对于二维) 或更多维度的网格。
3.  **计算频率:**
    *   统计数据集 A 落入每个网格的点的数量，除以 A 的总样本数，得到概率向量 $P$。
    *   统计数据集 B 落入每个网格的点的数量，除以 B 的总样本数，得到概率向量 $Q$。
    *   *注意：为了防止计算 KL 散度时出现 $\log(0)$，通常给每个 bin 加一个极小的平滑值（如 $\epsilon = 1e-10$）并重新归一化。*
4.  **计算 JS 散度:** 使用上述公式计算两个向量 $P$ 和 $Q$ 之间的 JS 散度。

**优点：** 计算速度快，易于实现。
**缺点：** 结果对网格大小（Bin size）非常敏感。网格太密会导致很多空桶，网格太疏会丢失分布细节。

### 方法二：核密度估计 (Kernel Density Estimation, KDE) —— 更精确

如果样本量较小或分布较平滑，直方图法可能太粗糙。KDE 可以从样本点拟合出连续的概率密度函数，然后在离散的参考点上采样。

**步骤：**

1.  **拟合模型:** 使用高斯核（Gaussian Kernel）分别对数据集 A 和数据集 B 进行拟合，得到两个估计函数 $f_A(x)$ 和 $f_B(x)$。
2.  **采样参考点:** 在数据空间内生成一组密集的网格点或随机采样点作为评估点。
3.  **计算概率:** 计算 $f_A$ 和 $f_B$ 在这些评估点上的概率密度值，并归一化，得到向量 $P$ 和 $Q$。
4.  **计算 JS 散度:** 基于 $P$ 和 $Q$ 计算。

**优点：** 比直方图更平滑，对样本量差异不敏感。
**缺点：** 在高维空间（>3维）计算量巨大且容易出现“维数灾难”。

### 方法三：k-近邻估计 (k-NN Estimation) —— 适用于高维

对于高维数据，基于网格的方法不再适用。可以使用基于 k-近邻的方法直接估计 KL 散度，进而估算 JS 散度。

这种方法比较复杂，通常直接使用现成的库（如 Python 的 `scikit-learn` 或专门的互信息估计库）。它的核心思想是利用点与其第 k 个最近邻居的距离来近似局部概率密度。

---

### Python 代码实现 (基于直方图法)

这是最通用且稳健的方法。为了演示，我将编写一个完整的 Python 脚本，包含数据生成、离散化处理和 JS 散度计算。

```python
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

def compute_js_divergence_2d(data_p, data_q, n_bins=20):
    """
    计算两个二维样本集之间的 JS 散度 (使用直方图法)。
    
    参数:
        data_p: 形状为 (N, 2) 的 numpy 数组，样本集 P
        data_q: 形状为 (M, 2) 的 numpy 数组，样本集 Q (N 可以不等于 M)
        n_bins: 直方图的网格数量 (例如 20x20)
        
    返回:
        js_div: JS 散度值 (0 到 1 之间，以 e 为底通常是 0-ln(2)，scipy 默认以 e 为底)
        注意：scipy.spatial.distance.jensenshannon 返回的是 JS 距离 (JS散度的平方根)
        如果需要 JS 散度，需要对其平方。
    """
    
    # 1. 确定公共的边界范围
    # 我们需要确保两个直方图是在完全相同的空间网格上计算的
    min_x = min(np.min(data_p[:, 0]), np.min(data_q[:, 0]))
    max_x = max(np.max(data_p[:, 0]), np.max(data_q[:, 0]))
    min_y = min(np.min(data_p[:, 1]), np.min(data_q[:, 1]))
    max_y = max(np.max(data_p[:, 1]), np.max(data_q[:, 1]))
    
    # 稍微扩大边界以防止边界点问题
    range_x = [min_x - 0.1, max_x + 0.1]
    range_y = [min_y - 0.1, max_y + 0.1]
    
    # 2. 计算二维直方图
    # density=True 确保积分（面积）为1，但这在离散概率向量计算中还不够，我们需要和为1
    hist_p, _, _ = np.histogram2d(data_p[:, 0], data_p[:, 1], bins=n_bins, range=[range_x, range_y])
    hist_q, _, _ = np.histogram2d(data_q[:, 0], data_q[:, 1], bins=n_bins, range=[range_x, range_y])
    
    # 3. 展平并转换为概率分布向量
    # 展平为 1D 数组
    p_probs = hist_p.flatten()
    q_probs = hist_q.flatten()
    
    # 归一化，使和为 1 (变成离散概率分布 PMF)
    p_probs = p_probs / np.sum(p_probs)
    q_probs = q_probs / np.sum(q_probs)
    
    # 4. 计算 JS 散度
    # scipy 的 jensenshannon 函数会自动处理除以0的情况（只要不是两个都是0）
    # 它返回的是 JS 距离 (metric)，即 sqrt(JS_divergence)
    # base=2 通常用于信息论 (单位 bit)，base=e 用于数学推导 (单位 nat)
    js_dist = jensenshannon(p_probs, q_probs, base=2)
    
    # 得到 JS 散度
    js_divergence = js_dist ** 2
    
    return js_divergence

# --- 测试示例 ---

# 设置随机种子
np.random.seed(42)

# 生成样本集 P (高斯分布，中心 [0,0])，样本数 1000
mean_p = [0, 0]
cov_p = [[1, 0], [0, 1]]
data_p = np.random.multivariate_normal(mean_p, cov_p, 1000)

# 生成样本集 Q (高斯分布，中心 [1,1])，样本数 500 (注意样本数不同)
mean_q = [1, 1]
cov_q = [[1.5, 0], [0, 1.5]]
data_q = np.random.multivariate_normal(mean_q, cov_q, 500)

# 计算
js_val = compute_js_divergence_2d(data_p, data_q, n_bins=30)

print(f"样本 P 数量: {len(data_p)}")
print(f"样本 Q 数量: {len(data_q)}")
print(f"计算得到的 JS 散度: {js_val:.5f}")

# 可视化验证
plt.figure(figsize=(8, 6))
plt.scatter(data_p[:, 0], data_p[:, 1], alpha=0.5, label='P (Target)', s=10)
plt.scatter(data_q[:, 0], data_q[:, 1], alpha=0.5, label='Q (Source)', s=10, color='red')
plt.legend()
plt.title(f"2D Samples (JS Div: {js_val:.4f})")
plt.show()
```

### 关键注意事项

1.  **Bin 的数量选择**:
    *   Bin 太少：JS 散度会偏低，因为分布被过度平滑，差异被掩盖。
    *   Bin 太多：JS 散度会偏高，因为样本落入不同 bin 的概率增加（特别是样本量不足时），导致算法误以为两个分布完全不重叠。
    *   **经验法则**：对于 $N$ 个样本，Bin 的数量通常取 $\sqrt{N}$ 左右，或者使用 Freedman-Diaconis 准则。

2.  **Scipy 的 `jensenshannon`**:
    *   一定要注意该函数返回的是 **距离**（Distance），它是散度的平方根。如果你需要严格的散度值，必须对结果求平方。
    *   该函数默认处理了 $P(i)=0$ 或 $Q(i)=0$ 的情况，你不需要手动加 $\epsilon$ (除非你自己手写公式计算 KL 散度)。

3.  **高维情况**:
    *   如果你的数据维度很高（例如 > 10维），`np.histogramdd` 会生成极其庞大的稀疏矩阵，内存会爆掉。
    *   **解决方案**：在高维情况下，不要直接算 JS 散度。通常的做法是先进行**降维**（如 PCA 或 t-SNE）降到 2D/3D，然后再计算直方图 JS 散度；或者使用专门针对高维的评估指标（如 Frechet Inception Distance - FID，这在图像生成中是标准做法）。

### 总结

对于样本数不同的二维/多维数据：
1.  **定义公共空间范围**。
2.  **网格化（直方图）**：将空间切分，统计落入每个格子的样本数。
3.  **归一化**：将计数除以各自的总样本数，得到两个长度相同的概率向量（长度等于网格总数）。
4.  **计算公式**：代入 JS 散度公式计算。