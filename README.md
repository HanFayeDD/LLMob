## TODO
- add record done 
- change critic prompt way done
- 命令行设置对比实验（默认参数）done
- init role prompt done
- llm as judge
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