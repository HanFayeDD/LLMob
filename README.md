## todo
- add record done 
- change critic prompt way done
- 命令行设置对比实验（默认参数）done
- init role prompt done

## 对比试验
- 有无critic
- 推荐地点算法
- 保存pattern中间结果，保证实验可复现性质

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
