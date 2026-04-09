FPATH = r"chathistory\normal\gp3.5turbo_PM\cho.txt"





with open(FPATH, "r", encoding="utf-8") as f:
    content = f.read()
content = content.split("\n")
idx_tag = -1
for idx, line in enumerate(content):
    if "#" in line:
        idx_tag = idx
        break
    
if idx_tag == -1:
    raise ValueError("No tag found in the file.")

content = content[idx_tag+1:]

res = []

cishu = (len(content) // 3)

for i in range(0, cishu, 1):
    p1 = int(content[3*i].split("/")[0].split("-")[-1])
    p2 = int(content[3*i+1].split(":")[-1])
    p3 = int(content[3*i+2].split(":")[-1])
    res.append((p1, p2, p3))
    
print(res)


r1, r2, r3 = zip(*res)
print(f"总轨迹数：{20*14}")
print(f"总调用次数：{sum(r1)}, 成功轨迹：{sum(r2)}, 失败轨迹：{sum(r3)}")

print(FPATH, 280, sum(r1), sum(r2), sum(r3), sep=",")