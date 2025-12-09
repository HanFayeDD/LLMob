import time
import openai
from functools import wraps

# ==========================================
# 新建：单线程版速率限制器 (SimpleRateLimiter)
# ==========================================
class SimpleRateLimiter():
    """
    一个简单的、非线程安全的速率限制装饰器。
    
    参数:
        max_calls (int): 时间窗口内允许的最大调用次数。
        period (float): 时间窗口的大小（秒）。
    """
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.timestamps = []  # 仅存储有效时间窗口内的时间戳

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # 1. 清理过期的时间戳
            # 保留那些在 [now - period, now] 范围内的记录
            # 例如：period=60，现在是 100，则保留 40 之后的记录
            self.timestamps = [t for t in self.timestamps if t > now - self.period]

            # 2. 检查是否达到限制
            if len(self.timestamps) >= self.max_calls:
                # 列表中最早的一次调用时间
                earliest_call = self.timestamps[0]
                
                # 计算需要等待的时间：直到最早的那次调用“过期”
                wait_time = (earliest_call + self.period) - now
                
                if wait_time > 0:
                    print(f"[RateLimit] 触发限流，休眠 {wait_time:.2f} 秒...")
                    time.sleep(wait_time)
                    
                    # 休眠结束后，更新当前时间
                    now = time.time()
                    # 再次清理（理论上最早的那个肯定过期了，腾出了位置）
                    self.timestamps = [t for t in self.timestamps if t > now - self.period]

            # 3. 记录本次执行的时间
            self.timestamps.append(now)
            
            # 4. 执行原函数
            return func(*args, **kwargs)
            
        return wrapper