#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将指定相对路径的 .pkl/.pickle 文件转换为 JSON 文件。

用法:
    python pkl2json.py <relative_path_to_pickle>

输出文件保存到相对路径的首层目录下，文件名与原文件同名，扩展名改为 .json。
例如:  data/2021/1481.pkl  ->  data/1481.json
"""

import sys
import os
import pickle
import json
import datetime
import pathlib
import numpy as np


def _default_serializer(obj):
    """处理无法直接 JSON 序列化的对象。"""
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return str(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.hex()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    # numpy 类型兼容
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    # pandas 类型兼容
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except ImportError:
        pass
    # 最终回退：尝试 __dict__，否则转 str
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def resolve_output_path(input_path: str) -> str:
    """
    根据相对路径计算输出路径：取首层目录 + 原文件名.json。
    例如 data/2021/1481.pkl -> data/1481.json
    若输入本身就在当前目录（无子目录），则输出到当前目录。
    """
    parts = pathlib.PurePosixPath(input_path.replace("\\", "/")).parts
    stem = pathlib.Path(input_path).stem
    if len(parts) > 1:
        root_dir = parts[0]
    else:
        root_dir = "."
    return os.path.join(root_dir, f"{stem}.json")


def main():
    if len(sys.argv) < 2:
        print(f"用法: python {os.path.basename(__file__)} <pickle_file>", file=sys.stderr)
        sys.exit(1)

    pkl_path = sys.argv[1]

    # 检查扩展名
    if not pkl_path.lower().endswith((".pkl", ".pickle")):
        print(f"错误: 输入文件必须是 .pkl 或 .pickle 文件，收到: {pkl_path}", file=sys.stderr)
        sys.exit(1)

    # 检查文件是否存在
    if not os.path.isfile(pkl_path):
        print(f"错误: 文件不存在: {pkl_path}", file=sys.stderr)
        sys.exit(1)

    # 读取 pickle
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, ImportError, ModuleNotFoundError) as e:
        print(f"错误: pickle 反序列化失败: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}", file=sys.stderr)
        sys.exit(1)

    # 序列化为 JSON
    try:
        json_str = json.dumps(data, ensure_ascii=False, indent=2, default=_default_serializer)
    except (TypeError, ValueError, OverflowError) as e:
        print(f"错误: JSON 序列化失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 写出 JSON
    output_path = resolve_output_path(pkl_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
    except OSError as e:
        print(f"错误: 写入文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"转换完成: {pkl_path} -> {output_path}")


if __name__ == "__main__":
    main()
