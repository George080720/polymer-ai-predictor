"""
多目标材料性能预测系统（贝叶斯优化版）
预测 + 智能推荐配方
"""

import streamlit as st
import sqlite3
import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')
import sqlite3
import datetime

# ---- 实验日志函数 ----
def init_db():
    conn = sqlite3.connect('experiment_log.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            smiles TEXT,
            inputs TEXT,
            results TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(smiles, inputs, results):
    try:
        conn = sqlite3.connect('experiment_log.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO predictions (timestamp, smiles, inputs, results)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            smiles,
            str(inputs),
            str(results)
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"保存失败：{e}")
        return False

def load_history():
    conn = sqlite3.connect('experiment_log.db')
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    return rows

st.set_page_config(page_title="多目标性能预测系统", layout="wide")
st.markdown("""
<style>
    /* 全局字体 */
    .stApp {
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    /* 主标题 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 2px solid #2ECC71;
        padding-bottom: 0.5rem;
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }
    /* 卡片样式 */
    .custom-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
        transition: 0.2s;
    }
    .custom-card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.10);
    }
    /* 侧边栏 */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    /* 按钮 */
    .stButton button {
        background-color: #1E3A5F;
        color: white;
        border-radius: 30px;
        font-weight: 600;
        border: none;
        transition: 0.2s;
    }
    .stButton button:hover {
        background-color: #2ECC71;
        color: #1E3A5F;
        transform: scale(1.02);
    }
    /* 分子图容器 */
    .mol-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
        text-align: center;
        margin: 0 auto;
        max-width: 600px;
    }
    /* 标签微调 */
    h1, h2, h3 {
        color: #1E3A5F;
    }
    /* 指标卡片 */
    .stMetric {
        background: #f8fafc;
        border-radius: 12px;
        padding: 8px;
    }
    .stMetric .stMetricValue {
        color: #1E3A5F !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 分子数据库
# ============================================================
MOLECULES = {
    "C=C": {
        "name": "乙烯", "formula": "C₂H₄", "weight": "28.05",
        "params": {"单体比例(%)": 90, "引发剂(phr)": 0.5, "聚合温度(°C)": 80, "聚合时间(h)": 4, "交联剂(phr)": 0.0},
        "atoms": {
            'C1': (150, 200, '#2ecc71', 'C'), 'C2': (350, 200, '#2ecc71', 'C'),
            'H1': (60, 120, '#3498db', 'H'), 'H2': (60, 280, '#3498db', 'H'),
            'H3': (440, 120, '#3498db', 'H'), 'H4': (440, 280, '#3498db', 'H')
        },
        "double_bonds": [(150, 185, 350, 185), (150, 215, 350, 215)],
        "single_bonds": [(150, 200, 60, 120), (150, 200, 60, 280),
                         (350, 200, 440, 120), (350, 200, 440, 280)]
    },
    "CCO": {
        "name": "乙醇", "formula": "C₂H₅OH", "weight": "46.07",
        "params": {"单体比例(%)": 85, "引发剂(phr)": 1.0, "聚合温度(°C)": 75, "聚合时间(h)": 6, "交联剂(phr)": 0.0},
        "atoms": {
            'C1': (120, 200, '#2ecc71', 'C'), 'C2': (250, 200, '#2ecc71', 'C'),
            'O1': (380, 200, '#e74c3c', 'O'),
            'H1': (60, 120, '#3498db', 'H'), 'H2': (60, 280, '#3498db', 'H'),
            'H3': (180, 120, '#3498db', 'H'), 'H4': (180, 280, '#3498db', 'H'),
            'H5': (320, 120, '#3498db', 'H'), 'H6': (440, 160, '#3498db', 'H')
        },
        "double_bonds": [],
        "single_bonds": [(120, 200, 60, 120), (120, 200, 60, 280),
                         (120, 200, 250, 200), (250, 200, 180, 120),
                         (250, 200, 180, 280), (250, 200, 380, 200),
                         (380, 200, 320, 120), (380, 200, 440, 160)]
    },
    "c1ccccc1": {
        "name": "苯", "formula": "C₆H₆", "weight": "78.11",
        "params": {"单体比例(%)": 80, "引发剂(phr)": 0.3, "聚合温度(°C)": 90, "聚合时间(h)": 8, "交联剂(phr)": 0.0},
        "atoms": {
            'C1': (200, 100, '#2ecc71', 'C'), 'C2': (130, 150, '#2ecc71', 'C'),
            'C3': (130, 230, '#2ecc71', 'C'), 'C4': (200, 280, '#2ecc71', 'C'),
            'C5': (270, 230, '#2ecc71', 'C'), 'C6': (270, 150, '#2ecc71', 'C'),
            'H1': (200, 40, '#3498db', 'H'), 'H2': (70, 120, '#3498db', 'H'),
            'H3': (70, 260, '#3498db', 'H'), 'H4': (200, 340, '#3498db', 'H'),
            'H5': (330, 260, '#3498db', 'H'), 'H6': (330, 120, '#3498db', 'H')
        },
        "double_bonds": [(200, 92, 130, 142), (200, 108, 130, 158),
                         (130, 222, 200, 272), (130, 238, 200, 288),
                         (262, 150, 262, 230), (278, 150, 278, 230)],
        "single_bonds": [(130, 150, 130, 230), (200, 280, 270, 230),
                         (200, 100, 270, 150), (200, 100, 200, 40),
                         (130, 150, 70, 120), (130, 230, 70, 260),
                         (200, 280, 200, 340), (270, 230, 330, 260),
                         (270, 150, 330, 120)]
    },
    "C=CC": {
        "name": "丙烯", "formula": "C₃H₆", "weight": "42.08",
        "params": {"单体比例(%)": 88, "引发剂(phr)": 0.8, "聚合温度(°C)": 85, "聚合时间(h)": 5, "交联剂(phr)": 0.0},
        "atoms": {
            'C1': (120, 200, '#2ecc71', 'C'), 'C2': (250, 200, '#2ecc71', 'C'),
            'C3': (380, 200, '#2ecc71', 'C'),
            'H1': (80, 140, '#3498db', 'H'), 'H2': (80, 260, '#3498db', 'H'),
            'H3': (280, 140, '#3498db', 'H'), 'H4': (280, 260, '#3498db', 'H'),
            'H5': (440, 140, '#3498db', 'H'), 'H6': (440, 260, '#3498db', 'H')
        },
        "double_bonds": [(120, 195, 250, 195), (120, 205, 250, 205)],
        "single_bonds": [(120, 200, 80, 140), (120, 200, 80, 260),
                         (250, 200, 280, 140), (250, 200, 280, 260),
                         (250, 200, 380, 200), (380, 200, 440, 140),
                         (380, 200, 440, 260)]
    },
    "C=CC=C": {
        "name": "丁二烯", "formula": "C₄H₆", "weight": "54.09",
        "params": {"单体比例(%)": 92, "引发剂(phr)": 0.6, "聚合温度(°C)": 70, "聚合时间(h)": 6, "交联剂(phr)": 0.0},
        "atoms": {
            'C1': (80, 200, '#2ecc71', 'C'), 'C2': (200, 200, '#2ecc71', 'C'),
            'C3': (320, 200, '#2ecc71', 'C'), 'C4': (440, 200, '#2ecc71', 'C'),
            'H1': (40, 140, '#3498db', 'H'), 'H2': (40, 260, '#3498db', 'H'),
            'H3': (160, 140, '#3498db', 'H'), 'H4': (160, 260, '#3498db', 'H'),
            'H5': (360, 140, '#3498db', 'H'), 'H6': (360, 260, '#3498db', 'H'),
            'H7': (480, 140, '#3498db', 'H'), 'H8': (480, 260, '#3498db', 'H')
        },
        "double_bonds": [(80, 195, 200, 195), (80, 205, 200, 205),
                         (320, 195, 440, 195), (320, 205, 440, 205)],
        "single_bonds": [(200, 200, 320, 200), (80, 200, 40, 140),
                         (80, 200, 40, 260), (200, 200, 160, 140),
                         (200, 200, 160, 260), (320, 200, 360, 140),
                         (320, 200, 360, 260), (440, 200, 480, 140),
                         (440, 200, 480, 260)]
    },
    "c1ccccc1C=C": {
        "name": "苯乙烯", "formula": "C₈H₈", "weight": "104.15",
        "params": {"单体比例(%)": 85, "引发剂(phr)": 0.5, "聚合温度(°C)": 80, "聚合时间(h)": 6, "交联剂(phr)": 0.05},
        "atoms": {
            'C1': (200, 100, '#2ecc71', 'C'), 'C2': (130, 150, '#2ecc71', 'C'),
            'C3': (130, 230, '#2ecc71', 'C'), 'C4': (200, 280, '#2ecc71', 'C'),
            'C5': (270, 230, '#2ecc71', 'C'), 'C6': (270, 150, '#2ecc71', 'C'),
            'C7': (330, 70, '#2ecc71', 'C'), 'C8': (400, 40, '#2ecc71', 'C'),
            'H2': (70, 120, '#3498db', 'H'), 'H3': (70, 260, '#3498db', 'H'),
            'H4': (200, 340, '#3498db', 'H'), 'H5': (330, 260, '#3498db', 'H'),
            'H6': (330, 120, '#3498db', 'H'), 'H7': (330, 20, '#3498db', 'H'),
            'H8': (455, 20, '#3498db', 'H'), 'H9': (455, 75, '#3498db', 'H')
        },
        "double_bonds": [(200, 92, 130, 142), (200, 108, 130, 158),
                         (130, 222, 200, 272), (130, 238, 200, 288),
                         (262, 150, 262, 230), (278, 150, 278, 230),
                         (324, 64, 394, 34), (336, 76, 406, 46)],
        "single_bonds": [(130, 150, 130, 230), (200, 280, 270, 230),
                         (200, 100, 270, 150), (130, 150, 70, 120),
                         (130, 230, 70, 260), (200, 280, 200, 340),
                         (270, 230, 330, 260), (270, 150, 330, 120),
                         (200, 100, 330, 70), (330, 70, 330, 20),
                         (400, 40, 455, 20), (400, 40, 455, 75)]
    },
    "c1ccc(O)cc1": {
    "name": "苯酚",
    "formula": "C₆H₅OH",
    "weight": "94.11",
    "params": {
        "单体比例(%)": 82,
        "引发剂(phr)": 0.4,
        "聚合温度(°C)": 75,
        "聚合时间(h)": 7,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        # 苯环六个碳（复用苯的坐标）
        'C1': (200, 100, '#2ecc71', 'C'),
        'C2': (130, 150, '#2ecc71', 'C'),
        'C3': (130, 230, '#2ecc71', 'C'),
        'C4': (200, 280, '#2ecc71', 'C'),
        'C5': (270, 230, '#2ecc71', 'C'),
        'C6': (270, 150, '#2ecc71', 'C'),
        # 羟基：水平方向（C1 → O 向右，O → H 向右上）
        'O1': (260, 100, '#e74c3c', 'O'),
        'H_OH': (310, 70, '#3498db', 'H'),
        # 苯环上的 H（C2-C6）
        'H2': (70, 120, '#3498db', 'H'),
        'H3': (70, 260, '#3498db', 'H'),
        'H4': (200, 340, '#3498db', 'H'),
        'H5': (330, 260, '#3498db', 'H'),
        'H6': (330, 120, '#3498db', 'H')
    },
    "double_bonds": [
        # 苯环双键（和苯完全一致）
        (200, 92, 130, 142),
        (200, 108, 130, 158),
        (130, 222, 200, 272),
        (130, 238, 200, 288),
        (262, 150, 262, 230),
        (278, 150, 278, 230)
    ],
    "single_bonds": [
        # 苯环骨架
        (130, 150, 130, 230),
        (200, 280, 270, 230),
        (200, 100, 270, 150),
        # C1-O1 键（水平向右）
        (200, 100, 260, 100),
        # O1-H_OH 键（向右上倾斜 45°）
        (260, 100, 310, 70),
        # 苯环 C-H
        (130, 150, 70, 120),
        (130, 230, 70, 260),
        (200, 280, 200, 340),
        (270, 230, 330, 260),
        (270, 150, 330, 120)
    ]
    },
    "C=CC#N": {
    "name": "丙烯腈",
    "formula": "C₃H₃N",
    "weight": "53.06",
    "params": {
        "单体比例(%)": 86,
        "引发剂(phr)": 0.7,
        "聚合温度(°C)": 65,
        "聚合时间(h)": 5,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        'C1': (100, 200, '#2ecc71', 'C'),
        'C2': (220, 200, '#2ecc71', 'C'),
        'C3': (340, 200, '#2ecc71', 'C'),
        'N1': (420, 200, '#e74c3c', 'N'),
        'H1': (60, 140, '#3498db', 'H'),
        'H2': (60, 260, '#3498db', 'H'),
        'H3': (180, 140, '#3498db', 'H')
    },
    "double_bonds": [
        (100, 195, 220, 195),
        (100, 205, 220, 205)
    ],
    "triple_bonds": [
    (340, 195, 420, 195),
    (340, 200, 420, 200),
    (340, 205, 420, 205)
    ],
    "single_bonds": [
        (220, 200, 340, 200),
        (100, 200, 60, 140),
        (100, 200, 60, 260),
        (220, 200, 180, 140),
        (340, 200, 420, 200)
    ]
    },
   "CC(=C)C(=O)OC": {
    "name": "甲基丙烯酸甲酯",
    "formula": "C₅H₈O₂",
    "weight": "100.12",
    "params": {
        "单体比例(%)": 84,
        "引发剂(phr)": 0.6,
        "聚合温度(°C)": 70,
        "聚合时间(h)": 6,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        # 双键碳（C1=C2）
        'C1': (100, 180, '#2ecc71', 'C'),
        'C2': (220, 180, '#2ecc71', 'C'),
        # 甲基（连在 C1 上，向上）
        'C3': (100, 80, '#2ecc71', 'C'),
        # 酯基碳（连在 C2 上，向右）
        'C4': (340, 180, '#2ecc71', 'C'),
        # 甲氧基（连在 C4 上，向右）
        'O1': (440, 180, '#e74c3c', 'O'),
        'C5': (520, 180, '#2ecc71', 'C'),
        # 酯基上的双键氧（C4=O2，向下）
        'O2': (340, 280, '#e74c3c', 'O'),
        # C1 上的两个氢（CH₂=）
        'H1': (60, 220, '#3498db', 'H'),
        'H2': (60, 140, '#3498db', 'H'),
        # C3 上的三个氢（CH₃）
        'H3': (70, 40, '#3498db', 'H'),
        'H4': (130, 40, '#3498db', 'H'),
        'H5': (100, 30, '#3498db', 'H'),
        # C5 上的三个氢（O-CH₃）
        'H6': (480, 140, '#3498db', 'H'),
        'H7': (560, 140, '#3498db', 'H'),
        'H8': (560, 220, '#3498db', 'H')
    },
    "double_bonds": [
        (100, 175, 220, 175),
        (100, 185, 220, 185),
        (332, 180, 332, 280),
        (348, 180, 348, 280)
    ],
    "single_bonds": [
        (100, 180, 100, 80),
        (220, 180, 340, 180),
        (340, 180, 440, 180),
        (440, 180, 520, 180),
        (100, 180, 60, 220),
        (100, 180, 60, 140),
        (100, 80, 70, 40),
        (100, 80, 130, 40),
        (100, 80, 100, 30),
        (520, 180, 480, 140),
        (520, 180, 560, 140),
        (520, 180, 560, 220)
    ]
    },
    "CC(=O)OC=C": {
    "name": "乙酸乙烯酯",
    "formula": "C₄H₆O₂",
    "weight": "86.09",
    "params": {
        "单体比例(%)": 88,
        "引发剂(phr)": 0.5,
        "聚合温度(°C)": 72,
        "聚合时间(h)": 5,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        # 甲基（最左侧）
        'C1': (80, 200, '#2ecc71', 'C'),
        # 羰基碳（中间偏左）
        'C2': (200, 200, '#2ecc71', 'C'),
        # 酯基氧（连接 C2 和 C3）
        'O1': (300, 200, '#e74c3c', 'O'),
        # 乙烯基碳（C3=C4，最右侧）
        'C3': (400, 200, '#2ecc71', 'C'),
        'C4': (500, 200, '#2ecc71', 'C'),
        # 羰基氧（C2=O2，向上）
        'O2': (200, 120, '#e74c3c', 'O'),
        # 甲基上的三个氢
        'H1': (40, 160, '#3498db', 'H'),
        'H2': (40, 240, '#3498db', 'H'),
        'H3': (80, 140, '#3498db', 'H'),
        # 乙烯基 C3 上的两个氢（C3=C4，C3 连接 H4 和 H5）
        'H4': (400, 140, '#3498db', 'H'),
        'H5': (400, 260, '#3498db', 'H'),
        # 乙烯基 C4 上的两个氢（C4 连接 H6 和 H7，相当于 CH₂）
        'H6': (540, 160, '#3498db', 'H'),
        'H7': (540, 240, '#3498db', 'H')
    },
    "double_bonds": [
        # C2=O2 双键（竖直向上）
        (196, 200, 196, 120),
        (204, 200, 204, 120),
        # C3=C4 双键（水平向右）
        (400, 195, 500, 195),
        (400, 205, 500, 205)
    ],
    "single_bonds": [
        # C1-C2 单键
        (80, 200, 200, 200),
        # C2-O1 单键
        (200, 200, 300, 200),
        # O1-C3 单键
        (300, 200, 400, 200),
        # C1 上的 C-H 键（甲基）
        (80, 200, 40, 160),
        (80, 200, 40, 240),
        (80, 200, 80, 140),
        # C3 上的 C-H 键
        (400, 200, 400, 140),
        (400, 200, 400, 260),
        # C4 上的 C-H 键
        (500, 200, 540, 160),
        (500, 200, 540, 240)
    ]
    },
    "C=Cl": {
    "name": "氯乙烯",
    "formula": "C₂H₃Cl",
    "weight": "62.50",
    "params": {
        "单体比例(%)": 90,
        "引发剂(phr)": 0.5,
        "聚合温度(°C)": 70,
        "聚合时间(h)": 5,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        'C1': (150, 200, '#2ecc71', 'C'),
        'C2': (300, 200, '#2ecc71', 'C'),
        'Cl1': (420, 200, '#9b59b6', 'Cl'),
        'H1': (110, 150, '#3498db', 'H'),
        'H2': (110, 250, '#3498db', 'H'),
        'H3': (340, 150, '#3498db', 'H')
    },
    "double_bonds": [
        (150, 195, 300, 195),
        (150, 205, 300, 205)
    ],
    "single_bonds": [
        (300, 200, 420, 200),
        (150, 200, 110, 150),
        (150, 200, 110, 250),
        (300, 200, 340, 150)
    ]
    },
    "c1cc(C(=O)O)ccc1C(=O)O": {
    "name": "对苯二甲酸",
    "formula": "C₈H₆O₄",
    "weight": "166.13",
    "params": {
        "单体比例(%)": 78,
        "引发剂(phr)": 0.6,
        "聚合温度(°C)": 120,
        "聚合时间(h)": 8,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        # 苯环（完整六边形）
        'C1': (300, 100, '#2ecc71', 'C'),
        'C2': (230, 150, '#2ecc71', 'C'),
        'C3': (230, 230, '#2ecc71', 'C'),
        'C4': (300, 280, '#2ecc71', 'C'),
        'C5': (370, 230, '#2ecc71', 'C'),
        'C6': (370, 150, '#2ecc71', 'C'),
        # 羧基 1（连在 C2 上，向左下）
        'C7': (150, 120, '#2ecc71', 'C'),
        'O1': (180, 60, '#e74c3c', 'O'),      # C=O 朝左上
        'O2': (90, 120, '#e74c3c', 'O'),      # O-H 朝左
        # 羧基 2（连在 C5 上，改成水平朝右）
        'C8': (450, 230, '#2ecc71', 'C'),      # 和 C5 同一水平线
        'O3': (520, 200, '#e74c3c', 'O'),      # C=O 朝右上
        'O4': (520, 260, '#e74c3c', 'O'),      # O-H 朝右下
        # 苯环上的 H
        'H1': (300, 40, '#3498db', 'H'),
        'H3': (170, 260, '#3498db', 'H'),
        'H4': (300, 340, '#3498db', 'H'),
        'H6': (430, 120, '#3498db', 'H'),
        # 羧基上的 H
        'H7': (90, 60, '#3498db', 'H'),
        'H8': (570, 250, '#3498db', 'H')
    },
    "double_bonds": [
        # 苯环双键
        (300, 92, 230, 142),
        (300, 108, 230, 158),
        (230, 222, 300, 272),
        (230, 238, 300, 288),
        (362, 150, 362, 230),
        (378, 150, 378, 230)],
    "single_bonds": [
        # 苯环骨架
        (230, 150, 230, 230),
        (300, 280, 370, 230),
        (300, 100, 370, 150),
        # C2-C7
        (230, 150, 150, 120),
        # C5-C8（水平连接）
        (370, 230, 450, 230),
        # C7-O2（O-H 朝左）
        (150, 120, 90, 120),
        # C8-O4（O-H 朝右下）
        (450, 230, 520, 260),
        # 苯环 C-H
        (300, 100, 300, 40),
        (230, 230, 170, 260),
        (300, 280, 300, 340),
        (370, 150, 430, 120),
        # O-H 键
        (90, 120, 90, 60),
        (520, 250, 570, 250)
    ]
    },
    "C(CO)O": {
    "name": "乙二醇",
    "formula": "C₂H₆O₂",
    "weight": "62.07",
    "params": {
        "单体比例(%)": 82,
        "引发剂(phr)": 0.7,
        "聚合温度(°C)": 85,
        "聚合时间(h)": 6,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        # 主链 C1-C2
        'C1': (150, 200, '#2ecc71', 'C'),
        'C2': (300, 200, '#2ecc71', 'C'),
        # 羟基 1（连在 C1 上，向左下，拉长 O-H）
        'O1': (80, 240, '#e74c3c', 'O'),
        'H1': (40, 280, '#3498db', 'H'),   # 从 (50,260) 改到 (40,280)，拉长 20px
        # 羟基 2（连在 C2 上，向右下，拉长 O-H）
        'O2': (370, 240, '#e74c3c', 'O'),
        'H2': (410, 280, '#3498db', 'H'),  # 从 (400,260) 改到 (410,280)，拉长 20px
        # C1 上的 H
        'H3': (150, 150, '#3498db', 'H'),
        'H4': (150, 250, '#3498db', 'H'),
        # C2 上的 H
        'H5': (300, 150, '#3498db', 'H'),
        'H6': (300, 250, '#3498db', 'H')
    },
    "double_bonds": [],
    "single_bonds": [
        (150, 200, 300, 200),
        (150, 200, 80, 240),
        (80, 240, 40, 280),      # O1-H1 拉长到 40,280
        (300, 200, 370, 240),
        (370, 240, 410, 280),    # O2-H2 拉长到 410,280
        (150, 200, 150, 150),
        (150, 200, 150, 250),
        (300, 200, 300, 150),
        (300, 200, 300, 250)
    ]
    },
    "C1CO1": {
    "name": "环氧乙烷",
    "formula": "C₂H₄O",
    "weight": "44.05",
    "params": {
        "单体比例(%)": 80,
        "引发剂(phr)": 0.3,
        "聚合温度(°C)": 60,
        "聚合时间(h)": 4,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        'C1': (150, 200, '#2ecc71', 'C'),
        'C2': (250, 200, '#2ecc71', 'C'),
        'O1': (200, 130, '#e74c3c', 'O'),
        'H1': (150, 250, '#3498db', 'H'),
        'H2': (150, 150, '#3498db', 'H'),
        'H3': (250, 250, '#3498db', 'H'),
        'H4': (250, 150, '#3498db', 'H')
    },
    "double_bonds": [],
    "single_bonds": [
        (150, 200, 250, 200),
        (150, 200, 200, 130),
        (250, 200, 200, 130),
        (150, 200, 150, 250),
        (150, 200, 150, 150),
        (250, 200, 250, 250),
        (250, 200, 250, 150)
    ]
    },
    "ABS": {
    "name": "ABS (丙烯腈-丁二烯-苯乙烯)",
    "formula": "C₃H₃N · C₄H₆ · C₈H₈",
    "weight": "约 150-200",
    "params": {
        "单体比例(%)": 85,
        "引发剂(phr)": 0.6,
        "聚合温度(°C)": 80,
        "聚合时间(h)": 7,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        # 用一条链表示共聚物：A-B-S 交替
        'M1': (100, 200, '#e74c3c', 'A'),
        'M2': (180, 200, '#3498db', 'B'),
        'M3': (260, 200, '#f1c40f', 'S'),
        'M4': (340, 200, '#e74c3c', 'A'),
        'M5': (420, 200, '#3498db', 'B'),
        'M6': (500, 200, '#f1c40f', 'S')
    },
    "double_bonds": [],
    "single_bonds": [
        (100, 200, 180, 200),
        (180, 200, 260, 200),
        (260, 200, 340, 200),
        (340, 200, 420, 200),
        (420, 200, 500, 200)
    ]
    },
    "SAN": {
    "name": "SAN (苯乙烯-丙烯腈)",
    "formula": "C₈H₈ · C₃H₃N",
    "weight": "约 100-150",
    "params": {
        "单体比例(%)": 85,
        "引发剂(phr)": 0.5,
        "聚合温度(°C)": 75,
        "聚合时间(h)": 6,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        'M1': (100, 200, '#f1c40f', 'S'),
        'M2': (180, 200, '#e74c3c', 'A'),
        'M3': (260, 200, '#f1c40f', 'S'),
        'M4': (340, 200, '#e74c3c', 'A'),
        'M5': (420, 200, '#f1c40f', 'S'),
        'M6': (500, 200, '#e74c3c', 'A')
    },
    "double_bonds": [],
    "single_bonds": [
        (100, 200, 180, 200),
        (180, 200, 260, 200),
        (260, 200, 340, 200),
        (340, 200, 420, 200),
        (420, 200, 500, 200)
    ]
    },
    "EVA": {
    "name": "EVA (乙烯-乙酸乙烯酯)",
    "formula": "C₂H₄ · C₄H₆O₂",
    "weight": "约 100-150",
    "params": {
        "单体比例(%)": 85,
        "引发剂(phr)": 0.5,
        "聚合温度(°C)": 70,
        "聚合时间(h)": 6,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        'M1': (100, 200, '#2ecc71', 'E'),
        'M2': (180, 200, '#3498db', 'V'),
        'M3': (260, 200, '#2ecc71', 'E'),
        'M4': (340, 200, '#3498db', 'V'),
        'M5': (420, 200, '#2ecc71', 'E'),
        'M6': (500, 200, '#3498db', 'V')
    },
    "double_bonds": [],
    "single_bonds": [
        (100, 200, 180, 200),
        (180, 200, 260, 200),
        (260, 200, 340, 200),
        (340, 200, 420, 200),
        (420, 200, 500, 200)
    ]
    },
    "SBS": {
    "name": "SBS (苯乙烯-丁二烯-苯乙烯)",
    "formula": "C₈H₈ · C₄H₆ · C₈H₈",
    "weight": "约 150-250",
    "params": {
        "单体比例(%)": 85,
        "引发剂(phr)": 0.6,
        "聚合温度(°C)": 75,
        "聚合时间(h)": 7,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        'M1': (100, 200, '#f1c40f', 'S'),
        'M2': (180, 200, '#3498db', 'B'),
        'M3': (260, 200, '#f1c40f', 'S'),
        'M4': (340, 200, '#3498db', 'B'),
        'M5': (420, 200, '#f1c40f', 'S'),
        'M6': (500, 200, '#3498db', 'B')
    },
    "double_bonds": [],
    "single_bonds": [
        (100, 200, 180, 200),
        (180, 200, 260, 200),
        (260, 200, 340, 200),
        (340, 200, 420, 200),
        (420, 200, 500, 200)
    ]
    },
    "NBR": {
    "name": "NBR (丁腈橡胶)",
    "formula": "C₄H₆ · C₃H₃N",
    "weight": "约 100-150",
    "params": {
        "单体比例(%)": 85,
        "引发剂(phr)": 0.5,
        "聚合温度(°C)": 80,
        "聚合时间(h)": 6,
        "交联剂(phr)": 0.0
    },
    "atoms": {
        'M1': (120, 200, '#3498db', 'B'),
        'M2': (200, 200, '#e74c3c', 'A'),
        'M3': (280, 200, '#3498db', 'B'),
        'M4': (360, 200, '#e74c3c', 'A'),
        'M5': (440, 200, '#3498db', 'B'),
        'M6': (520, 200, '#e74c3c', 'A')
    },
    "double_bonds": [],
    "single_bonds": [
        (120, 200, 200, 200),
        (200, 200, 280, 200),
        (280, 200, 360, 200),
        (360, 200, 440, 200),
        (440, 200, 520, 200)
    ]
}
}

# ============================================================
# 新手引导
# ============================================================
if 'first_time' not in st.session_state:
    st.session_state.first_time = True

if st.session_state.first_time:
    st.balloons()
    with st.expander("🎉 欢迎使用！点击这里查看快速入门", expanded=True):
        st.markdown("""
        ### 👋 欢迎来到材料性能预测与优化系统！
        
        **两大核心功能：**
        1. 🔮 **预测**：输入配方，预测 4 种性能
        2. 🧠 **优化**：输入目标性能，AI 自动推荐最优配方
        
        ---
        ### 🚀 三步上手
        1. **左侧边栏** → 确认特征和目标（默认已选好）
        2. **预测工作台** → 输入参数，看预测结果
        3. **优化工作台** → 设定目标，让 AI 推荐配方
        """)
        if st.button("👌 知道了，开始使用"):
            st.session_state.first_time = False
            st.rerun()
    st.stop()

st.title("🧪 多目标材料性能预测与优化系统")
st.markdown("预测性能 + AI 智能推荐配方")

# ============================================================
# 生成演示数据
# ============================================================
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(70, 100, n)
    x2 = np.random.uniform(0.1, 2.0, n)
    x3 = np.random.uniform(60, 120, n)
    x4 = np.random.uniform(2, 10, n)
    x5 = np.random.uniform(0, 0.5, n)
    Tg = 50 + 0.3*x1 + 80*x5 + np.random.normal(0, 3, n)
    y1 = 0.5 + 0.08*(Tg - 50)/10 + 2*x5 + np.random.normal(0, 0.1, n)
    y1 = np.maximum(y1, 0.1)
    y2 = 20 + 0.3*x1 + 5*x2 + 0.2*x3 + 10*x5 + np.random.normal(0, 2, n)
    y2 = np.maximum(y2, 10)
    y3 = 200 - 1.2*x1 - 0.5*x3 + 30*x5 + np.random.normal(0, 8, n)
    y3 = np.maximum(y3, 10)
    y4 = Tg + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({
        '单体比例(%)': x1, '引发剂(phr)': x2,
        '聚合温度(°C)': x3, '聚合时间(h)': x4, '交联剂(phr)': x5,
        '模量(GPa)': y1, '拉伸强度(MPa)': y2,
        '断裂伸长率(%)': y3, 'Tg(°C)': y4
    })
    return df

data = generate_data()

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.header("⚙️ 设置")
    with st.expander("📖 使用指南", expanded=False):
        st.markdown("""
        **预测模式**：输入配方 → 输出性能
        **优化模式**：输入目标性能 → 输出最优配方
        **批量预测**：上传 CSV → 一次性预测多组配方
        """)
    st.divider()
    st.subheader("📂 数据")
    uploaded_file = st.file_uploader("上传数据 (可选)", type=['xlsx', 'csv'])
    use_demo = st.checkbox("使用演示数据", value=True)
    if uploaded_file:
        data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ {len(data)} 行")
    elif use_demo:
        st.info("✅ 使用演示数据 (200 行)")
    else:
        st.warning("⚠️ 请上传数据或勾选演示数据")
    st.divider()
    st.subheader("🎯 选择列")
    all_cols = data.columns.tolist()
    default_feat = all_cols[:5]
    feat_cols = st.multiselect("特征列", all_cols, default=default_feat)
    default_target = all_cols[5:]
    target_cols = st.multiselect("目标列", all_cols, default=default_target)
    st.caption("💡 请注意，你所能查看的特征重要性仅针对那个你所放在所选目标列首位的，若要查看其他特征重要性，请注意顺序并重新勾选")
    test_size = st.slider("测试集比例", 0.1, 0.4, 0.25, 0.05)

if len(feat_cols) == 0 or len(target_cols) == 0:
    st.warning("⚠️ 请选择至少一个特征列和一个目标列")
    st.stop()

# ============================================================
# 训练模型
# ============================================================
X = data[feat_cols]
Y = data[target_cols]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

models = {}
predictions = {}
metrics = {}
for target in target_cols:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train[target])
    # 标准化键名：去掉所有空格，统一括号
    target_key = target.replace(" ", "").replace("（", "(").replace("）", ")")
    models[target_key] = model
    pred = model.predict(X_test)
    predictions[target_key] = pred
    metrics[target_key] = {
        'R2': r2_score(Y_test[target], pred),
        'RMSE': np.sqrt(mean_squared_error(Y_test[target], pred))
    }
    models[target] = model
    pred = model.predict(X_test)
    predictions[target] = pred
    metrics[target] = {
        'R2': r2_score(Y_test[target], pred),
        'RMSE': np.sqrt(mean_squared_error(Y_test[target], pred))
    }

# ============================================================
# Tab 模式：预测 + 优化（放在最前面）
# ============================================================
tab1, tab2 = st.tabs(["🔮 性能预测", "🧠 智能优化"])

# ---------- TAB 1: 预测 ----------
with tab1:
    st.subheader("输入新配方，预测所有性能")
    smiles_input = st.session_state.get("smiles_input", "C=C")
    # ---- 显示当前推荐的参数 ----
    if "recommended_params" in st.session_state:
        st.info(f"💡 当前已应用推荐参数：{st.session_state.recommended_params}")

    input_cols = st.columns(len(feat_cols))
    input_vals = []
    param_ranges = {
        '单体比例(%)': (70.0, 100.0), '引发剂(phr)': (0.1, 2.0),
        '聚合温度(°C)': (60.0, 120.0), '聚合时间(h)': (2.0, 10.0),
        '交联剂(phr)': (0.0, 0.5)
    }

    # ---- 共聚物特殊处理 ----
    if smiles_input in ["ABS", "SAN", "EVA", "SBS", "NBR"]:
        st.subheader("📊 单体组成设置")
    
        if smiles_input == "ABS":
            col_a, col_b, col_s = st.columns(3)
            with col_a:
                acrylonitrile_pct = st.slider("丙烯腈 (%)", 0, 100, 30, step=1)
            with col_b:
                butadiene_pct = st.slider("丁二烯 (%)", 0, 100, 20, step=1)
            with col_s:
                styrene_pct = st.slider("苯乙烯 (%)", 0, 100, 50, step=1)
            total = acrylonitrile_pct + butadiene_pct + styrene_pct
            if total != 100:
                st.warning(f"⚠️ 单体比例总和为 {total}%，请调整为 100%")
            else:
                st.success(f"✅ 单体比例：丙烯腈 {acrylonitrile_pct}% + 丁二烯 {butadiene_pct}% + 苯乙烯 {styrene_pct}% = 100%")
    
        elif smiles_input == "SAN":
            col_s, col_a = st.columns(2)
            with col_s:
                styrene_pct = st.slider("苯乙烯 (%)", 0, 100, 70, step=1)
            with col_a:
                acrylonitrile_pct = st.slider("丙烯腈 (%)", 0, 100, 30, step=1)
            total = styrene_pct + acrylonitrile_pct
            if total != 100:
                st.warning(f"⚠️ 单体比例总和为 {total}%，请调整为 100%")
            else:
                st.success(f"✅ 单体比例：苯乙烯 {styrene_pct}% + 丙烯腈 {acrylonitrile_pct}% = 100%")
                
        elif smiles_input == "EVA":
            col_e, col_v = st.columns(2)
            with col_e:
                ethylene_pct = st.slider("乙烯 (%)", 0, 100, 70, step=1)
            with col_v:
                va_pct = st.slider("乙酸乙烯酯 (%)", 0, 100, 30, step=1)
            total = ethylene_pct + va_pct
            if total != 100:
                st.warning(f"⚠️ 单体比例总和为 {total}%，请调整为 100%")
            else:
                st.success(f"✅ 单体比例：乙烯 {ethylene_pct}% + 乙酸乙烯酯 {va_pct}% = 100%")
    
        elif smiles_input == "SBS":
            col_s1, col_b, col_s2 = st.columns(3)
            with col_s1:
                styrene1_pct = st.slider("苯乙烯 (嵌段1) (%)", 0, 100, 30, step=1)
            with col_b:
                butadiene_pct = st.slider("丁二烯 (%)", 0, 100, 40, step=1)
            with col_s2:
                styrene2_pct = st.slider("苯乙烯 (嵌段2) (%)", 0, 100, 30, step=1)
            total = styrene1_pct + butadiene_pct + styrene2_pct
            if total != 100:
                st.warning(f"⚠️ 单体比例总和为 {total}%，请调整为 100%")
            else:
                st.success(f"✅ 单体比例：苯乙烯 {styrene1_pct}% + 丁二烯 {butadiene_pct}% + 苯乙烯 {styrene2_pct}% = 100%")
    
        elif smiles_input == "NBR":
            col_b, col_a = st.columns(2)
            with col_b:
                butadiene_pct = st.slider("丁二烯 (%)", 0, 100, 70, step=1)
            with col_a:
                acrylonitrile_pct = st.slider("丙烯腈 (%)", 0, 100, 30, step=1)
            total = butadiene_pct + acrylonitrile_pct
            if total != 100:
                st.warning(f"⚠️ 单体比例总和为 {total}%，请调整为 100%")
            else:
                st.success(f"✅ 单体比例：丁二烯 {butadiene_pct}% + 丙烯腈 {acrylonitrile_pct}% = 100%")
                
    # ---- 通用输入框 ----
    for i, col in enumerate(feat_cols):
        with input_cols[i]:
            min_val, max_val = param_ranges.get(col, (0.0, 10.0))
            if "recommended_params" in st.session_state:
                default_val = st.session_state.recommended_params.get(col, float(data[col].median()))
            else:
                default_val = float(data[col].median())
            val = st.number_input(
                col,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=0.1,
                format="%.2f",
                key=f"pred_{col}_{i}"
            )
            input_vals.append(val)

    # ---- 开始预测 ----
    if st.button("🚀 开始预测", use_container_width=True, type="primary", key="predict_button_v2"):
    # ... 预测代码 ...
        if len(input_vals) != len(feat_cols):
            st.error(f"输入特征数量错误：期望 {len(feat_cols)} 个，实际 {len(input_vals)} 个")
            st.stop()
        input_array = np.array([input_vals])
        results = {target: models[target].predict(input_array)[0] for target in target_cols}
        
        st.markdown("### ✅ 预测结果")
        # 保存到历史记录
        if save_prediction(smiles_input, input_vals, results):
            st.success("✅ 预测结果已保存到历史记录")
                # ---- 导出结果 ----
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            # 导出为 CSV
            result_data = {"目标": list(results.keys()), "预测值": list(results.values())}
            result_df_export = pd.DataFrame(result_data)
            csv_export = result_df_export.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 导出为 CSV",
                data=csv_export,
                file_name="prediction_results.csv",
                mime="text/csv",
                key="export_csv"
            )
                
        explanations = {
            '模量(GPa)': '🔹 模量越高，材料越“硬”',
            '拉伸强度(MPa)': '🔹 强度越高，抗拉力越大',
            '断裂伸长率(%)': '🔹 伸长率越高，材料越“韧”',
            'Tg(°C)': '🔹 Tg 越高，耐热性越好'
        }
        result_cols = st.columns(len(target_cols))
        colors = ['#1e3a5f', '#2a4a7f', '#3a5a9f', '#4a6abf']
        for i, (target, value) in enumerate(results.items()):
            with result_cols[i]:
                st.markdown(f"""
                <div style="background:{colors[i%len(colors)]}; padding:15px; border-radius:10px; text-align:center;">
                    <p style="color:#90caf9; font-size:14px; margin:0;">{target}</p>
                    <p style="color:#4fc3f7; font-size:30px; font-weight:bold; margin:5px 0;">{value:.3f}</p>
                    <p style="color:#b0c4de; font-size:12px; margin:0;">{explanations.get(target, '')}</p>
                </div>
                """, unsafe_allow_html=True)

        # ---- 特征重要性 ----
        st.markdown("---")
        st.subheader("📊 特征重要性分析")
        st.caption("显示每个工艺参数对预测性能的影响程度（数值越高影响越大）")

        target_list = list(models.keys())
        if target_list:
            selected_target = st.selectbox(
                "选择要分析的目标性能",
                target_list,
                key="importance_selector"
            )
            if selected_target in models:
                model = models[selected_target]
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    imp_df = pd.DataFrame({
                        '工艺参数': feat_cols,
                        '重要性分数': importances,
                        '相对重要性 (%)': (importances / importances.sum() * 100).round(1)
                    }).sort_values('重要性分数', ascending=False)
                    st.dataframe(imp_df, use_container_width=True, hide_index=True)
                    st.subheader("📈 重要性可视化")
                    st.bar_chart(imp_df.set_index('工艺参数')['重要性分数'], use_container_width=True)
                    st.caption("💡 数值越大，该参数对预测结果的影响越大")
                else:
                    st.info("当前模型不支持特征重要性分析")
            else:
                st.warning(f"目标 '{selected_target}' 不在模型中")
        else:
            st.info("暂无目标列数据")

    # ---- 批量预测 ----
    with st.expander("📁 批量预测（上传 CSV 批量预测多组配方）"):
        st.caption("上传一个包含多组配方的 CSV 文件，系统将一次性预测所有配方的性能")
        
        uploaded_batch = st.file_uploader(
            "上传批量预测文件 (CSV)",
            type=['csv'],
            key="batch_upload",
            help="CSV 文件应包含与特征列名称相同的列"
        )
        
        if uploaded_batch is not None:
            try:
                batch_data = pd.read_csv(uploaded_batch)
                st.info(f"✅ 已加载 {len(batch_data)} 组配方")
                
                missing_cols = [col for col in feat_cols if col not in batch_data.columns]
                if missing_cols:
                    st.error(f"❌ CSV 缺少以下列：{missing_cols}")
                else:
                    if st.button("🚀 开始批量预测", key="batch_predict_btn"):
                        with st.spinner(f"正在预测 {len(batch_data)} 组配方..."):
                            X_batch = batch_data[feat_cols].values
                            batch_results = {}
                            for target in target_cols:
                                if target in models:
                                    batch_results[target] = models[target].predict(X_batch)
                            
                            result_df = batch_data.copy()
                            for target, preds in batch_results.items():
                                result_df[f'预测_{target}'] = preds
                            
                            st.subheader("📊 批量预测结果")
                            st.dataframe(result_df, use_container_width=True)
                            
                            csv = result_df.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                label="📥 下载预测结果 (CSV)",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                                key="download_batch"
                            )
            except Exception as e:
                st.error(f"读取文件失败：{e}")
        
        # ---- 示例 CSV ----
        with st.expander("📄 查看示例 CSV 格式"):
            st.caption("下载示例 CSV 文件，了解批量预测所需的格式")
            sample_df = pd.DataFrame({
                '单体比例(%)': [85, 80, 90],
                '引发剂(phr)': [0.5, 0.6, 0.4],
                '聚合温度(°C)': [80, 85, 75],
                '聚合时间(h)': [6, 7, 5],
                '交联剂(phr)': [0.05, 0.0, 0.1]
            })
            sample_csv = sample_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📄 下载示例 CSV",
                data=sample_csv,
                file_name="sample_batch.csv",
                mime="text/csv",
                key="download_sample"
            )
        # ---- 实验日志 ----
    with st.expander("📋 实验日志（历史记录）"):
        st.caption("保存每次预测的配方和结果，方便回溯对比")
        
        # 初始化数据库
        def init_db():
            conn = sqlite3.connect('experiment_log.db')
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    smiles TEXT,
                    inputs TEXT,
                    results TEXT
                )
            ''')
            conn.commit()
            conn.close()
        
        def save_prediction(smiles, inputs, results):
            try:
                conn = sqlite3.connect('experiment_log.db')
                c = conn.cursor()
                c.execute('''
            INSERT INTO predictions (timestamp, smiles, inputs, results)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            smiles,
            str(inputs),
            str(results)
        ))
                conn.commit()
                conn.close()
                st.success("✅ 记录已保存！")  # 👈 加这行
                return True
            except Exception as e:
                st.error(f"❌ 保存失败：{e}")  # 👈 加这行
                return False
        
        def load_history():
            conn = sqlite3.connect('experiment_log.db')
            c = conn.cursor()
            c.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 50')
            rows = c.fetchall()
            conn.close()
            return rows
        
        # 初始化数据库
        init_db()
        
        # 显示历史记录
        if st.button("📋 查看历史记录", key="show_history"):
            rows = load_history()
            if rows:
                history_df = pd.DataFrame(rows, columns=['ID', '时间', '分子', '输入参数', '预测结果'])
                st.dataframe(history_df, use_container_width=True, hide_index=True)
                st.caption(f"共 {len(rows)} 条记录（最近50条）")
            else:
                st.info("暂无历史记录")
    
            # ---- DoE 辅助 ----
    with st.expander("🧪 DoE 辅助（推荐下一组实验）"):
        st.caption("基于当前模型的不确定性，推荐最值得做的下一组实验配方")
        
        if st.button("🎯 推荐下一组实验", key="doe_suggest"):
            # 在参数空间内随机采样
            n_samples = 100
            samples = []
            for col in feat_cols:
                min_val, max_val = param_ranges.get(col, (0.0, 10.0))
                samples.append(np.random.uniform(min_val, max_val, n_samples))
            
            X_samples = np.array(samples).T
            
            # 计算每个目标的不确定性
            variances = []
            for target in target_cols:
                if target in models:
                    preds = models[target].predict(X_samples)
                    variances.append(np.var(preds))
            
            if variances:
                avg_variances = np.mean(variances, axis=0)
                best_idx = np.argmax(avg_variances)
                recommended = X_samples[best_idx]
                
                st.markdown("### 🎯 推荐配方")
                
                # 用表格显示，不用 st.metric
                rec_df = pd.DataFrame({
                    "工艺参数": feat_cols,
                    "推荐值": [f"{x:.2f}" for x in recommended]
                })
                st.dataframe(rec_df, use_container_width=True, hide_index=True)
                
                st.caption("💡 这个配方的预测不确定性最高，做这个实验能最大程度提升模型精度")
            else:
                st.info("暂无足够数据生成推荐")
        # ---- 部分依赖图 ----
    with st.expander("📈 部分依赖图（PDP）"):
        st.caption("展示单个工艺参数对预测性能的影响")
        
        if len(feat_cols) > 0 and len(target_cols) > 0:
            col_pdp_target = st.selectbox("选择目标性能", target_cols, key="pdp_target")
            col_pdp_feature = st.selectbox("选择工艺参数", feat_cols, key="pdp_feature")
            
            if st.button("📊 生成部分依赖图", key="pdp_generate"):
                model = models[col_pdp_target]
                
                # 生成特征值网格
                min_val, max_val = param_ranges.get(col_pdp_feature, (0.0, 10.0))
                grid = np.linspace(min_val, max_val, 30)
                
                # 对每个网格点，保持其他特征不变，预测目标
                base_input = np.array([float(data[col].median()) for col in feat_cols])
                pdp_values = []
                
                for val in grid:
                    input_copy = base_input.copy()
                    feature_idx = feat_cols.index(col_pdp_feature)
                    input_copy[feature_idx] = val
                    pred = model.predict([input_copy])[0]
                    pdp_values.append(pred)
                
                # 绘图
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(grid, pdp_values, 'b-', linewidth=2, marker='o', markersize=4)
                ax.set_xlabel(col_pdp_feature, fontsize=12)
                ax.set_ylabel(col_pdp_target, fontsize=12)
                ax.set_title(f'部分依赖图：{col_pdp_feature} → {col_pdp_target}', fontsize=14)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
                
                st.caption("💡 其他特征固定为中位数，该曲线显示单一特征对预测的影响")
        else:
            st.info("暂无数据生成部分依赖图")
# ---------- TAB 2: 优化 ----------
with tab2:
    st.subheader("🎯 设定目标性能，AI 自动推荐最优配方")

    target_goal_cols = st.columns(len(target_cols))
    target_goals = {}
    for i, target in enumerate(target_cols):
        with target_goal_cols[i]:
            target_goals[target] = st.number_input(
                f"目标 {target}",
                min_value=float(Y[target].min()),
                max_value=float(Y[target].max()),
                value=float(Y[target].quantile(0.75)),
                step=0.1,
                format="%.2f",
                key=f"opt_goal_{target}_{i}"
            )

    def objective(x):
        x = np.array(x).reshape(1, -1)
        preds = []
        for target in target_cols:
            pred = models[target].predict(x)[0]
            preds.append((pred - target_goals[target])**2)
        return np.sum(preds)

    bounds = []
    for col in feat_cols:
        min_val, max_val = param_ranges.get(col, (0.0, 10.0))
        bounds.append((min_val, max_val))

    # 优化按钮（不加 key，让 Streamlit 自动管理）
    if st.button("🧠 开始优化", use_container_width=True, type="primary"):
        with st.spinner("AI 正在搜索最优配方..."):
            best_result = None
            best_score = float('inf')
            for _ in range(10):
                x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res

        if best_result and best_result.success:
            optimal_x = best_result.x
            optimal_input = np.array([optimal_x])
            optimal_preds = {}
            for target in target_cols:
                optimal_preds[target] = models[target].predict(optimal_input)[0]

            st.markdown("### ✅ 推荐的最优配方")
            recipe_df = pd.DataFrame({
                "参数": feat_cols,
                "推荐值": [f"{x:.2f}" for x in optimal_x]
            })
            st.dataframe(recipe_df, use_container_width=True, hide_index=True)

            st.markdown("### 📊 该配方预测的性能")
            perf_df = pd.DataFrame({
                "性能指标": list(optimal_preds.keys()),
                "预测值": [f"{v:.3f}" for v in optimal_preds.values()],
                "目标值": [f"{target_goals[t]:.3f}" for t in optimal_preds.keys()]
            })
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

            st.info("💡 提示：这是 AI 找到的最接近你目标的配方。如果想更精确，可以增加数据量或缩小参数范围。")
        else:
            st.error("❌ 优化失败，请检查目标值是否在合理范围内，或调整参数范围")
        # ---- 多目标权衡分析 ----
    with st.expander("📈 多目标权衡分析（帕累托前沿）"):
        st.caption("展示两个目标性能之间的权衡关系")
        
        if len(target_cols) >= 2:
            # 让用户选择两个目标
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                target1 = st.selectbox("选择目标 1", target_cols, key="pareto_target1")
            with col_t2:
                target2 = st.selectbox("选择目标 2", target_cols, key="pareto_target2")
            
            if target1 != target2:
                if st.button("📊 生成帕累托前沿图", key="pareto_generate"):
                    # 用全部数据生成帕累托前沿
                    X_all = data[feat_cols].values
                    preds1 = models[target1].predict(X_all)
                    preds2 = models[target2].predict(X_all)
                    
                    # 帕累托前沿：找所有非支配点
                    pareto_indices = []
                    for i in range(len(preds1)):
                        dominated = False
                        for j in range(len(preds1)):
                            if i != j and preds1[j] >= preds1[i] and preds2[j] >= preds2[i]:
                                dominated = True
                                break
                        if not dominated:
                            pareto_indices.append(i)
                    
                    # 绘图
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(preds1, preds2, alpha=0.3, label='所有配方', color='#b0c4de')
                    ax.scatter(
                        preds1[pareto_indices], 
                        preds2[pareto_indices], 
                        color='#e74c3c', 
                        s=80, 
                        label='帕累托前沿',
                        edgecolors='white',
                        linewidth=1
                    )
                    ax.set_xlabel(target1, fontsize=12)
                    ax.set_ylabel(target2, fontsize=12)
                    ax.set_title(f'{target1} vs {target2} 权衡分析', fontsize=14)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.caption(f"💡 红色点为帕累托最优配方，共 {len(pareto_indices)} 个")
            else:
                st.info("请选择两个不同的目标")
        else:
            st.info("至少需要两个目标才能进行多目标权衡分析")
# ============================================================
# 分子结构输入 + 展示（下拉菜单版）
# ============================================================
st.markdown("---")
st.subheader("🔬 分子结构 2D 展示")

# 分子名称 → SMILES 映射
MOLECULE_NAMES = {
    "乙烯": "C=C",
    "乙醇": "CCO",
    "苯": "c1ccccc1",
    "丙烯": "C=CC",
    "丁二烯": "C=CC=C",
    "苯乙烯": "c1ccccc1C=C",
    "苯酚": "c1ccc(O)cc1",
    "丙烯腈": "C=CC#N",
    "甲基丙烯酸甲酯": "CC(=C)C(=O)OC",
    "乙酸乙烯酯": "CC(=O)OC=C",
    "氯乙烯": "C=Cl",
    "对苯二甲酸": "c1cc(C(=O)O)ccc1C(=O)O",
    "乙二醇": "C(CO)O",
    "环氧乙烷": "C1CO1",
    "ABS(丙烯腈-丁二烯-苯乙烯)": "ABS",
    "SAN(苯乙烯-丙烯腈)": "SAN",
    "EVA(乙烯-乙酸乙烯酯)":"EVA",
    "SBS(苯乙烯-丁二烯-苯乙烯)":"SBS",
    "NBR(丁腈橡胶)":"NBR"
}
# ---- 共聚物类型选择（仅当选择的是共聚物时显示） ----
is_copolymer = smiles_input in ["ABS", "SAN", "EVA", "SBS", "NBR"]
if is_copolymer:
    copolymer_type = st.selectbox(
        "选择共聚链类型",
        ["无规共聚", "嵌段共聚", "交替共聚"],
        key="copolymer_type"
    )
else:
    copolymer_type = None

col_mol1, col_mol2 = st.columns([2, 1])

with col_mol1:
    mol_names = list(MOLECULE_NAMES.keys())
    selected_name = st.selectbox(
        "选择分子",
        mol_names,
        index=0,
        key="mol_selector"
    )
    st.session_state.smiles_input = MOLECULE_NAMES[selected_name]
    smiles_input = MOLECULE_NAMES[selected_name]
    st.caption(f"当前 SMILES: `{smiles_input}`")

with col_mol2:
    st.write("或")
    manual_smiles = st.text_input(
        "手动输入 SMILES",
        value="",
        placeholder="如 C=C",
        key="manual_smiles"
    )
    if manual_smiles:
        smiles_input = manual_smiles
def draw_copolymer_chain(draw, mol_data, copolymer_type, num_monomers=8):
    """绘制共聚物链结构"""
    # ---- 在函数内部加载字体 ----
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
        font_title = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = font
        font_title = font
    
    # 定义颜色和标签
    monomer_colors = {
        'A': '#e74c3c',  # 红色（丙烯腈）
        'B': '#3498db',  # 蓝色（丁二烯）
        'S': '#f1c40f',  # 黄色（苯乙烯）
        'E': '#2ecc71',  # 绿色（乙烯）
        'V': '#9b59b6',  # 紫色（乙酸乙烯酯）
    }
    monomer_labels = {
        'A': 'A (丙烯腈)', 'B': 'B (丁二烯)', 'S': 'S (苯乙烯)',
        'E': 'E (乙烯)', 'V': 'V (乙酸乙烯酯)'
    }
    
    # 确定单体的字母序列
    name = mol_data['name']
    if 'ABS' in name or '丙烯腈-丁二烯-苯乙烯' in name:
        letters = ['A', 'B', 'S']
    elif 'SAN' in name or '苯乙烯-丙烯腈' in name:
        letters = ['S', 'A']
    elif 'EVA' in name or '乙烯-乙酸乙烯酯' in name:
        letters = ['E', 'V']
    elif 'SBS' in name or '苯乙烯-丁二烯-苯乙烯' in name:
        letters = ['S', 'B']
    elif 'NBR' in name or '丁腈橡胶' in name:
        letters = ['B', 'A']
    else:
        letters = ['A', 'B', 'S']
    
    # 根据类型生成序列
    if copolymer_type == "无规共聚":
        np.random.seed(42)
        sequence = list(np.random.choice(letters, size=num_monomers))
    elif copolymer_type == "嵌段共聚":
        block_size = num_monomers // len(letters)
        sequence = []
        for i, letter in enumerate(letters):
            if i == len(letters) - 1:
                sequence.extend([letter] * (num_monomers - len(sequence)))
            else:
                sequence.extend([letter] * block_size)
    else:  # 交替共聚
        sequence = []
        for i in range(num_monomers):
            sequence.append(letters[i % len(letters)])
    
    # 绘制链
    spacing = 60
    start_x = 80
    y = 250
    radius = 20
    
    # 绘制连接线
    for i in range(num_monomers - 1):
        x1 = start_x + i * spacing
        x2 = start_x + (i + 1) * spacing
        draw.line([(x1 + radius, y), (x2 - radius, y)], fill='#7f8c8d', width=3)
    
    # 绘制每个单体
    for i, letter in enumerate(sequence):
        x = start_x + i * spacing
        color = monomer_colors.get(letter, '#95a5a6')
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], 
                     fill=color, outline='#2c3e50', width=2)
        # 使用函数内部加载的 font
        draw.text((x, y), letter, fill='white', font=font, anchor='mm')
    
    # 添加图例
    legend_x = 50
    legend_y = 380
    for letter in set(sequence):
        color = monomer_colors.get(letter, '#95a5a6')
        draw.rectangle([legend_x, legend_y, legend_x + 20, legend_y + 20], 
                      fill=color, outline='#2c3e50')
        draw.text((legend_x + 28, legend_y + 2), monomer_labels.get(letter, letter), 
                  fill='#2c3e50', font=font_small)
        legend_x += 150
    
    # 显示共聚类型
    draw.text((350, 320), f"共聚类型: {copolymer_type}", fill='#2c3e50', font=font_title, anchor='mt')
    
    return sequence

if smiles_input in MOLECULES:
    mol_data = MOLECULES[smiles_input]
    
    # 绘制分子结构图
    img = Image.new('RGB', (700, 500), color='white')
    draw = ImageDraw.Draw(img)
    is_copolymer = smiles_input in ["ABS", "SAN", "EVA", "SBS", "NBR"]
    if is_copolymer and copolymer_type:
        sequence = draw_copolymer_chain(draw, mol_data, copolymer_type,num_monomers=8)
    else:  
        for x1, y1, x2, y2 in mol_data["double_bonds"]:
            draw.line([(x1, y1), (x2, y2)], fill='#7f8c8d', width=4)
       # ===== 对苯二甲酸：强制画两个 C=O 双键（灰色，间距拉开）=====
        if smiles_input == "c1cc(C(=O)O)ccc1C(=O)O":
        # 左上 C7=O1（第一条线 + 第二条线偏移 8px）
            draw.line([(150, 120), (180, 60)], fill='#7f8c8d', width=3)
            draw.line([(158, 112), (188, 52)], fill='#7f8c8d', width=3)
        # 右下 C8=O3（第一条线 + 第二条线偏移 8px）
            draw.line([(450, 230), (520, 200)], fill='#7f8c8d', width=3)
            draw.line([(458, 222), (528, 192)], fill='#7f8c8d', width=3)
        for x1, y1, x2, y2 in mol_data["single_bonds"]:
            draw.line([(x1, y1), (x2, y2)], fill='#7f8c8d', width=3)
        for x1, y1, x2, y2 in mol_data.get("triple_bonds", []):
            draw.line([(x1, y1), (x2, y2)], fill='#7f8c8d', width=3)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        for atom_id, (x, y, color, label) in mol_data["atoms"].items():
            radius = 25 if label in ['C', 'O'] else 20
            draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], 
                         fill=color, outline='#2c3e50', width=2)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x - text_width//2, y - text_height//2), label, 
                      fill='white', font=font)
        
        legend_y = 450
        colors = [('#2ecc71', '碳 (C)'), ('#3498db', '氢 (H)'), ('#e74c3c', '氧 (O)')]
        x_offset = 50
        for color, label in colors:
            if any(color == atom[2] for atom in mol_data["atoms"].values()):
                draw.rectangle([x_offset, legend_y, x_offset+20, legend_y+20], 
                              fill=color, outline='#2c3e50')
                draw.text((x_offset+28, legend_y+2), label, fill='#2c3e50', font=font_small)
                x_offset += 130

    st.image(img, caption=f"{mol_data['name']} ({mol_data['formula']}) · 分子量 {mol_data['weight']} g/mol", use_container_width=False)
        
        # ---- 联动按钮（只保留这一个） ----
if "params" in mol_data:
    # 用 unique key 避免冲突
    if st.button(f"📥 应用 {mol_data['name']} 的推荐参数", key=f"apply_{smiles_input}"):
        st.session_state.recommended_params = mol_data["params"]
        st.success(f"✅ 已应用 {mol_data['name']} 的推荐参数！")
        st.rerun()
else:
    st.info(f"当前支持的分子：{', '.join(MOLECULES.keys())}。更多分子开发中...")

# ============================================================
# 高分子链结构展示
# ============================================================
st.subheader("🧬 高分子链结构展示")

chain_type = st.selectbox(
    "选择链结构类型",
    ["线型", "支化", "交联"]
)

img = Image.new('RGB', (700, 500), color='white')
draw = ImageDraw.Draw(img)

try:
    font_title = ImageFont.truetype("arial.ttf", 20)
    font_label = ImageFont.truetype("arial.ttf", 14)
except:
    font_title = ImageFont.load_default()
    font_label = ImageFont.load_default()

COLOR_MAIN = '#2ecc71'
COLOR_SIDE = '#3498db'
COLOR_CROSS = '#e74c3c'
COLOR_CROSS_POINT = '#e74c3c'

def draw_circle(draw, x, y, radius, color, outline='#2c3e50', width=2):
    draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill=color, outline=outline, width=width)

def draw_line(draw, x1, y1, x2, y2, color, width=3):
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

if chain_type == "线型":
    xs = np.linspace(50, 650, 10)
    ys = [250] * 10
    for i in range(len(xs)-1):
        draw_line(draw, xs[i], ys[i], xs[i+1], ys[i+1], COLOR_MAIN, 4)
    for x, y in zip(xs, ys):
        draw_circle(draw, x, y, 18, COLOR_MAIN)
    draw.text((350, 180), "线型高分子链", fill='#2c3e50', font=font_title, anchor='mt')
    draw_circle(draw, 80, 380, 14, COLOR_MAIN)
    draw.text((105, 375), "主链", fill='#2c3e50', font=font_label)

elif chain_type == "支化":
    xs = np.linspace(50, 650, 10)
    ys = [250] * 10
    for i in range(len(xs)-1):
        draw_line(draw, xs[i], ys[i], xs[i+1], ys[i+1], COLOR_MAIN, 4)
    for x, y in zip(xs, ys):
        draw_circle(draw, x, y, 18, COLOR_MAIN)
    for idx in [2, 5, 8]:
        x = xs[idx]
        draw_line(draw, x, y, x+30, y-50, COLOR_SIDE, 3)
        draw_circle(draw, x+30, y-50, 14, COLOR_SIDE)
        draw_line(draw, x, y, x+30, y+50, COLOR_SIDE, 3)
        draw_circle(draw, x+30, y+50, 14, COLOR_SIDE)
    draw.text((350, 160), "支化高分子链", fill='#2c3e50', font=font_title, anchor='mt')
    draw_circle(draw, 80, 380, 14, COLOR_MAIN)
    draw.text((105, 375), "主链", fill='#2c3e50', font=font_label)
    draw_circle(draw, 220, 380, 14, COLOR_SIDE)
    draw.text((245, 375), "侧链", fill='#2c3e50', font=font_label)

elif chain_type == "交联":
    xs = np.linspace(50, 650, 10)
    y1 = 180
    y2 = 320
    for i in range(len(xs)-1):
        draw_line(draw, xs[i], y1, xs[i+1], y1, COLOR_MAIN, 4)
        draw_line(draw, xs[i], y2, xs[i+1], y2, COLOR_MAIN, 4)
    for x in xs:
        draw_circle(draw, x, y1, 16, COLOR_MAIN)
        draw_circle(draw, x, y2, 16, COLOR_MAIN)
    for idx in [1, 4, 7, 9]:
        x = xs[idx]
        draw_line(draw, x, y1, x, y2, COLOR_CROSS, 2)
        draw_circle(draw, x, y1, 16, COLOR_CROSS_POINT)
        draw_circle(draw, x, y2, 16, COLOR_CROSS_POINT)
    draw.text((350, 100), "交联高分子链", fill='#2c3e50', font=font_title, anchor='mt')
    draw_circle(draw, 80, 380, 14, COLOR_MAIN)
    draw.text((105, 375), "主链", fill='#2c3e50', font=font_label)
    draw_line(draw, 220, 386, 260, 386, COLOR_CROSS, 3)
    draw.text((270, 380), "交联", fill='#2c3e50', font=font_label)
    draw_circle(draw, 360, 386, 14, COLOR_CROSS_POINT)
    draw.text((385, 380), "交联点", fill='#2c3e50', font=font_label)

st.image(img, caption=f"{chain_type}高分子链结构", use_container_width=False)
