"""
MaaS多阶段采纳决策的隐马尔可夫模型 (HMM) - 全数据模型 (Binary Stage 1)
Integrated calibration: 同时估计初始状态、转移概率和两阶段发射概率

数据组: 全部4个CSV (6,060 obs, 1:1 paired)
- Group 1 (选了MaaS): keep_positive + change_negative
- Group 2 (没选MaaS): keep_negative + change_positive
- 阶段1: 二元选择 (0=不转移, 1=MaaS)  ← 从5选项合并
- 阶段2: results ∈ {1,2,3,4,5} → 全5选项

模型动机:
AllData 5选项模型失败: Group 2 (56%) 全选 CHOICE_T1=0, 对M1-M4替代方案特定参数
(RailTime, TripTime, ASC) 提供零变异, 导致ASC_Diff2/3=0(状态识别失败),
Trans_Base饱和(~12.6)。合并为二元选择后, Group 2数据能贡献"不选MaaS vs 选MaaS"
的信息, 449人有组内变异。

模型结构:
- 基于Group 1 v4的参数结构, 阶段1简化为二元
- T=2 时间阶段, States=2 隐状态 (Skeptic, Enthusiast)
- 阶段1: 二元MaaS选择 (0=不转移, 1=MaaS) — 单一ASC_Diff识别约束
- 阶段2: 套餐订阅选择 (5选项: Bus First, Metro Access, Value Taxi, Ultra Access, PAYG)
- 两阶段均使用MNL (统一结构)
- 变量使用原始尺度 (撤销CSV预处理中的÷10和÷100)
- 总参数: 85 (Init:17 + Trans:11 + Stage1:13 + Stage2:44)

冷启动: 所有参数初始值为0 (ASC_Diff=1.0避免从边界开始)
"""

import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import logging
from biogeme.expressions import Beta, Variable, log, exp

# =============================================================================
# 1. 数据加载与预处理
# =============================================================================

def load_and_preprocess(filepath):
    """加载并预处理数据"""
    if filepath.endswith('.csv'):
        raw_df = pd.read_csv(filepath, encoding='gb2312')
    else:
        raw_df = pd.read_excel(filepath)

    df = raw_df.copy()

    # cost处理: 150以下为0, 150以上为1
    df['cost'] = df['cost'].replace([1,2,3,4,5,6], [0,0,1,1,1,1])

    # 阶段1 Choice (映射为 0-4)
    maas_map = {1:0, 4:1, 5:2, 6:3, 7:4}
    df['CHOICE_T1'] = df['maas'].map(maas_map)

    # 阶段2 Choice (映射为 0-4)
    df['CHOICE_T2'] = df['results'] - 1

    df = df.dropna(subset=['CHOICE_T1', 'CHOICE_T2'])

    int_cols = ['distance1', 'distance2', 'distance3', 'distance4', 'distance5',
                'income1', 'income2', 'income3',
                'age1', 'age2', 'age3', 'age4',
                'match_bus', 'match_metro', 'match_bike','first_car','first_taxi','first_pt','morning','evening','normal','late']
    for col in int_cols:
        if col in df.columns:
            if df[col].max() < 1 and df[col].min() >= 0:
                df[col] = (df[col] > 0.05).astype(int)
            else:
                df[col] = df[col].astype(int)

    return df

# ============ 数据模式开关 ============
USE_PAIRED = False  # True=1:1配对 (正确SE), False=笛卡尔积全量 (更多观测)
# ======================================

# 加载全部数据: Group 1 + Group 2
files = [
    'data/最终模型数据_keep_positive.csv',
    'data/最终模型数据_change_negative.csv',
    'data/最终模型数据_keep_negative.csv',
    'data/最终模型数据_change_positive.csv',
]
dfs = [load_and_preprocess(f) for f in files]
df_processed = pd.concat(dfs, ignore_index=True)

df_scores = pd.read_csv('data/factor_scores.csv')
df_processed = df_processed.merge(df_scores, on='peopleID')

# --- 1:1 顺序配对 (可选) ---
if USE_PAIRED:
    S1_LOS_COLS = ['M1ttimerail', 'M1triptime', 'M2ttime_rail', 'M2triptime',
                   'M3ttime_rail', 'M3triptime', 'M4ttime', 'maas']
    S2_LOS_COLS = ['taxi_12', 'price1', 'price2', 'price3', 'price4', 'results']
    paired_rows = []
    for pid, group in df_processed.groupby('peopleID', sort=False):
        group = group.reset_index(drop=True)
        n_s1 = len(group[S1_LOS_COLS].drop_duplicates())
        n_s2 = len(group[S2_LOS_COLS].drop_duplicates())
        n_pairs = min(n_s1, n_s2)
        for i in range(n_pairs):
            paired_rows.append(group.iloc[i * n_s2 + i])
    df_processed = pd.DataFrame(paired_rows).reset_index(drop=True)
    print(f"1:1配对后观测数: {len(df_processed)} (from {sum(len(d) for d in dfs)} Cartesian)")

# have_car编码从1/2改为0/1
df_processed['have_car'] = df_processed['have_car'] - 1

# 还原原始尺度: 撤销CSV中的÷10和÷100预处理
# binary变量(first_taxi等)已在threshold转换为0/1, 不需要还原

# ÷10 → ×10 (连续变量)
cols_x10 = ['M4ttime', 'time_savings', 'cost_savings']
for col in cols_x10:
    if col in df_processed.columns:
        df_processed[col] = df_processed[col] * 10

# ÷100 → ×100
cols_x100 = [
    'M1ttimerail', 'M1triptime', 'M2ttime_rail', 'M2triptime',
    'M3ttime_rail', 'M3triptime',
    'price1', 'price2', 'price3', 'price4',
    'taxi_12',
]
for col in cols_x100:
    if col in df_processed.columns:
        df_processed[col] = df_processed[col] * 100

# --- 二元阶段1选择 ---
# 0=不转移, 1=MaaS (M1/M2/M3/M4)
df_processed['CHOICE_T1_BIN'] = (df_processed['CHOICE_T1'] > 0).astype(int)

# 聚合LOS变量: M1/M2/M3 平均轨道时间 (已还原为原始尺度)
df_processed['avg_rail_time'] = df_processed[['M1ttimerail', 'M2ttime_rail', 'M3ttime_rail']].mean(axis=1)

# 数据验证
print("=" * 70)
print("全数据模型验证 (AllData Binary) - raw scale")
print("=" * 70)
print(f"总观测数: {len(df_processed)}")
print(f"\n阶段1选择分布 (原始5选项):")
print(df_processed['CHOICE_T1'].value_counts().sort_index())
print(f"\n阶段1选择分布 (二元):")
print(df_processed['CHOICE_T1_BIN'].value_counts().sort_index())
print(f"\n阶段2选择分布:")
print(df_processed['CHOICE_T2'].value_counts().sort_index())
print(f"\n新变量验证:")
print(f"  avg_rail_time: {df_processed['avg_rail_time'].describe()[['min','mean','max']].to_dict()}")
print(f"\n尺度验证:")
print(f"  M1ttimerail: {df_processed['M1ttimerail'].describe()[['min','max']].to_dict()}")
print(f"  price1:      {df_processed['price1'].describe()[['min','max']].to_dict()}")
print("=" * 70)

database = db.Database('MaaS_HMM_AllData_Binary', df_processed)

# 定义数据变量 (biogeme 3.3.x API)
CHOICE_T1_BIN = Variable('CHOICE_T1_BIN')
CHOICE_T2 = Variable('CHOICE_T2')
# 阶段1 LOS变量 (二元模型)
first_taxi = Variable('first_taxi')
first_pt = Variable('first_pt')
distance5 = Variable('distance5')
normal = Variable('normal')
avg_rail_time = Variable('avg_rail_time')
# 阶段2 LOS变量
taxi_12 = Variable('taxi_12')
price_12 = Variable('price_12')
price_3 = Variable('price_3')
price_4 = Variable('price_4')
price1 = Variable('price1')
price2 = Variable('price2')
price3 = Variable('price3')
price4 = Variable('price4')
cost = Variable('cost')
license = Variable('license')
have_car = Variable('have_car')
education = Variable('education')
# 个人属性变量
sex = Variable('sex')
age2 = Variable('age2')
age3 = Variable('age3')
age4 = Variable('age4')
income1 = Variable('income1')
income2 = Variable('income2')
occupy = Variable('occupy')
week_bus = Variable('week_bus')
week_metro = Variable('week_metro')
week_taxi = Variable('week_taxi')
week_ebike = Variable('week_ebike')
e_bike = Variable('e_bike')
travel_distance_work = Variable('travel_distance_work')
travel_distance_weekend = Variable('travel_distance_weekend')
MaasFamiliar = Variable('MaasFamiliar')
d6 = Variable('d6')
f6 = Variable('f6')
c6 = Variable('c6')
c7 = Variable('c7')
# 转移协变量
time_savings = Variable('time_savings')
cost_savings = Variable('cost_savings')
match_bus = Variable('match_bus')
match_metro = Variable('match_metro')
match_bike = Variable('match_bike')
match_e_bike = Variable('match_e_bike')
match_taxi = Variable('match_taxi')
match_price = Variable('match_price')
price_ratio = Variable('price_ratio')

FACTOR1 = Variable('FACTOR1')
FACTOR2 = Variable('FACTOR2')
FACTOR3 = Variable('FACTOR3')
FACTOR4 = Variable('FACTOR4')
FACTOR6 = Variable('FACTOR6')

# FACTOR1 Init 效应参数
B_FACTOR1_Init = Beta('B_FACTOR1_Init', 0, None, None, 0)
B_FACTOR2_Init = Beta('B_FACTOR2_Init', 0, None, None, 0)
B_FACTOR3_Init = Beta('B_FACTOR3_Init', 0, None, None, 0)
B_FACTOR4_Init = Beta('B_FACTOR4_Init', 0, None, None, 0)
B_FACTOR6_Init = Beta('B_FACTOR6_Init', 0, None, None, 0)

# =============================================================================
# 2. 参数定义 (Beta Parameters) - 冷启动: 所有初始值为0
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1 初始状态概率参数 (Initial State Probability) — 17 params
# -----------------------------------------------------------------------------
ASC_Init_S2 = Beta('ASC_Init_S2', 0, None, None, 0)

B_MaasFamiliar_Init = Beta('B_MaasFamiliar_Init', 0, None, None, 0)
B_Sex_Init = Beta('B_Sex_Init', 0, None, None, 0)
B_Age2_Init = Beta('B_Age2_Init', 0, None, None, 0)
B_Age4_Init = Beta('B_Age4_Init', 0, None, None, 0)
B_Income1_Init = Beta('B_Income1_Init', 0, None, None, 0)
B_Income2_Init = Beta('B_Income2_Init', 0, None, None, 0)
B_Education_Init = Beta('B_Education_Init', 0, None, None, 0)
B_Occupy_Init = Beta('B_Occupy_Init', 0, None, None, 0)
B_WeekMetro_Init = Beta('B_WeekMetro_Init', 0, None, None, 0)
B_WeekBus_Init = Beta('B_WeekBus_Init', 0, None, None, 0)
B_WeekTaxi_Init = Beta('B_WeekTaxi_Init', 0, None, None, 0)
B_WeekEbike_Init = Beta('B_WeekEbike_Init', 0, None, None, 0)
B_HaveCar_Init = Beta('B_HaveCar_Init', 0, None, None, 0)
B_Ebike_Init = Beta('B_Ebike_Init', 0, None, None, 0)
B_D6_Init = Beta('B_D6_Init', 0, None, None, 0)
B_F6_Init = Beta('B_F6_Init', 0, None, None, 0)

# -----------------------------------------------------------------------------
# 2.2 状态转移概率参数 (State Transition) — 11 params
# -----------------------------------------------------------------------------
Trans_Base_S1_to_S2 = Beta('Trans_Base_S1_to_S2', 0, None, None, 0)
Trans_Base_S2_to_S2 = Beta('Trans_Base_S2_to_S2', 0, None, None, 0)

G_TimeSavings = Beta('G_TimeSavings', 0, None, None, 0)
G_CostSavings = Beta('G_CostSavings', 0, None, None, 0)
G_MatchBus = Beta('G_MatchBus', 0, None, None, 0)
G_MatchMetro = Beta('G_MatchMetro', 0, None, None, 0)
G_MatchBike = Beta('G_MatchBike', 0, None, None, 0)
G_MatchEbike = Beta('G_MatchEbike', 0, None, None, 0)
G_MatchTaxi = Beta('G_MatchTaxi', 0, None, None, 0)
G_MatchPrice = Beta('G_MatchPrice', 0, None, None, 0)
G_PriceRatio = Beta('G_PriceRatio', 0, None, None, 0)

# -----------------------------------------------------------------------------
# 2.3 阶段1效用参数 - 二元MaaS选择 (State-Specific) — 13 params
# 从5选项合并: 删除ASC_M1-M3_S1, ASC_Diff1-3, ASC_M4_S2,
#              B_RailTime_S1/S2, B_TripTime_S1/S2, B_Normal_S2
# 新增: ASC_MaaS_S1, ASC_Diff (单一识别约束), B_AvgRailTime_S1/S2
# -----------------------------------------------------------------------------

# --- ASC: 二元MaaS选择 ---
ASC_MaaS_S1 = Beta('ASC_MaaS_S1', 0, None, None, 0)
ASC_Diff = Beta('ASC_Diff', 1.0, 0, None, 0)  # >= 0, 单一识别约束
ASC_MaaS_S2 = ASC_MaaS_S1 + ASC_Diff

# --- State 1 (Skeptic) 阶段1参数 ---
B_FirstTaxi_S1 = Beta('B_FirstTaxi_S1', 0, None, None, 0)
B_FirstPT_S1 = Beta('B_FirstPT_S1', 0, None, None, 0)
B_Distance5_S1 = Beta('B_Distance5_S1', 0, None, None, 0)
B_Normal_S1 = Beta('B_Normal_S1', 0, None, None, 0)
B_AvgRailTime_S1 = Beta('B_AvgRailTime_S1', 0, None, None, 0)

# --- State 2 (Enthusiast) 阶段1参数 ---
B_FirstTaxi_S2 = Beta('B_FirstTaxi_S2', 0, None, None, 0)
B_FirstPT_S2 = Beta('B_FirstPT_S2', 0, None, None, 0)
B_Distance5_S2 = Beta('B_Distance5_S2', 0, None, None, 0)
B_Normal_S2 = Beta('B_Normal_S2', 0, None, None, 0)
B_AvgRailTime_S2 = Beta('B_AvgRailTime_S2', 0, None, None, 0)

# -----------------------------------------------------------------------------
# 2.4 阶段2效用参数 - 套餐订阅选择 (State-Specific) — 44 params
# 与v4完全相同, 冷启动
# -----------------------------------------------------------------------------

# --- State 1 (Skeptic) 阶段2参数 ---
ASC_Bus_S1 = Beta('ASC_Bus_S1', 0, None, None, 0)
ASC_Metro_S1 = Beta('ASC_Metro_S1', 0, None, None, 0)
ASC_Taxi_S1 = Beta('ASC_Taxi_S1', 0, None, None, 0)
ASC_Ultra_S1 = Beta('ASC_Ultra_S1', 0, None, None, 0)
ASC_Payg_S1 = Beta('ASC_Payg_S1', 0, None, None, 0)

B_Taxi12_S1 = Beta('B_Taxi12_S1', 0, None, None, 0)
B_Price_S1 = Beta('B_Price_S1', 0, None, None, 0)
B_Price_S1_TU = Beta('B_Price_S1_TU', 0, None, None, 0)

B_WeekBus_S1 = Beta('B_WeekBus_S1', 0, None, None, 0)
B_Ebike_S1 = Beta('B_Ebike_S1', 0, None, None, 0)
B_Occupy_S1 = Beta('B_Occupy_S1', 0, None, None, 0)
B_Income1_S1 = Beta('B_Income1_S1', 0, None, None, 0)
B_Age4_S1 = Beta('B_Age4_S1', 0, None, None, 0)
B_TravelDistWork_S1 = Beta('B_TravelDistWork_S1', 0, None, None, 0)
B_WeekMetro_S1 = Beta('B_WeekMetro_S1', 0, None, None, 0)
B_C7_S1 = Beta('B_C7_S1', 0, None, None, 0)
B_TravelDistWeekend_S1 = Beta('B_TravelDistWeekend_S1', 0, None, None, 0)
B_WeekTaxi_S1 = Beta('B_WeekTaxi_S1', 0, None, None, 0)
B_Age3_S1 = Beta('B_Age3_S1', 0, None, None, 0)
B_Income2_S1 = Beta('B_Income2_S1', 0, None, None, 0)
B_C6_S1 = Beta('B_C6_S1', 0, None, None, 0)
B_License_S1 = Beta('B_License_S1', 0, None, None, 0)
B_HaveCar_S1 = Beta('B_HaveCar_S1', 0, None, None, 0)
B_PriceRatio_S1 = Beta('B_PriceRatio_S1', 0, None, None, 0)
B_Sex_S1 = Beta('B_Sex_S1', 0, None, None, 0)
B_Cost_S1 = Beta('B_Cost_S1', 0, None, None, 0)
B_Education_S1 = Beta('B_Education_S1', 0, None, None, 0)

# --- State 2 (Enthusiast) 阶段2参数 ---
ASC_Bus_S2 = Beta('ASC_Bus_S2', 0, None, None, 0)
ASC_Metro_S2 = Beta('ASC_Metro_S2', 0, None, None, 0)
ASC_Taxi_S2 = Beta('ASC_Taxi_S2', 0, None, None, 0)
ASC_Ultra_S2 = Beta('ASC_Ultra_S2', 0, None, None, 0)
ASC_Payg_S2 = Beta('ASC_Payg_S2', 0, None, None, 0)

B_Taxi12_S2 = Beta('B_Taxi12_S2', 0, None, None, 0)
B_PriceRatio_S2 = Beta('B_PriceRatio_S2', 0, None, None, 0)
B_Price_S2 = Beta('B_Price_S2', 0, None, None, 0)

B_WeekBus_S2 = Beta('B_WeekBus_S2', 0, None, None, 0)
B_Ebike_S2 = Beta('B_Ebike_S2', 0, None, None, 0)
B_Occupy_S2 = Beta('B_Occupy_S2', 0, None, None, 0)
B_Sex_S2 = Beta('B_Sex_S2', 0, None, None, 0)
B_Income1_S2 = Beta('B_Income1_S2', 0, None, None, 0)
B_Age4_S2 = Beta('B_Age4_S2', 0, None, None, 0)
B_TravelDistWork_S2 = Beta('B_TravelDistWork_S2', 0, None, None, 0)
B_WeekMetro_S2 = Beta('B_WeekMetro_S2', 0, None, None, 0)
B_C7_S2 = Beta('B_C7_S2', 0, None, None, 0)
B_TravelDistWeekend_S2 = Beta('B_TravelDistWeekend_S2', 0, None, None, 0)
B_WeekTaxi_S2 = Beta('B_WeekTaxi_S2', 0, None, None, 0)
B_Age3_S2 = Beta('B_Age3_S2', 0, None, None, 0)
B_Income2_S2 = Beta('B_Income2_S2', 0, None, None, 0)
B_Cost_S2 = Beta('B_Cost_S2', 0, None, None, 0)
B_License_S2 = Beta('B_License_S2', 0, None, None, 0)
B_HaveCar_S2 = Beta('B_HaveCar_S2', 0, None, None, 0)
B_Education_S2 = Beta('B_Education_S2', 0, None, None, 0)
B_C6_S2 = Beta('B_C6_S2', 0, None, None, 0)


# =============================================================================
# 3. 效用函数定义 (Utility Functions)
# =============================================================================

# -----------------------------------------------------------------------------
# 3.1 阶段1效用: 二元MaaS选择 (0=不转移, 1=MaaS)
# -----------------------------------------------------------------------------

# --- State 1 (Skeptic) ---
V1_NoMaaS_S1 = 0  # 参照

V1_MaaS_S1 = (ASC_MaaS_S1 +
              B_FirstPT_S1 * first_pt +
              B_FirstTaxi_S1 * first_taxi +
              B_Normal_S1 * normal +
              B_Distance5_S1 * distance5 +
              B_AvgRailTime_S1 * avg_rail_time)

V1_Map_S1 = {0: V1_NoMaaS_S1, 1: V1_MaaS_S1}

# --- State 2 (Enthusiast) ---
V1_NoMaaS_S2 = 0  # 参照

V1_MaaS_S2 = (ASC_MaaS_S2 +
              B_FirstPT_S2 * first_pt +
              B_FirstTaxi_S2 * first_taxi +
              B_Normal_S2 * normal +
              B_Distance5_S2 * distance5 +
              B_AvgRailTime_S2 * avg_rail_time)

V1_Map_S2 = {0: V1_NoMaaS_S2, 1: V1_MaaS_S2}

# -----------------------------------------------------------------------------
# 3.2 阶段2效用: 套餐订阅选择 (与v4完全相同)
# -----------------------------------------------------------------------------

# --- State 1 (Skeptic) Utilities ---
V2_0_S1 = (ASC_Bus_S1 +
           B_Taxi12_S1 * taxi_12 +
        #    B_PriceRatio_S1 * price_12 +
           B_Price_S1 * price1 +
           B_WeekBus_S1 * week_bus +
           B_Ebike_S1 * e_bike +
           B_Occupy_S1 * occupy +
           B_Sex_S1 * sex +
           B_Income1_S1 * income1 +
           B_Age4_S1 * age4)

V2_1_S1 = (
    ASC_Metro_S1 +
           B_Taxi12_S1 * taxi_12 +
        #    B_PriceRatio_S1 * price_12 +
           B_Price_S1 * price2 +
           B_TravelDistWork_S1 * travel_distance_work +
           B_WeekMetro_S1 * week_metro +
           B_Ebike_S1 * e_bike +
           B_Occupy_S1 * occupy +
           B_Sex_S1 * sex +
           B_C7_S1 * c7 +
           B_Income1_S1 * income1 +
           B_Age4_S1 * age4)

V2_2_S1 = (
    # ASC_Taxi_S1 +
        #    B_PriceRatio_S1 * price_3 +
           B_Price_S1_TU * price3 +
           B_TravelDistWeekend_S1 * travel_distance_weekend +
           B_WeekTaxi_S1 * week_taxi +
           B_Age3_S1 * age3 +
           B_Income2_S1 * income2)

V2_3_S1 = (
    # ASC_Ultra_S1 +
        #    B_PriceRatio_S1 * price_4 +
           B_Price_S1_TU * price4 +
           B_C6_S1 * c6 +
           B_Cost_S1 * cost +
           B_WeekTaxi_S1 * week_taxi +
           B_Age3_S1 * age3)

V2_4_S1 = (
        #    ASC_Payg_S1 +
           B_Cost_S1 * cost +
           B_License_S1 * license +
        #    B_HaveCar_S1 * have_car +
           B_Education_S1 * education)

V2_Map_S1 = {0: V2_0_S1, 1: V2_1_S1, 2: V2_2_S1, 3: V2_3_S1, 4: V2_4_S1}

# --- State 2 (Enthusiast) Utilities ---
V2_0_S2 = (ASC_Bus_S2 +
           B_Taxi12_S2 * taxi_12 +
        #    B_PriceRatio_S2 * price_12 +
           B_Price_S2 * price1 +
           B_WeekBus_S2 * week_bus +
           B_Ebike_S2 * e_bike +
           B_Occupy_S2 * occupy +
           B_Sex_S2 * sex +
           B_Income1_S2 * income1 +
           B_Age4_S2 * age4)

V2_1_S2 = (
    ASC_Metro_S2 +
           B_Taxi12_S2 * taxi_12 +
        #    B_PriceRatio_S2 * price_12 +
           B_Price_S2 * price2 +
           B_TravelDistWork_S2 * travel_distance_work +
           B_WeekMetro_S2 * week_metro +
           B_Ebike_S2 * e_bike +
           B_Occupy_S2 * occupy +
           B_Sex_S2 * sex +
           B_C7_S2 * c7 +
           B_Income1_S2 * income1 +
           B_Age4_S2 * age4)

V2_2_S2 = (
    # ASC_Taxi_S2 +
        #    B_PriceRatio_S2 * price_3 +
           B_Price_S2 * price3 +
           B_TravelDistWeekend_S2 * travel_distance_weekend +
           B_WeekTaxi_S2 * week_taxi +
           B_Age3_S2 * age3 +
           B_Income2_S2 * income2)

V2_3_S2 = (
    # ASC_Ultra_S2 +
        #    B_PriceRatio_S2 * price_4 +
           B_Price_S2 * price4 +
           B_C6_S2 * c6 +
           B_Cost_S2 * cost +
           B_WeekTaxi_S2 * week_taxi +
           B_Age3_S2 * age3)

V2_4_S2 = (
    # ASC_Payg_S2 +
    B_Cost_S2 * cost +
           B_License_S2 * license +
        #    B_HaveCar_S2 * have_car +
           B_Education_S2 * education)

V2_Map_S2 = {0: V2_0_S2, 1: V2_1_S2, 2: V2_2_S2, 3: V2_3_S2, 4: V2_4_S2}

# 可用性
av1 = {0: 1, 1: 1}  # 二元, 两个都可用
av2 = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

# =============================================================================
# 4. 概率计算 (Probabilities)
# =============================================================================

# 阶段1观测概率 (二元)
Prob_T1_Given_S1 = models.logit(V1_Map_S1, av1, CHOICE_T1_BIN)
Prob_T1_Given_S2 = models.logit(V1_Map_S2, av1, CHOICE_T1_BIN)

# 阶段2观测概率
Prob_T2_Given_S1 = models.logit(V2_Map_S1, av2, CHOICE_T2)
Prob_T2_Given_S2 = models.logit(V2_Map_S2, av2, CHOICE_T2)

# 初始状态概率 (17 params, 与v4相同)
V_Init_S2 = (ASC_Init_S2 +
            #  B_MaasFamiliar_Init * MaasFamiliar +
             B_Sex_Init * sex +
             B_Age2_Init * age2 +
             B_Age4_Init * age4 +
             B_Income1_Init * income1 +
             B_Income2_Init * income2 +
             B_Education_Init * education +
             B_Occupy_Init * occupy +
             B_WeekMetro_Init * week_metro +
             B_WeekBus_Init * week_bus +
             B_WeekTaxi_Init * week_taxi +
             B_WeekEbike_Init * week_ebike +
             B_HaveCar_Init * have_car +
            #  B_Ebike_Init * e_bike +
             B_D6_Init * d6 +
             B_F6_Init * f6 +
             B_FACTOR1_Init * FACTOR1 +
             B_FACTOR2_Init * FACTOR2 +
             B_FACTOR3_Init * FACTOR3 +
             B_FACTOR4_Init * FACTOR4 +
             B_FACTOR6_Init * FACTOR6)

Prob_Init_S1 = 1 / (1 + exp(V_Init_S2))
Prob_Init_S2 = exp(V_Init_S2) / (1 + exp(V_Init_S2))

# 状态转移概率 (11 params, 与v4相同)
Trans_Covariates = (G_TimeSavings * time_savings +
                    G_CostSavings * cost_savings +
                    G_MatchBus * match_bus +
                    G_MatchMetro * match_metro +
                    G_MatchBike * match_bike +
                    # G_MatchEbike * match_e_bike +
                    G_MatchTaxi * match_taxi +
                    G_MatchPrice * match_price +
                    G_PriceRatio * price_ratio
                    )

V_S1_to_S2 = Trans_Base_S1_to_S2 + Trans_Covariates
Prob_S1_to_S1 = 1 / (1 + exp(V_S1_to_S2))
Prob_S1_to_S2 = exp(V_S1_to_S2) / (1 + exp(V_S1_to_S2))

V_S2_to_S2 = Trans_Base_S2_to_S2 + Trans_Covariates
Prob_S2_to_S1 = 1 / (1 + exp(V_S2_to_S2))
Prob_S2_to_S2 = exp(V_S2_to_S2) / (1 + exp(V_S2_to_S2))

# =============================================================================
# 5. HMM似然函数构造 (Forward Algorithm)
# =============================================================================

Path_Start_S1 = Prob_Init_S1 * Prob_T1_Given_S1 * (
    Prob_S1_to_S1 * Prob_T2_Given_S1 +
    Prob_S1_to_S2 * Prob_T2_Given_S2
)

Path_Start_S2 = Prob_Init_S2 * Prob_T1_Given_S2 * (
    Prob_S2_to_S1 * Prob_T2_Given_S1 +
    Prob_S2_to_S2 * Prob_T2_Given_S2
)

Prob_Obs = Path_Start_S1 + Path_Start_S2
loglike = log(Prob_Obs)

# =============================================================================
# 6. 模型估计
# =============================================================================
logging.basicConfig(level=logging.INFO)

biogeme_obj = bio.BIOGEME(database, loglike)
biogeme_obj.model_name = "MaaS_HMM_AllData_Binary"

print("开始估计HMM模型 (全数据: AllData Binary) - raw scale, 冷启动")
print("=" * 70)
results = biogeme_obj.estimate()

# =============================================================================
# 7. 输出结果
# =============================================================================
print("\n" + "=" * 70)
print("模型估计结果 - 全数据模型 (AllData Binary, raw scale)")
print("=" * 70)

raw = results.raw_estimation_results
print(f'\n估计参数数量: {len(raw.beta_values)}')
print(f'最终对数似然: {raw.final_log_likelihood:.3f}')

print("\n" + "=" * 70)
print("估计参数")
print("=" * 70)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 120)

try:
    pandasResults = results.get_estimated_parameters()
    pandasResults = pandasResults.set_index('Name')
    print(pandasResults.to_string())

    # 显著性统计
    if 'Robust t-stat.' in pandasResults.columns:
        sig_count = (pandasResults['Robust t-stat.'].abs() >= 1.96).sum()
        total_params = len(pandasResults)
        print(f"\n显著参数 (|t|>=1.96): {sig_count}/{total_params}")

        # 不显著参数列表
        insig = pandasResults[pandasResults['Robust t-stat.'].abs() < 1.96]
        if len(insig) > 0:
            print(f"\n不显著参数 ({len(insig)}个):")
            for name, row in insig.iterrows():
                print(f"  {name:30s}: val={row['Value']:>10.4f}  t={row['Robust t-stat.']:>7.2f}")
    else:
        print("\n注意: 未计算标准误 (calculating_second_derivatives=never)")
        print("仅输出点估计值")
except Exception as e:
    print(f"get_estimated_parameters 失败: {e}")
    print("直接输出beta_values:")
    pandasResults = pd.DataFrame({
        'Value': raw.beta_values
    })
    pandasResults.index.name = 'Name'
    print(pandasResults.to_string())

# 状态识别验证 (单一 ASC_Diff)
print("\n--- 状态识别验证 ---")
if 'ASC_Diff' in pandasResults.index:
    val = pandasResults.loc['ASC_Diff', 'Value']
    if 'Robust t-stat.' in pandasResults.columns:
        t = pandasResults.loc['ASC_Diff', 'Robust t-stat.']
        ok = 'OK' if val > 0 and abs(t) >= 1.96 else 'FAIL'
        print(f"  ASC_Diff: val={val:.3f}, t={t:.2f} [{ok}]")
    else:
        ok = 'OK (正)' if val > 0 else 'FAIL (负!)'
        print(f"  ASC_Diff: val={val:.3f} [{ok}]")

# Trans_Base 饱和检查
print("\n--- Trans_Base 饱和检查 ---")
for name in ['Trans_Base_S1_to_S2', 'Trans_Base_S2_to_S2']:
    if name in pandasResults.index:
        val = pandasResults.loc[name, 'Value']
        saturated = '饱和!' if abs(val) > 10 else 'OK'
        print(f"  {name}: val={val:.3f} [{saturated}]")

# 与旧AllData参考比较
old_all_ll = -11138.718
all_ll = raw.final_log_likelihood
print(f"\n--- 与旧AllData (5选项) 参考比较 ---")
print(f"  旧AllData LL (6,060 obs, 84 params): {old_all_ll:.3f}")
print(f"  Binary LL ({len(df_processed)} obs, {len(raw.beta_values)} params): {all_ll:.3f}")
print(f"  LL差异: {all_ll - old_all_ll:.3f}")

pandasResults.to_csv('MaaS_HMM_AllData_Binary_results.csv')
print("\n结果已保存到 MaaS_HMM_AllData_Binary_results.csv")
