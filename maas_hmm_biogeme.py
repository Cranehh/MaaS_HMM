"""
MaaS多阶段采纳决策的隐马尔可夫模型 (HMM) - Biogeme完整版
融合PyMC模型中的所有变量到Biogeme框架

模型结构:
- T=2 时间阶段
- States=2 隐状态 (Skeptic, Enthusiast)
- 阶段1: MaaS模式选择 (5选项: 不转移, M1, M2, M3, M4)
- 阶段2: 套餐订阅选择 (5选项: Bus First, Metro Access, Value Taxi, Ultra Access, PAYG)

基于PyMC模型: build_hmm_multinomial()
"""

import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import biogeme.distributions as dist
import biogeme.results as res
import biogeme.messaging as msg
from biogeme.expressions import (
    Beta,
    DefineVariable,
    log,
    RandomVariable,
    Integrate,
    Elem,
    bioNormalCdf,
    exp,
    bioDraws,
    MonteCarlo
)

# =============================================================================
# 1. 数据加载与预处理 (保持与PyMC逻辑一致)
# =============================================================================

def load_and_preprocess(filepath):
    """加载并预处理数据"""
    # 读取数据
    if filepath.endswith('.csv'):
        raw_df = pd.read_csv(filepath, encoding='gb2312')
    else:
        raw_df = pd.read_excel(filepath)

    df = raw_df.copy()

    # --- 变量映射逻辑 (同PyMC) ---
    # cost处理: 150以下为0, 150以上为1
    df['cost'] = df['cost'].replace([1,2,3,4,5,6], [0,0,1,1,1,1])

    # 阶段1 Choice (映射为 0-4)
    # 1->0(No), 4->1(M1), 5->2(M2), 6->3(M3), 7->4(M4)
    maas_map = {1:0, 4:1, 5:2, 6:3, 7:4}
    df['CHOICE_T1'] = df['maas'].map(maas_map)

    # 阶段2 Choice (映射为 0-4)
    # results: 1->0(Bus), 2->1(Metro), 3->2(Taxi), 4->3(Ultra), 5->4(PAYG)
    df['CHOICE_T2'] = df['results'] - 1

    # 确保没有NA
    df = df.dropna(subset=['CHOICE_T1', 'CHOICE_T2'])

    # 确保关键变量为整数类型
    int_cols = ['distance1', 'distance2', 'distance3', 'distance4', 'distance5',
                'income1', 'income2', 'income3',
                'age1', 'age2', 'age3', 'age4',
                'match_bus', 'match_metro', 'match_bike']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df

# 加载数据
df_processed = load_and_preprocess('data/最终模型数据.csv')
database = db.Database('MaaS_HMM_Data', df_processed)

# 将Dataframe列转换为Biogeme变量对象
globals().update(database.variables)

##结构方程定义参数和系数
coef1_gender = Beta('coef1_gender', 0, None, None, 0)
coef2_gender = Beta('coef2_gender', 0, None, None, 0)
coef3_gender = Beta('coef3_gender', 0, None, None, 0)
coef4_gender = Beta('coef4_gender', 0, None, None, 0)
coef5_gender = Beta('coef5_gender', 0, None, None, 0)
coef6_gender = Beta('coef6_gender', 0, None, None, 0)

coef1_age1 = Beta('coef1_age1', 0, None, None, 0)
coef2_age1 = Beta('coef2_age1', 0, None, None, 0)
coef3_age1 = Beta('coef3_age1', 0, None, None, 0)
coef4_age1 = Beta('coef4_age1', 0, None, None, 0)
coef5_age1 = Beta('coef5_age1', 0, None, None, 0)
coef6_age1 = Beta('coef6_age1', 0, None, None, 0)

coef1_age2 = Beta('coef1_age2', 0, None, None, 0)
coef2_age2 = Beta('coef2_age2', 0, None, None, 0)
coef3_age2 = Beta('coef3_age2', 0, None, None, 0)
coef4_age2 = Beta('coef4_age2', 0, None, None, 0)
coef5_age2 = Beta('coef5_age2', 0, None, None, 0)
coef6_age2 = Beta('coef6_age2', 0, None, None, 0)

coef1_age3 = Beta('coef1_age3', 0, None, None, 0)
coef2_age3 = Beta('coef2_age3', 0, None, None, 0)
coef3_age3 = Beta('coef3_age3', 0, None, None, 0)
coef4_age3 = Beta('coef4_age3', 0, None, None, 0)
coef5_age3 = Beta('coef5_age3', 0, None, None, 0)
coef6_age3 = Beta('coef6_age3', 0, None, None, 0)

coef1_age4 = Beta('coef1_age4', 0, None, None, 0)
coef2_age4 = Beta('coef2_age4', 0, None, None, 0)
coef3_age4 = Beta('coef3_age4', 0, None, None, 0)
coef4_age4 = Beta('coef4_age4', 0, None, None, 0)
coef5_age4 = Beta('coef5_age4', 0, None, None, 0)
coef6_age4 = Beta('coef6_age4', 0, None, None, 0)

coef1_job = Beta('coef1_job', 0, None, None, 0)
coef2_job = Beta('coef2_job', 0, None, None, 0)
coef3_job = Beta('coef3_job', 0, None, None, 0)
coef4_job = Beta('coef4_job', 0, None, None, 0)
coef5_job = Beta('coef5_job', 0, None, None, 0)
coef6_job = Beta('coef6_job', 0, None, None, 0)

coef1_income1 = Beta('coef1_income1', 0, None, None, 0)
coef2_income1 = Beta('coef2_income1', 0, None, None, 0)
coef3_income1 = Beta('coef3_income1', 0, None, None, 0)
coef4_income1 = Beta('coef4_income1', 0, None, None, 0)
coef5_income1 = Beta('coef5_income1', 0, None, None, 0)
coef6_income1 = Beta('coef6_income1', 0, None, None, 0)

coef1_income2 = Beta('coef1_income2', 0, None, None, 0)
coef2_income2 = Beta('coef2_income2', 0, None, None, 0)
coef3_income2 = Beta('coef3_income2', 0, None, None, 0)
coef4_income2 = Beta('coef4_income2', 0, None, None, 0)
coef5_income2 = Beta('coef5_income2', 0, None, None, 0)
coef6_income2 = Beta('coef6_income2', 0, None, None, 0)

coef1_income3 = Beta('coef1_income3', 0, None, None, 0)
coef2_income3 = Beta('coef2_income3', 0, None, None, 0)
coef3_income3 = Beta('coef3_income3', 0, None, None, 0)
coef4_income3 = Beta('coef4_income3', 0, None, None, 0)
coef5_income3 = Beta('coef5_income3', 0, None, None, 0)
coef6_income3 = Beta('coef6_income3', 0, None, None, 0)

coef1_education = Beta('coef1_education', 0, None, None, 0)
coef2_education = Beta('coef2_education', 0, None, None, 0)
coef3_education = Beta('coef3_education', 0, None, None, 0)
coef4_education = Beta('coef4_education', 0, None, None, 0)
coef5_education = Beta('coef5_education', 0, None, None, 0)
coef6_education = Beta('coef6_education', 0, None, None, 0)

coef1_travel_num = Beta('coef1_travel_num', 0, None, None, 0)
coef2_travel_num = Beta('coef2_travel_num', 0, None, None, 0)
coef3_travel_num = Beta('coef3_travel_num', 0, None, None, 0)
coef4_travel_num = Beta('coef4_travel_num', 0, None, None, 0)
coef5_travel_num = Beta('coef5_travel_num', 0, None, None, 0)
coef6_travel_num = Beta('coef6_travel_num', 0, None, None, 0)

coef1_travel_distance_day = Beta('coef1_travel_distance_day', 0, None, None, 0)
coef2_travel_distance_day = Beta('coef2_travel_distance_day', 0, None, None, 0)
coef3_travel_distance_day = Beta('coef3_travel_distance_day', 0, None, None, 0)
coef4_travel_distance_day = Beta('coef4_travel_distance_day', 0, None, None, 0)
coef5_travel_distance_day = Beta('coef5_travel_distance_day', 0, None, None, 0)
coef6_travel_distance_day = Beta('coef6_travel_distance_day', 0, None, None, 0)

coef1_travel_distance_end = Beta('coef1_travel_distance_end', 0, None, None, 0)
coef2_travel_distance_end = Beta('coef2_travel_distance_end', 0, None, None, 0)
coef3_travel_distance_end = Beta('coef3_travel_distance_end', 0, None, None, 0)
coef4_travel_distance_end = Beta('coef4_travel_distance_end', 0, None, None, 0)
coef5_travel_distance_end = Beta('coef5_travel_distance_end', 0, None, None, 0)
coef6_travel_distance_end = Beta('coef6_travel_distance_end', 0, None, None, 0)

coef1_travel_aim = Beta('coef1_travel_aim', 0, None, None, 0)
coef2_travel_aim = Beta('coef2_travel_aim', 0, None, None, 0)
coef3_travel_aim = Beta('coef3_travel_aim', 0, None, None, 0)
coef4_travel_aim = Beta('coef4_travel_aim', 0, None, None, 0)
coef5_travel_aim = Beta('coef5_travel_aim', 0, None, None, 0)
coef6_travel_aim = Beta('coef6_travel_aim', 0, None, None, 0)

coef1_6a = Beta('coef1_6a', 0, None, None, 0)
coef2_6a = Beta('coef2_6a', 0, None, None, 0)
coef3_6a = Beta('coef3_6a', 0, None, None, 0)
coef4_6a = Beta('coef4_6a', 0, None, None, 0)
coef5_6a = Beta('coef5_6a', 0, None, None, 0)
coef6_6a = Beta('coef6_6a', 0, None, None, 0)

coef1_6b = Beta('coef1_6b', 0, None, None, 0)
coef2_6b = Beta('coef2_6b', 0, None, None, 0)
coef3_6b = Beta('coef3_6b', 0, None, None, 0)
coef4_6b = Beta('coef4_6b', 0, None, None, 0)
coef5_6b = Beta('coef5_6b', 0, None, None, 0)
coef6_6b = Beta('coef6_6b', 0, None, None, 0)

coef1_6c = Beta('coef1_6c', 0, None, None, 0)
coef2_6c = Beta('coef2_6c', 0, None, None, 0)
coef3_6c = Beta('coef3_6c', 0, None, None, 0)
coef4_6c = Beta('coef4_6c', 0, None, None, 0)
coef5_6c = Beta('coef5_6c', 0, None, None, 0)
coef6_6c = Beta('coef6_6c', 0, None, None, 0)

coef1_6d = Beta('coef1_6d', 0, None, None, 0)
coef2_6d = Beta('coef2_6d', 0, None, None, 0)
coef3_6d = Beta('coef3_6d', 0, None, None, 0)
coef4_6d = Beta('coef4_6d', 0, None, None, 0)
coef5_6d = Beta('coef5_6d', 0, None, None, 0)
coef6_6d = Beta('coef6_6d', 0, None, None, 0)

coef1_6e = Beta('coef1_6e', 0, None, None, 0)
coef2_6e = Beta('coef2_6e', 0, None, None, 0)
coef3_6e = Beta('coef3_6e', 0, None, None, 0)
coef4_6e = Beta('coef4_6e', 0, None, None, 0)
coef5_6e = Beta('coef5_6e', 0, None, None, 0)
coef6_6e = Beta('coef6_6e', 0, None, None, 0)

coef1_6f = Beta('coef1_6f', 0, None, None, 0)
coef2_6f = Beta('coef2_6f', 0, None, None, 0)
coef3_6f = Beta('coef3_6f', 0, None, None, 0)
coef4_6f = Beta('coef4_6f', 0, None, None, 0)
coef5_6f = Beta('coef5_6f', 0, None, None, 0)
coef6_6f = Beta('coef6_6f', 0, None, None, 0)

coef1_6g = Beta('coef1_6g', 0, None, None, 0)
coef2_6g = Beta('coef2_6g', 0, None, None, 0)
coef3_6g = Beta('coef3_6g', 0, None, None, 0)
coef4_6g = Beta('coef4_6g', 0, None, None, 0)
coef5_6g = Beta('coef5_6g', 0, None, None, 0)
coef6_6g = Beta('coef6_6g', 0, None, None, 0)

coef1_cost = Beta('coef1_cost', 0, None, None, 0)
coef2_cost = Beta('coef2_cost', 0, None, None, 0)
coef3_cost = Beta('coef3_cost', 0, None, None, 0)
coef4_cost = Beta('coef4_cost', 0, None, None, 0)
coef5_cost = Beta('coef5_cost', 0, None, None, 0)
coef6_cost = Beta('coef6_cost', 0, None, None, 0)

coef1_car_home = Beta('coef1_car_home', 0, None, None, 0)
coef2_car_home = Beta('coef2_car_home', 0, None, None, 0)
coef3_car_home = Beta('coef3_car_home', 0, None, None, 0)
coef4_car_home = Beta('coef4_car_home', 0, None, None, 0)
coef5_car_home = Beta('coef5_car_home', 0, None, None, 0)
coef6_car_home = Beta('coef6_car_home', 0, None, None, 0)


coef1_bus = Beta('coef1_bus', 0, None, None, 0)
coef2_bus = Beta('coef2_bus', 0, None, None, 0)
coef3_bus = Beta('coef3_bus', 0, None, None, 0)
coef4_bus = Beta('coef4_bus', 0, None, None, 0)
coef5_bus = Beta('coef5_bus', 0, None, None, 0)
coef6_bus = Beta('coef6_bus', 0, None, None, 0)

coef1_metro = Beta('coef1_metro', 0, None, None, 0)
coef2_metro = Beta('coef2_metro', 0, None, None, 0)
coef3_metro = Beta('coef3_metro', 0, None, None, 0)
coef4_metro = Beta('coef4_metro', 0, None, None, 0)
coef5_metro = Beta('coef5_metro', 0, None, None, 0)
coef6_metro = Beta('coef6_metro', 0, None, None, 0)

coef1_taxi = Beta('coef1_taxi', 0, None, None, 0)
coef2_taxi = Beta('coef2_taxi', 0, None, None, 0)
coef3_taxi = Beta('coef3_taxi', 0, None, None, 0)
coef4_taxi = Beta('coef4_taxi', 0, None, None, 0)
coef5_taxi = Beta('coef5_taxi', 0, None, None, 0)
coef6_taxi = Beta('coef6_taxi', 0, None, None, 0)

coef1_ebike = Beta('coef1_ebike', 0, None, None, 0)
coef2_ebike = Beta('coef2_ebike', 0, None, None, 0)
coef3_ebike = Beta('coef3_ebike', 0, None, None, 0)
coef4_ebike = Beta('coef4_ebike', 0, None, None, 0)
coef5_ebike = Beta('coef5_ebike', 0, None, None, 0)
coef6_ebike = Beta('coef6_ebike', 0, None, None, 0)

coef1_bike = Beta('coef1_bike', 0, None, None, 0)
coef2_bike = Beta('coef2_bike', 0, None, None, 0)
coef3_bike = Beta('coef3_bike', 0, None, None, 0)
coef4_bike = Beta('coef4_bike', 0, None, None, 0)
coef5_bike = Beta('coef5_bike', 0, None, None, 0)
coef6_bike = Beta('coef6_bike', 0, None, None, 0)
###########################################################
## 定义潜变量
omega = RandomVariable('omega')
density = dist.normalpdf(omega)
sigma_s1 = Beta('sigma_s1', 1, None, None, 0)
sigma_s2 = Beta('sigma_s2', 1, None, None, 0)
sigma_s3 = Beta('sigma_s3', 1, None, None, 0)
sigma_s4 = Beta('sigma_s4', 1, None, None, 0)
sigma_s5 = Beta('sigma_s5', 1, None, None, 0)
sigma_s6 = Beta('sigma_s6', 1, None, None, 0)

FACTOR1 = (
            # coef1_gender * sex
           coef1_age1 * age1
+  coef1_age2 * age2
+  coef1_age3 * age3
# +  coef1_age4 * age4
         +  coef1_job * occupy
            + coef1_income1 * income1
            + coef1_income2 * income2
            # + coef1_income3 * income3
         # +  coef1_education * education
         +  coef1_travel_num * travel_num
         +  coef1_travel_distance_day * travel_distance_work
         # +  coef1_travel_distance_end * travel_distance_weekend
         +  coef1_travel_aim * travel_aim
         # +  coef1_6a * a6
         # +  coef1_6b * b6
         # +  coef1_6c * c6
         # +  coef1_6d * d6
         # +  coef1_6e * e6
         # +  coef1_6f * f6
         # +  coef1_6g * g6
         # +  coef1_cost * cost
         # +  coef1_car_home * car_home
         +  coef1_bus * week_bus
         +  coef1_metro * week_metro
         +  coef1_taxi * week_taxi
         +  coef1_ebike * week_ebike
         +  coef1_bike * week_bike
         +  sigma_s1 * bioDraws('EC', 'NORMAL_MLHS')
)

FACTOR2 = (
            # coef2_gender * sex
             coef2_age1 * age1
            + coef2_age2 * age2
            + coef2_age3 * age3
            # + coef2_age4 * age4
         +  coef2_job * occupy
            + coef2_income1 * income1
            + coef2_income2 * income2
            # + coef2_income3 * income3
         # +  coef2_education * education
         +  coef2_travel_num * travel_num
         +  coef2_travel_distance_day * travel_distance_work
         # +  coef2_travel_distance_end * travel_distance_weekend
         +  coef2_travel_aim * travel_aim
         # +  coef2_6a * a6
         # +  coef2_6b * b6
         # +  coef2_6c * c6
         # +  coef2_6d * d6
         # +  coef2_6e * e6
         # +  coef2_6f * f6
         # +  coef2_6g * g6
         # +  coef2_cost * cost
         # +  coef2_car_home * car_home
         +  coef2_bus * week_bus
         +  coef2_metro * week_metro
         +  coef2_taxi * week_taxi
         +  coef2_ebike * week_ebike
         +  coef2_bike * week_bike
         +  sigma_s2 * bioDraws('EC', 'NORMAL_MLHS')
)

FACTOR3 = (
            # coef3_gender * sex
             coef3_age1 * age1
            + coef3_age2 * age2
            + coef3_age3 * age3
            # + coef3_age4 * age4
         +  coef3_job * occupy
            + coef3_income1 * income1
            + coef3_income2 * income2
            # + coef3_income3 * income3
         # +  coef3_education * education
         +  coef3_travel_num * travel_num
         +  coef3_travel_distance_day * travel_distance_work
         # +  coef3_travel_distance_end * travel_distance_weekend
         +  coef3_travel_aim * travel_aim
         # +  coef3_6a * a6
         # +  coef3_6b * b6
         # +  coef3_6c * c6
         # +  coef3_6d * d6
         # +  coef3_6e * e6
         # +  coef3_6f * f6
         # +  coef3_6g * g6
         # +  coef3_cost * cost
         # +  coef3_car_home * car_home
         +  coef3_bus * week_bus
         +  coef3_metro * week_metro
         +  coef3_taxi * week_taxi
         +  coef3_ebike * week_ebike
         +  coef3_bike * week_bike
         +  sigma_s3 * bioDraws('EC', 'NORMAL_MLHS')
)

FACTOR4 = (
            # coef4_gender * sex
             coef4_age1 * age1
            + coef4_age2 * age2
            + coef4_age3 * age3
            # + coef4_age4 * age4
         +  coef4_job * occupy
            + coef4_income1 * income1
            + coef4_income2 * income2
            # + coef4_income3 * income3
         # +  coef4_education * education
         +  coef4_travel_num * travel_num
         +  coef4_travel_distance_day * travel_distance_work
         # +  coef4_travel_distance_end * travel_distance_weekend
         +  coef4_travel_aim * travel_aim
         # +  coef4_6a * a6
         # +  coef4_6b * b6
         # +  coef4_6c * c6
         # +  coef4_6d * d6
         # +  coef4_6e * e6
         # +  coef4_6f * f6
         # +  coef4_6g * g6
         # +  coef4_cost * cost
         # +  coef4_car_home * car_home
         +  coef4_bus * week_bus
         +  coef4_metro * week_metro
         +  coef4_taxi * week_taxi
         +  coef4_ebike * week_ebike
         +  coef4_bike * week_bike
         +  sigma_s4 * bioDraws('EC', 'NORMAL_MLHS')
)

# FACTOR5 = (
#             # coef5_gender * sex
#              coef5_age1 * age1
#             + coef5_age2 * age2
#             + coef5_age3 * age3
#             + coef5_age4 * age4
#          +  coef5_job * occupy
#             + coef5_income1 * income1
#             + coef5_income2 * income2
#             + coef5_income3 * income3
#          # +  coef5_education * education
#          +  coef5_travel_num * travel_num
#          +  coef5_travel_distance_day * travel_distance_work
#          # +  coef5_travel_distance_end * travel_distance_weekend
#          +  coef5_travel_aim * travel_aim
#          +  coef5_6a * a6
#          +  coef5_6b * b6
#          +  coef5_6c * c6
#          +  coef5_6d * d6
#          +  coef5_6e * e6
#          +  coef5_6f * f6
#          +  coef5_6g * g6
#          # +  coef5_cost * cost
#          # +  coef5_car_home * car_home
#          +  coef5_bus * week_bus
#          +  coef5_metro * week_metro
#          +  coef5_taxi * week_taxi
#          +  coef5_ebike * week_ebike
#          +  coef5_bike * week_bike
#          +  sigma_s5 * omega
# )

FACTOR6 = (
            # coef6_gender * sex
             coef6_age1 * age1
            + coef6_age2 * age2
            + coef6_age3 * age3
            # + coef6_age4 * age4
         +  coef6_job * occupy
         +  coef6_income1 * income1
+  coef6_income2 * income2
# +  coef6_income3 * income3
         # +  coef6_education * education
         +  coef6_travel_num * travel_num
         +  coef6_travel_distance_day * travel_distance_work
         # +  coef6_travel_distance_end * travel_distance_weekend
         +  coef6_travel_aim * travel_aim
         # +  coef6_6a * a6
         # +  coef6_6b * b6
         # +  coef6_6c * c6
         # +  coef6_6d * d6
         # +  coef6_6e * e6
         # +  coef6_6f * f6
         # +  coef6_6g * g6
         # +  coef6_cost * cost
         # +  coef6_car_home * car_home
         +  coef6_bus * week_bus
         +  coef6_metro * week_metro
         +  coef6_taxi * week_taxi
         +  coef6_ebike * week_ebike
         +  coef6_bike * week_bike
         +  sigma_s6 * bioDraws('EC', 'NORMAL_MLHS')
)
###########################################################
##测量方程
##截距
INTER_at8 = Beta('INTER_at8', 0, None, None, 1)
INTER_at9 = Beta('INTER_at9', 0, None, None, 0)
INTER_at10 = Beta('INTER_at10', 0, None, None, 0)
INTER_at11 = Beta('INTER_at11', 0, None, None, 0)
INTER_at13 = Beta('INTER_at13', 0, None, None, 0)
INTER_at14 = Beta('INTER_at14', 0, None, None, 0)
INTER_at17 = Beta('INTER_at17', 0, None, None, 0)

INTER_at21 = Beta('INTER_at21', 0, None, None, 1)
INTER_at22 = Beta('INTER_at22', 0, None, None, 0)
INTER_at23 = Beta('INTER_at23', 0, None, None, 0)
INTER_at24 = Beta('INTER_at24', 0, None, None, 0)
INTER_at25 = Beta('INTER_at25', 0, None, None, 0)

INTER_at18 = Beta('INTER_at18', 0, None, None, 1)
INTER_at19 = Beta('INTER_at19', 0, None, None, 0)

INTER_at12 = Beta('INTER_at12', 0, None, None, 1)
INTER_at15 = Beta('INTER_at15', 0, None, None, 0)

INTER_at2 = Beta('INTER_at2', 0, None, None, 1)
INTER_at3 = Beta('INTER_at3', 0, None, None, 0)

INTER_at1 = Beta('INTER_at1', 0, None, None, 1)
INTER_at4 = Beta('INTER_at4', 0, None, None, 0)
INTER_at6 = Beta('INTER_at6', 0, None, None, 0)

###########################################################
# 态度指标和潜变量之间的系数关系
B_at8 = Beta('B_at8', 1, None, None, 1)
B_at9 = Beta('B_at9', 1, None, None, 0)
B_at10 = Beta('B_at10', 1, None, None, 0)
B_at11 = Beta('B_at11', 1, None, None, 0)
B_at13 = Beta('B_at13', 1, None, None, 0)
B_at14 = Beta('B_at14', 1, None, None, 0)
B_at17 = Beta('B_at17', 1, None, None, 0)

B_at21 = Beta('B_at21', 1, None, None, 1)
B_at22 = Beta('B_at22', 1, None, None, 0)
B_at23 = Beta('B_at23', 1, None, None, 0)
B_at24 = Beta('B_at24', 1, None, None, 0)
B_at25 = Beta('B_at25', 1, None, None, 0)

B_at18 = Beta('B_at18', 1, None, None, 1)
B_at19 = Beta('B_at19', 1, None, None, 0)

B_at12 = Beta('B_at12', 1, None, None, 1)
B_at15 = Beta('B_at15', 1, None, None, 0)

B_at2 = Beta('B_at2', 1, None, None, 0)
B_at3 = Beta('B_at3', 1, None, None, 0)

B_at1 = Beta('B_at1', 1, None, None, 1)
B_at4 = Beta('B_at4', 1, None, None, 0)
B_at6 = Beta('B_at6', 1, None, None, 0)

###########################################################
MODEL_at8 = INTER_at8 + B_at8 * FACTOR1
MODEL_at9 = INTER_at9 + B_at9 * FACTOR1
MODEL_at10 = INTER_at10 + B_at10 * FACTOR1
MODEL_at11 = INTER_at11 + B_at11 * FACTOR1
MODEL_at13 = INTER_at13 + B_at13 * FACTOR1
MODEL_at14 = INTER_at14 + B_at14 * FACTOR1
MODEL_at17 = INTER_at17 + B_at17 * FACTOR1

MODEL_at21 = INTER_at21 + B_at21 * FACTOR2
MODEL_at22 = INTER_at22 + B_at22 * FACTOR2
MODEL_at23 = INTER_at23 + B_at23 * FACTOR2
MODEL_at24 = INTER_at24 + B_at24 * FACTOR2
MODEL_at25 = INTER_at25 + B_at25 * FACTOR2


MODEL_at18 = INTER_at18 + B_at18 * FACTOR3
MODEL_at19 = INTER_at19 + B_at19 * FACTOR3

MODEL_at12 = INTER_at12 + B_at12 * FACTOR4
MODEL_at15 = INTER_at15 + B_at15 * FACTOR4

# MODEL_at2 = INTER_at2 + B_at2 * FACTOR5
# MODEL_at3 = INTER_at3 + B_at3 * FACTOR5

MODEL_at1 = INTER_at1 + B_at1 * FACTOR6
MODEL_at4 = INTER_at4 + B_at4 * FACTOR6
MODEL_at6 = INTER_at6 + B_at6 * FACTOR6
###########################################################
SIGMA_STAR_at8 = Beta('SIGMA_STAR_at8', 1, 1.0e-5, None, 1)
SIGMA_STAR_at9 = Beta('SIGMA_STAR_at9', 1, 1.0e-5, None, 0)
SIGMA_STAR_at10 = Beta('SIGMA_STAR_at10', 1, 1.0e-5, None, 0)
SIGMA_STAR_at11 = Beta('SIGMA_STAR_at11', 1, 1.0e-5, None, 0)
SIGMA_STAR_at13 = Beta('SIGMA_STAR_at13', 1, 1.0e-5, None, 0)
SIGMA_STAR_at14 = Beta('SIGMA_STAR_at14', 1, 1.0e-5, None, 0)
SIGMA_STAR_at17 = Beta('SIGMA_STAR_at17', 1, 1.0e-5, None, 0)

SIGMA_STAR_at21 = Beta('SIGMA_STAR_at21', 1, 1.0e-5, None, 1)
SIGMA_STAR_at22 = Beta('SIGMA_STAR_at22', 1, 1.0e-5, None, 0)
SIGMA_STAR_at23 = Beta('SIGMA_STAR_at23', 1, 1.0e-5, None, 0)
SIGMA_STAR_at24 = Beta('SIGMA_STAR_at24', 1, 1.0e-5, None, 0)
SIGMA_STAR_at25 = Beta('SIGMA_STAR_at25', 1, 1.0e-5, None, 0)

SIGMA_STAR_at18 = Beta('SIGMA_STAR_at18', 1, 1.0e-5, None, 1)
SIGMA_STAR_at19 = Beta('SIGMA_STAR_at19', 1, 1.0e-5, None, 0)

SIGMA_STAR_at12 = Beta('SIGMA_STAR_at12', 1, 1.0e-5, None, 1)
SIGMA_STAR_at15 = Beta('SIGMA_STAR_at15', 1, 1.0e-5, None, 0)

SIGMA_STAR_at2 = Beta('SIGMA_STAR_at2', 1, 1.0e-5, None, 1)
SIGMA_STAR_at3 = Beta('SIGMA_STAR_at3', 1, 1.0e-5, None, 0)

SIGMA_STAR_at1 = Beta('SIGMA_STAR_at1', 1, 1.0e-5, None, 1)
SIGMA_STAR_at4 = Beta('SIGMA_STAR_at4', 1, 1.0e-5, None, 0)
SIGMA_STAR_at6 = Beta('SIGMA_STAR_at6', 1, 1.0e-5, None, 0)

###########################################################
#不同梯度之间的关系
delta_1p = Beta('delta_1p', 0.1, 1.0e-5, None, 0)
delta_2p = Beta('delta_2p', 0.2, 1.0e-5, None, 0)
delta_3p = Beta('delta_3p', 0.3, 1.0e-5, None, 0)

tau_1p = 0
tau_2p = 0+delta_1p
tau_3p = tau_2p+delta_2p
tau_4p = tau_3p+delta_3p

at8_tau_1 = (tau_1p - MODEL_at8) / SIGMA_STAR_at8
at8_tau_2 = (tau_2p - MODEL_at8) / SIGMA_STAR_at8
at8_tau_3 = (tau_3p - MODEL_at8) / SIGMA_STAR_at8
at8_tau_4 = (tau_4p - MODEL_at8) / SIGMA_STAR_at8
Indat8 = {
    1: bioNormalCdf(at8_tau_1),
    2: bioNormalCdf(at8_tau_2) - bioNormalCdf(at8_tau_1),
    3: bioNormalCdf(at8_tau_3) - bioNormalCdf(at8_tau_2),
    4: bioNormalCdf(at8_tau_4) - bioNormalCdf(at8_tau_3),
    5: 1 - bioNormalCdf(at8_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at8 = Elem(Indat8, at8)

at9_tau_1 = (tau_1p - MODEL_at9) / SIGMA_STAR_at9
at9_tau_2 = (tau_2p - MODEL_at9) / SIGMA_STAR_at9
at9_tau_3 = (tau_3p - MODEL_at9) / SIGMA_STAR_at9
at9_tau_4 = (tau_4p - MODEL_at9) / SIGMA_STAR_at9
Indat9 = {
    1: bioNormalCdf(at9_tau_1),
    2: bioNormalCdf(at9_tau_2) - bioNormalCdf(at9_tau_1),
    3: bioNormalCdf(at9_tau_3) - bioNormalCdf(at9_tau_2),
    4: bioNormalCdf(at9_tau_4) - bioNormalCdf(at9_tau_3),
    5: 1 - bioNormalCdf(at9_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at9 = Elem(Indat9, at9)

at10_tau_1 = (tau_1p - MODEL_at10) / SIGMA_STAR_at10
at10_tau_2 = (tau_2p - MODEL_at10) / SIGMA_STAR_at10
at10_tau_3 = (tau_3p - MODEL_at10) / SIGMA_STAR_at10
at10_tau_4 = (tau_4p - MODEL_at10) / SIGMA_STAR_at10
Indat10 = {
    1: bioNormalCdf(at10_tau_1),
    2: bioNormalCdf(at10_tau_2) - bioNormalCdf(at10_tau_1),
    3: bioNormalCdf(at10_tau_3) - bioNormalCdf(at10_tau_2),
    4: bioNormalCdf(at10_tau_4) - bioNormalCdf(at10_tau_3),
    5: 1 - bioNormalCdf(at10_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at10 = Elem(Indat10, at10)



at11_tau_1 = (tau_1p - MODEL_at11) / SIGMA_STAR_at11
at11_tau_2 = (tau_2p - MODEL_at11) / SIGMA_STAR_at11
at11_tau_3 = (tau_3p - MODEL_at11) / SIGMA_STAR_at11
at11_tau_4 = (tau_4p - MODEL_at11) / SIGMA_STAR_at11
Indat11 = {
    1: bioNormalCdf(at11_tau_1),
    2: bioNormalCdf(at11_tau_2) - bioNormalCdf(at11_tau_1),
    3: bioNormalCdf(at11_tau_3) - bioNormalCdf(at11_tau_2),
    4: bioNormalCdf(at11_tau_4) - bioNormalCdf(at11_tau_3),
    5: 1 - bioNormalCdf(at11_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at11 = Elem(Indat11, at11)

at13_tau_1 = (tau_1p - MODEL_at13) / SIGMA_STAR_at13
at13_tau_2 = (tau_2p - MODEL_at13) / SIGMA_STAR_at13
at13_tau_3 = (tau_3p - MODEL_at13) / SIGMA_STAR_at13
at13_tau_4 = (tau_4p - MODEL_at13) / SIGMA_STAR_at13
Indat13 = {
    1: bioNormalCdf(at13_tau_1),
    2: bioNormalCdf(at13_tau_2) - bioNormalCdf(at13_tau_1),
    3: bioNormalCdf(at13_tau_3) - bioNormalCdf(at13_tau_2),
    4: bioNormalCdf(at13_tau_4) - bioNormalCdf(at13_tau_3),
    5: 1 - bioNormalCdf(at13_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at13 = Elem(Indat13, at13)

at14_tau_1 = (tau_1p - MODEL_at14) / SIGMA_STAR_at14
at14_tau_2 = (tau_2p - MODEL_at14) / SIGMA_STAR_at14
at14_tau_3 = (tau_3p - MODEL_at14) / SIGMA_STAR_at14
at14_tau_4 = (tau_4p - MODEL_at14) / SIGMA_STAR_at14
Indat14 = {
    1: bioNormalCdf(at14_tau_1),
    2: bioNormalCdf(at14_tau_2) - bioNormalCdf(at14_tau_1),
    3: bioNormalCdf(at14_tau_3) - bioNormalCdf(at14_tau_2),
    4: bioNormalCdf(at14_tau_4) - bioNormalCdf(at14_tau_3),
    5: 1 - bioNormalCdf(at14_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at14 = Elem(Indat14, at14)


at17_tau_1 = (tau_1p - MODEL_at17) / SIGMA_STAR_at17
at17_tau_2 = (tau_2p - MODEL_at17) / SIGMA_STAR_at17
at17_tau_3 = (tau_3p - MODEL_at17) / SIGMA_STAR_at17
at17_tau_4 = (tau_4p - MODEL_at17) / SIGMA_STAR_at17
Indat17 = {
    1: bioNormalCdf(at17_tau_1),
    2: bioNormalCdf(at17_tau_2) - bioNormalCdf(at17_tau_1),
    3: bioNormalCdf(at17_tau_3) - bioNormalCdf(at17_tau_2),
    4: bioNormalCdf(at17_tau_4) - bioNormalCdf(at17_tau_3),
    5: 1 - bioNormalCdf(at17_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at17 = Elem(Indat17, at17)
###########################################################
delta_1t = Beta('delta_1t', 0.1, 1.0e-5, None, 0)
delta_2t = Beta('delta_2t', 0.2, 1.0e-5, None, 0)
delta_3t = Beta('delta_3t', 0.3, 1.0e-5, None, 0)

tau_1t = 0
tau_2t = 0 + delta_1t
tau_3t = tau_2t + delta_2t
tau_4t = tau_3t + delta_3t

at21_tau_1 = (tau_1t - MODEL_at21) / SIGMA_STAR_at21
at21_tau_2 = (tau_2t - MODEL_at21) / SIGMA_STAR_at21
at21_tau_3 = (tau_3t - MODEL_at21) / SIGMA_STAR_at21
at21_tau_4 = (tau_4t - MODEL_at21) / SIGMA_STAR_at21
Indat21 = {
    1: bioNormalCdf(at21_tau_1),
    2: bioNormalCdf(at21_tau_2) - bioNormalCdf(at21_tau_1),
    3: bioNormalCdf(at21_tau_3) - bioNormalCdf(at21_tau_2),
    4: bioNormalCdf(at21_tau_4) - bioNormalCdf(at21_tau_3),
    5: 1 - bioNormalCdf(at21_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at21 = Elem(Indat21, at21)

at22_tau_1 = (tau_1t - MODEL_at22) / SIGMA_STAR_at22
at22_tau_2 = (tau_2t - MODEL_at22) / SIGMA_STAR_at22
at22_tau_3 = (tau_3t - MODEL_at22) / SIGMA_STAR_at22
at22_tau_4 = (tau_4t - MODEL_at22) / SIGMA_STAR_at22
Indat22 = {
    1: bioNormalCdf(at22_tau_1),
    2: bioNormalCdf(at22_tau_2) - bioNormalCdf(at22_tau_1),
    3: bioNormalCdf(at22_tau_3) - bioNormalCdf(at22_tau_2),
    4: bioNormalCdf(at22_tau_4) - bioNormalCdf(at22_tau_3),
    5: 1 - bioNormalCdf(at22_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at22 = Elem(Indat22, at22)

at23_tau_1 = (tau_1t - MODEL_at23) / SIGMA_STAR_at23
at23_tau_2 = (tau_2t - MODEL_at23) / SIGMA_STAR_at23
at23_tau_3 = (tau_3t - MODEL_at23) / SIGMA_STAR_at23
at23_tau_4 = (tau_4t - MODEL_at23) / SIGMA_STAR_at23
Indat23 = {
    1: bioNormalCdf(at23_tau_1),
    2: bioNormalCdf(at23_tau_2) - bioNormalCdf(at23_tau_1),
    3: bioNormalCdf(at23_tau_3) - bioNormalCdf(at23_tau_2),
    4: bioNormalCdf(at23_tau_4) - bioNormalCdf(at23_tau_3),
    5: 1 - bioNormalCdf(at23_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at23 = Elem(Indat23, at23)

at24_tau_1 = (tau_1t - MODEL_at24) / SIGMA_STAR_at24
at24_tau_2 = (tau_2t - MODEL_at24) / SIGMA_STAR_at24
at24_tau_3 = (tau_3t - MODEL_at24) / SIGMA_STAR_at24
at24_tau_4 = (tau_4t - MODEL_at24) / SIGMA_STAR_at24
Indat24 = {
    1: bioNormalCdf(at24_tau_1),
    2: bioNormalCdf(at24_tau_2) - bioNormalCdf(at24_tau_1),
    3: bioNormalCdf(at24_tau_3) - bioNormalCdf(at24_tau_2),
    4: bioNormalCdf(at24_tau_4) - bioNormalCdf(at24_tau_3),
    5: 1 - bioNormalCdf(at24_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at24 = Elem(Indat24, at24)

at25_tau_1 = (tau_1t - MODEL_at25) / SIGMA_STAR_at25
at25_tau_2 = (tau_2t - MODEL_at25) / SIGMA_STAR_at25
at25_tau_3 = (tau_3t - MODEL_at25) / SIGMA_STAR_at25
at25_tau_4 = (tau_4t - MODEL_at25) / SIGMA_STAR_at25
Indat25 = {
    1: bioNormalCdf(at25_tau_1),
    2: bioNormalCdf(at25_tau_2) - bioNormalCdf(at25_tau_1),
    3: bioNormalCdf(at25_tau_3) - bioNormalCdf(at25_tau_2),
    4: bioNormalCdf(at25_tau_4) - bioNormalCdf(at25_tau_3),
    5: 1 - bioNormalCdf(at25_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at25 = Elem(Indat25, at25)
###########################################################
delta_13 = Beta('delta_13', 0.1, 1.0e-5, None, 0)
delta_23 = Beta('delta_23', 0.2, 1.0e-5, None, 0)
delta_33 = Beta('delta_33', 0.3, 1.0e-5, None, 0)

tau_13 = 0
tau_23 = 0 + delta_13
tau_33 = tau_2t + delta_23
tau_43 = tau_3t + delta_33

at18_tau_1 = (tau_13 - MODEL_at18) / SIGMA_STAR_at18
at18_tau_2 = (tau_23 - MODEL_at18) / SIGMA_STAR_at18
at18_tau_3 = (tau_33 - MODEL_at18) / SIGMA_STAR_at18
at18_tau_4 = (tau_43 - MODEL_at18) / SIGMA_STAR_at18
Indat18 = {
    1: bioNormalCdf(at18_tau_1),
    2: bioNormalCdf(at18_tau_2) - bioNormalCdf(at18_tau_1),
    3: bioNormalCdf(at18_tau_3) - bioNormalCdf(at18_tau_2),
    4: bioNormalCdf(at18_tau_4) - bioNormalCdf(at18_tau_3),
    5: 1 - bioNormalCdf(at18_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at18 = Elem(Indat18, at18)

at19_tau_1 = (tau_13 - MODEL_at19) / SIGMA_STAR_at19
at19_tau_2 = (tau_23 - MODEL_at19) / SIGMA_STAR_at19
at19_tau_3 = (tau_33 - MODEL_at19) / SIGMA_STAR_at19
at19_tau_4 = (tau_43 - MODEL_at19) / SIGMA_STAR_at19
Indat19 = {
    1: bioNormalCdf(at19_tau_1),
    2: bioNormalCdf(at19_tau_2) - bioNormalCdf(at19_tau_1),
    3: bioNormalCdf(at19_tau_3) - bioNormalCdf(at19_tau_2),
    4: bioNormalCdf(at19_tau_4) - bioNormalCdf(at19_tau_3),
    5: 1 - bioNormalCdf(at19_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at19 = Elem(Indat19, at19)
###########################################################
delta_14 = Beta('delta_14', 0.1, 1.0e-5, None, 0)
delta_24 = Beta('delta_24', 0.2, 1.0e-5, None, 0)
delta_34 = Beta('delta_34', 0.3, 1.0e-5, None, 0)

tau_14 = 0
tau_24 = 0 + delta_14
tau_34 = tau_24 + delta_24
tau_44 = tau_34 + delta_34

at12_tau_1 = (tau_14 - MODEL_at12) / SIGMA_STAR_at12
at12_tau_2 = (tau_24 - MODEL_at12) / SIGMA_STAR_at12
at12_tau_3 = (tau_34 - MODEL_at12) / SIGMA_STAR_at12
at12_tau_4 = (tau_44 - MODEL_at12) / SIGMA_STAR_at12
Indat12 = {
    1: bioNormalCdf(at12_tau_1),
    2: bioNormalCdf(at12_tau_2) - bioNormalCdf(at12_tau_1),
    3: bioNormalCdf(at12_tau_3) - bioNormalCdf(at12_tau_2),
    4: bioNormalCdf(at12_tau_4) - bioNormalCdf(at12_tau_3),
    5: 1 - bioNormalCdf(at12_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at12 = Elem(Indat12, at12)

at15_tau_1 = (tau_14 - MODEL_at15) / SIGMA_STAR_at15
at15_tau_2 = (tau_24 - MODEL_at15) / SIGMA_STAR_at15
at15_tau_3 = (tau_34 - MODEL_at15) / SIGMA_STAR_at15
at15_tau_4 = (tau_44 - MODEL_at15) / SIGMA_STAR_at15
Indat15 = {
    1: bioNormalCdf(at15_tau_1),
    2: bioNormalCdf(at15_tau_2) - bioNormalCdf(at15_tau_1),
    3: bioNormalCdf(at15_tau_3) - bioNormalCdf(at15_tau_2),
    4: bioNormalCdf(at15_tau_4) - bioNormalCdf(at15_tau_3),
    5: 1 - bioNormalCdf(at15_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at15 = Elem(Indat15, at15)
###########################################################
# delta_15 = Beta('delta_15', 0.1, 1.0e-5, None, 0)
# delta_25 = Beta('delta_25', 0.2, 1.0e-5, None, 0)
# delta_35 = Beta('delta_35', 0.3, 1.0e-5, None, 0)
#
# tau_15 = 0
# tau_25 = 0 + delta_15
# tau_35 = tau_25 + delta_25
# tau_45 = tau_35 + delta_35
#
# at2_tau_1 = (tau_15 - MODEL_at2) / SIGMA_STAR_at2
# at2_tau_2 = (tau_25 - MODEL_at2) / SIGMA_STAR_at2
# at2_tau_3 = (tau_35 - MODEL_at2) / SIGMA_STAR_at2
# at2_tau_4 = (tau_45 - MODEL_at2) / SIGMA_STAR_at2
# Indat2 = {
#     1: bioNormalCdf(at2_tau_1),
#     2: bioNormalCdf(at2_tau_2) - bioNormalCdf(at2_tau_1),
#     3: bioNormalCdf(at2_tau_3) - bioNormalCdf(at2_tau_2),
#     4: bioNormalCdf(at2_tau_4) - bioNormalCdf(at2_tau_3),
#     5: 1 - bioNormalCdf(at2_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at2 = Elem(Indat2, at2)
#
# at3_tau_1 = (tau_15 - MODEL_at3) / SIGMA_STAR_at3
# at3_tau_2 = (tau_25 - MODEL_at3) / SIGMA_STAR_at3
# at3_tau_3 = (tau_35 - MODEL_at3) / SIGMA_STAR_at3
# at3_tau_4 = (tau_45 - MODEL_at3) / SIGMA_STAR_at3
# Indat3 = {
#     1: bioNormalCdf(at3_tau_1),
#     2: bioNormalCdf(at3_tau_2) - bioNormalCdf(at3_tau_1),
#     3: bioNormalCdf(at3_tau_3) - bioNormalCdf(at3_tau_2),
#     4: bioNormalCdf(at3_tau_4) - bioNormalCdf(at3_tau_3),
#     5: 1 - bioNormalCdf(at3_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at3 = Elem(Indat3, at3)
###########################################################
delta_16 = Beta('delta_16', 0.1, 1.0e-5, None, 0)
delta_26 = Beta('delta_26', 0.2, 1.0e-5, None, 0)
delta_36 = Beta('delta_36', 0.3, 1.0e-5, None, 0)

tau_16 = 0
tau_26 = 0 + delta_16
tau_36 = tau_26 + delta_26
tau_46 = tau_36 + delta_36

at1_tau_1 = (tau_16 - MODEL_at1) / SIGMA_STAR_at1
at1_tau_2 = (tau_26 - MODEL_at1) / SIGMA_STAR_at1
at1_tau_3 = (tau_36 - MODEL_at1) / SIGMA_STAR_at1
at1_tau_4 = (tau_46 - MODEL_at1) / SIGMA_STAR_at1
Indat1 = {
    1: bioNormalCdf(at1_tau_1),
    2: bioNormalCdf(at1_tau_2) - bioNormalCdf(at1_tau_1),
    3: bioNormalCdf(at1_tau_3) - bioNormalCdf(at1_tau_2),
    4: bioNormalCdf(at1_tau_4) - bioNormalCdf(at1_tau_3),
    5: 1 - bioNormalCdf(at1_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at1 = Elem(Indat1, at1)

at4_tau_1 = (tau_16 - MODEL_at4) / SIGMA_STAR_at4
at4_tau_2 = (tau_26 - MODEL_at4) / SIGMA_STAR_at4
at4_tau_3 = (tau_36 - MODEL_at4) / SIGMA_STAR_at4
at4_tau_4 = (tau_46 - MODEL_at4) / SIGMA_STAR_at4
Indat4 = {
    1: bioNormalCdf(at4_tau_1),
    2: bioNormalCdf(at4_tau_2) - bioNormalCdf(at4_tau_1),
    3: bioNormalCdf(at4_tau_3) - bioNormalCdf(at4_tau_2),
    4: bioNormalCdf(at4_tau_4) - bioNormalCdf(at4_tau_3),
    5: 1 - bioNormalCdf(at4_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at4 = Elem(Indat4, at4)

at6_tau_1 = (tau_16 - MODEL_at6) / SIGMA_STAR_at6
at6_tau_2 = (tau_26 - MODEL_at6) / SIGMA_STAR_at6
at6_tau_3 = (tau_36 - MODEL_at6) / SIGMA_STAR_at6
at6_tau_4 = (tau_46 - MODEL_at6) / SIGMA_STAR_at6
Indat6 = {
    1: bioNormalCdf(at6_tau_1),
    2: bioNormalCdf(at6_tau_2) - bioNormalCdf(at6_tau_1),
    3: bioNormalCdf(at6_tau_3) - bioNormalCdf(at6_tau_2),
    4: bioNormalCdf(at6_tau_4) - bioNormalCdf(at6_tau_3),
    5: 1 - bioNormalCdf(at6_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_at6 = Elem(Indat6, at6)

# =============================================================================
# 2. 参数定义 (Beta Parameters) - 完整版
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1 初始状态概率参数 (Initial State Probability)
# 对应PyMC: alpha_init, beta_init
# 使用MNL结构: P(S1) vs P(S2), S1为参考状态
# -----------------------------------------------------------------------------
ASC_Init_S2 = Beta('ASC_Init_S2', 0, None, None, 0)

# 初始状态协变量系数 (对应PyMC中X_init的所有变量)
B_MaasFamiliar_Init = Beta('B_MaasFamiliar_Init', 0, None, None, 0)
B_Sex_Init = Beta('B_Sex_Init', 0, None, None, 0)
B_Age2_Init = Beta('B_Age2_Init', 0, None, None, 0)
B_Age3_Init = Beta('B_Age3_Init', 0, None, None, 0)
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
B_TravelDistWork_Init = Beta('B_TravelDistWork_Init', 0, None, None, 0)
B_D6_Init = Beta('B_D6_Init', 0, None, None, 0)  # 共享汽车偏好
B_F6_Init = Beta('B_F6_Init', 0, None, None, 0)  # 共享单车偏好

# -----------------------------------------------------------------------------
# 2.2 状态转移概率参数 (State Transition)
# 对应PyMC: trans_logits_raw, gamma_trans
# 转移逻辑: Logit(P(S2|S_prev))
# -----------------------------------------------------------------------------
# 基础转移常数
Trans_Base_S1_to_S2 = Beta('Trans_Base_S1_to_S2', 0, None, None, 0)  # 从S1转到S2
Trans_Base_S2_to_S2 = Beta('Trans_Base_S2_to_S2', 1, None, None, 0)  # 从S2留在S2 (惯性)

# 状态转移协变量系数 (对应PyMC中X_trans的所有变量)
G_ChooseOptions = Beta('G_ChooseOptions', 0, None, None, 0)
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
# 2.3 阶段1效用参数 - MaaS模式选择 (State-Specific)
# 对应PyMC: ASC_t1, beta_xxx_t1
# -----------------------------------------------------------------------------

# 基准常数 (有序约束确保状态可识别)
# State 1 (Skeptic) 基准ASC
ASC_M1_S1 = Beta('ASC_M1_S1', 0, None, None, 0)
ASC_M2_S1 = Beta('ASC_M2_S1', 0, None, None, 0)
ASC_M3_S1 = Beta('ASC_M3_S1', 0, None, None, 0)
ASC_M4_S1 = Beta('ASC_M4_S1', 0, None, None, 0)
# State 2 = State 1 + Delta (强制正值确保有序)
ASC_Diff1 = Beta('ASC_Diff1', 1, 0, None, 0)  # lower bound 0 确保有序
ASC_Diff2 = Beta('ASC_Diff2', 1, 0, None, 0)
ASC_Diff3 = Beta('ASC_Diff3', 1, 0, None, 0)
ASC_Diff4 = Beta('ASC_Diff4', 1, 0, None, 0)

ASC_M1_S2 = ASC_M1_S1 + ASC_Diff1
ASC_M2_S2 = ASC_M2_S1 + ASC_Diff2
ASC_M3_S2 = ASC_M3_S1 + ASC_Diff3
ASC_M4_S2 = ASC_M4_S1 + ASC_Diff4


# --- State 1 (Skeptic) 阶段1 LOS参数 ---
B_FirstCar_S1 = Beta('B_FirstCar_S1', 0, None, None, 0)
B_FirstTaxi_S1 = Beta('B_FirstTaxi_S1', 0, None, None, 0)
B_FirstPT_S1 = Beta('B_FirstPT_S1', 0, None, None, 0)
B_Distance5_S1 = Beta('B_Distance5_S1', 0, None, None, 0)
B_RailTime_S1 = Beta('B_RailTime_S1', -0.01, None, None, 0)
B_TripTime_S1 = Beta('B_TripTime_S1', -0.01, None, None, 0)
B_Normal_S1 = Beta('B_Normal_S1', 0, None, None, 0)

# --- State 2 (Enthusiast) 阶段1 LOS参数 ---
B_FirstCar_S2 = Beta('B_FirstCar_S2', 0, None, None, 0)
B_FirstTaxi_S2 = Beta('B_FirstTaxi_S2', 0, None, None, 0)
B_FirstPT_S2 = Beta('B_FirstPT_S2', 0, None, None, 0)
B_Distance5_S2 = Beta('B_Distance5_S2', 0, None, None, 0)
B_RailTime_S2 = Beta('B_RailTime_S2', -0.01, None, None, 0)
B_TripTime_S2 = Beta('B_TripTime_S2', -0.01, None, None, 0)
B_Normal_S2 = Beta('B_Normal_S2', 0, None, None, 0)

# -----------------------------------------------------------------------------
# 2.4 阶段2效用参数 - 套餐订阅选择 (State-Specific)
# 对应PyMC: ASC_t2, beta_xxx_t2
# -----------------------------------------------------------------------------

# --- State 1 (Skeptic) 阶段2参数 ---
# ASC (PAYG为参考)
ASC_Bus_S1 = Beta('ASC_Bus_S1', 0, None, None, 0)
ASC_Metro_S1 = Beta('ASC_Metro_S1', 0, None, None, 0)
ASC_Taxi_S1 = Beta('ASC_Taxi_S1', 0, None, None, 0)
ASC_Ultra_S1 = Beta('ASC_Ultra_S1', 0, None, None, 0)

# 套餐属性系数
B_Taxi12_S1 = Beta('B_Taxi12_S1', 0, None, None, 0)
B_PriceRatio_S1 = Beta('B_PriceRatio_S1', 0, None, None, 0)
B_Price_S1 = Beta('B_Price_S1', -0.1, None, None, 0)

# 个人属性系数
B_WeekBus_S1 = Beta('B_WeekBus_S1', 0, None, None, 0)
B_Ebike_S1 = Beta('B_Ebike_S1', 0, None, None, 0)
B_Occupy_S1 = Beta('B_Occupy_S1', 0, None, None, 0)
B_Sex_S1 = Beta('B_Sex_S1', 0, None, None, 0)
B_Income1_S1 = Beta('B_Income1_S1', 0, None, None, 0)
B_Age4_S1 = Beta('B_Age4_S1', 0, None, None, 0)
B_TravelDistWork_S1 = Beta('B_TravelDistWork_S1', 0, None, None, 0)
B_WeekMetro_S1 = Beta('B_WeekMetro_S1', 0, None, None, 0)
B_C7_S1 = Beta('B_C7_S1', 0, None, None, 0)  # 多模式组合偏好
B_TravelDistWeekend_S1 = Beta('B_TravelDistWeekend_S1', 0, None, None, 0)
B_WeekTaxi_S1 = Beta('B_WeekTaxi_S1', 0, None, None, 0)
B_Age3_S1 = Beta('B_Age3_S1', 0, None, None, 0)
B_Income2_S1 = Beta('B_Income2_S1', 0, None, None, 0)
B_C6_S1 = Beta('B_C6_S1', 0, None, None, 0)  # 小汽车偏好
B_Cost_S1 = Beta('B_Cost_S1', -0.1, None, None, 0)
B_License_S1 = Beta('B_License_S1', 0, None, None, 0)
B_HaveCar_S1 = Beta('B_HaveCar_S1', 0, None, None, 0)
B_Education_S1 = Beta('B_Education_S1', 0, None, None, 0)

# --- State 2 (Enthusiast) 阶段2参数 ---
# ASC
ASC_Bus_S2 = Beta('ASC_Bus_S2', 0, None, None, 0)
ASC_Metro_S2 = Beta('ASC_Metro_S2', 0, None, None, 0)
ASC_Taxi_S2 = Beta('ASC_Taxi_S2', 0, None, None, 0)
ASC_Ultra_S2 = Beta('ASC_Ultra_S2', 0, None, None, 0)

# 套餐属性系数
B_Taxi12_S2 = Beta('B_Taxi12_S2', 0, None, None, 0)
B_PriceRatio_S2 = Beta('B_PriceRatio_S2', 0, None, None, 0)
B_Price_S2 = Beta('B_Price_S2', -0.1, None, None, 0)

# 个人属性系数
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
B_C6_S2 = Beta('B_C6_S2', 0, None, None, 0)
B_Cost_S2 = Beta('B_Cost_S2', -0.1, None, None, 0)
B_License_S2 = Beta('B_License_S2', 0, None, None, 0)
B_HaveCar_S2 = Beta('B_HaveCar_S2', 0, None, None, 0)
B_Education_S2 = Beta('B_Education_S2', 0, None, None, 0)

## 态度变量
B_FACTOR1 = Beta('B_FACTOR1',0,-1000,1000,0)
B_FACTOR2 = Beta('B_FACTOR2',0,-1000,1000,0)
B_FACTOR3 = Beta('B_FACTOR3',0,-1000,1000,0)
B_FACTOR4 = Beta('B_FACTOR4',0,-1000,1000,0)
B_FACTOR5 = Beta('B_FACTOR5',0,-1000,1000,0)
B_FACTOR6 = Beta('B_FACTOR6',0,-1000,1000,0)

# =============================================================================
# 3. 效用函数定义 (Utility Functions)
# =============================================================================

# --- 计算状态相关的基准ASC ---

# -----------------------------------------------------------------------------
# 3.1 阶段1效用: MaaS模式选择
# 选项: 0=No(不转移), 1=M1(地铁+公交), 2=M2(地铁+单车), 3=M3(地铁+网约车), 4=M4(共享汽车)
# -----------------------------------------------------------------------------

# --- State 1 (Skeptic) Utilities ---
# 不转移选项 (参考选项的效用)
V1_0_S1 = (B_FirstCar_S1 * first_car +
           B_FirstTaxi_S1 * first_taxi +
           B_Distance5_S1 * distance5)

# M1: 地铁+公交
V1_1_S1 = (ASC_M1_S1 +
           B_RailTime_S1 * M1ttimerail / 10 +
           B_TripTime_S1 * M1triptime / 10 +
           B_FirstPT_S1 * first_pt +
           B_Normal_S1 * normal)

# M2: 地铁+共享单车
V1_2_S1 = (ASC_M2_S1 +
           B_RailTime_S1 * M2ttime_rail / 10 +
           B_TripTime_S1 * M2triptime / 10 +
           B_FirstPT_S1 * first_pt +
           B_Normal_S1 * normal)

# M3: 地铁+网约车
V1_3_S1 = (ASC_M3_S1 +
           B_RailTime_S1 * M3ttime_rail / 10 +
           B_TripTime_S1 * M3triptime / 10 +
           B_FirstTaxi_S1 * first_taxi +
           B_Normal_S1 * normal)

# M4: 共享汽车
V1_4_S1 = (ASC_M4_S1 +
           B_TripTime_S1 * M4ttime / 10 +
           B_FirstTaxi_S1 * first_taxi +
           B_Distance5_S1 * distance5)

V1_Map_S1 = {0: V1_0_S1, 1: V1_1_S1, 2: V1_2_S1, 3: V1_3_S1, 4: V1_4_S1}

# --- State 2 (Enthusiast) Utilities ---
V1_0_S2 = (B_FirstCar_S2 * first_car +
           B_FirstTaxi_S2 * first_taxi +
           B_Distance5_S2 * distance5)

V1_1_S2 = (ASC_M1_S2 +
           B_RailTime_S2 * M1ttimerail / 10 +
           B_TripTime_S2 * M1triptime / 10 +
           B_FirstPT_S2 * first_pt +
           B_Normal_S2 * normal)

V1_2_S2 = (ASC_M2_S2 +
           B_RailTime_S2 * M2ttime_rail / 10 +
           B_TripTime_S2 * M2triptime / 10 +
           B_FirstPT_S2 * first_pt +
           B_Normal_S2 * normal)

V1_3_S2 = (ASC_M3_S2 +
           B_RailTime_S2 * M3ttime_rail / 10 +
           B_TripTime_S2 * M3triptime / 10 +
           B_FirstTaxi_S2 * first_taxi +
           B_Normal_S2 * normal)

V1_4_S2 = (ASC_M4_S2 +
           B_TripTime_S2 * M4ttime / 10 +
           B_FirstTaxi_S2 * first_taxi +
           B_Distance5_S2 * distance5)

V1_Map_S2 = {0: V1_0_S2, 1: V1_1_S2, 2: V1_2_S2, 3: V1_3_S2, 4: V1_4_S2}

# -----------------------------------------------------------------------------
# 3.2 阶段2效用: 套餐订阅选择
# 选项: 0=Bus First, 1=Metro Access, 2=Value Taxi, 3=Ultra Access, 4=PAYG
# -----------------------------------------------------------------------------

# --- State 1 (Skeptic) Utilities ---
# Bus First
V2_0_S1 = (ASC_Bus_S1 +
           B_Taxi12_S1 * taxi_12 +
           B_PriceRatio_S1 * price_12 +
           B_Price_S1 * price1 / 10 +
           B_WeekBus_S1 * week_bus +
           B_Ebike_S1 * e_bike +
           B_Occupy_S1 * occupy +
           B_Sex_S1 * sex +
           B_Income1_S1 * income1 +
           B_Age4_S1 * age4)

# Metro Access
V2_1_S1 = (ASC_Metro_S1 +
           B_Taxi12_S1 * taxi_12 +
           B_PriceRatio_S1 * price_12 +
           B_Price_S1 * price2 / 10 +
           B_TravelDistWork_S1 * travel_distance_work +
           B_WeekMetro_S1 * week_metro +
           B_Ebike_S1 * e_bike +
           B_Occupy_S1 * occupy +
           B_Sex_S1 * sex +
           B_C7_S1 * c7 +
           B_Income1_S1 * income1 +
           B_Age4_S1 * age4)

# Value Taxi
V2_2_S1 = (ASC_Taxi_S1 +
           B_PriceRatio_S1 * price_3 +
           B_Price_S1 * price3 / 10 +
           B_TravelDistWeekend_S1 * travel_distance_weekend +
           B_WeekTaxi_S1 * week_taxi +
           B_Age3_S1 * age3 +
           B_Income2_S1 * income2)

# Ultra Access
V2_3_S1 = (ASC_Ultra_S1 +
           B_PriceRatio_S1 * price_4 +
           B_Price_S1 * price4 / 10 +
           B_C6_S1 * c6 +
           B_Cost_S1 * cost +
           B_WeekTaxi_S1 * week_taxi +
           B_Age3_S1 * age3)

# PAYG (参考选项)
V2_4_S1 = (B_Cost_S1 * cost +
           B_License_S1 * license +
           B_HaveCar_S1 * have_car +
           B_Education_S1 * education)

V2_Map_S1 = {0: V2_0_S1, 1: V2_1_S1, 2: V2_2_S1, 3: V2_3_S1, 4: V2_4_S1}

# --- State 2 (Enthusiast) Utilities ---
V2_0_S2 = (ASC_Bus_S2 +
           B_Taxi12_S2 * taxi_12 +
           B_PriceRatio_S2 * price_12 +
           B_Price_S2 * price1 / 10 +
           B_WeekBus_S2 * week_bus +
           B_Ebike_S2 * e_bike +
           B_Occupy_S2 * occupy +
           B_Sex_S2 * sex +
           B_Income1_S2 * income1 +
           B_Age4_S2 * age4)

V2_1_S2 = (ASC_Metro_S2 +
           B_Taxi12_S2 * taxi_12 +
           B_PriceRatio_S2 * price_12 +
           B_Price_S2 * price2 / 10 +
           B_TravelDistWork_S2 * travel_distance_work +
           B_WeekMetro_S2 * week_metro +
           B_Ebike_S2 * e_bike +
           B_Occupy_S2 * occupy +
           B_Sex_S2 * sex +
           B_C7_S2 * c7 +
           B_Income1_S2 * income1 +
           B_Age4_S2 * age4)

V2_2_S2 = (ASC_Taxi_S2 +
           B_PriceRatio_S2 * price_3 +
           B_Price_S2 * price3 / 10 +
           B_TravelDistWeekend_S2 * travel_distance_weekend +
           B_WeekTaxi_S2 * week_taxi +
           B_Age3_S2 * age3 +
           B_Income2_S2 * income2)

V2_3_S2 = (ASC_Ultra_S2 +
           B_PriceRatio_S2 * price_4 +
           B_Price_S2 * price4 / 10 +
           B_C6_S2 * c6 +
           B_Cost_S2 * cost +
           B_WeekTaxi_S2 * week_taxi +
           B_Age3_S2 * age3)

V2_4_S2 = (B_Cost_S2 * cost +
           B_License_S2 * license +
           B_HaveCar_S2 * have_car +
           B_Education_S2 * education)

V2_Map_S2 = {0: V2_0_S2, 1: V2_1_S2, 2: V2_2_S2, 3: V2_3_S2, 4: V2_4_S2}

# 可用性 (假设所有选项对所有人都可用)
av1 = {0:1, 1:1, 2:1, 3:1, 4:1}
av2 = {0:1, 1:1, 2:1, 3:1, 4:1}

MU1 = Beta('MU1', 1, 0, 100, 0)
MU2 = Beta('MU2', 1, 0, 100, 0)
# MU3 = Beta('MU3', 1, 0, 100, 0)
# PT = MU1, [1,2,3]
# Car = 1, [4]
# Taxi = 1, [5]
# Bike = 1, [6]

##nested
PT = MU1, [0,1]
TAXI = 1, [2]
MORE = 1,[3]
NO = 1 ,[4]
nests1 = PT,TAXI, MORE,NO

PT2 = MU2, [0,1]
TAXI2 = 1, [2]
MORE2 = 1,[3]
NO2 = 1 ,[4]
nests2 = PT2, TAXI2, MORE2, NO2

# =============================================================================
# 4. 概率计算 (Probabilities)
# =============================================================================

# -----------------------------------------------------------------------------
# 4.1 观测概率 P(y | State) - 发射概率
# -----------------------------------------------------------------------------
# 阶段1观测概率
Prob_T1_Given_S1 = models.logit(V1_Map_S1, av1,  CHOICE_T1)
Prob_T1_Given_S2 = models.logit(V1_Map_S2, av1, CHOICE_T1)

# 阶段2观测概率
Prob_T2_Given_S1 = exp(models.lognested(V2_Map_S1, av2, nests1, CHOICE_T2))
Prob_T2_Given_S2 = models.logit(V2_Map_S2, av2, CHOICE_T2)

# -----------------------------------------------------------------------------
# 4.2 初始状态概率 P(State_1) - 使用MNL结构
# -----------------------------------------------------------------------------
# V_Init_S1 = 0 (参考状态)
# V_Init_S2 = ASC + beta * X
V_Init_S2 = (ASC_Init_S2 +
             B_MaasFamiliar_Init * MaasFamiliar +
             B_Sex_Init * sex +
             B_Age2_Init * age2 +
             B_Age3_Init * age3 +
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
             B_Ebike_Init * e_bike +
             B_TravelDistWork_Init * travel_distance_work +
             B_D6_Init * d6 +
             B_F6_Init * f6)

# Softmax for 2 states: P(S1) = 1/(1+exp(V_S2)), P(S2) = exp(V_S2)/(1+exp(V_S2))
Prob_Init_S1 = 1 / (1 + exp(V_Init_S2))
Prob_Init_S2 = exp(V_Init_S2) / (1 + exp(V_Init_S2))

# -----------------------------------------------------------------------------
# 4.3 状态转移概率 P(State_t | State_{t-1})
# -----------------------------------------------------------------------------
# 转移协变量效用
Trans_Covariates = (G_ChooseOptions * choose_options +
                    G_TimeSavings * time_savings +
                    G_CostSavings * cost_savings +
                    G_MatchBus * match_bus +
                    G_MatchMetro * match_metro +
                    G_MatchBike * match_bike +
                    G_MatchEbike * match_e_bike +
                    G_MatchTaxi * match_taxi +
                    G_MatchPrice * match_price +
                    G_PriceRatio * price_ratio +
                    B_FACTOR1 * FACTOR1 +
                    B_FACTOR2 * FACTOR2 +
                    B_FACTOR3 * FACTOR3 +
                    B_FACTOR4 * FACTOR4 +
                    # B_FACTOR5 * FACTOR5 +
                    B_FACTOR6 * FACTOR6)

# Case A: Previous State was S1 (Skeptic)
# V_Stay_S1 = 0 (参考)
# V_Goto_S2 = Trans_Base_S1_to_S2 + Trans_Covariates
V_S1_to_S2 = Trans_Base_S1_to_S2 + Trans_Covariates
Prob_S1_to_S1 = 1 / (1 + exp(V_S1_to_S2))
Prob_S1_to_S2 = exp(V_S1_to_S2) / (1 + exp(V_S1_to_S2))

# Case B: Previous State was S2 (Enthusiast)
# V_Goto_S1 = 0 (参考)
# V_Stay_S2 = Trans_Base_S2_to_S2 + Trans_Covariates
V_S2_to_S2 = Trans_Base_S2_to_S2 + Trans_Covariates
Prob_S2_to_S1 = 1 / (1 + exp(V_S2_to_S2))
Prob_S2_to_S2 = exp(V_S2_to_S2) / (1 + exp(V_S2_to_S2))

# =============================================================================
# 5. HMM似然函数构造 (Forward Algorithm)
# =============================================================================
# L = Sum_over_start_states [ P(start) * P(y1|start) * Sum_over_end_states [ P(end|start) * P(y2|end) ] ]

# 路径1: 初始状态为S1
Path_Start_S1 = Prob_Init_S1 * Prob_T1_Given_S1 * (
    Prob_S1_to_S1 * Prob_T2_Given_S1 +
    Prob_S1_to_S2 * Prob_T2_Given_S2
)

# 路径2: 初始状态为S2
Path_Start_S2 = Prob_Init_S2 * Prob_T1_Given_S2 * (
    Prob_S2_to_S1 * Prob_T2_Given_S1 +
    Prob_S2_to_S2 * Prob_T2_Given_S2
)

# 总观测概率 (边际化隐状态)
Prob_Obs = Path_Start_S1 + Path_Start_S2

# 总测量概率 (Likelihood of Measurement Equations)
Likelihood_Measurement = (
    P_at8 * P_at9 * P_at10 * P_at11 * P_at13 * P_at14 * P_at17 *
    P_at21 * P_at22 * P_at23 * P_at24 * P_at25 *
    P_at18 * P_at19 *
    P_at12 * P_at15 *
    P_at1 * P_at4 * P_at6
)

# 联合概率 = P(Observed Choices | Factors) * P(Measurements | Factors)
# 注意：由于使用了 bioDraws，这里必须是联合概率在 Draw 层面上的乘积
Joint_Prob = Prob_Obs * Likelihood_Measurement

# 对数似然 (使用 MonteCarlo 进行积分)
loglike = log(MonteCarlo(Joint_Prob))
# =============================================================================
# 6. 模型估计
# =============================================================================
# 设置日志级别
logger = msg.bioMessage()
logger.setGeneral()

# 创建Biogeme对象
biogeme_obj = bio.BIOGEME(database, loglike,numberOfDraws=500)
biogeme_obj.modelName = "MaaS_HMM_Full"

# 估计模型
print("开始估计HMM模型...")
print("=" * 70)
results = biogeme_obj.estimate()

# =============================================================================
# 7. 输出结果
# =============================================================================
print("\n" + "=" * 70)
print("模型估计结果")
print("=" * 70)

print(f'\n估计参数数量: {len(results.data.betaValues)}')
print(f'最终对数似然: {results.data.logLike:.3f}')
print(f'输出文件: {results.data.htmlFileName}')

# 写入LaTeX文件
results.writeLaTeX()
print(f'LaTeX文件: {results.data.latexFileName}')

# 输出估计参数
print("\n" + "=" * 70)
print("估计参数")
print("=" * 70)
pandasResults = results.getEstimatedParameters()
print(pandasResults)

# 保存结果到CSV
pandasResults.to_csv('MaaS_HMM_Full_results.csv')
print("\n结果已保存到 MaaS_HMM_Full_results.csv")