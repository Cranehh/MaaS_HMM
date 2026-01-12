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
import biogeme.messaging as msg
import biogeme.models as models
from biogeme.expressions import Beta, Variable, exp, log, Elem

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
ASC_Base_S1 = Beta('ASC_Base_S1', -1, None, None, 0)
# State 2 = State 1 + Delta (强制正值确保有序)
ASC_Diff = Beta('ASC_Diff', 1, 0, None, 0)  # lower bound 0 确保有序

# 模式特定偏差 (M1-M4相对于No的ASC差异, 跨状态共享)
M1_Delta = Beta('M1_Delta', 0, None, None, 0)
M2_Delta = Beta('M2_Delta', 0, None, None, 0)
M3_Delta = Beta('M3_Delta', 0, None, None, 0)
M4_Delta = Beta('M4_Delta', 0, None, None, 0)

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

# =============================================================================
# 3. 效用函数定义 (Utility Functions)
# =============================================================================

# --- 计算状态相关的基准ASC ---
ASC_Base_S2 = ASC_Base_S1 + ASC_Diff

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
V1_1_S1 = (ASC_Base_S1 + M1_Delta +
           B_RailTime_S1 * M1ttimerail / 10 +
           B_TripTime_S1 * M1triptime / 10 +
           B_FirstPT_S1 * first_pt +
           B_Normal_S1 * normal)

# M2: 地铁+共享单车
V1_2_S1 = (ASC_Base_S1 + M2_Delta +
           B_RailTime_S1 * M2ttime_rail / 10 +
           B_TripTime_S1 * M2triptime / 10 +
           B_FirstPT_S1 * first_pt +
           B_Normal_S1 * normal)

# M3: 地铁+网约车
V1_3_S1 = (ASC_Base_S1 + M3_Delta +
           B_RailTime_S1 * M3ttime_rail / 10 +
           B_TripTime_S1 * M3triptime / 10 +
           B_FirstTaxi_S1 * first_taxi +
           B_Normal_S1 * normal)

# M4: 共享汽车
V1_4_S1 = (ASC_Base_S1 + M4_Delta +
           B_TripTime_S1 * M4ttime / 10 +
           B_FirstTaxi_S1 * first_taxi +
           B_Distance5_S1 * distance5)

V1_Map_S1 = {0: V1_0_S1, 1: V1_1_S1, 2: V1_2_S1, 3: V1_3_S1, 4: V1_4_S1}

# --- State 2 (Enthusiast) Utilities ---
V1_0_S2 = (B_FirstCar_S2 * first_car +
           B_FirstTaxi_S2 * first_taxi +
           B_Distance5_S2 * distance5)

V1_1_S2 = (ASC_Base_S2 + M1_Delta +
           B_RailTime_S2 * M1ttimerail / 10 +
           B_TripTime_S2 * M1triptime / 10 +
           B_FirstPT_S2 * first_pt +
           B_Normal_S2 * normal)

V1_2_S2 = (ASC_Base_S2 + M2_Delta +
           B_RailTime_S2 * M2ttime_rail / 10 +
           B_TripTime_S2 * M2triptime / 10 +
           B_FirstPT_S2 * first_pt +
           B_Normal_S2 * normal)

V1_3_S2 = (ASC_Base_S2 + M3_Delta +
           B_RailTime_S2 * M3ttime_rail / 10 +
           B_TripTime_S2 * M3triptime / 10 +
           B_FirstTaxi_S2 * first_taxi +
           B_Normal_S2 * normal)

V1_4_S2 = (ASC_Base_S2 + M4_Delta +
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

# =============================================================================
# 4. 概率计算 (Probabilities)
# =============================================================================

# -----------------------------------------------------------------------------
# 4.1 观测概率 P(y | State) - 发射概率
# -----------------------------------------------------------------------------
# 阶段1观测概率
Prob_T1_Given_S1 = models.logit(V1_Map_S1, av1, CHOICE_T1)
Prob_T1_Given_S2 = models.logit(V1_Map_S2, av1, CHOICE_T1)

# 阶段2观测概率
Prob_T2_Given_S1 = models.logit(V2_Map_S1, av2, CHOICE_T2)
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
                    G_PriceRatio * price_ratio)

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

# 对数似然
loglike = log(Prob_Obs)

# =============================================================================
# 6. 模型估计
# =============================================================================
# 设置日志级别
logger = msg.bioMessage()
logger.setGeneral()

# 创建Biogeme对象
biogeme_obj = bio.BIOGEME(database, loglike)
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

# 输出简要摘要
print("\n" + "=" * 70)
print("模型摘要")
print("=" * 70)
print(results.shortSummary())

# 输出估计参数
print("\n" + "=" * 70)
print("估计参数")
print("=" * 70)
pandasResults = results.getEstimatedParameters()
print(pandasResults)

# 保存结果到CSV
pandasResults.to_csv('MaaS_HMM_Full_results.csv')
print("\n结果已保存到 MaaS_HMM_Full_results.csv")