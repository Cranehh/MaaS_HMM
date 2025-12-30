"""
MaaS多阶段采纳决策的隐马尔可夫模型 (HMM) - 完整版
================================================================

基于实际问卷数据结构，严格对应原有Biogeme模型的效用函数设定

阶段1 (Part 3 - one-trip场景选择):
    选择肢: 1=不转移, 4=M1(地铁+公交), 5=M2(地铁+单车), 6=M3(地铁+网约车), 7=M4(共享汽车)
    因变量: maas

阶段2 (Part 4 - 套餐订阅):
    选择肢: 1=Bus First, 2=Metro Access, 3=Value Taxi, 4=Ultra Access, 5=PAYG
    因变量: results

隐状态: 用户对MaaS的态度
    0: Skeptic (怀疑者)
    1: Neutral (中立者)
    2: Enthusiast (热衷者)

状态转移影响因素:
    - 套餐与出行模式匹配度
    - MaaS性价比感知
    - 第一阶段体验满意度代理变量
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. 数据加载和预处理
# =============================================================================

def load_data(filepath):
    """
    加载合并后的问卷数据
    
    Parameters
    ----------
    filepath : str
        数据文件路径
    """
    # 根据文件类型选择读取方式
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, encoding='gb2312')
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("不支持的文件格式")
    
    return df


def preprocess_data(df):
    """
    数据预处理，构建HMM所需的变量
    
    包括:
    1. 因变量处理 (y1, y2)
    2. 协变量标准化
    3. 构建状态转移的影响变量 (匹配度、性价比等)
    """
    
    data = df.copy()
    data['cost'] = data['cost'].replace(to_replace=[1,2,3,4,5,6],value=[0,0,1,1,1,1])
    # ========== 1. 因变量处理 ==========
    
    # 阶段1: maas选择 (1=不转移, 4-7=MaaS选项)
    # 简化为二元: 0=不转移(maas==1), 1=选择MaaS(maas in [4,5,6,7])
    data['y1_binary'] = (data['maas'] != 1).astype(int)
    
    # 保留原始多类别选择用于多项选择模型
    # maas: 1->0(不转移), 4->1(M1), 5->2(M2), 6->3(M3), 7->4(M4)
    maas_mapping = {1: 0, 4: 1, 5: 2, 6: 3, 7: 4}
    data['y1_multi'] = data['maas'].map(maas_mapping)
    
    # 阶段2: results (1-4=订阅套餐, 5=PAYG)
    # 简化为二元: 0=PAYG(results==5), 1=订阅套餐(results in [1,2,3,4])
    data['y2_binary'] = (data['results'] != 5).astype(int)
    
    # 保留原始多类别
    # results: 1->0(Bus First), 2->1(Metro Access), 3->2(Value Taxi), 4->3(Ultra Access), 5->4(PAYG)
    data['y2_multi'] = data['results'] - 1
    
    # ========== 2. 构建状态转移的关键变量 ==========
    
    # --- 2.1 套餐与出行模式匹配度 ---
    # 基于用户的出行模式偏好和套餐特征计算匹配度
    
    # # Bus First套餐匹配度: 高频公交用户匹配度高
    # data['match_bus_first'] = (
    #     data['week_bus'] * 0.4 +          # 公交使用频率
    #     data['a6'] * 0.3 +                 # 公交偏好
    #     (1 - data['have_car']) * 0.3      # 无车用户
    # )
    
    # # Metro Access套餐匹配度: 高频地铁用户匹配度高
    # data['match_metro_access'] = (
    #     data['week_metro'] * 0.4 +         # 地铁使用频率
    #     data['travel_distance_work'] * 0.3 + # 工作日通勤距离
    #     data['c7'] * 0.3                    # 多模式组合偏好
    # )
    
    # # Value Taxi套餐匹配度: 出租车/网约车用户匹配度高
    # data['match_value_taxi'] = (
    #     data['week_taxi'] * 0.4 +          # 出租车使用频率
    #     data['b6'] * 0.3 +                 # 出租车偏好
    #     data['travel_distance_weekend'] * 0.3  # 周末出行距离
    # )
    
    # # Ultra Access套餐匹配度: 多模式重度用户
    # data['match_ultra_access'] = (
    #     (data['week_metro'] + data['week_taxi'] + data['week_ebike']) / 3 * 0.4 +
    #     data['cost'] * 0.3 +              # 高出行花费
    #     data['travel_num'] * 0.3          # 高出行频率
    # )
    
    # # 综合匹配度 (取最大匹配度)
    # data['bundle_match'] = data[['match_bus_first', 'match_metro_access', 
    #                               'match_value_taxi', 'match_ultra_access']].max(axis=1)
    
    # # --- 2.2 MaaS性价比感知 ---
    # # 基于套餐价格与日常出行花费的比较
    
    # # 套餐平均价格 (根据选择的情景)
    # data['avg_bundle_price'] = (data['price1'] + data['price2'] + 
    #                              data['price3'] + data['price4']) / 4 * 100  # 还原缩放
    
    # # 性价比 = 日常花费 / 套餐价格 (值越大性价比越高)
    # # 使用cost作为日常花费的代理
    # cost_mapping = {0: 100, 1: 300}  # 0=150以下, 1=150以上
    # data['monthly_cost_proxy'] = data['cost'].map(cost_mapping).fillna(200)
    
    # # 标准化的性价比指标
    # data['price_value_ratio'] = (data['monthly_cost_proxy'] - data['avg_bundle_price']) / 100
    # data['price_value_ratio'] = data['price_value_ratio'].clip(-2, 2)  # 截断极值
    
    # # --- 2.3 第一阶段体验满意度代理 ---
    # # 基于MaaS选项与原有方式的时间/价格差异
    
    # # 如果选择了MaaS，计算相对于原方式的改善程度
    # # 时间节省 (负值表示MaaS更快)
    # data['time_saving'] = np.where(
    #     data['first_pt'] == 1,
    #     (data['M1triptime'] + data['M2triptime']) / 2 - data['C1triptimePT'],
    #     np.where(
    #         data['first_taxi'] == 1,
    #         (data['M3triptime'] + data['M4ttime']) / 2 - data['B1triptime'],
    #         (data['M4ttime'] - data['A1ttimeCar'])
    #     )
    # )
    # data['time_saving'] = -data['time_saving'] / 10  # 标准化，正值表示节省时间
    
    # # 价格节省
    # data['price_saving'] = np.where(
    #     data['first_pt'] == 1,
    #     (data['M1price'] + data['M2price']) / 2 - data['C1pricePT'],
    #     np.where(
    #         data['first_taxi'] == 1,
    #         (data['M3price'] + data['M4price']) / 2 - data['B1priceTaxi'],
    #         (data['M4price'] - data['A1priceCar'])
    #     )
    # )
    # data['price_saving'] = -data['price_saving'] / 10  # 标准化，正值表示省钱
    
    # # 综合满意度代理
    # data['trial_satisfaction'] = 0.5 * data['time_saving'] + 0.5 * data['price_saving']
    # data['trial_satisfaction'] = data['trial_satisfaction'].clip(-2, 2)
    
    # # ========== 3. 变量缩放 ==========
    
    # # 时间变量缩放 (除以10)
    # time_vars = ['M1triptime', 'M2triptime', 'M3triptime', 'M4ttime',
    #              'M1ttimerail', 'M2ttime_rail', 'M3ttime_rail',
    #              'C1triptimePT', 'B1triptime', 'A1ttimeCar']
    # for var in time_vars:
    #     if var in data.columns:
    #         data[var + '_scaled'] = data[var] / 10
    
    # # 价格变量缩放 (除以10)
    # price_vars = ['M1price', 'M2price', 'M3price', 'M4price',
    #               'price1', 'price2', 'price3', 'price4']
    # for var in price_vars:
    #     if var in data.columns:
    #         data[var + '_scaled'] = data[var] / 10
    
    return data


# =============================================================================
# 2. HMM模型定义 - 二元观测版本
# =============================================================================

def build_hmm_binary(data, n_states=3):
    """
    构建二元观测的HMM模型
    
    观测简化为:
    - y1: 是否选择MaaS (0/1)
    - y2: 是否订阅套餐 (0/1)
    """
    
    n_obs = len(data)
    
    # ========== 准备协变量 ==========
    
    # 初始状态协变量 (影响用户初始态度)
    # 参考原模型的个人特征变量
    X_init = np.column_stack([
        data['sex'].values,               # 性别
        data['age2'].values,              # 25-34岁
        data['age3'].values,              # 35-44岁
        data['age4'].values,              # 45岁以上
        data['income1'].values,           # 低收入
        data['income2'].values,           # 中收入
        data['education'].values,         # 本科以上
        data['occupy'].values,            # 出行多的工作
        data['week_metro'].values,        # 地铁使用频率
        data['week_bus'].values,          # 公交使用频率
        data['week_taxi'].values,         # 出租车使用频率
        data['week_ebike'].values,        # 电动车使用频率
        data['have_car'].values,          # 是否有车
        data['e_bike'].values,            # 是否有电动车
        data['travel_distance_work'].values,  # 工作日出行距离
        data['d6'].values,                # 共享汽车偏好
        data['f6'].values,                # 共享单车偏好
    ]).astype(np.float64)
    
    # 状态转移协变量 (影响态度转变)
    X_trans = np.column_stack([
        data['bundle_match'].values,          # 套餐匹配度
        data['price_value_ratio'].values,     # 性价比
        data['trial_satisfaction'].values,    # 试用满意度代理
        data['travel_num'].values,            # 出行频率
        data['cost'].values,                  # 月出行花费
    ]).astype(np.float64)
    
    # 观测数据
    y1 = data['y1_binary'].values.astype(np.int32)
    y2 = data['y2_binary'].values.astype(np.int32)
    
    n_init_cov = X_init.shape[1]
    n_trans_cov = X_trans.shape[1]
    
    # ========== 构建PyMC模型 ==========
    
    with pm.Model() as hmm_model:
        
        # ---------- 初始状态分布 P(z1|X_init) ----------
        alpha_init = pm.Normal('alpha_init', mu=0, sigma=0.5, shape=n_states-1)
        beta_init = pm.Normal('beta_init', mu=0, sigma=0.5, 
                              shape=(n_init_cov, n_states-1))
        
        logits_init = alpha_init + pt.dot(X_init, beta_init)
        logits_init_full = pt.concatenate([pt.zeros((n_obs, 1)), logits_init], axis=1)
        pi_init = pm.math.softmax(logits_init_full, axis=1)
        
        # ---------- 发射概率 (有序约束) ----------
        # t=1: 选择MaaS的概率
        emit_base_t1 = pm.Normal('emit_base_t1', mu=-1.5, sigma=0.5)
        emit_diff_t1 = pm.HalfNormal('emit_diff_t1', sigma=0.5, shape=n_states-1)
        emit_logit_t1 = pt.concatenate([
            pt.atleast_1d(emit_base_t1),
            emit_base_t1 + pt.cumsum(emit_diff_t1)
        ])
        emission_t1 = pm.math.sigmoid(emit_logit_t1)
        
        # t=2: 订阅套餐的概率
        emit_base_t2 = pm.Normal('emit_base_t2', mu=-1.5, sigma=0.5)
        emit_diff_t2 = pm.HalfNormal('emit_diff_t2', sigma=0.5, shape=n_states-1)
        emit_logit_t2 = pt.concatenate([
            pt.atleast_1d(emit_base_t2),
            emit_base_t2 + pt.cumsum(emit_diff_t2)
        ])
        emission_t2 = pm.math.sigmoid(emit_logit_t2)
        
        # ---------- 状态转移矩阵 P(z2|z1, X_trans) ----------
        trans_logits_raw = pm.Normal('trans_logits_raw', mu=0, sigma=0.5,
                                      shape=(n_states, n_states-1))
        
        # 协变量影响转移
        gamma_trans = pm.Normal('gamma_trans', mu=0, sigma=0.5, shape=n_trans_cov)
        trans_modifier = pt.dot(X_trans, gamma_trans)
        
        # 构建转移概率
        trans_logits_base = pt.concatenate([
            pt.zeros((n_states, 1)),
            trans_logits_raw
        ], axis=1)
        
        trans_logits = trans_logits_base[None, :, :] + pt.zeros((n_obs, n_states, n_states))
        # 协变量促进向更积极状态转移
        trans_logits = pt.set_subtensor(
            trans_logits[:, :, -1],
            trans_logits[:, :, -1] + trans_modifier[:, None]
        )
        trans_probs = pm.math.softmax(trans_logits, axis=2)
        
        # ---------- 边际似然 (Forward算法) ----------
        emit_1 = pt.where(y1[:, None] == 1, emission_t1[None, :], 1 - emission_t1[None, :])
        emit_2 = pt.where(y2[:, None] == 1, emission_t2[None, :], 1 - emission_t2[None, :])
        
        alpha_1 = pi_init * emit_1
        alpha_2 = pt.sum(alpha_1[:, :, None] * trans_probs * emit_2[:, None, :], axis=1)
        
        marginal = pt.sum(alpha_2, axis=1)
        log_lik = pt.sum(pt.log(marginal + 1e-10))
        
        pm.Potential('log_likelihood', log_lik)
        
        # 保存用于分析
        pm.Deterministic('emission_probs_t1', emission_t1)
        pm.Deterministic('emission_probs_t2', emission_t2)
        pm.Deterministic('pi_init_mean', pt.mean(pi_init, axis=0))
        
    return hmm_model


# =============================================================================
# 3. HMM模型定义 - 多类别观测版本 (更贴近原模型)
# =============================================================================

def build_hmm_multinomial(data, n_states=3):
    """
    构建多类别观测的HMM模型
    
    阶段1: 5个选择肢 (不转移, M1, M2, M3, M4)
    阶段2: 5个选择肢 (Bus First, Metro Access, Value Taxi, Ultra Access, PAYG)
    
    效用函数设定参考原Biogeme模型
    """
    
    n_obs = len(data)
    n_alt_t1 = 5  # 阶段1选择肢数量
    n_alt_t2 = 5  # 阶段2选择肢数量
    
    # ========== 准备LOS变量 (阶段1) ==========
    
    # 不转移选项的属性 (选项0)
    # 使用first_car, first_taxi, first_pt作为标识
    
    # 不转移选项
    X_no = np.column_stack([
        data['first_car'].values,
        data['first_taxi'].values,
        data['distance5'].values,
    ]).astype(np.float64)

    # M1: 地铁+公交
    X_M1 = np.column_stack([
        data['M1ttimerail'].values,   # 地铁时间
        data['M1triptime'].values ,     # 总出行时间
        data['first_pt'].values,            # 原方式是PT
        data['normal'].values,              # 正常时段
    ]).astype(np.float64)
    
    # M2: 地铁+共享单车
    X_M2 = np.column_stack([
        data['M2ttime_rail'].values,
        data['M2triptime'].values,
        data['first_pt'].values,
        data['normal'].values,
    ]).astype(np.float64)
    
    # M3: 地铁+网约车
    X_M3 = np.column_stack([
        data['M3ttime_rail'].values,
        data['M3triptime'].values,
        data['first_taxi'].values,
        data['normal'].values,
    ]).astype(np.float64)
    
    # M4: 共享汽车
    X_M4 = np.column_stack([
        data['M4ttime'].values,
        data['first_taxi'].values,
        data['distance5'].values,
    ]).astype(np.float64)
    
    
    # ========== 准备LOS变量 (阶段2) ==========
    
    # Bus First
    X_B1 = np.column_stack([
        data['taxi_12'].values,           # 出租车配额
        data['price_12'].values,          # 价格比例
        data['price1'].values,            # 价格
        data['week_bus'].values,          # 公交使用频率
        data['e_bike'].values,            # 有电动车
        data['occupy'].values,            # 工作类型
        data['sex'].values,               # 性别
        data['income1'].values,           # 收入
        data['age4'].values,              # 年龄
    ]).astype(np.float64)
    
    # Metro Access
    X_B2 = np.column_stack([
        data['taxi_12'].values,
        data['price_12'].values,
        data['price2'].values,
        data['travel_distance_work'].values,
        data['week_metro'].values,
        data['e_bike'].values,
        data['occupy'].values,
        data['sex'].values,
        data['c7'].values,                # 多模式组合偏好
        data['income1'].values,
        data['age4'].values,
    ]).astype(np.float64)
    
    # Value Taxi
    X_B3 = np.column_stack([
        data['price_3'].values,
        data['price3'].values,
        data['travel_distance_weekend'].values,
        data['week_taxi'].values,
        data['age3'].values,
        data['income2'].values,
    ]).astype(np.float64)
    
    # Ultra Access
    X_B4 = np.column_stack([
        data['price_4'].values,
        data['price4'].values,
        data['c6'].values,                # 小汽车偏好
        data['cost'].values,
        data['week_taxi'].values,
        data['age3'].values,
    ]).astype(np.float64)
    
    # PAYG
    X_PAYG = np.column_stack([
        data['cost'].values,
        data['license'].values,
        data['have_car'].values,
        data['education'].values,
    ]).astype(np.float64)
    
    # 初始状态协变量
    X_init = np.column_stack([
        data['MaasFamiliar'].values,
        data['sex'].values,
        data['age2'].values,
        data['age3'].values,
        data['income1'].values,
        data['income2'].values,
        data['education'].values,
        data['week_metro'].values,
        data['week_bus'].values,
        data['have_car'].values,
        data['f6'].values,
        data['d6'].values,
    ]).astype(np.float64)
    
    # 状态转移协变量
    X_trans = np.column_stack([
        data['choose_options'].values,
        data['time_savings'].values,
        data['cost_savings'].values,
        data['match_bus'].values,
        data['match_metro'].values,
        data['match_bike'].values,
        data['match_e_bike'].values,
        data['match_taxi'].values,
        data['match_price'].values,
        data['price_ratio'].values,
    ]).astype(np.float64)
    
    # 观测
    y1 = data['y1_multi'].values.astype(np.int32)
    y2 = data['y2_multi'].values.astype(np.int32)
    
    n_init_cov = X_init.shape[1]
    n_trans_cov = X_trans.shape[1]
    
    with pm.Model() as hmm_model:
        
        # ---------- 初始状态分布 ----------
        alpha_init = pm.Normal('alpha_init', mu=0, sigma=0.5, shape=n_states-1)
        beta_init = pm.Normal('beta_init', mu=0, sigma=0.5, 
                              shape=(n_init_cov, n_states-1))
        
        logits_init = alpha_init + pt.dot(X_init, beta_init)
        logits_init_full = pt.concatenate([pt.zeros((n_obs, 1)), logits_init], axis=1)
        pi_init = pm.math.softmax(logits_init_full, axis=1)
        
        # ---------- 阶段1发射概率 (多项Logit) ----------
        # 每个隐状态下各选项的效用参数不同
        
        # ASC for each alternative (state-specific)
        ASC_t1 = pm.Normal('ASC_t1', mu=0, sigma=0.5, shape=(n_states, n_alt_t1-1))
        
        # LOS参数 (简化: 跨状态共享，但scale不同)
        # beta_time_t1 = pm.Normal('beta_time_t1', mu=-0.1, sigma=0.1, shape=n_states)
        # beta_rail_t1 = pm.Normal('beta_rail_t1', mu=-0.05, sigma=0.1, shape=n_states)

        beta_firstcar_t1 = pm.Normal('beta_firstcar_t1', mu=0, sigma=0.5, shape=n_states)
        beta_firsttaxi_t1 = pm.Normal('beta_firsttaxi_t1', mu=0, sigma=0.5, shape=n_states)
        beta_firstpt_t1 = pm.Normal('beta_firstpt_t1', mu=0, sigma=0.5, shape=n_states)
        beta_distance5_t1 = pm.Normal('beta_distance5_t1', mu=-0.1, sigma=0.1, shape=n_states)
        beta_railtime_t1 = pm.Normal('beta_railtime_t1', mu=-0.1, sigma=0.1, shape=n_states)
        beta_triptime_t1 = pm.Normal('beta_triptime_t1', mu=-0.1, sigma=0.1, shape=n_states)
        beta_normal_t1 = pm.Normal('beta_normal_t1', mu=0, sigma=0.5, shape=n_states)
        
        # 状态特定的发射概率
        def compute_emit_probs_t1(state_idx):
            """计算状态state_idx下阶段1各选项的选择概率"""
            # 不转移 (参考选项)
            V_no = beta_firstcar_t1[state_idx] * X_no[:, 0] + \
                   beta_firsttaxi_t1[state_idx] * X_no[:, 1] + \
                   beta_distance5_t1[state_idx] * X_no[:, 2]
            
            # M1
            V_M1 = ASC_t1[state_idx, 0] + \
                    beta_railtime_t1[state_idx] * X_M1[:, 0] + \
                    beta_triptime_t1[state_idx] * X_M1[:, 1] + \
                    beta_firstpt_t1[state_idx] * X_M1[:, 2] + \
                    beta_normal_t1[state_idx] * X_M1[:, 3]
                    
            
            # M2
            V_M2 = ASC_t1[state_idx, 1] + \
                    beta_railtime_t1[state_idx] * X_M2[:, 0] + \
                    beta_triptime_t1[state_idx] * X_M2[:, 1] + \
                    beta_firstpt_t1[state_idx] * X_M2[:, 2] + \
                    beta_normal_t1[state_idx] * X_M2[:, 3]
                    
            
            # M3
            V_M3 = ASC_t1[state_idx, 2] + \
                    beta_railtime_t1[state_idx] * X_M3[:, 0] + \
                    beta_triptime_t1[state_idx] * X_M3[:, 1] + \
                    beta_firsttaxi_t1[state_idx] * X_M3[:, 2] + \
                    beta_normal_t1[state_idx] * X_M3[:, 3]
            
            # M4
            V_M4 = ASC_t1[state_idx, 3] + \
                    beta_triptime_t1[state_idx] * X_M4[:, 0] + \
                    beta_firsttaxi_t1[state_idx] * X_M4[:, 1] + \
                    beta_distance5_t1[state_idx] * X_M4[:, 2]
            
            V_all = pt.stack([V_no, V_M1, V_M2, V_M3, V_M4], axis=1)
            return pm.math.softmax(V_all, axis=1)
        
        # 计算各状态的发射概率
        emit_probs_t1_list = [compute_emit_probs_t1(k) for k in range(n_states)]
        emit_probs_t1 = pt.stack(emit_probs_t1_list, axis=0)  # (n_states, n_obs, n_alt)
        
        # ---------- 阶段2发射概率 (多项Logit) ----------
        ASC_t2 = pm.Normal('ASC_t2', mu=0, sigma=0.5, shape=(n_states, n_alt_t2-1))
        beta_taxi12_t2 = pm.Normal('beta_taxi12_t2', mu=0, sigma=0.5, shape=n_states)
        beta_priceratio_t2 = pm.Normal('beta_priceratio_t2', mu=0, sigma=0.5, shape=n_states)
        beta_price_t2 = pm.Normal('beta_price_t2', mu=0, sigma=0.5, shape=n_states)
        beta_weekbus_t2 = pm.Normal('beta_weekbus_t2', mu=0, sigma=0.5, shape=n_states)
        beta_ebike_t2 = pm.Normal('beta_ebike_t2', mu=0, sigma=0.5, shape=n_states)
        beta_occupy_t2 = pm.Normal('beta_occupy_t2', mu=0, sigma=0.5, shape=n_states)
        beta_sex_t2 = pm.Normal('beta_sex_t2', mu=0, sigma=0.5, shape=n_states)
        beta_income1_t2 = pm.Normal('beta_income1_t2', mu=0, sigma=0.5, shape=n_states)
        beta_age4_t2 = pm.Normal('beta_age4_t2', mu=0, sigma=0.5, shape=n_states)
        beta_traveldistancework_t2 = pm.Normal('beta_traveldistancework_t2', mu=0, sigma=0.5, shape=n_states)
        beta_weekmetro_t2 = pm.Normal('beta_weekmetro_t2', mu=0, sigma=0.5, shape=n_states)
        beta_c7_t2 = pm.Normal('beta_c7_t2', mu=0, sigma=0.5, shape=n_states)
        beta_traveldistanceweekend_t2 = pm.Normal('beta_traveldistanceweekend_t2', mu=0, sigma=0.5, shape=n_states)
        beta_weektaxi_t2 = pm.Normal('beta_weektaxi_t2', mu=0, sigma=0.5, shape=n_states)
        beta_age3_t2 = pm.Normal('beta_age3_t2', mu=0, sigma=0.5, shape=n_states)
        beta_income2_t2 = pm.Normal('beta_income2_t2', mu=0, sigma=0.5, shape=n_states)
        beta_c6_t2 = pm.Normal('beta_c6_t2', mu=0, sigma=0.5, shape=n_states)
        beta_cost_t2 = pm.Normal('beta_cost_t2', mu=0, sigma=0.5, shape=n_states)
        beta_license_t2 = pm.Normal('beta_license_t2', mu=0, sigma=0.5, shape=n_states)
        beta_havecar_t2 = pm.Normal('beta_havecar_t2', mu=0, sigma=0.5, shape=n_states)
        beta_education_t2 = pm.Normal('beta_education_t2', mu=0, sigma=0.5, shape=n_states)
        
        def compute_emit_probs_t2(state_idx):
            """计算状态state_idx下阶段2各选项的选择概率"""
            # Bus First
            V_B1 = ASC_t2[state_idx, 0] + \
                    beta_taxi12_t2[state_idx] * X_B1[:, 0] + \
                    beta_priceratio_t2[state_idx] * X_B1[:, 1] + \
                    beta_price_t2[state_idx] * X_B1[:, 2] + \
                    beta_weekbus_t2[state_idx] * X_B1[:, 3] + \
                    beta_ebike_t2[state_idx] * X_B1[:, 4] + \
                    beta_occupy_t2[state_idx] * X_B1[:, 5] + \
                    beta_sex_t2[state_idx] * X_B1[:, 6] + \
                    beta_income1_t2[state_idx] * X_B1[:, 7] + \
                    beta_age4_t2[state_idx] * X_B1[:, 8]
            # Metro Access
            V_B2 = ASC_t2[state_idx, 1] + \
                    beta_taxi12_t2[state_idx] * X_B2[:, 0] + \
                    beta_priceratio_t2[state_idx] * X_B2[:, 1] + \
                    beta_price_t2[state_idx] * X_B2[:, 2] + \
                    beta_traveldistancework_t2[state_idx] * X_B2[:, 3] + \
                    beta_weekmetro_t2[state_idx] * X_B2[:, 4] + \
                    beta_ebike_t2[state_idx] * X_B2[:, 5] + \
                    beta_occupy_t2[state_idx] * X_B2[:, 6] + \
                    beta_sex_t2[state_idx] * X_B2[:, 7] + \
                    beta_c7_t2[state_idx] * X_B2[:, 8] + \
                    beta_income1_t2[state_idx] * X_B2[:, 9] + \
                    beta_age4_t2[state_idx] * X_B2[:, 10]
            
            # Value Taxi
            V_B3 = ASC_t2[state_idx, 2] + \
                    beta_priceratio_t2[state_idx] * X_B3[:, 0] + \
                    beta_price_t2[state_idx] * X_B3[:, 1] + \
                    beta_traveldistanceweekend_t2[state_idx] * X_B3[:, 2] + \
                    beta_weektaxi_t2[state_idx] * X_B3[:, 3] + \
                    beta_age3_t2[state_idx] * X_B3[:, 4] + \
                    beta_income2_t2[state_idx] * X_B3[:, 5]
            
            # Ultra Access
            V_B4 = ASC_t2[state_idx, 3] + \
                    beta_priceratio_t2[state_idx] * X_B4[:, 0] + \
                    beta_price_t2[state_idx] * X_B4[:, 1] + \
                    beta_c6_t2[state_idx] * X_B4[:, 2] + \
                    beta_cost_t2[state_idx] * X_B4[:, 3] + \
                    beta_weektaxi_t2[state_idx] * X_B4[:, 4] + \
                    beta_age3_t2[state_idx] * X_B4[:, 5]
            
            # PAYG (参考选项)
            V_PAYG = beta_cost_t2[state_idx] * X_PAYG[:, 0] + \
                     beta_license_t2[state_idx] * X_PAYG[:, 1] + \
                     beta_havecar_t2[state_idx] * X_PAYG[:, 2] + \
                     beta_education_t2[state_idx] * X_PAYG[:, 3]
            
            V_all = pt.stack([V_B1, V_B2, V_B3, V_B4, V_PAYG], axis=1)
            return pm.math.softmax(V_all, axis=1)
        
        emit_probs_t2_list = [compute_emit_probs_t2(k) for k in range(n_states)]
        emit_probs_t2 = pt.stack(emit_probs_t2_list, axis=0)
        
        # ---------- 状态转移矩阵 ----------
        trans_logits_raw = pm.Normal('trans_logits_raw', mu=0, sigma=0.5,
                                      shape=(n_states, n_states-1))
        gamma_trans = pm.Normal('gamma_trans', mu=0, sigma=0.5, shape=n_trans_cov)
        trans_modifier = pt.dot(X_trans, gamma_trans)
        
        trans_logits_base = pt.concatenate([
            pt.zeros((n_states, 1)),
            trans_logits_raw
        ], axis=1)
        
        trans_logits = trans_logits_base[None, :, :] + pt.zeros((n_obs, n_states, n_states))
        trans_logits = pt.set_subtensor(
            trans_logits[:, :, -1],
            trans_logits[:, :, -1] + trans_modifier[:, None]
        )
        trans_probs = pm.math.softmax(trans_logits, axis=2)
        
        # ---------- 边际似然 ----------
        # 对于多类别观测，需要索引正确的概率
        
        # P(y1=j | z1=k) for observed j
        emit_1 = pt.zeros((n_obs, n_states))
        for k in range(n_states):
            emit_1 = pt.set_subtensor(
                emit_1[:, k],
                emit_probs_t1[k, pt.arange(n_obs), y1]
            )
        
        # P(y2=j | z2=k) for observed j
        emit_2 = pt.zeros((n_obs, n_states))
        for k in range(n_states):
            emit_2 = pt.set_subtensor(
                emit_2[:, k],
                emit_probs_t2[k, pt.arange(n_obs), y2]
            )
        
        # Forward algorithm
        alpha_1 = pi_init * emit_1
        alpha_2 = pt.sum(alpha_1[:, :, None] * trans_probs * emit_2[:, None, :], axis=1)
        
        marginal = pt.sum(alpha_2, axis=1)
        # ===== 修正部分 =====
        # 逐样本对数似然（用于WAIC/LOO）
        log_lik_per_obs = pt.log(marginal + 1e-10)  # shape: (n_obs,)
        pm.Deterministic('log_lik', log_lik_per_obs)  # 保存逐样本似然

        # 总对数似然（用于采样）
        total_log_lik = pt.sum(log_lik_per_obs)
        pm.Potential('log_likelihood', total_log_lik)
        
        pm.Deterministic('pi_init_mean', pt.mean(pi_init, axis=0))
        
    return hmm_model


# =============================================================================
# 4. 模型估计和分析函数
# =============================================================================

def fit_model(model, draws=1000, tune=500, chains=2, target_accept=0.95):
    """MCMC采样"""
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=42,
            progressbar=True,
            init='adapt_diag',
        )
    return trace


def analyze_results(trace, data, model_type='binary'):
    """分析后验结果"""
    
    print("\n" + "="*70)
    print("MaaS多阶段采纳决策HMM模型 - 后验分析结果")
    print("="*70)
    
    states = ['Skeptic', 'Neutral', 'Enthusiast']
    
    if model_type == 'binary':
        # 发射概率
        emit_t1 = trace.posterior['emission_probs_t1'].mean(dim=['chain', 'draw']).values
        emit_t2 = trace.posterior['emission_probs_t2'].mean(dim=['chain', 'draw']).values
        
        print("\n【发射概率】P(选择MaaS/订阅套餐 | 隐状态)")
        print("-"*60)
        print(f"{'隐状态':<15} {'阶段1:选择MaaS':<20} {'阶段2:订阅套餐':<20}")
        for i, state in enumerate(states):
            print(f"{state:<15} {emit_t1[i]:.3f}{'':<17} {emit_t2[i]:.3f}")
    
    # 初始状态分布
    pi_mean = trace.posterior['pi_init_mean'].mean(dim=['chain', 'draw']).values
    print(f"\n【平均初始状态分布】")
    print("-"*60)
    for i, state in enumerate(states):
        print(f"  {state}: {pi_mean[i]:.3f} ({pi_mean[i]*100:.1f}%)")
    
    # 初始状态协变量影响
    print(f"\n【初始状态的协变量影响】(相对于Skeptic)")
    print("-"*60)
    alpha = trace.posterior['alpha_init'].mean(dim=['chain', 'draw']).values
    beta = trace.posterior['beta_init'].mean(dim=['chain', 'draw']).values
    
    print(f"截距: Neutral={alpha[0]:+.3f}, Enthusiast={alpha[1]:+.3f}")
    
    if model_type == 'binary':
        cov_names = ['sex', 'age2', 'age3', 'age4', 'income1', 'income2', 
                     'education', 'occupy', 'week_metro', 'week_bus', 
                     'week_taxi', 'week_ebike', 'have_car', 'e_bike',
                     'travel_dist_work', 'd6', 'f6']
    else:
        cov_names = ['sex', 'age2', 'age3', 'income1', 'income2', 
                     'education', 'week_metro', 'week_bus', 'have_car', 
                     'f6', 'd6']
    
    print(f"\n显著协变量 (|coef| > 0.2):")
    for j, cov in enumerate(cov_names):
        if j < beta.shape[0]:
            if abs(beta[j, 0]) > 0.2 or abs(beta[j, 1]) > 0.2:
                print(f"  {cov:<18} → Neutral: {beta[j,0]:+.3f}, Enthusiast: {beta[j,1]:+.3f}")
    
    # 状态转移影响
    gamma = trace.posterior['gamma_trans'].mean(dim=['chain', 'draw']).values
    print(f"\n【状态转移的协变量影响】")
    print("-"*60)
    trans_cov_names = ['bundle_match', 'price_value_ratio', 'trial_satisfaction', 
                       'travel_num', 'cost'] if model_type == 'binary' else \
                      ['bundle_match', 'price_value_ratio', 'trial_satisfaction', 'travel_num']
    for j, cov in enumerate(trans_cov_names):
        if j < len(gamma):
            print(f"  {cov:<25}: {gamma[j]:+.3f}")
    
    # 诊断
    print(f"\n【MCMC诊断】")
    print("-"*60)
    var_names = ['alpha_init', 'gamma_trans']
    if model_type == 'binary':
        var_names += ['emit_base_t1', 'emit_base_t2']
    summary = az.summary(trace, var_names=var_names)
    print(summary[['mean', 'sd', 'r_hat', 'ess_bulk']].to_string())
    
    return trace


def compute_posterior_states(trace, data, model_type='binary'):
    """计算隐状态后验分布"""
    
    print("\n" + "="*70)
    print("隐状态后验推断")
    print("="*70)
    
    n_obs = len(data)
    n_states = 3
    states = ['Skeptic', 'Neutral', 'Enthusiast']
    
    # 获取参数后验均值
    pi_mean = trace.posterior['pi_init_mean'].mean(dim=['chain', 'draw']).values
    trans_raw = trace.posterior['trans_logits_raw'].mean(dim=['chain', 'draw']).values
    trans_logits = np.concatenate([np.zeros((n_states, 1)), trans_raw], axis=1)
    trans_probs = softmax(trans_logits, axis=1)
    
    if model_type == 'binary':
        emit_t1 = trace.posterior['emission_probs_t1'].mean(dim=['chain', 'draw']).values
        emit_t2 = trace.posterior['emission_probs_t2'].mean(dim=['chain', 'draw']).values
        y1 = data['y1_binary'].values
        y2 = data['y2_binary'].values
        
        z1_posterior = np.zeros((n_obs, n_states))
        z2_posterior = np.zeros((n_obs, n_states))
        
        for i in range(n_obs):
            emit_1_i = emit_t1 if y1[i] == 1 else (1 - emit_t1)
            emit_2_i = emit_t2 if y2[i] == 1 else (1 - emit_t2)
            
            joint = np.zeros((n_states, n_states))
            for k in range(n_states):
                for j in range(n_states):
                    joint[k, j] = pi_mean[k] * emit_1_i[k] * trans_probs[k, j] * emit_2_i[j]
            
            z1_posterior[i] = joint.sum(axis=1) / (joint.sum() + 1e-10)
            z2_posterior[i] = joint.sum(axis=0) / (joint.sum() + 1e-10)
    
        data['z1_pred'] = np.argmax(z1_posterior, axis=1)
        data['z2_pred'] = np.argmax(z2_posterior, axis=1)
        
        for i, state in enumerate(states):
            data[f'z1_prob_{state.lower()}'] = z1_posterior[:, i]
            data[f'z2_prob_{state.lower()}'] = z2_posterior[:, i]
    
    # 统计
    print("\n【阶段1隐状态分布】")
    z1_dist = pd.Series(data['z1_pred']).value_counts(normalize=True).sort_index()
    for i in range(n_states):
        if i in z1_dist.index:
            print(f"  {states[i]}: {z1_dist[i]:.1%}")
    
    print("\n【阶段2隐状态分布】")
    z2_dist = pd.Series(data['z2_pred']).value_counts(normalize=True).sort_index()
    for i in range(n_states):
        if i in z2_dist.index:
            print(f"  {states[i]}: {z2_dist[i]:.1%}")
    
    print("\n【状态转移统计】")
    trans_empirical = pd.crosstab(data['z1_pred'], data['z2_pred'], normalize='index')
    trans_empirical.index = [states[i] for i in trans_empirical.index]
    trans_empirical.columns = [states[i] for i in trans_empirical.columns]
    print(trans_empirical.round(3).to_string())
    
    return data


# =============================================================================
# 5. 主程序
# =============================================================================

def create_simulated_data(n=500):
    """创建模拟数据用于测试"""
    np.random.seed(42)
    
    data = {
        'peopleID': np.arange(n),
        'maas': np.random.choice([1, 4, 5, 6, 7], n, p=[0.6, 0.1, 0.1, 0.1, 0.1]),
        'results': np.random.choice([1, 2, 3, 4, 5], n, p=[0.15, 0.15, 0.15, 0.15, 0.4]),
        
        # 个人特征
        'sex': np.random.binomial(1, 0.44, n),
        'age1': np.zeros(n), 'age2': np.zeros(n), 'age3': np.zeros(n), 'age4': np.zeros(n),
        'income1': np.zeros(n), 'income2': np.zeros(n), 'income3': np.zeros(n),
        'education': np.random.binomial(1, 0.7, n),
        'occupy': np.random.binomial(1, 0.3, n),
        'have_car': np.random.binomial(1, 0.48, n),
        'license': np.random.binomial(1, 0.7, n),
        'e_bike': np.random.binomial(1, 0.65, n),
        
        # 出行频率
        'week_bus': np.random.binomial(1, 0.35, n),
        'week_metro': np.random.binomial(1, 0.45, n),
        'week_taxi': np.random.binomial(1, 0.2, n),
        'week_ebike': np.random.binomial(1, 0.25, n),
        'week_bike': np.random.binomial(1, 0.3, n),
        
        # 出行特征
        'travel_num': np.random.binomial(1, 0.5, n),
        'travel_distance_work': np.random.binomial(1, 0.45, n),
        'travel_distance_weekend': np.random.binomial(1, 0.4, n),
        'travel_aim': np.random.binomial(1, 0.6, n),
        'cost': np.random.binomial(1, 0.4, n),
        
        # 偏好
        'a6': np.random.binomial(1, 0.5, n),
        'b6': np.random.binomial(1, 0.25, n),
        'c6': np.random.binomial(1, 0.4, n),
        'd6': np.random.binomial(1, 0.15, n),
        'e6': np.random.binomial(1, 0.35, n),
        'f6': np.random.binomial(1, 0.3, n),
        'g6': np.random.binomial(1, 0.45, n),
        'c7': np.random.binomial(1, 0.3, n),
        
        # 第一选择
        'first_car': np.zeros(n),
        'first_taxi': np.zeros(n),
        'first_pt': np.zeros(n),
        
        # 出发时间
        'morning': np.zeros(n), 'evening': np.zeros(n), 
        'normal': np.zeros(n), 'late': np.zeros(n),
        
        # 距离
        'distance1': np.zeros(n), 'distance2': np.zeros(n),
        'distance3': np.zeros(n), 'distance4': np.zeros(n), 'distance5': np.zeros(n),
        
        # LOS变量 - 阶段1
        'M1ttimerail': np.random.uniform(10, 40, n),
        'M1triptime': np.random.uniform(30, 80, n),
        'M1price': np.random.uniform(5, 20, n),
        'M2ttime_rail': np.random.uniform(10, 40, n),
        'M2triptime': np.random.uniform(25, 70, n),
        'M2price': np.random.uniform(5, 25, n),
        'M3ttime_rail': np.random.uniform(10, 35, n),
        'M3triptime': np.random.uniform(30, 60, n),
        'M3price': np.random.uniform(20, 60, n),
        'M4ttime': np.random.uniform(20, 50, n),
        'M4price': np.random.uniform(30, 80, n),
        
        # 传统方式
        'C1triptimePT': np.random.uniform(40, 90, n),
        'C1pricePT': np.random.uniform(3, 10, n),
        'B1triptime': np.random.uniform(25, 50, n),
        'B1priceTaxi': np.random.uniform(30, 80, n),
        'A1ttimeCar': np.random.uniform(30, 70, n),
        'A1priceCar': np.random.uniform(20, 50, n),
        
        # 套餐属性 - 阶段2
        'price1': np.random.uniform(0.5, 1.5, n),
        'price2': np.random.uniform(1.5, 3, n),
        'price3': np.random.uniform(5, 10, n),
        'price4': np.random.uniform(10, 15, n),
        'price_12': np.random.uniform(0.8, 1.2, n),
        'price_3': np.random.uniform(0.8, 1.2, n),
        'price_4': np.random.uniform(0.8, 1.2, n),
        'taxi_12': np.random.uniform(0, 5, n),
        'taxi_3': np.random.uniform(3, 8, n),
        'taxi_4': np.random.uniform(5, 10, n),
    }
    
    # 生成分组变量
    age_group = np.random.choice([1, 2, 3, 4], n, p=[0.25, 0.4, 0.25, 0.1])
    data['age1'] = (age_group == 1).astype(int)
    data['age2'] = (age_group == 2).astype(int)
    data['age3'] = (age_group == 3).astype(int)
    data['age4'] = (age_group == 4).astype(int)
    
    income_group = np.random.choice([1, 2, 3], n, p=[0.45, 0.45, 0.1])
    data['income1'] = (income_group == 1).astype(int)
    data['income2'] = (income_group == 2).astype(int)
    data['income3'] = (income_group == 3).astype(int)
    
    first_choice = np.random.choice([1, 2, 3], n, p=[0.3, 0.2, 0.5])
    data['first_car'] = (first_choice == 1).astype(int)
    data['first_taxi'] = (first_choice == 2).astype(int)
    data['first_pt'] = (first_choice == 3).astype(int)
    
    depart = np.random.choice([1, 2, 3, 4], n, p=[0.3, 0.3, 0.3, 0.1])
    data['morning'] = (depart == 1).astype(int)
    data['evening'] = (depart == 2).astype(int)
    data['normal'] = (depart == 3).astype(int)
    data['late'] = (depart == 4).astype(int)
    
    dist = np.random.choice([1, 2, 3, 4, 5], n, p=[0.2, 0.25, 0.25, 0.2, 0.1])
    data['distance1'] = (dist == 1).astype(int)
    data['distance2'] = (dist == 2).astype(int)
    data['distance3'] = (dist == 3).astype(int)
    data['distance4'] = (dist == 4).astype(int)
    data['distance5'] = (dist == 5).astype(int)
    
    return pd.DataFrame(data)


def compute_aic_bic_from_trace(trace, n_obs):
    """
    从trace计算AIC和BIC
    
    Parameters
    ----------
    trace : arviz.InferenceData
    n_obs : int
        样本量
    """
    
    # 获取逐样本对数似然的后验均值
    log_lik_samples = trace.posterior['log_lik'].values  # shape: (chains, draws, n_obs)
    
    # 方法1: 使用后验均值参数计算的似然（点估计）
    log_lik_mean = log_lik_samples.mean(axis=(0, 1))  # 每个样本的平均对数似然
    total_log_lik = log_lik_mean.sum()
    
    # 方法2: 使用后验均值的总似然
    # total_log_lik = log_lik_samples.sum(axis=2).mean()
    
    # 计算参数数量
    # 需要统计模型中的自由参数
    k = count_parameters(trace)
    
    # AIC 和 BIC
    aic = -2 * total_log_lik + 2 * k
    bic = -2 * total_log_lik + k * np.log(n_obs)
    
    return {
        'log_likelihood': total_log_lik,
        'k': k,
        'n': n_obs,
        'AIC': aic,
        'BIC': bic
    }


def count_parameters(trace):
    """统计模型参数数量"""
    k = 0
    
    # 遍历后验中的参数
    for var_name in trace.posterior.data_vars:
        if var_name.startswith('log_lik') or var_name.endswith('_mean'):
            continue  # 跳过派生变量
        
        var_shape = trace.posterior[var_name].shape[2:]  # 去掉chain和draw维度
        k += np.prod(var_shape)
    
    return int(k)


if __name__ == "__main__":
    
    print("="*70)
    print("MaaS多阶段采纳决策HMM模型")
    print("="*70)
    
    # 1. 创建/加载数据
    print("\n[Step 1] 加载数据...")
    
    # 使用模拟数据测试 (实际使用时替换为真实数据)
    data = load_data('data/最终模型数据.csv')
    # data = data.sample(n=500, random_state=42).reset_index(drop=True)
    # data = create_simulated_data(n=500)
    
    # 2. 预处理
    print("\n[Step 2] 数据预处理...")
    data = preprocess_data(data)
    
    print(f"  样本量: {len(data)}")
    
    # 3. 构建模型
    print("\n[Step 3] 构建HMM模型 (多元观测版本)...")
    model = build_hmm_multinomial(data, n_states=3)
    
    # 4. 先验检查
    print("\n[Step 4] 先验预测检查...")
    with model:
        prior = pm.sample_prior_predictive(samples=100, random_seed=42)
    
    # 5. MCMC采样
    print("\n[Step 5] MCMC采样...")
    trace = fit_model(model, draws=2000, tune=2000, chains=4, target_accept=0.99)
    
    # 6. 结果分析
    analyze_results(trace, data, model_type='multi')
    
    # # 7. 隐状态推断
    # data = compute_posterior_states(trace, data, model_type='multi')
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)

    # 8. 保存结果
    print("\n[Step 8] 保存模型结果...")
    import os
    output_dir = './maas_hmm_results'
    os.makedirs(output_dir, exist_ok=True)

    name = 'less_parameters'
    # 保存trace
    az.to_netcdf(trace, f'{output_dir}/trace_{name}.nc')

    # 保存摘要
    summary = az.summary(trace)
    summary.to_csv(f'{output_dir}/summary_{name}.csv')

    # 保存数据
    data.to_csv(f'{output_dir}/data_with_states_{name}.csv', index=False)

    print(f"结果已保存到 {output_dir}/")

    # 计算AIC/BIC
    results = compute_aic_bic_from_trace(trace, n_obs=len(data))

    print(f"Log-likelihood: {results['log_likelihood']:.2f}")
    print(f"Parameters (k): {results['k']}")
    print(f"AIC: {results['AIC']:.2f}")
    print(f"BIC: {results['BIC']:.2f}")