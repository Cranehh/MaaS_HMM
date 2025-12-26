## 去掉了结构方程里的主要方式
import pandas as pd
df = pd.read_csv(f'../data/MaaS被调查者面对情景.csv',encoding = 'gb2312')
ls = []
for i in range(len(df)):
    if df.iloc[i,1] < 3:
        ques1 = df.iloc[i,2]
        ques2 = df.iloc[i,8]
        ques3 = df.iloc[i,14]
        ques4 = df.iloc[i,20]
        ques5 = df.iloc[i,26]
        anws1 = df.iloc[i,3]
        anws2 = df.iloc[i,9]
        anws3 = df.iloc[i,15]
        anws4 = df.iloc[i,21]
        anws5 = df.iloc[i,27]
        ls.append([df.iloc[i,0],1,ques1,anws1])
        ls.append([df.iloc[i,0],1,8+ques2,anws2])
        ls.append([df.iloc[i,0],1,16+ques3,anws3])
        ls.append([df.iloc[i,0],1,24+ques4,anws4])
        ls.append([df.iloc[i,0],1,32+ques5,anws5])
    elif df.iloc[i,1] < 5:
        ques1 = df.iloc[i,4]
        ques2 = df.iloc[i,10]
        ques3 = df.iloc[i,16]
        ques4 = df.iloc[i,22]
        ques5 = df.iloc[i,28]
        anws1 = df.iloc[i,5]
        anws2 = df.iloc[i,11]
        anws3 = df.iloc[i,17]
        anws4 = df.iloc[i,23]
        anws5 = df.iloc[i,29]
        ls.append([df.iloc[i,0],2,40+ques1,anws1])
        ls.append([df.iloc[i,0],2,48+ques2,anws2])
        ls.append([df.iloc[i,0],2,56+ques3,anws3])
        ls.append([df.iloc[i,0],2,64+ques4,anws4])
        ls.append([df.iloc[i,0],2,72+ques5,anws5])
    else:
        ques1 = df.iloc[i,6]
        ques2 = df.iloc[i,12]
        ques3 = df.iloc[i,18]
        ques4 = df.iloc[i,24]
        ques5 = df.iloc[i,30]
        anws1 = df.iloc[i,7]
        anws2 = df.iloc[i,13]
        anws3 = df.iloc[i,19]
        anws4 = df.iloc[i,25]
        anws5 = df.iloc[i,31]
        ls.append([df.iloc[i,0],3,80+ques1,anws1])
        ls.append([df.iloc[i,0],3,88+ques2,anws2])
        ls.append([df.iloc[i,0],3,96+ques3,anws3])
        ls.append([df.iloc[i,0],3,104+ques4,anws4])
        ls.append([df.iloc[i,0],3,112+ques5,anws5])
data = pd.DataFrame(ls,columns = ['index','week_taxi','bundle_index','results'])
question_bundle_data = pd.read_csv(f'../data/question_bundle_data.csv',encoding = 'gb2312')
question_bundle_data = question_bundle_data.rename(columns = {'index':'bundle_index','class':'week_taxi'})
data = pd.merge(data,question_bundle_data,on = ['week_taxi','bundle_index'])
data = data.sort_values('index')
people_data = pd.read_csv(f'../data/MaaS被调查者聚类加个人属性加态度.csv',encoding = 'gb2312')
##年龄哑变量化
people_data['age'] = people_data['age'].replace(to_replace=[1,2,3,4,5,6,7],value=[1,1,2,2,3,3,4])
age = pd.get_dummies(people_data['age'])
age.columns = ['age1','age2','age3','age4']
people_data = pd.concat([people_data,age],axis = 1)
##收入哑变量化
people_data['income_individual'] = people_data['income_individual'].replace(to_replace=[1,2,3,4,5,6,7,8],value=[1,1,1,2,2,2,2,3])
income = pd.get_dummies(people_data['income_individual'])
income.columns = ['income1','income2','income3']
people_data = pd.concat([people_data,income],axis = 1)
data = data.drop(columns=['week_taxi','bundle_index'])
data = data.rename(columns={'index':'peopleID'})
model_data = pd.merge(data,people_data,on = 'peopleID')
#将1、2、3转换为0、1、2
model_data['sex'] = model_data['sex']-1
model_data['travel_aim'] = model_data['travel_aim']-1
model_data['home_people'] = model_data['home_people']-1
model_data['license'] = 2-model_data['license']
model_data['drive'] = 2-model_data['drive']
model_data['e-bike'] = 2-model_data['e-bike']
#负数补0
model_data.loc[model_data['car_home']<0,'car_home'] = 0
model_data.loc[model_data['drive']<0,'drive'] = 0
##变量哑元化
#每月花费，150以下为0
model_data['cost'] = model_data['cost'].replace(to_replace=[1,2,3,4,5,6],value=[0,0,1,1,1,1])
#年龄，34以下为0
model_data['age'] = model_data['age'].replace(to_replace=[1,2,3,4,5,6,7],value=[0,0,0,1,1,1,1])
#教育，本科以上为1
model_data['education'] = model_data['education'].replace(to_replace=[1,2,3,4,5,6],value=[1,1,1,0,0,0])
#个人收入，1W以下为0
model_data['income_individual'] = model_data['income_individual'].replace(to_replace=[1,2,3,4,5,6,7,8],value=[0,0,0,0,0,1,1,1])
#家庭年收入，30W以下为0
model_data['income_home'] = model_data['income_home'].replace(to_replace=[1,2,3,4,5,6,7],value=[0,0,0,0,1,1,1])
#工作，出行少的工作为0
model_data['occupy'] = model_data['occupy'].replace(to_replace=[1,2,3,4,5,6,7,8,9,10,11],value=[0,0,0,0,0,0,0,1,1,1,1])
##变量缩放
#共享电动车除10
model_data['ebike_12'] = model_data['ebike_12'] / 10
model_data['ebike_3'] = model_data['ebike_3'] / 10
model_data['ebike_4'] = model_data['ebike_4'] / 10
#出租车除100
model_data['taxi_12'] = model_data['taxi_12'] / 100
model_data['taxi_3'] = model_data['taxi_3'] / 100
model_data['taxi_4'] = model_data['taxi_4'] / 100
#价格除100
model_data['price1'] = model_data['price1'] / 100
model_data['price2'] = model_data['price2'] / 100
model_data['price3'] = model_data['price3'] / 100
model_data['price4'] = model_data['price4'] / 100
##列改名
model_data = model_data.rename(columns={'6a':'a6','6b':'b6','6c':'c6','6d':'d6','6e':'e6','6f':'f6','6g':'g6'})
model_data = model_data.rename(columns={'7a':'a7','7b':'b7','7c':'c7','7d':'d7'})
model_data = model_data.rename(columns={'e-bike':'e_bike'})
import sys
import pandas as pd
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
database = db.Database('HCM', model_data)
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

##方式变量
ASC_1 = Beta('ASC_1',0,-1000,1000,0)
ASC_2 = Beta('ASC_2',0,-1000,1000,0)
ASC_3 = Beta('ASC_3',0,-1000,1000,0)
ASC_4 = Beta('ASC_4',0,-1000,1000,0)
ASC_5 = Beta('ASC_5',0,-1000,1000,1)


##套餐次数与价格系数（对MaaS采用的直接影响）
B_BUN_EBIKE = Beta('B_BUN_EBIKE',0,-1000,1000,0)
B_BUN_TAXI = Beta('B_BUN_TAXI',0,-1000,1000,0)
B_BUN_PRICERATIO = Beta('B_BUN_PRICERATIO',0,-1000,1000,0)
B_BUN_PRICE = Beta('B_BUN_PRICE',0,-1000,1000,0)


##日常出行模式系数(分析什么样的人群可能会采用MaaS)
B_TRAVEL_NUM = Beta('B_TRAVEL_NUM',0,-1000,1000,0)
B_TRAVEL_DISTANCE_WORK = Beta('B_TRAVEL_DISTANCE_WORK',0,-1000,1000,0)
B_TRAVEL_DISTANCE_END = Beta('B_TRAVEL_DISTANCE_END',0,-1000,1000,0)
B_TRAVEL_AIM = Beta('B_TRAVEL_AIM',0,-1000,1000,0)
B_PT = Beta('B_PT',0,-1000,1000,0)
B_TAXI = Beta('B_TAXI',0,-1000,1000,0)
B_CAR = Beta('B_CAR',0,-1000,1000,0)
B_SHARECAR = Beta('B_SHARECAR',0,-1000,1000,0)
B_SHAREBIKE = Beta('B_SHAREBIKE',0,-1000,1000,0)
B_BIKE = Beta('B_BIKE',0,-1000,1000,0)
B_WALK = Beta('B_WALK',0,-1000,1000,0)
B_COST = Beta('B_COST',0,-1000,1000,0)
B_BUS_NUM = Beta('B_BUS_NUM',0,-1000,1000,0)
B_METRO_NUM = Beta('B_METRO_NUM',0,-1000,1000,0)
B_BIKE_NUM = Beta('B_BIKE_NUM',0,-1000,1000,0)
B_EBIKE_NUM = Beta('B_EBIKE_NUM',0,-1000,1000,0)
B_TAXI_NUM = Beta('B_TAXI_NUM',0,-1000,1000,0)
B_COMBINE_SHAREBIKE = Beta('B_COMBINE_SHAREBIKE',0,-1000,1000,0)

##社会经济属性系数（分析人群对选择套餐的偏好，有助于针对性的调整套餐）
B_SEX = Beta('B_SEX',0,-1000,1000,0)
B_AGE = Beta('B_AGE',0,-1000,1000,0)
B_EDUCATION = Beta('B_EDUCATION',0,-1000,1000,0)
B_INCOME_INDIVIDUAL = Beta('B_INCOME_INDIVIDUAL',0,-1000,1000,0)
B_CAR_HOME = Beta('B_CAR_HOME',0,-1000,1000,0)
B_EBIKE_HOME = Beta('B_EBIKE_HOME',0,-1000,1000,0)
B_OCCUPY = Beta('B_OCCUPY',0,-1000,1000,0)
B_HAVECAR = Beta('B_HAVECAR',0,-1000,1000,0)
B_LICENSE = Beta('B_LICENSE',0,-1000,1000,0)
B_AGE1 = Beta('B_AGE1',0,-1000,1000,0)
B_AGE2 = Beta('B_AGE2',0,-1000,1000,0)
B_AGE3 = Beta('B_AGE3',0,-1000,1000,0)
B_AGE4 = Beta('B_AGE4',0,-1000,1000,0)
B_INCOME1 = Beta('B_INCOME1',0,-1000,1000,0)
B_INCOME2 = Beta('B_INCOME2',0,-1000,1000,0)
B_INCOME3 = Beta('B_INCOME3',0,-1000,1000,0)

##人群分类
B_TRAVELCLASS = Beta('B_TRAVELCLASS',0,-1000,1000,0)
B_MAASCLASS = Beta('B_MAASCLASS',0,-1000,1000,0)

## 个人属性分类
B_PEOPLECLASS1 = Beta('B_PEOPLECLASS1',0,-1000,1000,0)
B_PEOPLECLASS2 = Beta('B_PEOPLECLASS2',0,-1000,1000,0)
B_PEOPLECLASS3 = Beta('B_PEOPLECLASS3',0,-1000,1000,0)

## 态度变量
B_FACTOR1 = Beta('B_FACTOR1',0,-1000,1000,0)
B_FACTOR2 = Beta('B_FACTOR2',0,-1000,1000,0)
B_FACTOR3 = Beta('B_FACTOR3',0,-1000,1000,0)
B_FACTOR4 = Beta('B_FACTOR4',0,-1000,1000,0)
B_FACTOR5 = Beta('B_FACTOR5',0,-1000,1000,0)
B_FACTOR6 = Beta('B_FACTOR6',0,-1000,1000,0)


##面板误差效应
Epo_1 = Beta('Epo_1', 0, None, None, 1)
Epo_1S = Beta('Epo_1S', 1, None, None, 0)
Epo_1_RND = Epo_1 + Epo_1S * bioDraws('Epo_1_RND', 'NORMAL')

Epo_2 = Beta('Epo_2', 0, None, None, 1)
Epo_2S = Beta('Epo_2S', 1, None, None, 0)
Epo_2_RND = Epo_2 + Epo_2S * bioDraws('Epo_2_RND', 'NORMAL')

Epo_3 = Beta('Epo_3', 0, None, None, 1)
Epo_3S = Beta('Epo_3S', 1, None, None, 0)
Epo_3_RND = Epo_3 + Epo_3S * bioDraws('Epo_3_RND', 'NORMAL')

Epo_4 = Beta('Epo_4', 0, None, None, 1)
Epo_4S = Beta('Epo_4S', 1, None, None, 0)
Epo_4_RND = Epo_4 + Epo_4S * bioDraws('Epo_4_RND', 'NORMAL')

Epo_5 = Beta('Epo_5', 0, None, None, 1)
Epo_5S = Beta('Epo_5S', 1, None, None, 0)
Epo_5_RND = Epo_5 + Epo_5S * bioDraws('Epo_5_RND', 'NORMAL')
# Utility functions

#If the person has a GA (season ticket) her incremental cost is actually 0
#rather than the cost value gathered from the
# network data.

# SM_COST =  SM_CO   * (  GA   ==  0  )
# TRAIN_COST =  TRAIN_CO   * (  GA   ==  0  )

# For numerical reasons, it is good practice to scale the data to
# that the values of the parameters are around 1.0.
# A previous estimation with the unscaled data has generated
# parameters around -0.01 for both cost and time. Therefore, time and
# cost are multipled my 0.01.

# The following statements are designed to preprocess the data. It is
# like creating a new columns in the data file. This should be
# preferred to the statement like
# TRAIN_TT_SCALED = TRAIN_TT / 100.0
# which will cause the division to be reevaluated again and again,
# throuh the iterations. For models taking a long time to estimate, it
# may make a significant difference.

##缩放
# HIGH_COST_SCALED = DefineVariable('HIGH_COST_SCALED', HWCost / 100.0, database)
# TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED', TCost / 100.0, database)
# PLANE_COST_SCALED = DefineVariable('Plane_COST_SCALED', PCost/ 100.0, database)
# CAR_COST_SCALED = DefineVariable('CAR_COST_SCALED', CCost / 100.0, database)

V1 = (ASC_1\
# + B_BUN_EBIKE * ebike_12\
+ B_BUN_TAXI * taxi_12\
+ B_BUN_PRICERATIO * price_12\
+ B_BUN_PRICE * price1\
# + B_PT * a6\
# +B_TRAVEL_DISTANCE_WORK * travel_distance_work\
# +B_TRAVEL_AIM * travel_aim\
# +B_SHAREBIKE * e6\
+B_BUS_NUM * week_bus\
+B_EBIKE_HOME * e_bike\
+B_OCCUPY * occupy\
#       +B_AGE * age\
      +B_SEX * sex\
#       +B_COMBINE_SHAREBIKE * c7\
# +B_PEOPLECLASS2 * people_class_prob2\
#       +B_MAASCLASS * attitude_class_prob\
      +B_INCOME1 * income1\
      +B_AGE4 * age4\
#       +B_EDUCATION * education\
#       +B_AGE2 * age2\
# +Epo_1_RND

+B_FACTOR2 * FACTOR2\
+B_FACTOR4 * FACTOR4\
+B_FACTOR3 * FACTOR3\
+B_FACTOR6 * FACTOR6)

V2 = (ASC_2\
# + B_BUN_EBIKE * ebike_12\
+ B_BUN_TAXI * taxi_12\
+ B_BUN_PRICERATIO * price_12\
+ B_BUN_PRICE * price2\
# + B_PT * a6\
+B_TRAVEL_DISTANCE_WORK * travel_distance_work\
# +B_TRAVEL_AIM * travel_aim\
# +B_SHAREBIKE * e6\
+B_METRO_NUM * week_metro\
+B_EBIKE_HOME * e_bike\
+B_OCCUPY * occupy\
#       +B_AGE * age\
      +B_SEX * sex\
      +B_COMBINE_SHAREBIKE * c7\
# +B_TRAVELCLASS * travel_mode_class_prob\
#       +B_MAASCLASS * attitude_class_prob\
      +B_INCOME1 * income1\
      +B_AGE4 * age4\
#       +B_EDUCATION * education\
#       +B_AGE2 * age2\
# +Epo_2_RND

+B_FACTOR2 * FACTOR2\
+B_FACTOR4 * FACTOR4\
+B_FACTOR3 * FACTOR3\
+B_FACTOR6 * FACTOR6)

V3 = (ASC_3\
# + B_BUN_EBIKE * ebike_3\
# + B_BUN_TAXI * taxi_3\
+ B_BUN_PRICERATIO * price_3\
+ B_BUN_PRICE * price3\
# + B_TAXI * b6\
# +B_COST * cost\
+B_TRAVEL_DISTANCE_END * travel_distance_weekend\
+B_TAXI_NUM * week_taxi\
# +B_CAR_HOME * car_home\
# +B_INCOME_INDIVIDUAL * income_individual\
# +B_EDUCATION * education\
# +B_AGE * age\
# +B_TRAVELCLASS * travel_mode_class_prob\
#       +B_MAASCLASS * attitude_class_prob\
      +B_AGE3 * age3\
      +B_INCOME2 * income2\
#       +B_INCOME3 * income3\
#       +B_AGE2 * age2\
#       +B_TRAVEL_AIM * travel_aim\
# +Epo_3_RND

+B_FACTOR2 * FACTOR2\
+B_FACTOR3 * FACTOR3\
+B_FACTOR6 * FACTOR6)

V4 = (ASC_4\
# + B_BUN_EBIKE * ebike_4\
# + B_BUN_TAXI * taxi_4\
+ B_BUN_PRICERATIO * price_4\
+ B_BUN_PRICE * price4\
# + B_TAXI * b6\
+ B_CAR * c6\
+B_COST * cost\
# +B_TRAVEL_DISTANCE_END * travel_distance_weekend\
+B_TAXI_NUM * week_taxi\
# +B_CAR_HOME * car_home\
# +B_INCOME_INDIVIDUAL * income_individual\
# +B_EDUCATION * education\
# +B_AGE * age\
# +B_SEX * sex\
# +B_TRAVELCLASS * travel_mode_class_prob\
#       +B_MAASCLASS * attitude_class_prob\
      +B_AGE3 * age3\
#       +B_INCOME3 * income3\
#       +B_TRAVEL_AIM * travel_aim\
# +Epo_4_RND

+B_FACTOR2 * FACTOR2\
+B_FACTOR3 * FACTOR3\
+B_FACTOR6 * FACTOR6\
      # +B_FACTOR5 * FACTOR5
      )

V5 = (ASC_5\
+B_COST * cost\
# +B_CAR_HOME * car_home\
# +B_AGE * age\
      +B_LICENSE * license\
      +B_HAVECAR * have_car\
      +B_EDUCATION * education\
# +B_PEOPLECLASS1 * people_class_prob1\
#     +B_AGE4 * age4\
#       +B_INCOME3 * income3\
#       +B_INCOME1 * income1\
#       +B_INCOME_INDIVIDUAL * income_individual\
#       +B_TRAVEL_AIM * travel_aim\
# +Epo_5_RND

+B_FACTOR1 * FACTOR1)
# +B_FACTOR5 * FACTOR5)


# Associate utility functions with the numbering of alternatives
V = {1: V1,
     2: V2
    ,3:V3
    ,4:V4
    ,5:V5}

# Associate the availability conditions with the alternatives
# CAR_AV_SP =  DefineVariable('CAR_AV_SP',CAR_AV  * (  SP   !=  0  ))
# TRAIN_AV_SP =  DefineVariable('TRAIN_AV_SP',TRAIN_AV  * (  SP   !=  0  ))

av = {1: 1,
      2: 1
     ,3: 1
     ,4: 1
     ,5: 1}


MU1 = Beta('MU1', 1, 0, 100, 0)
MU2 = Beta('MU2', 1, 0, 100, 0)
# MU3 = Beta('MU3', 1, 0, 100, 0)
# PT = MU1, [1,2,3]
# Car = 1, [4]
# Taxi = 1, [5]
# Bike = 1, [6]

##nested
PT = MU1, [1,2]
TAXI = 1, [3]
MORE = 1,[4]
NO = 1 ,[5]
nests = PT,TAXI, MORE,NO
# 不加面板效应
condprob = exp(models.lognested(V, av,nests, results))
condlike = (
    P_at8
    * P_at9
    * P_at10
    * P_at11
    * P_at13
    * P_at14
    * P_at17
    *P_at21
    *P_at22
    *P_at23
    *P_at24
    *P_at25
    *P_at18
    * P_at19
    * P_at12
    *P_at15
    # *P_at2
    # *P_at3
    *P_at1
    *P_at4
    *P_at6
    * condprob
)
#加入面板效应
# prob = models.loglogit(V, av, results)
# condlike = (
#     P_at8
#     * P_at9
#     * P_at10
#     * P_at11
#     * P_at13
#     * P_at14
#     * P_at17
#     *P_at21
#     *P_at22
#     *P_at23
#     *P_at24
#     *P_at25
#     *P_at18
#     * P_at19
#     * P_at12
#     *P_at15
#     *P_at2
#     *P_at3
#     *P_at1
#     *P_at4
#     *P_at6
#     * log(MonteCarlo(prob))
# )
# We integrate over B_TIME_RND using Monte-Carlo

# We integrate over B_TIME_RND using Monte-Carlo

# The choice model is a logit, with availability conditions
loglike = log(MonteCarlo(condlike))

# Define level of verbosity
logger = msg.bioMessage()
# logger.setSilent()
# logger.setWarning()
logger.setGeneral()
# logger.setDetailed()

# Create the Biogeme object
# biogeme = bio.BIOGEME(database, loglike)
biogeme = bio.BIOGEME(database, loglike, numberOfDraws=500)
biogeme.modelName = "HCM"

# Estimate the parameters
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(f'Estimated betas: {len(results.data.betaValues)}')
print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')
results.writeLaTeX()
print(f'LaTeX file: {results.data.latexFileName}')
