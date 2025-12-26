#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_excel(f'../单次出行数据.xlsx')
people_data = pd.read_csv(f'../MaaS被调查者聚类加个人属性加态度.csv',encoding = 'gb2312')
people_data = people_data[['peopleID','week_bus','week_metro', 'week_bike', 'week_ebike', 'week_taxi']]
people_data.columns = ['Uid','week_bus','week_metro', 'week_bike', 'week_ebike', 'week_taxi']
df = pd.merge(df,people_data)


df = df[-df['cheekall']]
#不支持MaaS的表述正向化
reject = [2,3,5,20]
for i in reject:
    df['at'+str(i)] = 6 - df['at'+str(i)]
first_choice = pd.get_dummies(df['nonmaas'])

first_choice.columns = ['first_car','first_taxi','first_pt']

df = pd.concat([df,first_choice],axis = 1)

departtime = pd.get_dummies(df['departtime'])
departtime.columns = ['morning','evening','normal','late']
df = pd.concat([df,departtime],axis = 1)

def get_big_penalty(r,column,av):
    if r[av] == 1:
        return r[column]
    else:
        return 1000

##出行模式哑元化
df['week_bus'] = df['week_bus'].replace(to_replace=[1,2,3,4,5],value=[0,0,0,1,1])
df['week_metro'] = df['week_metro'].replace(to_replace=[1,2,3,4,5],value=[0,0,0,1,1])
df['week_bike'] = df['week_bike'].replace(to_replace=[1,2,3,4,5],value=[0,0,0,1,1])
df['week_ebike'] = df['week_ebike'].replace(to_replace=[1,2,3,4,5],value=[0,0,0,1,1])
df['week_taxi'] = df['week_taxi'].replace(to_replace=[1,2,3,4,5],value=[0,0,0,1,1])

df['A1ttimeCar'] = df.apply(lambda r:get_big_penalty(r,'A1ttimeCar','Carav'),axis = 1)
df['A1priceCar'] = df.apply(lambda r:get_big_penalty(r,'A1priceCar','Carav'),axis = 1)

df['C1busrail'] = df.apply(lambda r:get_big_penalty(r,'C1busrail','PTav'),axis = 1)
df['C1ttimePT'] = df.apply(lambda r:get_big_penalty(r,'C1ttimePT','PTav'),axis = 1)
df['C1wtimePT'] = df.apply(lambda r:get_big_penalty(r,'C1wtimePT','PTav'),axis = 1)
df['C1distancePT_walk'] = df.apply(lambda r:get_big_penalty(r,'C1distancePT_walk','PTav'),axis = 1)
df['C1triptimePT'] = df.apply(lambda r:get_big_penalty(r,'C1triptimePT','PTav'),axis = 1)
df['C1pricePT'] = df.apply(lambda r:get_big_penalty(r,'C1pricePT','PTav'),axis = 1)

df['M1ttimerail'] = df.apply(lambda r:get_big_penalty(r,'M1ttimerail','Mav'),axis = 1)
df['M1ttime_bus'] = df.apply(lambda r:get_big_penalty(r,'M1ttime_bus','Mav'),axis = 1)
df['M1wtime'] = df.apply(lambda r:get_big_penalty(r,'M1wtime','Mav'),axis = 1)
df['M1distance_walk'] = df.apply(lambda r:get_big_penalty(r,'M1distance_walk','Mav'),axis = 1)
df['M1triptime'] = df.apply(lambda r:get_big_penalty(r,'M1triptime','Mav'),axis = 1)
df['M1price'] = df.apply(lambda r:get_big_penalty(r,'M1price','Mav'),axis = 1)

df['M2ttime_rail'] = df.apply(lambda r:get_big_penalty(r,'M2ttime_rail','Mav'),axis = 1)
df['M2ttime_bus'] = df.apply(lambda r:get_big_penalty(r,'M2ttime_bus','Mav'),axis = 1)
df['M2wtime'] = df.apply(lambda r:get_big_penalty(r,'M2wtime','Mav'),axis = 1)
df['M2distance_bike'] = df.apply(lambda r:get_big_penalty(r,'M2distance_bike','Mav'),axis = 1)
df['M2triptime'] = df.apply(lambda r:get_big_penalty(r,'M2triptime','Mav'),axis = 1)
df['M2price'] = df.apply(lambda r:get_big_penalty(r,'M2price','Mav'),axis = 1)

df['M3ttime_rail'] = df.apply(lambda r:get_big_penalty(r,'M3ttime_rail','PTav'),axis = 1)
df['M3ttime_taxi'] = df.apply(lambda r:get_big_penalty(r,'M3ttime_taxi','PTav'),axis = 1)
df['M3wtime'] = df.apply(lambda r:get_big_penalty(r,'M3wtime','PTav'),axis = 1)
df['M3triptime'] = df.apply(lambda r:get_big_penalty(r,'M3triptime','PTav'),axis = 1)
df['M3price'] = df.apply(lambda r:get_big_penalty(r,'M3price','PTav'),axis = 1)


def get_M1difference_time_prive(r):
    if r['first_car'] == 1:
        return (r['M1triptime'] - r['A1ttimeCar']), (r['M1price'] - r['A1priceCar'])
    if r['first_taxi'] == 1:
        return (r['M1triptime'] - r['B1triptime']), (r['M1price'] - r['B1priceTaxi'])
    if r['first_pt'] == 1:
        return (r['M1triptime'] - r['C1triptimePT']), (r['M1price'] - r['C1pricePT'])

def get_M2difference_time_prive(r):
    if r['first_car'] == 1:
        return (r['M2triptime'] - r['A1ttimeCar']), (r['M2price'] - r['A1priceCar'])
    if r['first_taxi'] == 1:
        return (r['M2triptime'] - r['B1triptime']), (r['M2price'] - r['B1priceTaxi'])
    if r['first_pt'] == 1:
        return (r['M2triptime'] - r['C1triptimePT']), (r['M2price'] - r['C1pricePT'])


def get_M3difference_time_prive(r):
    if r['first_car'] == 1:
        return (r['M3triptime'] - r['A1ttimeCar']), (r['M3price'] - r['A1priceCar'])
    if r['first_taxi'] == 1:
        return (r['M3triptime'] - r['B1triptime']), (r['M3price'] - r['B1priceTaxi'])
    if r['first_pt'] == 1:
        return (r['M3triptime'] - r['C1triptimePT']), (r['M3price'] - r['C1pricePT'])

def get_M4difference_time_prive(r):
    if r['first_car'] == 1:
        return (r['M4ttime'] - r['A1ttimeCar']), (r['M4price'] - r['A1priceCar'])
    if r['first_taxi'] == 1:
        return (r['M4ttime'] - r['B1triptime']), (r['M4price'] - r['B1priceTaxi'])
    if r['first_pt'] == 1:
        return (r['M4ttime'] - r['C1triptimePT']), (r['M4price'] - r['C1pricePT'])

def get_notransfer(r):
    if r['first_car'] == 1:
        return r['A1ttimeCar'], r['A1priceCar']
    if r['first_taxi'] == 1:
        return r['B1triptime'], r['B1priceTaxi']
    if r['first_pt'] == 1:
        return r['C1triptimePT'], r['C1pricePT']


df['no_triptime'], df['no_price'] = df.apply(get_notransfer,axis = 1).apply(lambda r:r[0]), df.apply(get_notransfer,axis = 1).apply(lambda r:r[1])


df['M1dif_triptime'], df['M1dif_price'] = df.apply(get_M1difference_time_prive,axis = 1).apply(lambda r:r[0]), df.apply(get_M1difference_time_prive,axis = 1).apply(lambda r:r[1])
df['M2dif_triptime'], df['M2dif_price'] = df.apply(get_M2difference_time_prive,axis = 1).apply(lambda r:r[0]), df.apply(get_M2difference_time_prive,axis = 1).apply(lambda r:r[1])
df['M3dif_triptime'], df['M3dif_price'] = df.apply(get_M3difference_time_prive,axis = 1).apply(lambda r:r[0]), df.apply(get_M3difference_time_prive,axis = 1).apply(lambda r:r[1])
df['M4dif_triptime'], df['M4dif_price'] = df.apply(get_M4difference_time_prive,axis = 1).apply(lambda r:r[0]), df.apply(get_M4difference_time_prive,axis = 1).apply(lambda r:r[1])


##轨道交通时间占比
df['M1metroratio'] = df['M1ttimerail']/df['M1triptime']
df['M2metroratio'] = df['M2ttime_rail']/df['M2triptime']
df['M3metroratio'] = df['M3ttime_rail']/df['M3triptime']


df['C1busrail'] = df['C1busrail'] - 1


##年龄哑变量化
df['age'] = df['age'].replace(to_replace=[1,2,3,4,5,6,7],value=[1,1,2,2,3,3,4])
age = pd.get_dummies(df['age'])
age.columns = ['age1','age2','age3','age4']
df = pd.concat([df,age],axis = 1)
##个人收入哑元化
df['income_m'] = df['income_m'].replace(to_replace=[1,2,3,4,5,6,7,8],value=[1,1,1,1,2,2,2,3])
income = pd.get_dummies(df['income_m'])
income.columns = ['income1','income2','income3']
df = pd.concat([df,income],axis = 1)
#个人收入，1W以下为0
df['income_m'] = df['income_m'].replace(to_replace=[1,2,3,4,5,6,7,8],value=[0,0,0,0,0,1,1,1])
#出行距离哑元化
distance = pd.get_dummies(df['distance'])
distance.columns = ['distance1','distance2','distance3','distance4','distance5']
df = pd.concat([df,distance],axis = 1)
#将1、2、3转换为0、1、2
df['gender'] = df['gender']-1
df['MainPurpose'] = df['MainPurpose']-1
df['member'] = df['member'].replace(to_replace=[1,2,3,4],value=[0,1,1,1])
df['DriveLicense'] = 2-df['DriveLicense']
df['DriveProf'] = 2-df['DriveProf']
df['bikeown'] = 2-df['bikeown']
df['MaasFamiliar'] = 2-df['MaasFamiliar']
#负数补0
df.loc[df['carnum']<0,'carnum'] = 0
df.loc[df['DriveLicense']<0,'DriveLicense'] = 0
##变量哑元化
#每月花费，150以下为0
# model_data['cost'] = model_data['cost'].replace(to_replace=[1,2,3,4,5,6],value=[0,0,1,1,1,1])
#年龄，34以下为0
df['age'] = df['age'].replace(to_replace=[1,2,3,4,5,6,7],value=[0,0,0,1,1,1,1])
#教育，本科以上为1
df['graduate'] = df['graduate'].replace(to_replace=[1,2,3,4,5,6],value=[1,1,1,0,0,0])


#家庭年收入，30W以下为0
df['income_y'] = df['income_y'].replace(to_replace=[1,2,3,4,5,6,7],value=[0,0,0,0,1,1,1])
#工作，出行少的工作为0
df['occupy'] = df['occupy'].replace(to_replace=[1,2,3,4,5,6,7,8,9,10,11],value=[0,0,0,0,0,0,0,1,1,1,1])

df['Carown'] = 2 - df['Carown']

df['TripsPerWeek'] = df['TripsPerWeek'].replace(to_replace=[1,2,3,4],value = [0,0,1,1])
df['DistanceWorkday'] = df['DistanceWorkday'].replace(to_replace=[1,2,3,4,5,6],value = [0,0,1,1,1,1])
df['DistanceWeekend'] = df['DistanceWeekend'].replace(to_replace=[1,2,3,4,5,6],value = [0,0,1,1,1,1])

df['DistanceWeekend'].value_counts()



df['maas'] = df['maas'].replace(to_replace=[1,2,3],value=[1,1,1])


dataframe = df.drop(columns=['cheek1', 'cheek2','cheekall'])


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
    MonteCarlo,
    
)
database = db.Database('HCM', dataframe)
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

asc_structure = Beta('asc_structure', 0, None, None, 0)
FACTOR1 = (
# asc_structure
          coef1_gender * gender
         +  coef1_age1 * age1
         +  coef1_age2 * age2
         +  coef1_age3 * age3
         # +  coef1_age4 * age4
         +  coef1_job * occupy
         + coef1_income1 * income1
         + coef1_income2 * income2
         # + coef1_income3 * income3
         +  coef1_education * graduate
         +  coef1_travel_num * TripsPerWeek
         +  coef1_travel_distance_day * DistanceWorkday
         # +  coef1_travel_distance_end * DistanceWeekend
         # +  coef1_travel_aim * MainPurpose
         # +  coef1_6a * Al_PT
         # +  coef1_6b * Al_taxi
         # +  coef1_6c * Al_car
         +  coef1_6d * Al_shareedcar
         +  coef1_6e * Al_bike
         +  coef1_6f * Al_sharedbike
         +  coef1_6g * Al_walk
         +  coef1_car_home * Carown
         # +  coef1_bus * week_bus
         +  coef1_metro * week_metro
         # +  coef1_taxi * week_taxi
         +  coef1_ebike * week_ebike
         # +  coef1_bike * week_bike
         +  sigma_s1 * omega
)

# FACTOR2 = (
#             coef2_gender * gender
#          +  coef2_age1 * age1
#          +  coef2_age2 * age2
#          +  coef2_age3 * age3
#          +  coef2_age4 * age4
#          +  coef2_job * occupy
#          + coef2_income1 * income1
#          + coef2_income2 * income2
#          + coef2_income3 * income3
#          +  coef2_education * graduate
#          +  coef2_travel_num * TripsPerWeek
#          +  coef2_travel_distance_day * DistanceWorkday
#          +  coef2_travel_distance_end * DistanceWeekend
#          +  coef2_travel_aim * MainPurpose
#          +  coef2_6a * Al_PT
#          +  coef2_6b * Al_taxi
#          +  coef2_6c * Al_car
#          +  coef2_6d * Al_shareedcar
#          +  coef2_6e * Al_bike
#          +  coef2_6f * Al_sharedbike
#          +  coef2_6g * Al_walk
#          +  coef2_car_home * Carown
#          +  coef2_bus * week_bus
#          +  coef2_metro * week_metro
#          +  coef2_taxi * week_taxi
#          +  coef2_ebike * week_ebike
#          +  coef2_bike * week_bike
#          +  sigma_s2 * omega
# )

# FACTOR3 = (
#             coef3_gender * gender
#          +  coef3_age1 * age1
#          +  coef3_age2 * age2
#          +  coef3_age3 * age3
#          +  coef3_age4 * age4
#          +  coef3_job * occupy
#          + coef3_income1 * income1
#          + coef3_income2 * income2
#          + coef3_income3 * income3
#          +  coef3_education * graduate
#          +  coef3_travel_num * TripsPerWeek
#          +  coef3_travel_distance_day * DistanceWorkday
#          +  coef3_travel_distance_end * DistanceWeekend
#          +  coef3_travel_aim * MainPurpose
#          +  coef3_6a * Al_PT
#          +  coef3_6b * Al_taxi
#          +  coef3_6c * Al_car
#          +  coef3_6d * Al_shareedcar
#          +  coef3_6e * Al_bike
#          +  coef3_6f * Al_sharedbike
#          +  coef3_6g * Al_walk
#          +  coef3_car_home * Carown
#          +  coef3_bus * week_bus
#          +  coef3_metro * week_metro
#          +  coef3_taxi * week_taxi
#          +  coef3_ebike * week_ebike
#          +  coef3_bike * week_bike
#          +  sigma_s3 * omega
# )
# #
# FACTOR4 = (
#             coef4_gender * gender
#          +  coef4_age1 * age1
#          +  coef4_age2 * age2
#          +  coef4_age3 * age3
#          +  coef4_age4 * age4
#          +  coef4_job * occupy
#          + coef4_income1 * income1
#          + coef4_income2 * income2
#          + coef4_income3 * income3
#          +  coef4_education * graduate
#          +  coef4_travel_num * TripsPerWeek
#          +  coef4_travel_distance_day * DistanceWorkday
#          +  coef4_travel_distance_end * DistanceWeekend
#          +  coef4_travel_aim * MainPurpose
#          +  coef4_6a * Al_PT
#          +  coef4_6b * Al_taxi
#          +  coef4_6c * Al_car
#          +  coef4_6d * Al_shareedcar
#          +  coef4_6e * Al_bike
#          +  coef4_6f * Al_sharedbike
#          +  coef4_6g * Al_walk
#          +  coef4_car_home * Carown
#          +  coef4_bus * week_bus
#          +  coef4_metro * week_metro
#          +  coef4_taxi * week_taxi
#          +  coef4_ebike * week_ebike
#          +  coef4_bike * week_bike
#          +  sigma_s4 * omega
# )
#
# FACTOR5 = (
#             coef5_gender * gender
#          +  coef5_age1 * age1
#          +  coef5_age2 * age2
#          +  coef5_age3 * age3
#          +  coef5_age4 * age4
#          +  coef5_job * occupy
#          + coef5_income1 * income1
#          + coef5_income2 * income2
#          + coef5_income3 * income3
#          +  coef5_education * graduate
#          +  coef5_travel_num * TripsPerWeek
#          +  coef5_travel_distance_day * DistanceWorkday
#          +  coef5_travel_distance_end * DistanceWeekend
#          +  coef5_travel_aim * MainPurpose
#          +  coef5_6a * Al_PT
#          +  coef5_6b * Al_taxi
#          +  coef5_6c * Al_car
#          +  coef5_6d * Al_shareedcar
#          +  coef5_6e * Al_bike
#          +  coef5_6f * Al_sharedbike
#          +  coef5_6g * Al_walk
#          +  coef5_car_home * Carown
#          +  coef5_bus * week_bus
#          +  coef5_metro * week_metro
#          +  coef5_taxi * week_taxi
#          +  coef5_ebike * week_ebike
#          +  coef5_bike * week_bike
#          +  sigma_s5 * omega
# )
#
# FACTOR6 = (
#             coef6_gender * gender
#          +  coef6_age1 * age1
#          +  coef6_age2 * age2
#          +  coef6_age3 * age3
#          +  coef6_age4 * age4
#          +  coef6_job * occupy
#          + coef6_income1 * income1
#          + coef6_income2 * income2
#          + coef6_income3 * income3
#          +  coef6_education * graduate
#          +  coef6_travel_num * TripsPerWeek
#          +  coef6_travel_distance_day * DistanceWorkday
#          +  coef6_travel_distance_end * DistanceWeekend
#          +  coef6_travel_aim * MainPurpose
#          +  coef6_6a * Al_PT
#          +  coef6_6b * Al_taxi
#          +  coef6_6c * Al_car
#          +  coef6_6d * Al_shareedcar
#          +  coef6_6e * Al_bike
#          +  coef6_6f * Al_sharedbike
#          +  coef6_6g * Al_walk
#          +  coef6_car_home * Carown
#          +  coef6_bus * week_bus
#          +  coef6_metro * week_metro
#          +  coef6_taxi * week_taxi
#          +  coef6_ebike * week_ebike
#          +  coef6_bike * week_bike
#          +  sigma_s6 * omega
# )

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
# MODEL_at13 = INTER_at13 + B_at13 * FACTOR1
MODEL_at14 = INTER_at14 + B_at14 * FACTOR1
MODEL_at17 = INTER_at17 + B_at17 * FACTOR1

# MODEL_at21 = INTER_at21 + B_at21 * FACTOR2
# MODEL_at22 = INTER_at22 + B_at22 * FACTOR2
# MODEL_at23 = INTER_at23 + B_at23 * FACTOR2
# MODEL_at24 = INTER_at24 + B_at24 * FACTOR2
# MODEL_at25 = INTER_at25 + B_at25 * FACTOR2


# MODEL_at18 = INTER_at18 + B_at18 * FACTOR3
# MODEL_at19 = INTER_at19 + B_at19 * FACTOR3
# #
# MODEL_at12 = INTER_at12 + B_at12 * FACTOR4
# MODEL_at15 = INTER_at15 + B_at15 * FACTOR4
#
# MODEL_at2 = INTER_at2 + B_at2 * FACTOR5
# MODEL_at3 = INTER_at3 + B_at3 * FACTOR5
#
# MODEL_at1 = INTER_at1 + B_at1 * FACTOR6
# MODEL_at4 = INTER_at4 + B_at4 * FACTOR6
# MODEL_at6 = INTER_at6 + B_at6 * FACTOR6
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

# at13_tau_1 = (tau_1p - MODEL_at13) / SIGMA_STAR_at13
# at13_tau_2 = (tau_2p - MODEL_at13) / SIGMA_STAR_at13
# at13_tau_3 = (tau_3p - MODEL_at13) / SIGMA_STAR_at13
# at13_tau_4 = (tau_4p - MODEL_at13) / SIGMA_STAR_at13
# Indat13 = {
#     1: bioNormalCdf(at13_tau_1),
#     2: bioNormalCdf(at13_tau_2) - bioNormalCdf(at13_tau_1),
#     3: bioNormalCdf(at13_tau_3) - bioNormalCdf(at13_tau_2),
#     4: bioNormalCdf(at13_tau_4) - bioNormalCdf(at13_tau_3),
#     5: 1 - bioNormalCdf(at13_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at13 = Elem(Indat13, at13)

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
# delta_1t = Beta('delta_1t', 0.1, 1.0e-5, None, 0)
# delta_2t = Beta('delta_2t', 0.2, 1.0e-5, None, 0)
# delta_3t = Beta('delta_3t', 0.3, 1.0e-5, None, 0)
#
# tau_1t = 0
# tau_2t = 0 + delta_1t
# tau_3t = tau_2t + delta_2t
# tau_4t = tau_3t + delta_3t
#
# at21_tau_1 = (tau_1t - MODEL_at21) / SIGMA_STAR_at21
# at21_tau_2 = (tau_2t - MODEL_at21) / SIGMA_STAR_at21
# at21_tau_3 = (tau_3t - MODEL_at21) / SIGMA_STAR_at21
# at21_tau_4 = (tau_4t - MODEL_at21) / SIGMA_STAR_at21
# Indat21 = {
#     1: bioNormalCdf(at21_tau_1),
#     2: bioNormalCdf(at21_tau_2) - bioNormalCdf(at21_tau_1),
#     3: bioNormalCdf(at21_tau_3) - bioNormalCdf(at21_tau_2),
#     4: bioNormalCdf(at21_tau_4) - bioNormalCdf(at21_tau_3),
#     5: 1 - bioNormalCdf(at21_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at21 = Elem(Indat21, at21)
#
# at22_tau_1 = (tau_1t - MODEL_at22) / SIGMA_STAR_at22
# at22_tau_2 = (tau_2t - MODEL_at22) / SIGMA_STAR_at22
# at22_tau_3 = (tau_3t - MODEL_at22) / SIGMA_STAR_at22
# at22_tau_4 = (tau_4t - MODEL_at22) / SIGMA_STAR_at22
# Indat22 = {
#     1: bioNormalCdf(at22_tau_1),
#     2: bioNormalCdf(at22_tau_2) - bioNormalCdf(at22_tau_1),
#     3: bioNormalCdf(at22_tau_3) - bioNormalCdf(at22_tau_2),
#     4: bioNormalCdf(at22_tau_4) - bioNormalCdf(at22_tau_3),
#     5: 1 - bioNormalCdf(at22_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at22 = Elem(Indat22, at22)
#
# at23_tau_1 = (tau_1t - MODEL_at23) / SIGMA_STAR_at23
# at23_tau_2 = (tau_2t - MODEL_at23) / SIGMA_STAR_at23
# at23_tau_3 = (tau_3t - MODEL_at23) / SIGMA_STAR_at23
# at23_tau_4 = (tau_4t - MODEL_at23) / SIGMA_STAR_at23
# Indat23 = {
#     1: bioNormalCdf(at23_tau_1),
#     2: bioNormalCdf(at23_tau_2) - bioNormalCdf(at23_tau_1),
#     3: bioNormalCdf(at23_tau_3) - bioNormalCdf(at23_tau_2),
#     4: bioNormalCdf(at23_tau_4) - bioNormalCdf(at23_tau_3),
#     5: 1 - bioNormalCdf(at23_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at23 = Elem(Indat23, at23)
#
# at24_tau_1 = (tau_1t - MODEL_at24) / SIGMA_STAR_at24
# at24_tau_2 = (tau_2t - MODEL_at24) / SIGMA_STAR_at24
# at24_tau_3 = (tau_3t - MODEL_at24) / SIGMA_STAR_at24
# at24_tau_4 = (tau_4t - MODEL_at24) / SIGMA_STAR_at24
# Indat24 = {
#     1: bioNormalCdf(at24_tau_1),
#     2: bioNormalCdf(at24_tau_2) - bioNormalCdf(at24_tau_1),
#     3: bioNormalCdf(at24_tau_3) - bioNormalCdf(at24_tau_2),
#     4: bioNormalCdf(at24_tau_4) - bioNormalCdf(at24_tau_3),
#     5: 1 - bioNormalCdf(at24_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at24 = Elem(Indat24, at24)
#
# at25_tau_1 = (tau_1t - MODEL_at25) / SIGMA_STAR_at25
# at25_tau_2 = (tau_2t - MODEL_at25) / SIGMA_STAR_at25
# at25_tau_3 = (tau_3t - MODEL_at25) / SIGMA_STAR_at25
# at25_tau_4 = (tau_4t - MODEL_at25) / SIGMA_STAR_at25
# Indat25 = {
#     1: bioNormalCdf(at25_tau_1),
#     2: bioNormalCdf(at25_tau_2) - bioNormalCdf(at25_tau_1),
#     3: bioNormalCdf(at25_tau_3) - bioNormalCdf(at25_tau_2),
#     4: bioNormalCdf(at25_tau_4) - bioNormalCdf(at25_tau_3),
#     5: 1 - bioNormalCdf(at25_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at25 = Elem(Indat25, at25)
###########################################################
# delta_13 = Beta('delta_13', 0.1, 1.0e-5, None, 0)
# delta_23 = Beta('delta_23', 0.2, 1.0e-5, None, 0)
# delta_33 = Beta('delta_33', 0.3, 1.0e-5, None, 0)
#
# tau_13 = 0
# tau_23 = 0 + delta_13
# tau_33 = tau_23 + delta_23
# tau_43 = tau_33 + delta_33
#
# at18_tau_1 = (tau_13 - MODEL_at18) / SIGMA_STAR_at18
# at18_tau_2 = (tau_23 - MODEL_at18) / SIGMA_STAR_at18
# at18_tau_3 = (tau_33 - MODEL_at18) / SIGMA_STAR_at18
# at18_tau_4 = (tau_43 - MODEL_at18) / SIGMA_STAR_at18
# Indat18 = {
#     1: bioNormalCdf(at18_tau_1),
#     2: bioNormalCdf(at18_tau_2) - bioNormalCdf(at18_tau_1),
#     3: bioNormalCdf(at18_tau_3) - bioNormalCdf(at18_tau_2),
#     4: bioNormalCdf(at18_tau_4) - bioNormalCdf(at18_tau_3),
#     5: 1 - bioNormalCdf(at18_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at18 = Elem(Indat18, at18)
#
# at19_tau_1 = (tau_13 - MODEL_at19) / SIGMA_STAR_at19
# at19_tau_2 = (tau_23 - MODEL_at19) / SIGMA_STAR_at19
# at19_tau_3 = (tau_33 - MODEL_at19) / SIGMA_STAR_at19
# at19_tau_4 = (tau_43 - MODEL_at19) / SIGMA_STAR_at19
# Indat19 = {
#     1: bioNormalCdf(at19_tau_1),
#     2: bioNormalCdf(at19_tau_2) - bioNormalCdf(at19_tau_1),
#     3: bioNormalCdf(at19_tau_3) - bioNormalCdf(at19_tau_2),
#     4: bioNormalCdf(at19_tau_4) - bioNormalCdf(at19_tau_3),
#     5: 1 - bioNormalCdf(at19_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at19 = Elem(Indat19, at19)
# # ###########################################################
# delta_14 = Beta('delta_14', 0.1, 1.0e-5, None, 0)
# delta_24 = Beta('delta_24', 0.2, 1.0e-5, None, 0)
# delta_34 = Beta('delta_34', 0.3, 1.0e-5, None, 0)
#
# tau_14 = 0
# tau_24 = 0 + delta_14
# tau_34 = tau_24 + delta_24
# tau_44 = tau_34 + delta_34
#
# at12_tau_1 = (tau_14 - MODEL_at12) / SIGMA_STAR_at12
# at12_tau_2 = (tau_24 - MODEL_at12) / SIGMA_STAR_at12
# at12_tau_3 = (tau_34 - MODEL_at12) / SIGMA_STAR_at12
# at12_tau_4 = (tau_44 - MODEL_at12) / SIGMA_STAR_at12
# Indat12 = {
#     1: bioNormalCdf(at12_tau_1),
#     2: bioNormalCdf(at12_tau_2) - bioNormalCdf(at12_tau_1),
#     3: bioNormalCdf(at12_tau_3) - bioNormalCdf(at12_tau_2),
#     4: bioNormalCdf(at12_tau_4) - bioNormalCdf(at12_tau_3),
#     5: 1 - bioNormalCdf(at12_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at12 = Elem(Indat12, at12)
#
# at15_tau_1 = (tau_14 - MODEL_at15) / SIGMA_STAR_at15
# at15_tau_2 = (tau_24 - MODEL_at15) / SIGMA_STAR_at15
# at15_tau_3 = (tau_34 - MODEL_at15) / SIGMA_STAR_at15
# at15_tau_4 = (tau_44 - MODEL_at15) / SIGMA_STAR_at15
# Indat15 = {
#     1: bioNormalCdf(at15_tau_1),
#     2: bioNormalCdf(at15_tau_2) - bioNormalCdf(at15_tau_1),
#     3: bioNormalCdf(at15_tau_3) - bioNormalCdf(at15_tau_2),
#     4: bioNormalCdf(at15_tau_4) - bioNormalCdf(at15_tau_3),
#     5: 1 - bioNormalCdf(at15_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at15 = Elem(Indat15, at15)
# ###########################################################
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
# ###########################################################
# delta_16 = Beta('delta_16', 0.1, 1.0e-5, None, 0)
# delta_26 = Beta('delta_26', 0.2, 1.0e-5, None, 0)
# delta_36 = Beta('delta_36', 0.3, 1.0e-5, None, 0)
#
# tau_16 = 0
# tau_26 = 0 + delta_16
# tau_36 = tau_26 + delta_26
# tau_46 = tau_36 + delta_36
#
# at1_tau_1 = (tau_16 - MODEL_at1) / SIGMA_STAR_at1
# at1_tau_2 = (tau_26 - MODEL_at1) / SIGMA_STAR_at1
# at1_tau_3 = (tau_36 - MODEL_at1) / SIGMA_STAR_at1
# at1_tau_4 = (tau_46 - MODEL_at1) / SIGMA_STAR_at1
# Indat1 = {
#     1: bioNormalCdf(at1_tau_1),
#     2: bioNormalCdf(at1_tau_2) - bioNormalCdf(at1_tau_1),
#     3: bioNormalCdf(at1_tau_3) - bioNormalCdf(at1_tau_2),
#     4: bioNormalCdf(at1_tau_4) - bioNormalCdf(at1_tau_3),
#     5: 1 - bioNormalCdf(at1_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at1 = Elem(Indat1, at1)
#
# at4_tau_1 = (tau_16 - MODEL_at4) / SIGMA_STAR_at4
# at4_tau_2 = (tau_26 - MODEL_at4) / SIGMA_STAR_at4
# at4_tau_3 = (tau_36 - MODEL_at4) / SIGMA_STAR_at4
# at4_tau_4 = (tau_46 - MODEL_at4) / SIGMA_STAR_at4
# Indat4 = {
#     1: bioNormalCdf(at4_tau_1),
#     2: bioNormalCdf(at4_tau_2) - bioNormalCdf(at4_tau_1),
#     3: bioNormalCdf(at4_tau_3) - bioNormalCdf(at4_tau_2),
#     4: bioNormalCdf(at4_tau_4) - bioNormalCdf(at4_tau_3),
#     5: 1 - bioNormalCdf(at4_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at4 = Elem(Indat4, at4)
#
# at6_tau_1 = (tau_16 - MODEL_at6) / SIGMA_STAR_at6
# at6_tau_2 = (tau_26 - MODEL_at6) / SIGMA_STAR_at6
# at6_tau_3 = (tau_36 - MODEL_at6) / SIGMA_STAR_at6
# at6_tau_4 = (tau_46 - MODEL_at6) / SIGMA_STAR_at6
# Indat6 = {
#     1: bioNormalCdf(at6_tau_1),
#     2: bioNormalCdf(at6_tau_2) - bioNormalCdf(at6_tau_1),
#     3: bioNormalCdf(at6_tau_3) - bioNormalCdf(at6_tau_2),
#     4: bioNormalCdf(at6_tau_4) - bioNormalCdf(at6_tau_3),
#     5: 1 - bioNormalCdf(at6_tau_4),
#     6: 1.0,
#     -1: 1.0,
#     -2: 1.0,
# }
#
# P_at6 = Elem(Indat6, at6)

## 态度变量
C_B_FACTOR1 = Beta('C_B_FACTOR1',0,-1000,1000,0)
C_B_FACTOR2 = Beta('C_B_FACTOR2',0,-1000,1000,0)
C_B_FACTOR3 = Beta('C_B_FACTOR3',0,-1000,1000,0)
C_B_FACTOR4 = Beta('C_B_FACTOR4',0,-1000,1000,0)
C_B_FACTOR5 = Beta('C_B_FACTOR5',0,-1000,1000,0)
C_B_FACTOR6 = Beta('C_B_FACTOR6',0,-1000,1000,0)

##潜在分类，出行模式+个人属性
# B_TRAVEL_NUM = Beta('B_TRAVEL_NUM',0,-1000,1000,0)
C_ASC = Beta('C_ASC', 1, -1000, 1000, 0)
C_B_TRAVEL_DISTANCE_WORK = Beta('C_B_TRAVEL_DISTANCE_WORK', 1, -1000, 1000, 0)
C_B_TRAVEL_DISTANCE_END = Beta('C_B_TRAVEL_DISTANCE_END', 1, -1000, 1000, 0)
C_B_TRAVEL_AIM = Beta('C_B_TRAVEL_AIM', 1, -1000, 1000, 0)
C_B_PT = Beta('C_B_PT', 1, -1000, 1000, 0)
C_B_TAXI = Beta('C_B_TAXI', 1, -1000, 1000, 0)
C_B_CAR = Beta('C_B_CAR', 1, -1000, 1000, 0)
C_B_SHARECAR = Beta('C_B_SHARECAR', 1, -1000, 1000, 0)
C_B_SHAREBIKE = Beta('C_B_SHAREBIKE', 1, -1000, 1000, 0)
C_B_BIKE = Beta('C_B_BIKE', 1, -1000, 1000, 0)
C_B_WALK = Beta('C_B_WALK', 1, -1000, 1000, 0)
C_B_COST = Beta('C_B_COST', 1, -1000, 1000, 0)
C_B_BUS_NUM = Beta('C_B_BUS_NUM', 1, -1000, 1000, 0)
C_B_METRO_NUM = Beta('C_B_METRO_NUM', 1, -1000, 1000, 0)
C_B_BIKE_NUM = Beta('C_B_BIKE_NUM', 1, -1000, 1000, 0)
C_B_EBIKE_NUM = Beta('C_B_EBIKE_NUM', 1, -1000, 1000, 0)
C_B_TAXI_NUM = Beta('C_B_TAXI_NUM', 1, -1000, 1000, 0)
C_B_COMBINE_SHAREBIKE = Beta('C_B_COMBINE_SHAREBIKE', 1, -1000, 1000, 0)

C_B_SEX = Beta('C_B_SEX', 0, -1000, 1000, 0)
C_B_AGE = Beta('C_B_AGE', 0, -1000, 1000, 0)
C_B_EDUCATION = Beta('C_B_EDUCATION', 0, -1000, 1000, 0)
C_B_INCOME_INDIVIDUAL = Beta('C_B_INCOME_INDIVIDUAL', 0, -1000, 1000, 0)
C_B_CAR_HOME = Beta('C_B_CAR_HOME', 0, -1000, 1000, 0)
C_B_EBIKE_HOME = Beta('C_B_EBIKE_HOME', 0, -1000, 1000, 0)
C_B_OCCUPY = Beta('C_B_OCCUPY', 0, -1000, 1000, 0)
C_B_HAVECAR = Beta('C_B_HAVECAR', 0, -1000, 1000, 0)
C_B_LICENSE = Beta('C_B_LICENSE', 0, -1000, 1000, 0)
C_B_AGE1 = Beta('C_B_AGE1', 0, -1000, 1000, 0)
C_B_AGE2 = Beta('C_B_AGE2', 0, -1000, 1000, 0)
C_B_AGE3 = Beta('C_B_AGE3', 0, -1000, 1000, 0)
C_B_AGE4 = Beta('C_B_AGE4', 0, -1000, 1000, 0)
C_B_INCOMEM = Beta('C_B_INCOMEM', 0, -1000, 1000, 0)
C_B_INCOMEY = Beta('C_B_INCOMEY', 0, -1000, 1000, 0)
C_B_INCOME1 = Beta('C_B_INCOME1', 0, -1000, 1000, 0)
C_B_INCOME2 = Beta('C_B_INCOME2', 0, -1000, 1000, 0)
C_B_INCOME3 = Beta('C_B_INCOME3', 0, -1000, 1000, 0)
C_B_MAASFAMILAR = Beta('C_B_MAASFAMILAR', 0, -1000, 1000, 0)

##潜在类别概率
utilityClass1 = exp(C_ASC + \
                    #                     C_B_SEX* gender + \
                    C_B_MAASFAMILAR * MaasFamiliar + \
                    #                     C_B_AGE1* age1 + \
                    #                     C_B_AGE2* age2 + \
                    #                     C_B_AGE3* age3 + \
                    #                     C_B_AGE4* age4 + \
                    #                     C_B_INCOME1* income1 +\
                    #                     C_B_INCOME2* income2 +\
                    #                     C_B_INCOME3* income3 +\
                    #                     C_B_INCOMEY* income_y +\
                    #                     C_B_CAR * Al_car +\
                    C_B_TAXI * Al_taxi + \
                    C_B_PT * Al_PT + \
                    C_B_BIKE * Al_bike + \
                    C_B_HAVECAR * Carown + \
                    #                     C_B_EBIKE_HOME * bikeown+\
                    #                     C_B_OCCUPY * occupy+\
                    #                     C_B_EDUCATION * graduate+\
                    #                     C_B_SHAREBIKE * Al_sharedbike+\
                    #                     C_B_TRAVEL_DISTANCE_WORK * DistanceWorkday+\
                    C_B_TRAVEL_DISTANCE_END * DistanceWeekend +\
                    C_B_FACTOR1 * FACTOR1
                    # C_B_FACTOR3 * FACTOR3
                    )

utilityClass2 = exp(0)

sumutilityclass = utilityClass1 + utilityClass2

probClass1 = utilityClass1 / sumutilityclass
probClass2 = utilityClass2 / sumutilityclass

###巢1
##方式变量
ASC_11 = Beta('ASC_11', 0, -1000, 1000, 1)
ASC_21 = Beta('ASC_21', 0, -1000, 1000, 0)
ASC_31 = Beta('ASC_31', 0, -1000, 1000, 0)
ASC_41 = Beta('ASC_41', 0, -1000, 1000, 0)
ASC_51 = Beta('ASC_51', 0, -1000, 1000, 0)
ASC_61 = Beta('ASC_61', 0, -1000, 1000, 0)
ASC_71 = Beta('ASC_71', 0, -1000, 1000, 0)

##后一次选择肢本身的影响
B_WAIT_TIME1 = Beta('B_WAIT_TIME1', 0, -1000, 1000, 0)
B_TRIP_TIME_NO_1 = Beta('B_TRIP_TIME_NO_1', 0, -1000, 1000, 0)
B_TRIP_TIME1 = Beta('B_TRIP_TIME1', 0, -1000, 1000, 0)
B_PRICE1 = Beta('B_PRICE1', 0, -1000, 1000, 0)
B_PRICE_NO_1 = Beta('B_PRICE_NO_1', 0, -1000, 1000, 0)
B_PRICE_TRANS_1 = Beta('B_PRICE_TRANS_1', 0, -1000, 1000, 0)
B_WALK_DISTANCE1 = Beta('B_WALK_DISTANCE1', 0, -1000, 1000, 0)
B_BUSORRAIL1 = Beta('B_BUSORRAIL1', 0, -1000, 1000, 0)
B_RAIL_TIME1 = Beta('B_RAIL_TIME1', 0, -1000, 1000, 0)
B_RAIL_RATIO1 = Beta('B_RAIL_RATIO1', 0, -1000, 1000, 0)
B_TAXI_TIME1 = Beta('B_TAXI_TIME1', 0, -1000, 1000, 0)
B_BIKE_DISTANCE1 = Beta('B_BIKE_DISTANCE1', 0, -1000, 1000, 0)
B_BIKE_AVAI1 = Beta('B_BIKE_AVAI1', 0, -1000, 1000, 0)
B_WALK_AVAI1 = Beta('B_WALK_AVAI1', 0, -1000, 1000, 0)

##增益变量
B_DIF_TIME1 = Beta('B_DIF_TIME1', 0, -1000, 1000, 0)
B_DIF_TIME_M12_1 = Beta('B_DIF_TIME_M12_1', 0, -1000, 1000, 0)
B_DIF_TIME_M3_1 = Beta('B_DIF_TIME_M3_1', 0, -1000, 1000, 0)
B_DIF_TIME_M4_1 = Beta('B_DIF_TIME_M4_1', 0, -1000, 1000, 0)
B_DIF_PRICE1 = Beta('B_DIF_PRICE1', 0, -1000, 1000, 0)
B_DIF_PRICE_M12_1 = Beta('B_DIF_PRICE_M12_1', 0, -1000, 1000, 0)
B_DIF_PRICE_M3_1 = Beta('B_DIF_PRICE_M3_1', 0, -1000, 1000, 0)
B_DIF_PRICE_M4_1 = Beta('B_DIF_PRICE_M4_1', 0, -1000, 1000, 0)

##跨阶段出行变量
B_SAME_CHOICE1 = Beta('B_SAME_CHOICE1', 0, -1000, 1000, 0)
B_SAME_CHOICE_CAR1 = Beta('B_SAME_CHOICE_CAR1', 0, -1000, 1000, 0)
B_SAME_CHOICE_TAXI1 = Beta('B_SAME_CHOICE_TAXI1', 0, -1000, 1000, 0)
B_SAME_CHOICE_PT1 = Beta('B_SAME_CHOICE_PT1', 0, -1000, 1000, 0)
B_FIRST_CAR1 = Beta('B_FIRST_CAR1', 0, -1000, 1000, 0)
B_FIRST_TAXI1 = Beta('B_FIRST_TAXI1', 0, -1000, 1000, 0)
B_FIRST_PT1 = Beta('B_FIRST_PT1', 0, -1000, 1000, 0)

##出行目的、出行场景变量
B_PURPOSE1 = Beta('B_PURPOSE1', 0, -1000, 1000, 0)
B_DEPARTTIME11 = Beta('B_DEPARTTIME11', 0, -1000, 1000, 0)
B_DEPARTTIME21 = Beta('B_DEPARTTIME21', 0, -1000, 1000, 0)
B_DEPARTTIME31 = Beta('B_DEPARTTIME31', 0, -1000, 1000, 0)
B_DEPARTTIME41 = Beta('B_DEPARTTIME41', 0, -1000, 1000, 0)
B_DISTANCE11 = Beta('B_DISTANCE11', 0, -1000, 1000, 0)
B_DISTANCE21 = Beta('B_DISTANCE21', 0, -1000, 1000, 0)
B_DISTANCE31 = Beta('B_DISTANCE31', 0, -1000, 1000, 0)
B_DISTANCE41 = Beta('B_DISTANCE41', 0, -1000, 1000, 0)
B_DISTANCE51 = Beta('B_DISTANCE51', 0, -1000, 1000, 0)

###巢2
##方式变量
ASC_12 = Beta('ASC_12', 0, -1000, 1000, 1)
ASC_22 = Beta('ASC_22', 0, -1000, 1000, 0)
ASC_32 = Beta('ASC_32', 0, -1000, 1000, 0)
ASC_42 = Beta('ASC_42', 0, -1000, 1000, 0)
ASC_52 = Beta('ASC_52', 0, -1000, 1000, 0)
ASC_62 = Beta('ASC_62', 0, -1000, 1000, 0)
ASC_72 = Beta('ASC_72', 0, -1000, 1000, 0)

##后一次选择肢本身的影响
B_WAIT_TIME2 = Beta('B_WAIT_TIME2', 0, -1000, 1000, 0)
B_TRIP_TIME_NO_2 = Beta('B_TRIP_TIME_NO_2', 0, -1000, 1000, 0)
B_TRIP_TIME2 = Beta('B_TRIP_TIME2', 0, -1000, 1000, 0)
B_PRICE2 = Beta('B_PRICE2', 0, -1000, 1000, 0)
B_PRICE_NO_2 = Beta('B_PRICE_NO_2', 0, -1000, 1000, 0)
B_PRICE_TRANS_2 = Beta('B_PRICE_TRANS_2', 0, -1000, 1000, 0)
B_WALK_DISTANCE2 = Beta('B_WALK_DISTANCE2', 0, -1000, 1000, 0)
B_BUSORRAIL2 = Beta('B_BUSORRAIL2', 0, -1000, 1000, 0)
B_RAIL_TIME2 = Beta('B_RAIL_TIME2', 0, -1000, 1000, 0)
B_RAIL_RATIO2 = Beta('B_RAIL_RATIO2', 0, -1000, 1000, 0)
B_TAXI_TIME2 = Beta('B_TAXI_TIME2', 0, -1000, 1000, 0)
B_BIKE_DISTANCE2 = Beta('B_BIKE_DISTANCE2', 0, -1000, 1000, 0)
B_BIKE_AVAI2 = Beta('B_BIKE_AVAI2', 0, -1000, 1000, 0)
B_WALK_AVAI2 = Beta('B_WALK_AVAI2', 0, -1000, 1000, 0)

##增益变量
B_DIF_TIME2 = Beta('B_DIF_TIME2', 0, -1000, 1000, 0)
B_DIF_TIME_M12_2 = Beta('B_DIF_TIME_M12_2', 0, -1000, 1000, 0)
B_DIF_TIME_M3_2 = Beta('B_DIF_TIME_M3_2', 0, -1000, 1000, 0)
B_DIF_TIME_M4_2 = Beta('B_DIF_TIME_M4_2', 0, -1000, 1000, 0)
B_DIF_PRICE2 = Beta('B_DIF_PRICE2', 0, -1000, 1000, 0)
B_DIF_PRICE_M12_2 = Beta('B_DIF_PRICE_M12_2', 0, -1000, 1000, 0)
B_DIF_PRICE_M3_2 = Beta('B_DIF_PRICE_M3_2', 0, -1000, 1000, 0)
B_DIF_PRICE_M4_2 = Beta('B_DIF_PRICE_M4_2', 0, -1000, 1000, 0)

##跨阶段出行变量
B_SAME_CHOICE2 = Beta('B_SAME_CHOICE2', 0, -1000, 1000, 0)
B_SAME_CHOICE_CAR2 = Beta('B_SAME_CHOICE_CAR2', 0, -1000, 1000, 0)
B_SAME_CHOICE_TAXI2 = Beta('B_SAME_CHOICE_TAXI2', 0, -1000, 1000, 0)
B_SAME_CHOICE_PT2 = Beta('B_SAME_CHOICE_PT2', 0, -1000, 1000, 0)
B_FIRST_CAR2 = Beta('B_FIRST_CAR2', 0, -1000, 1000, 0)
B_FIRST_TAXI2 = Beta('B_FIRST_TAXI2', 0, -1000, 1000, 0)
B_FIRST_PT2 = Beta('B_FIRST_PT2', 0, -1000, 1000, 0)

##出行目的、出行场景变量
B_PURPOSE2 = Beta('B_PURPOSE2', 0, -1000, 1000, 0)
B_DEPARTTIME12 = Beta('B_DEPARTTIME12', 0, -1000, 1000, 0)
B_DEPARTTIME22 = Beta('B_DEPARTTIME22', 0, -1000, 1000, 0)
B_DEPARTTIME32 = Beta('B_DEPARTTIME32', 0, -1000, 1000, 0)
B_DEPARTTIME42 = Beta('B_DEPARTTIME42', 0, -1000, 1000, 0)
B_DISTANCE12 = Beta('B_DISTANCE12', 0, -1000, 1000, 0)
B_DISTANCE22 = Beta('B_DISTANCE22', 0, -1000, 1000, 0)
B_DISTANCE32 = Beta('B_DISTANCE32', 0, -1000, 1000, 0)
B_DISTANCE42 = Beta('B_DISTANCE42', 0, -1000, 1000, 0)
B_DISTANCE52 = Beta('B_DISTANCE52', 0, -1000, 1000, 0)
##日常出行模式系数(分析什么样的人群可能会采用MaaS)


##面板误差效应
# Epo_1 = Beta('Epo_1', 0, None, None, 1)
# Epo_1S = Beta('Epo_1S', 1, None, None, 0)
# Epo_1_RND = Epo_1 + Epo_1S * bioDraws('Epo_1_RND', 'NORMAL')

# Epo_2 = Beta('Epo_2', 0, None, None, 1)
# Epo_2S = Beta('Epo_2S', 1, None, None, 0)
# Epo_2_RND = Epo_2 + Epo_2S * bioDraws('Epo_2_RND', 'NORMAL')

# Epo_3 = Beta('Epo_3', 0, None, None, 1)
# Epo_3S = Beta('Epo_3S', 1, None, None, 0)
# Epo_3_RND = Epo_3 + Epo_3S * bioDraws('Epo_3_RND', 'NORMAL')

# Epo_4 = Beta('Epo_4', 0, None, None, 1)
# Epo_4S = Beta('Epo_4S', 1, None, None, 0)
# Epo_4_RND = Epo_4 + Epo_4S * bioDraws('Epo_4_RND', 'NORMAL')

# Epo_5 = Beta('Epo_5', 0, None, None, 1)
# Epo_5S = Beta('Epo_5S', 1, None, None, 0)
# Epo_5_RND = Epo_5 + Epo_5S * bioDraws('Epo_5_RND', 'NORMAL')


##缩放
# HIGH_COST_SCALED = DefineVariable('HIGH_COST_SCALED', HWCost / 100.0, database)
# TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED', TCost / 100.0, database)
# PLANE_COST_SCALED = DefineVariable('Plane_COST_SCALED', PCost/ 100.0, database)
# CAR_COST_SCALED = DefineVariable('CAR_COST_SCALED', CCost / 100.0, database)
'''
Uid
Qid,purpose,departtime,distance,Carown,Carlimit,Carav,PTav,Mav,nonmaas
maas,A1ttimeCar,A1priceCar,B1ttimeTaxi,B1wtimeTaxi,B1triptime,B1priceTaxi,C1busrail,C1ttimePT,C1wtimePT
C1distancePT_walk,C1triptimePT,C1pricePT,M1ttimerail,M1ttime_bus,M1wtime,M1distance_walk,M1triptime,M1price,M2ttime_rail
M2ttime_bus,M2wtime,M2distance_bike,M2triptime,M2price,M3ttime_rail,M3ttime_taxi,M3wtime,M3triptime,M3price
M4ttime,M4price,TripsPerWeek,DistanceWorkday,DistanceWeekend,MainPurpose,fellows,Al_PT,Al_taxi,Al_car
Al_shareedcar,Al_bike,Al_sharedbike,Al_walk,C_PT_car,C_PT_taxi,C_PT_sharedbike,C_taxi_sharedbike,MaasFamiliar,at1
at2,at3,at4,at5,at6,at7,at8,at9,at10,at11
at12,at13,at14,at15,at16,at17,at18,at19,at20,at21
at22,at23,at24,at25,gender,age,graduate,occupy,income_m,income_y
member,carnum,DriveLicense,DriveProf,bikeown,first_car,first_taxi,first_pt,morning,evening
normal,late,M1dif_triptime,M1dif_price,M2dif_triptime,M2dif_price,M3dif_triptime,M3dif_price,M4dif_triptime,M4dif_price
results,
'''

##类别1的选择
V11 = (ASC_11 \
       #       + B_TRIP_TIME1 * no_triptime\
       #       + B_PRICE1 * no_price\
       + B_SAME_CHOICE_CAR1 * first_car \
       + B_SAME_CHOICE_TAXI1 * first_taxi \
       #       + B_SAME_CHOICE_PT1 * first_pt\
       #       + B_PURPOSE1 * purpose * first_taxi\
       #       + B_TRAVEL_NUM * TripsPerWeek\
       #       + B_TRAVEL_DISTANCE_END * DistanceWeekend\
       #       + B_CAR * Al_car\
       #       + B_TAXI * Al_taxi\
       #       + B_PT * Al_PT\
       #       + B_HAVECAR * Carown\
       #       + B_AGE1 * age1
       #       + B_AGE2 * age2\
       #       + B_AGE3 * age3\
       #       + B_INCOME1 * income_m\
       #        + B_DISTANCE11 * distance1\
       #        + B_DISTANCE21 * distance2\
       #        + B_DISTANCE31 * distance3\
       #        + B_DISTANCE41 * distance4\
       + B_DISTANCE51 * distance5 \
       #        + B_DEPARTTIME11 * morning\
       #        + B_DEPARTTIME21 * evening\
       #        + B_DEPARTTIME31 * normal\
       #        + B_DEPARTTIME41 * late\
       )
# V2 = (ASC_2\
#       + B_TRIP_TIME * B1triptime\
#       + B_WAIT_TIME * B1wtimeTaxi\
#       + B_PRICE * B1priceTaxi\
# #       + B_SAME_CHOICE * first_taxi\
#       + B_TAXI * Al_taxi\
#       + B_AGE2 * age2\
#       + B_AGE3 * age3
#       )
# V3 = (ASC_3\
#       + B_BUSORRAIL * C1busrail\
#       + B_TRIP_TIME * C1triptimePT\
#       + B_WAIT_TIME * C1wtimePT\
#       + B_WALK_DISTANCE * C1distancePT_walk\
#       + B_PRICE * C1pricePT\
# #       + B_SAME_CHOICE * first_pt\
#       + B_TRAVEL_DISTANCE_WORK * DistanceWorkday\
#       + B_PT * Al_PT\
#       + B_AGE1 * age1
#       )
V41 = (0 \
       + ASC_41 \
       + B_RAIL_TIME1 * M1ttimerail \
       #        + B_RAIL_RATIO1 * M1metroratio\
       + B_TRIP_TIME1 * M1triptime \
       #       + B_WAIT_TIME1 * M1wtime\
       #       + B_WALK_DISTANCE1 * M1distance_walk\
       #       + B_PRICE1 * M1price\
       + B_FIRST_PT1 * first_pt \
       #       + B_DIF_TIME_M12_1 * M1dif_triptime \
       #       + B_DIF_PRICE_M12_1 * (M1dif_price * first_pt)\
       #       + B_PURPOSE1 * purpose\
       #       + B_TRAVEL_DISTANCE_WORK * DistanceWorkday\
       #       + B_SHAREBIKE * Al_sharedbike\
       #       + B_BIKE * Al_bike\
       #       + B_AGE1 * age1\
       #       + B_FIRST_CAR1 * first_car\
       #       + B_FIRST_TAXI * first_taxi\
       #        + B_DEPARTTIME11 * morning\
       #        + B_DEPARTTIME21 * evening\
       + B_DEPARTTIME31 * normal \
       #        + B_DEPARTTIME41 * late\
       #        + B_DISTANCE11 * distance1\
       #        + B_DISTANCE21 * distance2\
       #        + B_DISTANCE31 * distance3\
       #        + B_DISTANCE41 * distance4\
       #        + B_DISTANCE51 * distance5\
       )
V51 = (0 \
       + ASC_51 \
       + B_RAIL_TIME1 * M2ttime_rail \
       #        + B_RAIL_RATIO1 * M2metroratio\
       + B_TRIP_TIME1 * M2triptime \
       #       + B_WAIT_TIME1 * M2wtime\
       #       + B_BIKE_DISTANCE1 * M2distance_bike\
       #       + B_PRICE1 * M2price\
       + B_FIRST_PT1 * first_pt \
       #       + B_DIF_TIME_M12_1 * M2dif_triptime \
       #       + B_DIF_PRICE_M12_1 * (M2dif_price * first_pt)\
       #       + B_PURPOSE1 * purpose\
       #       + B_TRAVEL_DISTANCE_WORK * DistanceWorkday\
       #       + B_SHAREBIKE * Al_sharedbike\
       #       + B_BIKE * Al_bike\
       #       + B_AGE1 * age1\
       #       + B_FIRST_CAR1 * first_car\
       #        + B_DEPARTTIME11 * morning\
       #        + B_DEPARTTIME21 * evening\
       + B_DEPARTTIME31 * normal \
       #        + B_DEPARTTIME41 * late\
       #       + B_FIRST_TAXI * first_taxi\
       #        + B_DISTANCE11 * distance1\
       #        + B_DISTANCE21 * distance2\
       #        + B_DISTANCE31 * distance3\
       #        + B_DISTANCE41 * distance4\
       #        + B_DISTANCE51 * distance5\
       )
V61 = (ASC_61 \
       + B_RAIL_TIME1 * M3ttime_rail \
       #       + B_RAIL_RATIO1 * M3metroratio\
       #       + B_TAXI_TIME1 * M3ttime_taxi\
       + B_TRIP_TIME1 * M3triptime \
       #       + B_WAIT_TIME1 * M3wtime\
       #       + B_PRICE1 * M3price\
       #       + B_FIRST_PT1 * first_pt\
       + B_FIRST_TAXI1 * first_taxi \
       #       + B_DIF_TIME_M12_1 * M3dif_triptime\
       #       + B_DIF_PRICE_M12_1 * (M3dif_price * (first_pt+first_taxi))\
       #       + B_PURPOSE1 * purpose\
       #       + B_TRAVEL_DISTANCE_WORK * DistanceWorkday\
       #       + B_SHAREBIKE * Al_sharedbike\
       #       + B_BIKE * Al_bike\
       #       + B_AGE1 * age1\
       #       + B_FIRST_CAR1 * first_car\
       #       + B_FIRST_TAXI1 * first_taxi\
       #        + B_DEPARTTIME11 * morning\
       #        + B_DEPARTTIME21 * evening\
       + B_DEPARTTIME31 * normal \
       #        + B_DEPARTTIME41 * late\
       #        + B_DISTANCE11 * distance1\
       #        + B_DISTANCE21 * distance2\
       #        + B_DISTANCE31 * distance3\
       #        + B_DISTANCE41 * distance4\
       #        + B_DISTANCE51 * distance5\
       )
V71 = (0 \
       + ASC_71 \
       + B_TRIP_TIME1 * M4ttime \
       #       + B_PRICE1 * M4price\
       #       + B_DIF_TIME_M4_1 * M4dif_triptime\
       #       + B_DIF_PRICE_M4_1 * M4dif_price\
       #       + B_PURPOSE1 * purpose\
       #       + B_TRAVEL_DISTANCE_WORK * DistanceWorkday\
       #       + B_SHARECAR * Al_shareedcar\
       #       + B_AGE1 * age1\
       #       + B_FIRST_PT * first_pt\
       #       + B_FIRST_CAR * first_car\
       + B_FIRST_TAXI1 * first_taxi \
       #       + B_FIRST_CAR1 * first_car\
       #        + B_DEPARTTIME11 * morning\
       #        + B_DEPARTTIME21 * evening\
       #        + B_DEPARTTIME31 * normal\
       #        + B_DEPARTTIME41 * late\
       #        + B_DISTANCE11 * distance1\
       #        + B_DISTANCE21 * distance2\
       #        + B_DISTANCE31 * distance3\
       #        + B_DISTANCE41 * distance4\
       + B_DISTANCE51 * distance5 \

       )

# Associate utility fu nctions with the numbering of alternatives
V1 = {
    1: V11,
    #     2: V2,
    #     3: V3,
    4: V41,
    5: V51,
    6: V61,
    7: V71
}

V12 = (ASC_12 \
       #       + B_TRIP_TIME1 * no_triptime\
       #       + B_PRICE1 * no_price\
       + B_SAME_CHOICE_CAR2 * first_car \
       + B_SAME_CHOICE_TAXI2 * first_taxi \
       #       + B_SAME_CHOICE_PT1 * first_pt\
       #       + B_PURPOSE2 * purpose * first_taxi\
       #       + B_TRAVEL_NUM * TripsPerWeek\
       #       + B_TRAVEL_DISTANCE_END * DistanceWeekend\
       #       + B_CAR * Al_car\
       #       + B_TAXI * Al_taxi\
       #       + B_PT * Al_PT\
       #       + B_HAVECAR * Carown\
       #       + B_AGE1 * age1
       #       + B_AGE2 * age2\
       #       + B_AGE3 * age3\
       #       + B_INCOME1 * income_m\
       #        + B_DISTANCE11 * distance1\
       #        + B_DISTANCE21 * distance2\
       #        + B_DISTANCE31 * distance3\
       #        + B_DISTANCE41 * distance4\
       + B_DISTANCE52 * distance5 \
       #        + B_DEPARTTIME11 * morning\
       #        + B_DEPARTTIME21 * evening\
       #        + B_DEPARTTIME31 * normal\
       #        + B_DEPARTTIME42 * late\
       )
# V2 = (ASC_2\
#       + B_TRIP_TIME * B1triptime\
#       + B_WAIT_TIME * B1wtimeTaxi\
#       + B_PRICE * B1priceTaxi\
# #       + B_SAME_CHOICE * first_taxi\
#       + B_TAXI * Al_taxi\
#       + B_AGE2 * age2\
#       + B_AGE3 * age3
#       )
# V3 = (ASC_3\
#       + B_BUSORRAIL * C1busrail\
#       + B_TRIP_TIME * C1triptimePT\
#       + B_WAIT_TIME * C1wtimePT\
#       + B_WALK_DISTANCE * C1distancePT_walk\
#       + B_PRICE * C1pricePT\
# #       + B_SAME_CHOICE * first_pt\
#       + B_TRAVEL_DISTANCE_WORK * DistanceWorkday\
#       + B_PT * Al_PT\
#       + B_AGE1 * age1
#       )
V42 = (0 \
       + ASC_42 \
       + B_RAIL_TIME2 * M1ttimerail \
       #        + B_RAIL_RATIO1 * M1metroratio\
       + B_TRIP_TIME2 * M1triptime \
       #       + B_WAIT_TIME1 * M1wtime\
       #       + B_WALK_DISTANCE1 * M1distance_walk\
       #       + B_PRICE1 * M1price\
       + B_FIRST_PT2 * first_pt \
       #       + B_DIF_TIME_M12_2 * M1dif_triptime \
       #       + B_DIF_PRICE_M12_1 * (M1dif_price * first_pt)\
       #       + B_PURPOSE1 * purpose\
       #       + B_TRAVEL_DISTANCE_WORK * DistanceWorkday\
       #       + B_SHAREBIKE * Al_sharedbike\
       #       + B_BIKE * Al_bike\
       #       + B_AGE1 * age1\
       #       + B_FIRST_CAR1 * first_car\
       #       + B_FIRST_TAXI * first_taxi\
       #        + B_DEPARTTIME11 * morning\
       #        + B_DEPARTTIME22 * evening\
       + B_DEPARTTIME32 * normal \
       #        + B_DEPARTTIME41 * late\
       #        + B_DISTANCE11 * distance1\
       #        + B_DISTANCE21 * distance2\
       #        + B_DISTANCE31 * distance3\
       #        + B_DISTANCE41 * distance4\
       #        + B_DISTANCE51 * distance5\
       )
V52 = (0 \
       + ASC_52 \
       + B_RAIL_TIME2 * M2ttime_rail \
       #        + B_RAIL_RATIO1 * M2metroratio\
       + B_TRIP_TIME2 * M2triptime \
       #       + B_WAIT_TIME1 * M2wtime\
       #       + B_BIKE_DISTANCE1 * M2distance_bike\
       #       + B_PRICE1 * M2price\
       + B_FIRST_PT2 * first_pt \
       #       + B_DIF_TIME_M12_2 * M2dif_triptime \
       #       + B_DIF_PRICE_M12_1 * (M2dif_price * first_pt)\
       #       + B_PURPOSE1 * purpose\
       #       + B_TRAVEL_DISTANCE_WORK * DistanceWorkday\
       #       + B_SHAREBIKE * Al_sharedbike\
       #       + B_BIKE * Al_bike\
       #       + B_AGE1 * age1\
       #       + B_FIRST_CAR1 * first_car\
       #        + B_DEPARTTIME12 * morning\
       #        + B_DEPARTTIME22 * evening\
       + B_DEPARTTIME32 * normal \
       #        + B_DEPARTTIME41 * late\
       #       + B_FIRST_TAXI * first_taxi\
       #        + B_DISTANCE11 * distance1\
       #        + B_DISTANCE21 * distance2\
       #        + B_DISTANCE31 * distance3\
       #        + B_DISTANCE41 * distance4\
       #        + B_DISTANCE51 * distance5\
       )
V62 = (ASC_62 \
       + B_RAIL_TIME2 * M3ttime_rail \
       #       + B_RAIL_RATIO1 * M3metroratio\
       #       + B_TAXI_TIME1 * M3ttime_taxi\
       + B_TRIP_TIME2 * M3triptime \
       #       + B_WAIT_TIME1 * M3wtime\
       #       + B_PRICE1 * M3price\
       #       + B_FIRST_PT1 * first_pt\
       + B_FIRST_TAXI2 * first_taxi \
       #       + B_DIF_TIME_M12_2 * M3dif_triptime\
       #       + B_DIF_PRICE_M12_1 * (M3dif_price * (first_pt+first_taxi))\
       #       + B_PURPOSE1 * purpose\
       #       + B_TRAVEL_DISTANCE_WORK * DistanceWorkday\
       #       + B_SHAREBIKE * Al_sharedbike\
       #       + B_BIKE * Al_bike\
       #       + B_AGE1 * age1\
       #       + B_FIRST_CAR1 * first_car\
       #       + B_FIRST_TAXI1 * first_taxi\
       #        + B_DEPARTTIME11 * morning\
       #        + B_DEPARTTIME21 * evening\
       + B_DEPARTTIME32 * normal \
       #        + B_DEPARTTIME41 * late\
       #        + B_DISTANCE11 * distance1\
       #        + B_DISTANCE21 * distance2\
       #        + B_DISTANCE31 * distance3\
       #        + B_DISTANCE41 * distance4\
       #        + B_DISTANCE51 * distance5\
       )
V72 = (0 \
       + ASC_72 \
       + B_TRIP_TIME2 * M4ttime \
       #       + B_PRICE1 * M4price\
       #       + B_DIF_TIME_M4_1 * M4dif_triptime\
       #       + B_DIF_PRICE_M4_1 * M4dif_price\
       #       + B_PURPOSE1 * purpose\
       #       + B_TRAVEL_DISTANCE_WORK * DistanceWorkday\
       #       + B_SHARECAR * Al_shareedcar\
       #       + B_AGE1 * age1\
       #       + B_FIRST_PT * first_pt\
       #       + B_FIRST_CAR * first_car\
       + B_FIRST_TAXI2 * first_taxi \
       #       + B_FIRST_CAR1 * first_car\
       #        + B_DEPARTTIME11 * morning\
       #        + B_DEPARTTIME21 * evening\
       #        + B_DEPARTTIME31 * normal\
       #        + B_DEPARTTIME41 * late\
       #        + B_DISTANCE11 * distance1\
       #        + B_DISTANCE21 * distance2\
       #        + B_DISTANCE31 * distance3\
       #        + B_DISTANCE41 * distance4\
       + B_DISTANCE52 * distance5 \
       )

# Associate utility functions with the numbering of alternatives
V2 = {
    1: V12,
#     2: V2,
#     3: V3,
    4: V42,
    5: V52,
    6: V62,
    7: V72
}


av = {1: 1
#       ,2: 1
#      ,3: PTav
     ,4: 1
     ,5: 1
     ,6: 1 
     ,7: 1}

prob1 = models.logit(V1, av, maas)
prob2 = models.logit(V2, av, maas)


condprob = probClass1 * prob1 + probClass2 * prob2

condlike = (
    P_at8
    * P_at9
    * P_at10
    * P_at11
    # * P_at13
    * P_at14
    * P_at17
    # P_at21
    # *P_at22
    # *P_at23
    # *P_at24
    # *P_at25
    # P_at18
    # * P_at19
    # * P_at12
    # *P_at15
    # *P_at2
    # *P_at3
    # P_at1
    # *P_at4
    # *P_at6
    * condprob
)

loglike = log(Integrate(condlike * density, 'omega'))
# Define level of verbosity
logger = msg.bioMessage()
# logger.setSilent()
# logger.setWarning()
logger.setGeneral()
# logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, loglike)
biogeme.modelName = 'LC-HCM'

# Estimate the parameters
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(f'Estimated betas: {len(results.data.betaValues)}')
print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')
results.writeLaTeX()
print(f'LaTeX file: {results.data.latexFileName}')


# In[ ]:




