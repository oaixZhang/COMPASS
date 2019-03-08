import pandas as pd
import numpy as np


def preprocess():
    df = pd.read_csv('./data/ADNI_Training_Q1.csv')
    bldata = df[df.VISCODE == 'bl'].copy()
    # gender
    gender = bldata['PTGENDER'].copy()
    gender[bldata.PTGENDER == 'Male'] = 1
    gender[bldata.PTGENDER == 'Female'] = 0
    bldata['PTGENDER'] = gender
    # APOE gene type
    bldata.index = range(0, len(bldata))
    genotype = bldata['APOE Genotype']
    APOE2 = np.zeros(len(bldata), dtype=np.int32)
    for i in range(0, len(bldata)):
        APOE2[i] = genotype[i].count('2')
    bldata['APOE2'] = APOE2
    # MMSE and MMSE24
    mmse = bldata.pop('MMSE')
    mmse24 = df[df.VISCODE == 'm24'].pop('MMSE')
    bldata['MMSE'] = mmse.values
    bldata['MMSE24'] = mmse24.values
    # CN-->1 , MCI-->2 , AD-->3
    bldata['DX_bl'] = bldata['DX_bl'].map({'CN': 1, 'LMCI': 2, 'EMCI': 2, 'AD': 3})
    # drop columns
    bldata.drop(columns=['PTID', 'EXAMDATE', 'VISCODE', 'APOE Genotype'], inplace=True)
    # save
    bldata.to_csv('./data/data_processed.csv', index=0)


# whole data for classification,groups not split
def data4classification():
    df = pd.read_csv('./data/data_processed.csv')
    # delta mmse
    deltaMMSE = df['MMSE24'] - df['MMSE']
    declined = [int(i <= -3) for i in deltaMMSE]
    df['DECLINED'] = declined
    df.drop(columns=['RID', 'MMSE24'], inplace=True)

    return df


# split into CN,MCI,AD groups for classification
def split4classification():
    df = data4classification()
    CN = df[df.DX_bl == 1].copy()
    CN.drop(columns=['DX_bl'], inplace=True)
    CN.to_csv('./data/clf_CN.csv', index=0)

    MCI = df[df.DX_bl == 2].copy()
    MCI.drop(columns=['DX_bl'], inplace=True)
    MCI.to_csv('./data/clf_MCI.csv', index=0)

    AD = df[df.DX_bl == 3].copy()
    AD.drop(columns=['DX_bl'], inplace=True)
    AD.to_csv('./data/clf_AD.csv', index=0)


# whole data for regression, groups not divided
def data4regression():
    df = pd.read_csv('./data/data_processed.csv')
    # delta mmse
    deltaMMSE = df['MMSE24'] - df['MMSE']
    df['deltaMMSE'] = deltaMMSE
    df.drop(columns=['RID', 'MMSE24'], inplace=True)
    return df


# split into CN,MCI,AD groups for regression
def split4regression():
    df = data4regression()
    CN = df[df.DX_bl == 1].copy()
    CN.drop(columns=['DX_bl'], inplace=True)
    CN.to_csv('./data/reg_CN.csv', index=0)

    MCI = df[df.DX_bl == 2].copy()
    MCI.drop(columns=['DX_bl'], inplace=True)
    MCI.to_csv('./data/reg_MCI.csv', index=0)

    AD = df[df.DX_bl == 3].copy()
    AD.drop(columns=['DX_bl'], inplace=True)
    AD.to_csv('./data/reg_AD.csv', index=0)


# ADNI-MEM,ADNI-EF data
def merge_mem_ef_data():
    data = pd.read_csv('./data/data_processed.csv')
    df = pd.read_csv('./data/ADNI_MEM_EF.csv')
    data = pd.merge(data, df, how='left', on='RID')
    data.drop(columns=['RID'], inplace=True)
    data.to_csv('./data/data_with_MEM_EF.csv', index=0)
    return df


# data with MEM and EF
def extra4classification():
    df = pd.read_csv('./data/data_with_MEM_EF.csv')
    # delta mmse
    deltaMMSE = df['MMSE24'] - df['MMSE']
    declined = [int(i <= -3) for i in deltaMMSE]
    df['DECLINED'] = declined
    df.drop(columns=['MMSE24'], inplace=True)
    # split into groups
    CN = df[df.DX_bl == 1].copy()
    CN.drop(columns=['DX_bl'], inplace=True)
    CN.to_csv('./data/clf_CN_extra_data.csv', index=0)
    MCI = df[df.DX_bl == 2].copy()
    MCI.drop(columns=['DX_bl'], inplace=True)
    MCI.to_csv('./data/clf_MCI_extra_data.csv', index=0)
    AD = df[df.DX_bl == 3].copy()
    AD.drop(columns=['DX_bl'], inplace=True)
    AD.to_csv('./data/clf_AD_extra_data.csv', index=0)
    return df


# data with MEM and EF
def extra4regression():
    df = pd.read_csv('./data/data_with_MEM_EF.csv')
    # delta mmse
    deltaMMSE = df['MMSE24'] - df['MMSE']
    df['deltaMMSE'] = deltaMMSE
    df.drop(columns=['MMSE24'], inplace=True)
    # split into groups
    CN = df[df.DX_bl == 1].copy()
    CN.drop(columns=['DX_bl'], inplace=True)
    CN.to_csv('./data/reg_CN_extra_data.csv', index=0)
    MCI = df[df.DX_bl == 2].copy()
    MCI.drop(columns=['DX_bl'], inplace=True)
    MCI.to_csv('./data/reg_MCI_extra_data.csv', index=0)
    AD = df[df.DX_bl == 3].copy()
    AD.drop(columns=['DX_bl'], inplace=True)
    AD.to_csv('./data/reg_AD_extra_data.csv', index=0)
    return df


if __name__ == '__main__':
    preprocess()
    data4classification()
    split4classification()
    data4regression()
    split4regression()
    merge_mem_ef_data()
    extra4classification()
    extra4regression()
