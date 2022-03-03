# %%
import numpy as np
import pandas as pd
import itertools
from datetime import datetime, timedelta

def filtering(df):
    '''
    전체 Organism에 대한 전처리 함수
    '''
    organism_list = ['staphylococcus aureus', 'enterococcus faecium', 'enterococcus faecalis', 'acinetobacter baumannii', 'pseudomonas aeruginosa', 'escherichia coli', 'klebsiella pneumoniae']

    df['anti_susc_month'] = pd.to_datetime(df['anti_susc_month'])
    df['anti_susc_month'] = df['anti_susc_month'].apply(lambda x:x.strftime('%Y-%m')[-2:])

    time_table = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    total_count_dict = dict.fromkeys(time_table, 0)
    total_unique_count_dict = dict.fromkeys(time_table, 0)
    anti_count_dict = dict.fromkeys(time_table, 0)

    total_unique_count_list = []

    for organism in organism_list:

        ## 특정 Organism에 해당하는 내성균 List
        if 'staphylococcus aureus' == organism:
            resistants = ['oxacillin', 'cefoxitin', 'cefoxitin screen', 'cefoxitin screen +']
        elif 'enterococcus faecium' == organism or 'enterococcus faecalis' == organism:
            resistants = ['vancomycin']
        elif 'acinetobacter baumannii' == organism or 'pseudomonas aeruginosa' == organism:
            resistants = ['imipenem', 'meropenem']
        elif 'escherichia coli' == organism or 'klebsiella pneumoniae' == organism:
            resistants = ['beta lactamase']
        else:
            continue
        
        # Organism에 해당하는 내성균이 R과 I인 행 Count
        for _, row in df.iterrows():
            processed_row = row[resistants][np.logical_not(pd.isna(row[resistants]))] # NaN값 처리
            
            # 내성균이 모두 Nan값을 가지는 경우
            if len(processed_row) == 0:
                continue
            else:
                # 총 분리 수
                total_count_dict[row['anti_susc_month']] += 1

                # 총 분리가 발생한 환자 수 (총 분리 수의 Unique한 환자)
                person_month_pair = (row['person_id'], row['anti_susc_month'])
                if person_month_pair not in total_unique_count_list:
                    total_unique_count_dict[row['anti_susc_month']] += 1
                    total_unique_count_list.append(person_month_pair)

            # 내성균이 r과 i값을 가지는 경우 (총 감염 발생 신환 수)
            if ('r' or 'i') in list(itertools.chain(*[eval(x) for x in processed_row])):
                anti_count_dict[row['anti_susc_month']] += 1
                
        # break

    return list(total_count_dict.values()), list(total_unique_count_dict.values()), list(anti_count_dict.values())


def filtering_V2(df, organism):
    '''
    특정 Organism에 대한 전처리 함수
    '''    
    ## 특정 Organism에 해당하는 내성균 List
    if 'staphylococcus aureus' == organism:
        resistants = ['oxacillin', 'cefoxitin', 'cefoxitin screen', 'cefoxitin screen +']
    elif 'enterococcus faecium / enterococcus faecalis' == organism:
        resistants = ['vancomycin']
    elif 'acinetobacter baumannii' == organism or 'pseudomonas aeruginosa' == organism:
        resistants = ['imipenem', 'meropenem']
    elif 'escherichia coli' == organism or 'klebsiella pneumoniae' == organism:
        resistants = ['beta lactamase']
    else:
        print('No Matching Organism')
 
    df['anti_susc_month'] = pd.to_datetime(df['anti_susc_month'])
    df['anti_susc_month'] = df['anti_susc_month'].apply(lambda x:x.strftime('%Y-%m')[-2:])

    # 연도 & 월별 Count Dictionary
    time_table = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    total_count_dict = dict.fromkeys(time_table, 0)
    total_unique_count_dict = dict.fromkeys(time_table, 0)
    anti_count_dict = dict.fromkeys(time_table, 0)

    total_unique_count_list = []


    # Organism에 해당하는 내성균이 R과 I인 행 Count
    for _, row in df.iterrows():
        processed_row = row[resistants][np.logical_not(pd.isna(row[resistants]))] # NaN값 처리
        
        # 내성균이 모두 Nan값을 가지는 경우
        if len(processed_row) == 0:
            continue
        else:
            # 총 분리 수
            total_count_dict[row['anti_susc_month']] += 1

            # 총 분리가 발생한 환자 수 (총 분리 수의 Unique한 환자)
            person_month_pair = (row['person_id'], row['anti_susc_month'])
            if person_month_pair not in total_unique_count_list:
                total_unique_count_dict[row['anti_susc_month']] += 1
                total_unique_count_list.append(person_month_pair)

        # 내성균이 r과 i값을 가지는 경우 (총 감염 발생 신환 수)
        if ('r' or 'i') in list(itertools.chain(*[eval(x) for x in processed_row])):
            anti_count_dict[row['anti_susc_month']] += 1

    return list(total_count_dict.values()), list(total_unique_count_dict.values()), list(anti_count_dict.values())


def filtering_new_patient(df):
    '''
    전체 Organism에 대한 신환 수 계산 전처리 함수
    '''    

    df['anti_susc_month'] = pd.to_datetime(df['anti_susc_month'])
    df['anti_susc_month'] = df['anti_susc_month'].apply(lambda x:x.strftime('%Y-%m')[-2:])
        
    # 연도 & 월별 Count Dictionary
    time_table = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    total_count_dict = dict.fromkeys(time_table, 0)
    new_patient_count_dict = dict.fromkeys(time_table, 0)

    # 병동 Count Dictionary
    location_table = list(df['inhospital_location'].value_counts().sort_index().index)
    time_by_location_count_dict = dict.fromkeys(list(itertools.product(time_table, location_table)), 0)

    # 조건
    organism_list = ['staphylococcus aureus', 'enterococcus faecium', 'enterococcus faecalis', 'acinetobacter baumannii', 'pseudomonas aeruginosa', 'escherichia coli', 'klebsiella pneumoniae']
    person_list = []
    anti_susc_time_list= []

    for idx, row in df.iterrows():
        total_count_dict[row['anti_susc_month']] += 1
        
        # 해당 person_id가 이전에 48시간 이내 검사에서 음성이 나온 사람일 경우
        if row['person_id'] in person_list:
            if row['anti_susc_time'] in anti_susc_time_list: # 동일 시간에 2개 이상의 균이 나오는 경우 처리
                continue   
            elif row['organism'] in organism_list:   # Organism이 발생했을 경우
                month = str(row['anti_susc_month']).zfill(2)
                location = row['inhospital_location']
                new_patient_count_dict[month] += 1
                time_by_location_count_dict[(month, location)] += 1

        # 48시간 이내 검사에서 균 6종에 대해 음성이 나온 경우 person_id와 검사시간 기록
        if row.inspection_time_from_visit <= 48 and row.organism not in organism_list:
            person_list.append(row.person_id)
            anti_susc_time_list.append(row.anti_susc_time)
        
    return list(total_count_dict.values()), list(new_patient_count_dict.values()), list(time_by_location_count_dict.values()), location_table



def filteringV2_new_patient(df, organism):
    '''
    특정 Organism에 대한 신환 수 계산 전처리 함수
    '''
    # 특정 Organism에 해당하는 내성균 List
    if 'staphylococcus aureus' == organism:
        resistants = ['oxacillin', 'cefoxitin', 'cefoxitin screen', 'cefoxitin screen +']
    elif 'enterococcus faecium / enterococcus faecalis' == organism:
        resistants = ['vancomycin']
        organism = ['enterococcus faecium', 'enterococcus faecalis']
    elif 'acinetobacter baumannii' == organism or 'pseudomonas aeruginosa' == organism:
        resistants = ['imipenem', 'meropenem']
    elif 'escherichia coli' == organism or 'klebsiella pneumoniae' == organism:
        resistants = ['beta lactamase']
    
    if type(organism) != list:
        organism = [organism]

    df['anti_susc_month'] = pd.to_datetime(df['anti_susc_month'])
    df['anti_susc_month'] = df['anti_susc_month'].apply(lambda x:x.strftime('%Y-%m')[-2:])      

    # 연도 & 월별 Count Dictionary
    time_table = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    total_count_dict = dict.fromkeys(time_table, 0)
    new_patient_count_dict = dict.fromkeys(time_table, 0)

    # 병동 Count Dictionary
    # location_table = list(df['inhospital_location'].value_counts().sort_index().index)
    location_table = ["52W", "53W", "54W", "55W", "57W", "61W", "62W", "63W", "64W", "65W", "66W", "71W", "72W", "73W", "74W", "75W", "76W", "78W8W", "81W", "82W", "83W", "84W", "85W", "86W", "88W", "CCU", "CIC", "DER", "EICU", "LAF", "MIC", "OBS", "SIC"]
    time_by_location_count_dict = dict.fromkeys(list(itertools.product(time_table, location_table)), 0)

    person_list = []
    anti_susc_time_list= []

    for idx, row in df.iterrows():
        total_count_dict[row['anti_susc_month']] += 1   # 입원 환자 수
        
        # 해당 person_id가 이전에 48시간 이내 검사에서 음성이 나온 사람일 경우
        if row['person_id'] in person_list:
            if row['anti_susc_time'] in anti_susc_time_list:    # 동일 시간에 2개 이상의 균이 나오는 경우 처리
                continue
            elif row['organism'] == organism:     # 발생했을 경우 
                month = str(row['anti_susc_month']).zfill(2)
                location = row['inhospital_location']
                new_patient_count_dict[month] += 1
                time_by_location_count_dict[(month, location)] += 1

        # 48시간 이내 검사에서 특정균(organism)에 대해 음성이 나온 경우 person_id와 검사시간 기록
        if (row.inspection_time_from_visit <= 48) and (row.organism not in organism):
            person_list.append(row.person_id)
            anti_susc_time_list.append(row.anti_susc_time)
                        
    return list(total_count_dict.values()), list(new_patient_count_dict.values()), list(time_by_location_count_dict.values()), location_table