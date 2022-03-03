from numpy.lib.type_check import real
import pandas as pd
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.db import connections
import simplejson as json
import random
from .processing import filtering, filtering_V2, filtering_new_patient, filteringV2_new_patient
from .infection import *

def null_index(request):
    return render(request, "null_index.html")


def resistant(request):
	example = 'hello AJAX'
	context = {"hello":  example}

	return HttpResponse(json.dumps(context), content_type="applications/json")


####################################################################### INDEX A ####################################################################################


def indexA(request):
	'''
	Get total years from database
	'''

	df = pd.read_csv('data/db1.csv')
	df = df.loc[:,~df.columns.duplicated()]    # 중복컬럼 제거
	total_count, total_unique_count, anti_count = filtering(df)
	anti_rate = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count, total_count)]


	########## 1000 재원 환자당 발생률 & 신환 발생 분포도 (병실 시각화, df2 & df3 merge)
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거


	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])

	hospitalization_count, new_patient_count, time_by_location_count_dict, location_table = filtering_new_patient(df_merge)
	new_patient_rate = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count, hospitalization_count)]

	ctx = {"total": total_count , 'total_unique':total_unique_count, "anti": anti_count, "rate": anti_rate, 'hospitalization':hospitalization_count, 'new_patient':new_patient_count, 'new_patient_rate':new_patient_rate,
			'location_table':location_table, 'location_count_values':time_by_location_count_dict, 
			"selected_year": "2002-2018", "selected_organism": "All organism"}

	return render(request, "www/indexA.html", ctx)


def get_choiceA(request):
	'''
	Get choiced year and organism from database
	'''
	year = request.GET.get('year')
	organism = request.GET.get('organism')
	if organism == 'enterococcus faecium / enterococcus faecalis' : organism_query = ['enterococcus faecium', 'enterococcus faecalis']
	else: organism_query = [f'{organism}']

	#총 분리수, 내성균 분리수
	df = pd.read_csv('data/db1.csv')
	df = df.loc[:, ~df.columns.duplicated()]    # 중복컬럼 제거

	## 내성균 조건
	df = df[df.organism.isin(organism_query)]
 
	## 연도 조건
	df['year'] = df['date'].apply(lambda x:x[:4])
	df = df[df['year'] == year]

	total_count, total_unique_count, anti_count = filtering_V2(df, organism=organism)
	anti_rate = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count, total_count)]


	########## 1000 재원 환자당 발생률 ###########
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거
 
	## 내성균 조건
	df2 = df2[df2.organism.isin(organism_query)]
 
	## 연도 조건
	df2['year'] = df2['date'].apply(lambda x:x[:4])
	df2 = df2[df2['year'] == year]


	# Visit occurence table
	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])

	hospitalization_count, new_patient_count, time_by_location_count, location_table = filteringV2_new_patient(df_merge, organism=organism)
	new_patient_rate = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count, hospitalization_count)]

	ctx = {"total":total_count, 'total_unique':total_unique_count, "anti": anti_count, "rate": anti_rate, 'hospitalization':hospitalization_count, 
			'new_patient':new_patient_count, 'new_patient_rate':new_patient_rate, "selected_year": year, "selected_organism": organism.title(),
			'location_table':location_table, 'location_count_values':time_by_location_count}
	
	return render(request, "www/indexA.html", ctx)

####################################################################### INDEX A ####################################################################################


####################################################################### INDEX B ####################################################################################

def indexB(request):
	'''
	Get total years from database
	'''
	df = pd.read_csv('data/db1.csv')
	################################# INDEX B #################################
	random.seed(100)
	sample = random.sample(list(range(len(df))), int(len(df)*.7))
	df = df.loc[sample].reset_index()
	################################# INDEX B #################################

	df = df.loc[:,~df.columns.duplicated()]    # 중복컬럼 제거
	total_count, total_unique_count, anti_count = filtering(df)
	anti_rate = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count, total_count)]


	########## 1000 재원 환자당 발생률 & 신환 발생 분포도 (병실 시각화, df2 & df3 merge)
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거

	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])
	################################# INDEX B #################################
	random.seed(100)
	sample = random.sample(list(range(len(df_merge))), int(len(df_merge)*.7)) # Sample 70% from 2002~2018
	df_merge = df_merge.loc[sample].reset_index()
	################################# INDEX B #################################


	hospitalization_count, new_patient_count, time_by_location_count_dict, location_table = filtering_new_patient(df_merge)
	new_patient_rate = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count, hospitalization_count)]

	ctx = {"total": total_count , 'total_unique':total_unique_count, "anti": anti_count, "rate": anti_rate, 'hospitalization':hospitalization_count, 'new_patient':new_patient_count, 'new_patient_rate':new_patient_rate,
			'location_table':location_table, 'location_count_values':time_by_location_count_dict, 
			"selected_year": "2002-2018", "selected_organism": "All organism"}
	return render(request, "www/indexB.html", ctx)



def get_choiceB(request):
	'''
	Get choiced year and organism from database
	'''
	year = request.GET.get('year')
	organism = request.GET.get('organism')
	if organism == 'enterococcus faecium / enterococcus faecalis' : organism_query = ['enterococcus faecium', 'enterococcus faecalis']
	else: organism_query = [f'{organism}']

	#총 분리수, 내성균 분리수
	df = pd.read_csv('data/db1.csv')
	################################# INDEX B #################################
	random.seed(100)
	sample = random.sample(list(range(len(df))), int(len(df)*.7))
	df = df.loc[sample].reset_index()
	################################# INDEX B #################################

	df = df.loc[:, ~df.columns.duplicated()]    # 중복컬럼 제거

	## 내성균 조건 (선택한 특정 Organism)
	df = df[df.organism.isin(organism_query)]

	## 연도 조건
	df['year'] = df['date'].apply(lambda x:x[:4])
	df = df[df['year'] == year]

	total_count, total_unique_count, anti_count = filtering_V2(df, organism=organism)
	anti_rate = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count, total_count)]


	########## 1000 재원 환자당 발생률 ###########
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거
 
	## 내성균 조건
	df2 = df2[df2.organism.isin(organism_query)]
 
	## 연도 조건
	df2['year'] = df2['date'].apply(lambda x:x[:4])
	df2 = df2[df2['year'] == year]


	# Visit occurence table
	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])
	################################# INDEX B #################################
	random.seed(100)
	sample = random.sample(list(range(len(df_merge))), int(len(df_merge)*.7)) # Sample 70% from 2002~2018
	df_merge = df_merge.loc[sample].reset_index()
	################################# INDEX B #################################

	hospitalization_count, new_patient_count, time_by_location_count, location_table = filteringV2_new_patient(df_merge, organism=organism)
	new_patient_rate = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count, hospitalization_count)]

	ctx = {"total":total_count, 'total_unique':total_unique_count, "anti": anti_count, "rate": anti_rate, 'hospitalization':hospitalization_count, 
			'new_patient':new_patient_count, 'new_patient_rate':new_patient_rate, "selected_year": year, "selected_organism": organism.title(),
			'location_table':location_table, 'location_count_values':time_by_location_count}
	
	return render(request, "www/indexB.html", ctx)

####################################################################### INDEX B ####################################################################################

####################################################################### INDEX C ####################################################################################

def indexC(request):
	'''
	Get total years from database
	'''

	df = pd.read_csv('data/db1.csv')
	df = df.loc[:,~df.columns.duplicated()]    # 중복컬럼 제거

	################################# INDEX C #################################
	df['year'] = df['date'].apply(lambda x:x[:4])
	df = df[df['year'] >= '2008'].reset_index(drop=True)
	random.seed(100)
	sample = random.sample(list(range(len(df))), int(len(df)*.7))
	df = df.loc[sample].reset_index()
	################################# INDEX C #################################

	total_count, total_unique_count, anti_count = filtering(df)
	anti_rate = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count, total_count)]

	########## 1000 재원 환자당 발생률 & 신환 발생 분포도 (병실 시각화, df2 & df3 merge)
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거

	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])

	################################# INDEX C #################################
	df_merge['year'] = df_merge['date'].apply(lambda x:x[:4])
	df_merge = df_merge[df_merge['year'] >= '2008'].reset_index(drop=True)
	random.seed(100)
	sample   = random.sample(list(range(len(df_merge))), int(len(df_merge)*.7))
	df_merge = df_merge.loc[sample].reset_index()
	################################# INDEX C #################################  

	hospitalization_count, new_patient_count, time_by_location_count_dict, location_table = filtering_new_patient(df_merge)
	new_patient_rate = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count, hospitalization_count)]

	ctx = {"total": total_count , 'total_unique':total_unique_count, "anti": anti_count, "rate": anti_rate, 'hospitalization':hospitalization_count, 'new_patient':new_patient_count, 'new_patient_rate':new_patient_rate,
			'location_table':location_table, 'location_count_values':time_by_location_count_dict, 
			"selected_year": "2008-2017", "selected_organism": "All organism"}

	return render(request, "www/indexC.html", ctx)

def get_choiceC(request):
	'''
	Get choiced year and organism from database
	'''
	year = request.GET.get('year')
	organism = request.GET.get('organism')
	if organism == 'enterococcus faecium / enterococcus faecalis' : organism_query = ['enterococcus faecium', 'enterococcus faecalis']
	else: organism_query = [f'{organism}']

	#총 분리수, 내성균 분리수
	df = pd.read_csv('data/db1.csv')
	################################# INDEX C #################################
	df['year'] = df['date'].apply(lambda x:x[:4])
	df = df[df['year'] >= '2008'].reset_index(drop=True)
	random.seed(100)
	sample = random.sample(list(range(len(df))), int(len(df)*.7))
	df = df.loc[sample].reset_index()
	################################# INDEX C #################################

	## 내성균 조건 (선택한 특정 Organism)
	df = df[df.organism.isin(organism_query)]

	total_count, total_unique_count, anti_count = filtering_V2(df, organism=organism)
	anti_rate = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count, total_count)]



	########## 1000 재원 환자당 발생률 ###########
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거
	## 내성균 조건
	df2 = df2[df2.organism.isin(organism_query)]
	## 연도 조건
	df2['year'] = df2['date'].apply(lambda x:x[:4])
	df2 = df2[df2['year'] == year]


	# Visit occurence table
	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])
	################################# INDEX C #################################
	df_merge['year'] = df_merge['date'].apply(lambda x:x[:4])
	df_merge = df_merge[df_merge['year'] >= '2008'].reset_index(drop=True)
	random.seed(100)
	sample = random.sample(list(range(len(df_merge))), int(len(df_merge)*.7))
	df_merge = df_merge.loc[sample].reset_index()
	################################# INDEX C #################################

	hospitalization_count, new_patient_count, time_by_location_count, location_table = filteringV2_new_patient(df_merge, organism=organism)
	new_patient_rate = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count, hospitalization_count)]

	ctx = {"total":total_count, 'total_unique':total_unique_count, "anti": anti_count, "rate": anti_rate, 'hospitalization':hospitalization_count, 
			'new_patient':new_patient_count, 'new_patient_rate':new_patient_rate, "selected_year": year, "selected_organism": organism.title(),
			'location_table':location_table, 'location_count_values':time_by_location_count}
	
	return render(request, "www/indexC.html", ctx)

####################################################################### INDEX C ####################################################################################


####################################################################### INDEX D ####################################################################################
def indexD(request):
    data = pd.read_csv('data/infection_data.csv')[['date','organism','감염위험도']]  # date와 감염위험도 Column만 사용
    data.columns = ['date', 'organism', 'infection_risk']
    data['date'] = pd.to_datetime(data['date'])
    data = data.drop([0,1,2,3,4,5], axis=0).set_index('date')            # 6개 organism에 대한 첫 달의 감염위험도는 -1이므로 제거
    
    ctx = {}
    organism_list = ['staphylococcus aureus', 'enterococcus faecium / enterococcus faecalis', 'acinetobacter baumannii', 
                 'pseudomonas aeruginosa', 'escherichia coli', 'klebsiella pneumoniae']
    
    run_opt = request.GET.get('run_opt')
    # Train New ARIMA model
    if run_opt == 'train':
        model_dict = {}
        for i, organism in enumerate(tqdm(organism_list)):
            # 1. Train/Test 분리 및 시계열 분해
            train_data, test_data = data_split(data, organism=organism_list[i])
            observed, trend, seasonal, resid = get_decomposition(train_data, visualize=False)
            
            # 2. Train ARIMA
            model = arima_train(train_data)
            model = search_best_arima(train_data, model_type='sarima')
            
            # 3. attach trained model
            model_dict[organism] = model
            
        # 4. save trained model
        with open("model_weight/SARIMA_models_trained.pkl", "wb") as f:
            pickle.dump(model_dict, f)
            
    # Get Inference from ARIMA model
    if run_opt == 'train': # load saved model if opt is no train
        pass
    else: 
        with open("model_weight/SARIMA_models.pkl", "rb") as f:
            model_dict = pickle.load(f)
    
    for i, organism in enumerate(organism_list):
        ctx['predicted_values'+str(i)] = []
        ctx['predicted_lowerbounds'+str(i)] = []
        ctx['predicted_upperbounds'+str(i)] = []
        ctx['real_values'+str(i)] = []
        
        # 0. organism 모델 선택
        model = model_dict[organism]

        # 1. Train/Test 분리 및 시계열 분해
        train_data, test_data = data_split(data, organism=organism_list[i])
        observed, trend, seasonal, resid = get_decomposition(train_data, visualize=False)
        
        # 2. Infer Infection Risk with the model
        pred_value, pred_upper, pred_lower, pred_index = arima_predict(model, test_data, train_data, visualize=False, extension=12)
        test_r2 = r2_score(test_data['infection_risk'].values, pred_value[:len(test_data)].values)
        test_mae = (np.abs(test_data['infection_risk'].values - pred_value[:len(test_data)].values)).mean()
        
        real_data = pd.concat([train_data, test_data])
        real_index_year = [i.date().year for i in list(real_data.index)]
        real_index_month = [i.date().month for i in list(real_data.index)]
        pred_index_year = [i.date().year for i in pred_index]
        pred_index_month = [i.date().month for i in pred_index]
        total_index_year = [i.date().year for i in sorted(list(set(list(real_data.index) + list(pred_index))))]
        total_index_month = [i.date().month for i in sorted(list(set(list(real_data.index) + list(pred_index))))]
        
                
        ctx['predicted_values'+str(i)] = [-515] * (12 * 8) + list(np.array(pred_value).squeeze())
        ctx['predicted_upperbounds'+str(i)] = [-515] * (12 * 8) + list(np.array(pred_upper).squeeze())
        ctx['predicted_lowerbounds'+str(i)] = [-515] * (12 * 8) + list(np.array(pred_lower).squeeze())
        
        ctx['real_values'+str(i)] = list(np.array(real_data.values).squeeze())
        
        ctx['observed'+str(i)] = list(np.array(observed.values).squeeze())
        ctx['trend'+str(i)] = list(np.array(trend.values).squeeze())
        ctx['seasonal'+str(i)] = list(np.array(seasonal.values).squeeze())
        ctx['resid'+str(i)] = list(np.array(resid.values).squeeze())
        ctx['r2_score'+str(i)] = round(test_r2, 2)
        ctx['mae_score'+str(i)] = round(test_mae, 2)
        
                
    # ctx['organism_list'] = organism_list
    for key in ctx.keys():
        ctx[key] = str(ctx[key]).replace("'", '"')
        if 'predicted_' in key:
            ctx[key] = ctx[key].replace('-515', 'null')
        if 'observed' in key or 'trend' in key or 'seasonal' in key or 'resid' in key:
            ctx[key] = ctx[key].replace('nan', 'null')
            
    return render(request, "www/indexD.html", ctx)    
        
      
        
####################################################################### INDEX E ####################################################################################

def indexE(request):
	ctx = {"selected_year": "2002-2018", "selected_organism": "All organism"}
	'''
	Get total years from database -- indexB
	'''
	df = pd.read_csv('data/db1.csv')
	df = df.loc[:,~df.columns.duplicated()]    # 중복컬럼 제거
	total_count_A, total_unique_count_A, anti_count_A = filtering(df)
	anti_rate_A = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count_A, total_count_A)]


	########## 1000 재원 환자당 발생률 & 신환 발생 분포도 (병실 시각화, df2 & df3 merge)
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거


	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])

	hospitalization_count_A, new_patient_count_A, time_by_location_count_dict_A, location_table_A = filtering_new_patient(df_merge)
	new_patient_rate_A = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count_A, hospitalization_count_A)]
 
	non_anti_count_A = [i - j for i, j in zip(total_count_A, anti_count_A)]
	ctx["non_anti_A"] = non_anti_count_A	
 
	ctx["total_A"] = total_count_A
	ctx["anti_A"]  = anti_count_A
	ctx["rate_A"]  = anti_rate_A
	ctx["new_patient_A"] = new_patient_count_A
	ctx["new_patient_rate_A"] = new_patient_rate_A
	
	print('indexE, A done')

	'''
	Get total years from database -- indexB
	'''
	df = pd.read_csv('data/db1.csv')
	################################# INDEX B #################################
	random.seed(100)
	sample = random.sample(list(range(len(df))), int(len(df)*.7))
	df = df.loc[sample].reset_index()
	################################# INDEX B #################################

	df = df.loc[:,~df.columns.duplicated()]    # 중복컬럼 제거
	total_count_B, total_unique_count_B, anti_count_B = filtering(df)
	anti_rate_B = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count_B, total_count_B)]


	########## 1000 재원 환자당 발생률 & 신환 발생 분포도 (병실 시각화, df2 & df3 merge)
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거

	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])
	################################# INDEX B #################################
	random.seed(100)
	sample = random.sample(list(range(len(df_merge))), int(len(df_merge)*.7)) # Sample 70% from 2002~2018
	df_merge = df_merge.loc[sample].reset_index()
	################################# INDEX B #################################
 
	hospitalization_count_B, new_patient_count_B, time_by_location_count_dict_B, location_table_B = filtering_new_patient(df_merge)
	new_patient_rate_B = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count_B, hospitalization_count_B)]

	non_anti_count_B = [i - j for i, j in zip(total_count_B, anti_count_B)]
	ctx["non_anti_B"] = non_anti_count_B

	ctx["total_B"] = total_count_B
	ctx["anti_B"]  = anti_count_B
	ctx["rate_B"]  = anti_rate_B
	ctx["new_patient_B"] = new_patient_count_B
	ctx["new_patient_rate_B"] = new_patient_rate_B
 
	print('indexE, B done')
 
	'''
	Get total years from database -- indexC
	'''
	df = pd.read_csv('data/db1.csv')
	df = df.loc[:,~df.columns.duplicated()]    # 중복컬럼 제거
	################################# INDEX C #################################
	df['year'] = df['date'].apply(lambda x:x[:4])
	df = df[df['year'] >= '2008'].reset_index(drop=True)
	random.seed(100)
	sample = random.sample(list(range(len(df))), int(len(df)*.7))
	df = df.loc[sample].reset_index()
	################################# INDEX C #################################

	total_count_C, total_unique_count_C, anti_count_C = filtering(df)
	anti_rate_C = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count_C, total_count_C)]

	########## 1000 재원 환자당 발생률 & 신환 발생 분포도 (병실 시각화, df2 & df3 merge)
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거

	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])

	################################# INDEX C #################################
	df_merge['year'] = df_merge['date'].apply(lambda x:x[:4])
	df_merge = df_merge[df_merge['year'] >= '2008'].reset_index(drop=True)
	random.seed(100)
	sample   = random.sample(list(range(len(df_merge))), int(len(df_merge)*.7))
	df_merge = df_merge.loc[sample].reset_index()
	################################# INDEX C #################################  

	hospitalization_count_C, new_patient_count_C, time_by_location_count_dict_C, location_table_C = filtering_new_patient(df_merge)
	new_patient_rate_C = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count_C, hospitalization_count_C)]
 
	non_anti_count_C = [i - j for i, j in zip(total_count_C, anti_count_C)]
	ctx["non_anti_C"] = non_anti_count_C
 
	ctx["total_C"] = total_count_C
	ctx["anti_C"]  = anti_count_C
	ctx["rate_C"]  = anti_rate_C
	ctx["new_patient_C"] = new_patient_count_C
	ctx["new_patient_rate_C"] = new_patient_rate_C

	print('indexE, C done')
 
	return render(request, "www/indexE.html", ctx)


def get_choiceE(request):
	year = request.GET.get('year')
	organism = request.GET.get('organism')
	if organism == 'enterococcus faecium / enterococcus faecalis' : organism_query = ['enterococcus faecium', 'enterococcus faecalis']
	else: organism_query = [f'{organism}']
     
	ctx = {"selected_year": year, "selected_organism": organism.title()}
 
	'''
	Get choiced year and organism from database -- index A
	'''

	#총 분리수, 내성균 분리수
	df = pd.read_csv('data/db1.csv')
	df = df.loc[:, ~df.columns.duplicated()]    # 중복컬럼 제거

	## 내성균 조건
	df = df[df.organism.isin(organism_query)]
 
	## 연도 조건
	df['year'] = df['date'].apply(lambda x:x[:4])
	df = df[df['year'] == year]

	total_count_A, total_unique_count_A, anti_count_A = filtering_V2(df, organism=organism)
	anti_rate_A = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count_A, total_count_A)]


	########## 1000 재원 환자당 발생률 ###########
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거
 
	## 내성균 조건
	df2 = df2[df2.organism.isin(organism_query)]
 
	## 연도 조건
	df2['year'] = df2['date'].apply(lambda x:x[:4])
	df2 = df2[df2['year'] == year]


	# Visit occurence table
	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])

	hospitalization_count_A, new_patient_count_A, time_by_location_count_A, location_table_A = filteringV2_new_patient(df_merge, organism=organism)
	new_patient_rate_A = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count_A, hospitalization_count_A)]

	non_anti_count_A = [i - j for i, j in zip(total_count_A, anti_count_A)]
	ctx["non_anti_A"] = non_anti_count_A

	ctx["total_A"] = total_count_A
	ctx["anti_A"]  = anti_count_A
	ctx["rate_A"]  = anti_rate_A
	ctx["new_patient_A"] = new_patient_count_A
	ctx["new_patient_rate_A"] = new_patient_rate_A

	print('getChoiceA done')
 
	'''
	Get choiced year and organism from database -- index B
	'''

	#총 분리수, 내성균 분리수
	df = pd.read_csv('data/db1.csv')
	################################# INDEX B #################################
	random.seed(100)
	sample = random.sample(list(range(len(df))), int(len(df)*.7))
	df = df.loc[sample].reset_index()
	################################# INDEX B #################################

	df = df.loc[:, ~df.columns.duplicated()]    # 중복컬럼 제거

	## 내성균 조건 (선택한 특정 Organism)
	df = df[df.organism.isin(organism_query)]

	## 연도 조건
	df['year'] = df['date'].apply(lambda x:x[:4])
	df = df[df['year'] == year]

	total_count_B, total_unique_count_B, anti_count_B = filtering_V2(df, organism=organism)
	anti_rate_B = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count_B, total_count_B)]


	########## 1000 재원 환자당 발생률 ###########
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거
 
	## 내성균 조건
	df2 = df2[df2.organism.isin(organism_query)]
 
	## 연도 조건
	df2['year'] = df2['date'].apply(lambda x:x[:4])
	df2 = df2[df2['year'] == year]


	# Visit occurence table
	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])
	################################# INDEX B #################################
	random.seed(100)
	sample = random.sample(list(range(len(df_merge))), int(len(df_merge)*.7)) # Sample 70% from 2002~2018
	df_merge = df_merge.loc[sample].reset_index()
	################################# INDEX B #################################

	hospitalization_count_B, new_patient_count_B, time_by_location_count_B, location_table_B = filteringV2_new_patient(df_merge, organism=organism)
	new_patient_rate_B = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count_B, hospitalization_count_B)]
 
	non_anti_count_B = [i - j for i, j in zip(total_count_B, anti_count_B)]
	ctx["non_anti_B"] = non_anti_count_B
	
	ctx["total_B"] = total_count_B
	ctx["anti_B"]  = anti_count_B
	ctx["rate_B"]  = anti_rate_B
	ctx["new_patient_B"] = new_patient_count_B
	ctx["new_patient_rate_B"] = new_patient_rate_B
 
	print('getChoiceB done')
 
	'''
	Get choiced year and organism from database -- indexC
	'''

	#총 분리수, 내성균 분리수
	df = pd.read_csv('data/db1.csv')
	################################# INDEX C #################################
	df['year'] = df['date'].apply(lambda x:x[:4])
	df = df[df['year'] >= '2008'].reset_index(drop=True)
	random.seed(100)
	sample = random.sample(list(range(len(df))), int(len(df)*.7))
	df = df.loc[sample].reset_index()
	################################# INDEX C #################################

	## 내성균 조건 (선택한 특정 Organism)
	df = df[df.organism.isin(organism_query)]

	total_count_C, total_unique_count_C, anti_count_C = filtering_V2(df, organism=organism)
	anti_rate_C = [round(float(anti)/total*100,2) if total!=0 else 0.0 for anti, total in zip(anti_count_C, total_count_C)]



	########## 1000 재원 환자당 발생률 ###########
	df2 = pd.read_csv('data/db2.csv')
	df2 = df2.loc[:,~df2.columns.duplicated()]    # 중복컬럼 제거
	## 내성균 조건
	df2 = df2[df2.organism.isin(organism_query)]
	## 연도 조건
	df2['year'] = df2['date'].apply(lambda x:x[:4])
	df2 = df2[df2['year'] == year]


	# Visit occurence table
	df3 = pd.read_csv('data/db3.csv')
	df3 = df3.drop_duplicates('visit_occurrence_id', keep='first')

	df_merge = df2.merge(df3, on='visit_occurrence_id', how='inner')  # df3의 visit_occ_id가 df2에는 없는 경우가 있음.
	df_merge['inhospital_location'] = df_merge['inhospital_location_source_subunit'].apply(lambda x:x[:-2])
	################################# INDEX C #################################
	df_merge['year'] = df_merge['date'].apply(lambda x:x[:4])
	df_merge = df_merge[df_merge['year'] >= '2008'].reset_index(drop=True)
	random.seed(100)
	sample = random.sample(list(range(len(df_merge))), int(len(df_merge)*.7))
	df_merge = df_merge.loc[sample].reset_index()
	################################# INDEX C #################################

	hospitalization_count_C, new_patient_count_C, time_by_location_count_C, location_table_C = filteringV2_new_patient(df_merge, organism=organism)
	new_patient_rate_C = [round(float(new)/total*1000,2) if total!=0 else 0.0 for new, total in zip(new_patient_count_C, hospitalization_count_C)]
 
	non_anti_count_C = [i - j for i, j in zip(total_count_C, anti_count_C)]
	ctx["non_anti_C"] = non_anti_count_C

	ctx["total_C"] = total_count_C
	ctx["anti_C"]  = anti_count_C
	ctx["rate_C"]  = anti_rate_C
	ctx["new_patient_C"] = new_patient_count_C
	ctx["new_patient_rate_C"] = new_patient_rate_C

	print('getChoiceC done')
  	
	return render(request, "www/indexE.html", ctx)

####################################################################### INDEX E ####################################################################################
