import pandas as pd
def create_df_from_json(evaluation):
    for i,el in enumerate(evaluation.turn):
        if i == 0:
            dictionary = {key : [] for key in el[0].keys()}
        for line in el:
            for key in line.keys():
                if key != 'number':
                    dictionary[key].append(line[key])
                else:
                    dictionary[key].append(str(evaluation.number.iloc[i])+'_'+str(line[key]))
    return pd.DataFrame.from_dict(dictionary)

def load_trec_topics(trec_path, years = [2019,2020]):
    if trec_path[-1] != '/':
        trec_path+='/'
    if 2019 in years : 
        _2019 = pd.read_csv(f'{trec_path}evaluation2019.csv')
        manual_2019 = pd.read_csv(f'{trec_path}treccast/test_manual_utterance.tsv', sep = '\t', names=['qid','manual_rewritten_utterance'])
        _2019 = _2019.merge(manual_2019,on='qid')
    if 2020 in years :
        _2020 = create_df_from_json(pd.read_json(f'{trec_path}treccast/2020_manual_evaluation_topics_v1.0.json'))
        _2020 = _2020.rename(columns={'number':'qid'})
        _2020.insert(1,'conv_id',_2020['qid'].apply(lambda x: int(x.split('_')[0])))
        _2020.insert(2,'turn',_2020['qid'].apply(lambda x: int(x.split('_')[1])))
    if 2021 in years :
        _2021 = create_df_from_json(pd.read_json(f'{trec_path}treccast/2021_manual_evaluation_topics_v1.0.json'))
        _2021 = _2021.rename(columns={'number':'qid'})
        _2021.insert(1,'conv_id',_2021['qid'].apply(lambda x: int(x.split('_')[0])))
        _2021.insert(2,'turn',_2021['qid'].apply(lambda x:int( x.split('_')[1])))
    if 2019 in years and 2020 in years and 2021 in years:
        return _2019,_2020,_2021
    elif 2019 in years and 2020 in years:
        return _2019,_2020
    elif 2020 in years and 2021 in years:
        return _2020,_2021
    elif 2019 in years and 2021 in years:
        return _2019,_2021
    elif 2019 in years:
        return _2019
    elif 2020 in years:
        return _2020
    elif 2021 in years:
        return _2021 
