#%%
import pandas as pd
#%%
special_tokens = {'context': '>>INTRODUCTION<<', 'answer': '>>ANSWER<<', 'question': '>>QUESTION<<'}

do_qrecc = True
if do_qrecc:
    ## import qrecc dataset ##
    qrecc_test = pd.read_json('~/llms/data/qrecc/qrecc-test.json')
    qrecc_train = pd.read_json('~/llms/data/qrecc/qrecc-training.json')
    original_context_test = []
    for conv_id in qrecc_test.Conversation_no.unique():
        conversation = qrecc_test[qrecc_test.Conversation_no == conv_id]
        for turn in conversation.Turn_no:
                original_context_test.append(conversation[(1 < turn) & (conversation.Turn_no < turn)].Question.tolist())
    qrecc_test['raw_context'] = original_context_test
    original_context_train = []
    for conv_id in qrecc_train.Conversation_no.unique():
        conversation = qrecc_train[qrecc_train.Conversation_no == conv_id]
        for turn in conversation.Turn_no:
                original_context_train.append(conversation[(1 < turn) & (conversation.Turn_no < turn)].Question.tolist())
    qrecc_train['raw_context'] = original_context_train
    context_list = []
    ### create text_train ###
    text_train = [special_tokens['context'] + " ".join(x[::2]) + special_tokens['question'] + y + special_tokens['answer'] + z   for x,y,z in zip(qrecc_train['Context'].tolist(),qrecc_train['Question'].tolist(),qrecc_train['Rewrite'].tolist())]
    ### create test_train ###
    text_test = [special_tokens['context'] + " ".join(x[::2]) + special_tokens['question'] + y + special_tokens['answer'] + z   for x,y,z in zip(qrecc_test['Context'].tolist(),qrecc_test['Question'].tolist(),qrecc_test['Rewrite'].tolist())]
    train_qrec = pd.DataFrame({'text':text_train})
    test_qrec = pd.DataFrame({'text':text_test})
    train_qrec.to_csv(f'qrecc_train_with_context.csv',index=False)
    test_qrec.to_csv(f'qrecc_test_with_context.csv',index=False)
    ### Create dataframe with the context raw and not the rewritten one ###
    text_train = [special_tokens['context'] + " ".join(x[:]) + special_tokens['question'] + y + special_tokens['answer'] + z   for x,y,z in zip(qrecc_train['raw_context'].tolist(),qrecc_train['Question'].tolist(),qrecc_train['Rewrite'].tolist())]
    text_test = [special_tokens['context'] + " ".join(x[:]) + special_tokens['question'] + y + special_tokens['answer'] + z   for x,y,z in zip(qrecc_test['raw_context'].tolist(),qrecc_test['Question'].tolist(),qrecc_test['Rewrite'].tolist())]
    train_qrec = pd.DataFrame({'text':text_train})
    test_qrec = pd.DataFrame({'text':text_test})
    train_qrec.to_csv(f'qrecc_train_with_raw_context.csv',index=False)
    test_qrec.to_csv(f'qrecc_test_with_raw_context.csv',index=False)

    ###
    text_train = [special_tokens['context'] + " ".join(x[::2]) + special_tokens['question'] + y + special_tokens['answer'] + z + " " + k   for x,y,z,k in zip(qrecc_train['Context'].tolist(),qrecc_train['Question'].tolist(),qrecc_train['Rewrite'].tolist(),qrecc_train['Answer'].tolist())]
    text_test = [special_tokens['context'] + " ".join(x[::2]) + special_tokens['question'] + y + special_tokens['answer'] + z + " " + k   for x,y,z,k in zip(qrecc_test['Context'].tolist(),qrecc_test['Question'].tolist(),qrecc_test['Rewrite'].tolist(),qrecc_test['Answer'].tolist())]
    train_qrec = pd.DataFrame({'text':text_train})
    test_qrec = pd.DataFrame({'text':text_test})
    train_qrec.to_csv(f'qrecc_train_context_and_answer.csv',index=False)
    test_qrec.to_csv(f'qrecc_test_context_and_answer.csv',index=False)
#%%

from huggingface_hub import login
login(token='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')

    
# %%
from methods import *
trec19,trec20,trec21 =  load_trec_topics('/home/guidorocchietti/llms/data/trec/',[2019,2020,2021])
# %%
def extract_trec_for_rewriting(trec_df):
    raw_context = []
    rewritten_context = []
    question = []
    qid = []
    conv_ids = []
    turns = []
    for conv_id in trec_df.conv_id.unique():
        conversation = trec_df[trec_df.conv_id == conv_id]
        for turn in conversation.turn:
            #if turn != 1:
                raw_context.append(conversation[(1 < turn) & (conversation.turn < turn)].raw_utterance.tolist())
                rewritten_context.append(conversation[(1 < turn) & (conversation.turn < turn)].manual_rewritten_utterance.tolist())
                question.append(conversation[conversation.turn == turn].raw_utterance.iloc[0])
                qid.append(str(conv_id)+'_'+str(turn))
                conv_ids.append(conv_id)
                turns.append(turn)
    prompt_raw_context = [special_tokens['context'] + " ".join(x) + special_tokens['question'] + y + special_tokens['answer']   for x,y in zip(raw_context,question)]
    prompt_rewritten_context = [special_tokens['context'] + " ".join(x) + special_tokens['question'] + y + special_tokens['answer']   for x,y in zip(rewritten_context,question)]

    return pd.DataFrame({'qid': qid, 'conv_id': conv_ids, 'turn': turns, 'raw_context': raw_context, 'rewritten_context': rewritten_context, 'question': question, 'prompt_raw_context': prompt_raw_context, 'prompt_rewritten_context': prompt_rewritten_context})

df19 = extract_trec_for_rewriting(trec19)
df20 = extract_trec_for_rewriting(trec20)
df21 = extract_trec_for_rewriting(trec21)
df19.to_csv('/home/guidorocchietti/llms/data/trec/trec19_for_rewriting.csv',index=False)
df20.to_csv('/home/guidorocchietti/llms/data/trec/trec20_for_rewriting.csv',index=False)
df21.to_csv('/home/guidorocchietti/llms/data/trec/trec21_for_rewriting.csv',index=False)
    #group df_19 by conv_id skipping turn == 1
#%%
# %%
## Check if elements in trec19.raw_utterance are in qrecc_test.Question
elements_test_19 = []
elements_test_20 = []
elements_test_21 = []
for x in qrecc_test.Question.tolist():
    for y in trec19.raw_utterance.tolist():
        if x in y or y in x:
            elements_test_19.append(x)
    for y in trec20.raw_utterance.tolist():
        if x in y or y in x:
            elements_test_20.append(x)
    for y in trec21.raw_utterance.tolist():
        if x in y or y in x:
            elements_test_21.append(x)

# %%
elements_train_19 = []
elements_train_20 = []
elements_train_21 = []
for x in qrecc_train.Question.tolist():
    for y in trec19.raw_utterance.tolist():
        if x in y or y in x:
            elements_train_19.append(x)
    for y in trec20.raw_utterance.tolist():
        if x in y or y in x:
            elements_train_20.append(x)
    for y in trec21.raw_utterance.tolist():
        if x in y or y in x:
            elements_train_21.append(x)

# %%
