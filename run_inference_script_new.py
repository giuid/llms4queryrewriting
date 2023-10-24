import os, re, torch
os.environ["CUDA_VISIBLE_DEVICES"]="4,5"
### define data path ###
if 'llms' in os.getcwd():
    data_path = re.sub('llms.*','llms/data/',os.getcwd())
else: 
    data_path = '/home/guidorocchietti/llms/data/'
os.environ['TRANSFORMERS_CACHE'] = f'{data_path}/cache'
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import argparse
import pandas as pd
import re
from tqdm import tqdm

def instantiate_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    return pipeline, tokenizer

def instantiate_pipeline(model_id,temperature = 0.7,max_new_tokens=512,repetition_penalty=1.1 , bnb_config=None):
    #model_config = transformers.AutoConfig.from_pretrained(model_id)
    if bnb_config is not None:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            #config=model_config,
            quantization_config=bnb_config,
            device_map='auto')
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            #config=model_config,
            device_map='auto')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id)
    generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=max_new_tokens,  # mex number of tokens to generate in the output
            repetition_penalty=repetition_penalty  # without this output begins repeating)
            )
    return generate_text, tokenizer , model


def do_rewriting(df, pipeline,tokenizer):#, temperature =0.8): 
    print('Doing rewriting...')
    results = []
    cleaned_results = []
    whole_answer = []
    prompts = df.prompt_rewritten_context.tolist()[:]
    #for prompt in df19[df19.conv_id ==31].prompt_rewritten_context.tolist():
    import time 
    start = time.time()
    sequences = pipeline(
        prompts,
        #max_length=128,
        max_new_tokens = 48,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        )
    end = time.time()
    elapsed = end - start
    for seq,prompt in zip(sequences,prompts):
        results.append(seq[0]['generated_text'])
        cleaned = re.sub(r'.*>>ANSWER<<','',seq[0]['generated_text'])
        whole_answer.append(cleaned)
        cleaned = re.sub(r'\?.*','?',cleaned)
        cleaned = re.sub(r'\..*','.',cleaned)
        cleaned_results.append(cleaned)
    results_df = pd.DataFrame({'qid': df.qid.tolist(), 'prompt':prompts,'result':results,'cleaned_result':cleaned_results, 'answer':whole_answer})
    #results_df.to_csv('rewriting19_context128tk.csv',index=False)
    return results_df, elapsed/len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,help='Model name (from huggingface) or path. (Default == tiiuae/falcon-7b)', default='/llms/code/rewriting/models/llama-2-13b_qrecc_raw_context_128_tk_lr_5e-06_16/checkpoint-7186/')
    parser.add_argument("--trec_path", type=str, help='Path to the trec folder. the script is looking for trec19_for_rewriting.csv kind of file. It has the following columns: qid,conv_id,turn,raw_context,rewritten_context,question,prompt_raw_context,prompt_rewritten_context  (Default == /home/guidorocchietti/llms/data/trec/)', default='/home/guidorocchietti/llms/data/trec/')
    parser.add_argument("--year", type=int, help='Year of the trec dataset to use. You can set year as 0 to perform rewriting form 2019,2020 and 2021 TRECCaST (Default == 0)', default=0)
    parser.add_argument('--output_path', type=str,help='Path to save the results. (Default == ./rewritings/)', default='/home/guidorocchietti/llms/code/rewriting/rewritings/llama-2-13b/')
    parser.add_argument('--n_samples', type=int,help='Number of samples to generate. (Default == 1)', default=1)
    parser.add_argument('--bnb', type=bool,help='Use Bits and Bytes quantization (4bits). (Default == False)', default=False,action=argparse.BooleanOptionalAction)
    parser.add_argument('--temperature', type=float,help='Temperature to use for generation. (Default == 0.7)', default=0.7)
    parser.add_argument('--max_new_tokens', type=int,help='Max number of tokens to generate. (Default == 512)', default=512)
    parser.add_argument('--repetition_penalty', type=float,help='Repetition penalty to use for generation. (Default == 1.1)', default=1.1)
    parser.add_argument('--file_name', type=str,help='File name to save the results. (Default == None)', default=None)
    
    ### parse arguments ###
    
    args = parser.parse_args()
    model_name = args.model_name
    trec_path = args.trec_path
    year = args.year
    output_path = args.output_path
    samples = args.n_samples
    bnb = args.bnb 
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    repetition_penalty = args.repetition_penalty
    file_name = args.file_name
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)
    ### print arguments ###
    print(f'model selected : {model_name}')
    print(f'year selected : {year}' if year != 0 else 'years selected : 2019,2020,2021')
    print(f'rewritings will be saved in : {output_path}')
    
    if samples >1:print(f'number of samples to generate : {samples}')
    print(f'Bits and Bytes quantization : {bnb}')
    
    
    ### instantiate bnb_config if bnb is True ###
    bnb_config = transformers.BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16
                                ) if bnb else     None
    
    #pipeline, tokenizer = instantiate_model(model_name)
    pipeline, tokenizer, model = instantiate_pipeline(model_name,temperature = temperature,max_new_tokens=max_new_tokens,repetition_penalty=repetition_penalty, bnb_config=bnb_config)
    df19 = pd.read_csv(f'{trec_path}trec19_for_rewriting.csv')
    df20 = pd.read_csv(f'{trec_path}trec20_for_rewriting.csv')
    df21 = pd.read_csv(f'{trec_path}trec21_for_rewriting.csv')
    name = 'llama-2-13b' if 'llama-2-13b' in model_name.lower() else 'falcon-7b' if 'falcon-7b' in model_name.lower() else 'falcon-40b' if 'falcon-40b' in model_name.lower() else 'llama-2-7b' if 'llama-2-7b' in model_name.lower() else model_name
    if file_name is not None:
        name = file_name + '_' + name
    all_results = {}
    do19 =True if year == 0 or year ==2019 else False
    do20 = True if year == 0 or year ==2020 else False
    do21 = True if year == 0 or year ==2021 else False

    ### set tqdm bar printing the temperature (temp) ###

    for i in range(samples):
        if samples > 1:
            temperature = str(temperature) + '_sample_' +  str(i)
        if do19:
            results, elapsed = do_rewriting(df19, pipeline, tokenizer)
            results.to_csv(f'{output_path}rewriting19_{name}_temp_{str(temperature)}.csv',index=False)
            with open(f'{output_path}Elapsed_{name}_temp_{str(temperature)}_TREC_19.txt','w') as f:
                f.write(f'{elapsed}')
            print ('elapsed time per sample in 2019: ',elapsed)
        if do20:
            results, elapsed = do_rewriting(df20, pipeline, tokenizer)
            results.to_csv(f'{output_path}rewriting20_{name}_temp_{str(temperature)}.csv',index=False)
            with open(f'{output_path}Elapsed_{name}_temp_{str(temperature)}_TREC_20.txt','w') as f:
                f.write(f'{elapsed}')
            print ('elapsed time per sample in 2020: ',elapsed)

        if do21:
            results, elapsed = do_rewriting(df21, pipeline, tokenizer)
            results.to_csv(f'{output_path}rewriting21_{name}_temp_{str(temperature)}.csv',index=False)
            with open(f'{output_path}Elapsed_{name}_temp_{str(temperature)}_TREC_21.txt','w') as f:
                f.write(f'{elapsed}')
            print ('elapsed time per sample in 2021: ',elapsed)





if __name__ == "__main__":
    main()
else:
    main()

    

# %%
