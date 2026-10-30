#%%
import os, re
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, BitsAndBytesConfig, AutoConfig, AutoModel
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
### define data path ###
if 'llms' in os.getcwd():
    data_path = re.sub('llms.*','llms/data/',os.getcwd())
else: 
    data_path = '/home/guidorocchietti/llms/data/'
os.environ['TRANSFORMERS_CACHE'] = f'{data_path}/cache'

from huggingface_hub import login
login(token=os.environ.get('HF_TOKEN'))  # set HF_TOKEN in your environment

#%%
from datasets import load_dataset, DatasetDict
import argparse

## def main():
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help= 'model name to fine-tune default: meta-llama/Llama-2-70b-hf', default='meta-llama/Llama-2-70b-hf')
    parser.add_argument('--dataset', type=str, help= 'dataset name to fine-tune on default: giuid/qrecc_raw_context', default='giuid/qrecc_raw_context')  
    parser.add_argument("--context_length", type=int,help='Length of the context to be used for the model.', default=128)
    parser.add_argument("--batch_size", type=int,help='Batch size to be used for the model.', default=32)
    parser.add_argument("--learning_rate", type=float,help='Learning rate to be used for the model.', default=5e-6)
    parser.add_argument("--num_epochs", type=int,help='Number of epochs to be used for the model.', default=5)
    parser.add_argument('--resume_from_checkpoint', type=str,help='Path to checkpoint to resume training from.', default=None)
    parser.add_argument('--quantize', type=bool,help='Whether to quantize the model or not. Default == False', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--n_bits', type=int,help='Number of bits to use for quantization. Available options [4,8]. Default == 4', default=4)
    parser.add_argument('--save_strategy', type=str,help='Save strategy to use for the model. Available options ["no","steps","epoch"]. Default == "steps"', default='steps')
    parser.add_argument('--save_total_limit' , type=int,help='Maximum number of checkpoint to store. Default == 2', default=2)
    parser.add_argument('--save_steps' , type=int,help='Number of steps between each checkpoint. Default == 500', default=500)
    parser.add_argument('--model_output_path' , type=str,help='Path to save the model. Default == None', default=None)
    args = parser.parse_args()
    
    ### parse arguments ###
    model_name = args.model_name
    dataset_name = args.dataset
    context_length = args.context_length
    learning_rate = args.learning_rate
    per_device_train_batch_size = args.batch_size
    num_train_epochs = args.num_epochs
    resume_from_checkpoint = args.resume_from_checkpoint
    save_strategy = args.save_strategy
    save_total_limit = args.save_total_limit
    save_steps = args.save_steps
    n_bits = args.n_bits
    quantize = args.quantize
    model_output_path = args.model_output_path
    

    
    #print("Qunatization: ",quantize)
    name = 'llama2-70b' if 'llama2-70b' in model_name.lower() else 'llama-2-13b' if 'llama-2-13b' in model_name.lower() else 'falcon-7b' if 'falcon-7b' in model_name.lower() else 'falcon-40b' if 'falcon-40b' in model_name.lower() else 'llama-2-7b' if 'llama-2-7b' in model_name.lower() else model_name
    if model_output_path is None:
        model_output_path = f"models/{name}_{dataset_name.replace('giuid/','')}_{context_length}_tk_lr_{str(learning_rate)}_batch_{per_device_train_batch_size}"
        ###check if model_output_path exists and resume_from_checkpoint is none. If it exists add a number###
        if os.path.exists(model_output_path) and resume_from_checkpoint is None:
            print(f'Model output path {model_output_path} already exists. The folder will be renamed to {model_output_path}_1')
            model_output_path = f"{model_output_path}_1"
    elif model_output_path[-1] != '/':
        model_output_path+='/'
    ###LOAD DATASET###
    #Load Shuffled Dataset Training and Validation
    print(f'Training {name} on {dataset_name} with context length {context_length}...')
    print(f'Learning rate: {learning_rate}, Batch size: {per_device_train_batch_size}, Number of epochs: {num_train_epochs}')
    if  resume_from_checkpoint is not None:
        print(f'Resuming training from {resume_from_checkpoint}')
    print(f'Model output path: {model_output_path}')

    ds = load_dataset(dataset_name, split="train").shuffle(seed=42)
    ### select first 90% of the dataset as training set ###
    ds_train = ds.select(range(int(len(ds)*0.9)))
    ### select last 10% of the dataset as validation set ###
    ds_val = ds.select(range(int(len(ds)*0.9),len(ds)))
    #Load Test Dataset
    ds_test = load_dataset(dataset_name, split="test")
    #Create Raw dataset
    raw_datasets = DatasetDict(
        {
            "train": ds_train,  # .shuffle().select(range(50000)),
            "valid": ds_val,  # .shuffle().select(range(500))
            "test": ds_test,  # .shuffle().select(range(500))
        }
    )
    ### LOAD TOKENIZER ###
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {'c_token': '>>INTRODUCTION<<', 'a_token': '>>ANSWER<<', 'q_token': '>>QUESTION<<'}
    tokenizer.add_special_tokens({'additional_special_tokens': [special_tokens[key] for key in special_tokens.keys()]})
    tokenizer.pad_token = tokenizer.eos_token  
    def tokenize(element, context_length= 128):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
            padding="max_length",
            return_attention_mask=False
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}
    #Tokenize datasets
    import torch
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Tokenizing datasets...')
    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    tokenized_datasets
    tokenized_datasets = tokenized_datasets.with_format("torch", device=device)
    #Load Model
    from peft import LoraConfig, get_peft_model
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    
    ###bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True if args.n_bits == 8 else False,
        load_in_4bit=True if args.n_bits == 4 else False,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )    
        
    ###Lora Config
    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        #target_modules=["query_key_value"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if quantize:
        if resume_from_checkpoint is not None:
            from peft import PeftModel, PeftConfig
            config = PeftConfig.from_pretrained(resume_from_checkpoint)
            if "t5" in model_name:
                model = AutoModel.from_pretrained(model_name,
                config = config.base_model_name_or_path,
                device_map='auto',
                load_in_4bit=True,
                trust_remote_code=True,
                quantization_config=bnb_config,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config = config.base_model_name_or_path,
                device_map='auto',
                load_in_4bit=True,
                trust_remote_code=True,
                quantization_config=bnb_config,
                )
            model = PeftModel.from_pretrained(model, resume_from_checkpoint)
            model = model.merge_and_unload()
            model.resize_token_embeddings(len(tokenizer))
        else:
            if 't5' in model_name:
                model = AutoModel.from_pretrained(model_name,
                config = config.base_model_name_or_path,
                device_map='auto',
                load_in_4bit=True,
                trust_remote_code=True,
                quantization_config=bnb_config,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=bnb_config, device_map='auto')
            model.resize_token_embeddings(len(tokenizer))
            ### Prepare Model for Training qith peft
            from peft import prepare_model_for_kbit_training

            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            print('Reducing model to qlora reduction')
            model = get_peft_model(model, config)
            #model.to(device,device_map='auto')
            print_trainable_parameters(model)
            print(f'The qlora config is {config}')

    else:
        if resume_from_checkpoint is not None:
            if "t5" in model_name:
                model = AutoModel.from_pretrained(resume_from_checkpoint, device_map='auto',  trust_remote_code=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(resume_from_checkpoint, trust_remote_code=True,)
        else:
            if "t5" in model_name:
                model = AutoModel.from_pretrained(model_name, device_map='auto',  trust_remote_code=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,)

   
    #empty cuda cache
    torch.cuda.empty_cache()

    #Prepare Data Collator and Training Arguments
    from transformers import DataCollatorForLanguageModeling
    from transformers import TrainingArguments
    #name = 'llama2' if 'lama' in model_name else 'falcon'

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    print('Preparing training arguments...')
    
    


    
    if resume_from_checkpoint is not None:
        training_args = TrainingArguments(model_output_path,
                                    learning_rate = learning_rate,
                                    per_device_train_batch_size=per_device_train_batch_size,
                                    num_train_epochs=num_train_epochs,
                                    #save_strategy="epoch",
                                    save_strategy = save_strategy,
                                    evaluation_strategy="steps",
                                    save_steps=save_steps,
                                    save_total_limit = save_total_limit,
                                    resume_from_checkpoint=resume_from_checkpoint,
                                    #load_best_model_at_end=False,
                                    optim="paged_adamw_8bit",
                                    load_best_model_at_end= True
                                    )
    else:
        training_args = TrainingArguments(model_output_path,
                                    learning_rate = learning_rate,
                                    per_device_train_batch_size=per_device_train_batch_size,
                                    num_train_epochs=num_train_epochs,
                                    #save_strategy="epoch",
                                    save_strategy = save_strategy,
                                    evaluation_strategy="steps",
                                    save_steps=save_steps,
                                    save_total_limit = save_total_limit,
                                    #load_best_model_at_end=False,
                                    optim="paged_adamw_8bit",
                                    load_best_model_at_end= True
                                    )
        
    print(f'Training model with {training_args.num_train_epochs} epochs, {training_args.per_device_train_batch_size} batch size and {training_args.learning_rate} learning rate')
    ### Instantiate Trainer
    from transformers import Trainer
    #from  torch.nn import DistributedDataParallel
    #model = DistributedDataParallel(model)
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    torch.cuda.empty_cache()

    ### Train The Model
    trainer.train()
    

    ### Print parametes to file ###
    with open(f'{model_output_path}/args.txt','w') as f:
        f.write(f'model_name: {model_name}\n')
        f.write(f'dataset_name: {dataset_name}\n')
        f.write(f'context_length: {context_length}\n')
        f.write(f'learning_rate: {learning_rate}\n')
        f.write(f'per_device_train_batch_size: {per_device_train_batch_size}\n')
        f.write(f'num_train_epochs: {num_train_epochs}\n')
        f.write(f'resume_from_checkpoint: {resume_from_checkpoint}\n')
        f.write(f'save_strategy: {save_strategy}\n')
        f.write(f'save_total_limit: {save_total_limit}\n')
        f.write(f'save_steps: {save_steps}\n')
        f.write(f'name: {name}\n')    
    ### save qlora config ###
    #with open (f'{model_output_path}/qlora_cofig.txt','w') as f: f.write(config)
    
    from telegram_sender import send_message
    send_message("First Finetuning Completed")


if __name__ == "__main__":
    main()
#else:
#    main()


### python finetuning_script.py --model_name  /home/guidorocchietti/llms/data/cache/models--tiiuae--falcon-7b/snapshots/378337427557d1df3e742264a2901a49f25d4eb1 --dataset giuid/qrecc_context_and_answers --context_length 128 --batch_size 16 --learning_rate 5e-6 --num_epochs 5
