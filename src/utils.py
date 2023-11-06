import datetime
from datasets import load_dataset
from transformers import AutoTokenizer


def get_name_with_date(name):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    new_filename = f"{name}_{formatted_datetime}"
    return new_filename


def get_datasets(dataset_name):
    ds = load_dataset(dataset_name, split="train").shuffle(seed=42)
    # select first 90% of the dataset as training set
    ds_train = ds.select(range(int(len(ds) * 0.9)))
    # select last 10% of the dataset as validation set
    ds_val = ds.select(range(int(len(ds) * 0.9), len(ds)))
    # Load Test Dataset
    ds_test = load_dataset(dataset_name, split="test")
    return ds_train, ds_val, ds_test


# def tokenize(element, model_name, context_length=128):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     special_tokens = {
#         "c_token": ">>INTRODUCTION<<",
#         "a_token": ">>ANSWER<<",
#         "q_token": ">>QUESTION<<",
#     }
#     tokenizer.add_special_tokens(
#         {
#             "additional_special_tokens": [
#                 special_tokens[key] for key in special_tokens.keys()
#             ]
#         }
#     )
#     tokenizer.pad_token = tokenizer.eos_token
#     outputs = tokenizer(
#         element["text"],
#         truncation=True,
#         max_length=context_length,
#         return_overflowing_tokens=True,
#         return_length=True,
#         padding="max_length",
#         return_attention_mask=False,
#     )
#     input_batch = []
#     for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
#         if length == context_length:
#             input_batch.append(input_ids)
#     return {"input_ids": input_batch}


# def print_trainable_parameters(model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#     )
