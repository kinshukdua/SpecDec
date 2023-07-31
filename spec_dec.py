from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_models(model_name: str, 
                      assistant_name:str) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    print("Initializing models")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    assistant_model = AutoModelForCausalLM.from_pretrained(assistant_name).to(device)

    return model, assistant_model, tokenizer



def get_n_candidates(assistant, candidate_count, new_candidates):
    print("Generating candidate tokens")
    for i in range(candidate_count):
        assistant_outputs = assistant(new_candidates)
        new_candidates = assistant_outputs.logits.argmax(-1)
        new_candidates = torch.cat((new_candidates, 
                                          new_candidates[:,-1:]),dim=1)
        if torch.eq(new_candidates[:,-1:], assistant.generation_config.eos_token_id):
            break
        # print(tokenizer.batch_decode(new_candidates[:,inp_len:]))
    return new_candidates

def get_next_tokens(model, new_candidates):
    print("Generating batch tokens")
    model_outputs = model(new_candidates)
    next_tokens = model_outputs.logits.argmax(-1)
    return next_tokens
    # tokenizer.batch_decode(next_tokens[:,inp_len-1:])


def speculative_loop(model, assistant, tokenizer, prompt, max_new_tokens):
    print("Starting speculative loop")
    total_tokens = 0
    total_matches = 0
    candidate_count = 10
    # Get inputs
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    current_tokens = inputs['input_ids']
    inp_len = current_tokens.shape[-1]
    # Get candidate
    while current_tokens.shape[-1]-inp_len <= max_new_tokens:
        new_candidates = current_tokens.detach.clone()
        curr_len = current_tokens.shape[-1]
        new_candidates = get_n_candidates(assistant, candidate_count, new_candidates)
        next_tokens = get_next_tokens(model, new_candidates)
        # slice out inputs
        selected_tokens = next_tokens[:,curr_len-1:]
        candidate_new_tokens = new_candidates[:,curr_len:]
        # check for match
        n_matches = ((~(candidate_new_tokens == selected_tokens[:,:-1]))
                    .cumsum(dim=-1) < 1).sum()
        # if all candidates match
        total_tokens += candidate_count
        total_matches += n_matches
        if n_matches == candidate_count:
            # increase candidate count
            candidate_count += 2
            # add new og model token prediction to output
            new_candidates = torch.cat((new_candidates, selected_tokens[:,-1:]),dim=1)
            if torch.eq(selected_tokens[:,-1:], assistant.generation_config.eos_token_id):
                break
        else:
            if candidate_count > 1:
                candidate_count -= 1 
            # only add matched tokens
            new_candidates = torch.cat((new_candidates, selected_tokens[:,:n_matches+1]),dim=1)

    return tokenizer.batch_decode(new_candidates), total_tokens, total_matches



def main():
    print("Starting execution")
    model_name = "facebook/opt-1.3b"
    assistant_name = "facebook/opt-125m"
    model, assistant, tokenizer = init_models(model_name, assistant_name)
    prompt = "Hi!"
    output, total, matches = speculative_loop(model, assistant, tokenizer, prompt, 50)
    print(output)
    print(f"{matches}/{total}")


