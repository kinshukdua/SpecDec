from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

checkpoint = "facebook/opt-1.3b"
assistant_checkpoint = "facebook/opt-125m"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# inputs = tokenizer(prompt, return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
assistant = AutoModelForCausalLM.from_pretrained(assistant_checkpoint).to(device)

# outputs = model.generate(**inputs, assistant_model=assistant_model)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
prompts = [] # TODO ADD PROMPTS
results = []
for prompt in prompts:
    prompt = "Hi"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    current_tokens = inputs['input_ids']
    inp_len = current_tokens.shape[-1]
    total_tokens = 0
    total_matches = 0
    candidate_count = 5

    while curr_len < 50:
        new_candidates = current_tokens.detach().clone()
        curr_len = current_tokens.shape[-1]
        for i in range(candidate_count):
            assistant_outputs = assistant(new_candidates)
            output_tokens = assistant_outputs.logits.argmax(-1)
            new_candidates = torch.cat((new_candidates, 
                                                output_tokens[:,-1:]),dim=1)
            if torch.eq(output_tokens[:,-1:], assistant.generation_config.eos_token_id):
                break
        # print(tokenizer.batch_decode(new_candidates))
        model_outputs = model(new_candidates)
        next_tokens = model_outputs.logits.argmax(-1)
        # print(tokenizer.batch_decode(next_tokens))
        selected_tokens = next_tokens[:,curr_len-1:]
        candidate_new_tokens = new_candidates[:,curr_len:]
        # check for match
        n_matches = ((~(candidate_new_tokens == selected_tokens[:,:-1]))
                    .cumsum(dim=-1) < 1).sum()
        # print(n_matches)
        total_tokens += candidate_count
        total_matches += n_matches
        current_tokens = torch.cat((current_tokens, selected_tokens[:,:n_matches+1]),dim=1)
        if n_matches == candidate_count:
            # increase candidate count
            candidate_count += 2
            # add new og model token prediction to output
            
            if torch.eq(selected_tokens[:,-1:], assistant.generation_config.eos_token_id):
                print("EOS")
        else:
            if candidate_count > 1:
                candidate_count -= 1 
            # only add matched tokens
        print(tokenizer.batch_decode(current_tokens))
        print(total_matches/total_tokens*100)
    results.append((tokenizer.batch_decode(current_tokens),total_matches,total_tokens))
    