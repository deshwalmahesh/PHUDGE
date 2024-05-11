from torch.utils.data import Dataset
import numpy as np
import torch

sys_ref = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate and a reference answer that gets a score of 5 are given.
1. Write a detailed feedback that assess the quality of the response.
2. After writing a feedback, write a score that is an integer between 1 and 5. You can refer to the reference answer (which has a perfect score of 5) to get the idea for scoring.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations."""

sys_rubric = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations."""

sys_only = """###Task Description:
An instruction (might include an Input inside it) and a response to evaluate are given.
1. Write a detailed feedback that assess the quality of the response strictly.
2. After writing a feedback, write a score that is an integer between 1 and 5.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations."""

sys_ref_rubric = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general. You can also refer to the reference answer (which has a perfect score of 5) to get the idea for scoring.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations."""

sys_map_prom = {"sys_only":sys_only, "ref_only":sys_ref, "rubric_only":sys_rubric, "ref_rub": sys_ref_rubric}



def create_prometheus_prompt(inst:str, completion:str, include_rubric:bool, include_ref_ans:bool, task:str):
    """
    task: [CAUSAL, CLASSIF, REG]
    """

    inst = inst[inst.find("###The instruction to evaluate"):]
    ref_ans_index = inst.find("###Reference Answer")
    rubric_index = inst.find("###Score Rubrics")
    
    que_ans = "<|user|>\n" + inst[:ref_ans_index].strip() + "\n\n"
    ref_ans = inst[ref_ans_index:rubric_index].strip() + "\n\n"
    rubric = inst[rubric_index:inst.find("\n\n###Feedback:")].strip()
    
    if include_rubric and include_ref_ans: 
        sys_prompt = sys_map_prom["ref_rub"]
        
    elif include_rubric and (not include_ref_ans): 
        sys_prompt = sys_map_prom["rubric_only"]
        ref_ans = ""
        
    elif include_ref_ans and (not include_rubric): 
        sys_prompt = sys_map_prom["ref_only"]
        rubric = ""
        ref_ans = ref_ans.strip()
        
    else: 
        sys_prompt = sys_map_prom["sys_only"]
        ref_ans = ""
        rubric = ""
        que_ans = que_ans.strip()
    
    prompt = "<|system|>\n" + sys_prompt + "<|end|>\n\n" + que_ans + ref_ans + rubric + "<|end|>\n\n<|assistant|>\n"
    if task == "CAUSAL": prompt += completion.strip()
    
    return prompt
    

class TextDataset(Dataset):
    def __init__(self, instructions, completions_or_labels, task, tokenizer, MAX_LENGTH, ref_drop_thresh = 0.5, rub_drop_thresh = 0.5):
        assert task in ["CAUSAL", "CLASSIF", "REG"], "task must be in [CAUSAL, CLASSIF, REG]"
        self.instructions = instructions # It is Text list for Regression / Classification and TEXT completion for CAUSAL
        self.task = task
        self.completions_or_labels = completions_or_labels
        self.tokenizer = tokenizer
        self.MAX_LENGTH = MAX_LENGTH
        self.ref_drop_thresh = ref_drop_thresh
        self.rub_drop_thresh = rub_drop_thresh
        self.len = len(instructions)
        
        if task == "CAUSAL": 
            if isinstance(self.completions_or_labels[0], (int, float)) : raise ValueError("When task == 'CAUSAL', `completions_or_labels` must be a string")
        else:
            if isinstance(self.completions_or_labels[0], str): raise ValueError("When task is REG or CLASSIF, `completions_or_labels` must be a int or float")
            

    def show_random_example(self):
        idx = np.random.choice(range(self.len-1))
        return create_prometheus_prompt(self.instructions[idx], self.completions_or_labels[idx], True, True, self.task)
        
    
    def tokenize_prompt(self, prompt, label = None): 
        result = self.tokenizer(prompt, truncation=True, max_length = self.MAX_LENGTH, padding = "max_length")
        
        if self.task == "CAUSAL":
            result["labels"] = result["input_ids"].copy()
        elif self.task == "REG":
            result["label"] = float(label)
        else:
            result["label"] = int(label) - 1
            
        return result       
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        include_rubric, include_ref_ans = True, True
        a,b = np.random.rand(2)
        if a >= self.rub_drop_thresh: include_rubric = False
        if b >= self.ref_drop_thresh: include_ref_ans = False
        
        prompt = create_prometheus_prompt(self.instructions[idx], self.completions_or_labels[idx], include_rubric, include_ref_ans, self.task)
        
        if self.task == "CAUSAL": prompt += self.tokenizer.eos_token
        tokenized_prompt = self.tokenize_prompt(prompt, label = self.completions_or_labels[idx])
        
        return {key: torch.tensor(val) for key, val in tokenized_prompt.items()}
