import json
import torch
import re
from transformers import AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList


class StopOnBoxed(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pattern = re.compile(r'\\boxed\{[A-E]\}')  # matches \boxed{A} ... \boxed{E}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # decode current sequence
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # return True once a boxed answer is seen
        return bool(self.pattern.search(decoded))


class LLMCom:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate_response_pipeline(self, prompt, max_new_tokens=1024, temperature=0.3):
        stopping_criteria = StoppingCriteriaList([StopOnBoxed(self.tokenizer)])

        response = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=False,  # deterministic for math
            return_full_text=False,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )

        return response[0]["generated_text"]

    def chat(self, user_message, method="pipeline", **kwargs):
        # Wrap the user message in LLaMA 3.1 chat format automatically
        messages = [
            {"role": "user", "content": user_message}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return self.generate_response_pipeline(prompt, **kwargs)
    

def load_questions():
    with open("benchmarking/key_pair.json", "r", encoding = "utf-8") as f:
        return json.load(f)
        
def extract_answer(text: str):
    match = re.search(r'\$?\\boxed\{\s*([^}]*)\s*\}\$?', text)
    if match:
        return match.group(1).strip()

    match = re.search(r'[\(\[\s]?([A-E])[\)\]\s]?', text)
    if match:
        return match.group(1).strip()

    match = re.search(r'Answer:\s*([A-E0-9]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None

if __name__ == "__main__":

    score = 0

    answers = []

    llama = LLMCom()
    
    contest_instructions = "Solve the following math question. Of A, B, C, D and E, only one of these is correct. Avoid unecessary brute forcing. Figures are not necessarily drawn to scale.\n"
    
    answer_instructions = "You are to format your answer with \\boxed{X}, where X is the letter associated with the answer choice you are most confident with.\n"
    
    questions = load_questions()

    for i in range(25):
        title = questions[i]["title"]
        question = questions[i]["question"]
        full_prompt = f"{contest_instructions} {answer_instructions} The question is given as follows: \n {title}: {question}"
        
        print(f"Prompt to be fed: {full_prompt}")
        response = llama.chat(full_prompt, temperature=0.3) 
        print("Response:", response)

        answer = extract_answer(response)

        with open ("benchmarking/responses.txt","a") as f:
            f.write(f"{title}: \n {question} \n")
            f.write(f"Solution: {answer}\n\n")

        if answer == questions[i]["answer"]:
            score += 6
            answers.append(1)
        else:
            score += 0
            answers.append(0)
        print("=========================================")
        print(f"CURRENT QUESTION: {i+1} | SCORE: {score}")
        print("=========================================")




