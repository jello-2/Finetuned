import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LLMCom:
    def __init__ (self, model_name = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate_response_pipeline(self, user_message, max_new_tokens=768, temperature=0.3):
        response = self.pipe(
            user_message,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=False,
            return_full_text=False
        )

        return response[0]["generated_text"]
        
    def chat(self, user_message, method="pipeline", **kwargs):
        
        return self.generate_response_pipeline(user_message,  **kwargs)
    


def load_questions():
    with open("benchmarking/key_pair.json", "r", encoding = "utf-8") as f:
        return json.load(f)
        
def extract_answer(text):
    match = re.search(r'\( *([A-E]) *\)', text)
    if match:
        return match.group(1)


if __name__ == "__main__":

    score = 0

    answers = []

    llama = LLMCom()
    
    contest_instructions = "Solve the following math question. Of A, B, C, D and E, only one of these is correct. This will not require the use of a calculator. Avoid unecessary brute forcing. Figures are not necessarily drawn to scale.\n"
    
    answer_instructions = "You are to reply concisely with your answer with the label \\boxed{} of the **letter** corresponding to the answer choice you are most confident with.\n"
    
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




