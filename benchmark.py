import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


contest_instructions = "This is a 25-question, multiple choice test. Each question is followed by answers marked A, B, C, D and E. Only one of these is correct. You will receive 6 points for a correct answer, and 0 points for each incorrect answer, for a maximum of 150 points. No aids are permitted other than scratch paper, graph paper, ruler, compass, protractor and erasers (and calculators that are accepted for use on the test if before 2006. No problems on the test will <i>require</i> the use of a calculator). Avoid unecessary brute forcing. Figures are not necessarily drawn to scale."
answer_instructions = "You are to reply your answer with a box, with the label \\boxed{} of the letter corresponding to the answer choice you are most confident with."

class LLMCom:
    def __init__ (self, model_name = "meta-llama/Llama-3.1-8B-Instruct"):
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

    def format_chat_prompt(self, user_message, system_message=None):
        messages = []
    
        messages.append({"role": "user", "content": user_message})
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def generate_response_pipeline(self, user_message, system_message=None, max_new_tokens=512, temperature=0.7):
        messages = []
        
        messages.append({"role": "user", "content": user_message})
        
        response = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            return_full_text=False
        )

        return response[0]["generated_text"]
    


def load():
    with open("key_pair.json", "r", encoding = "utf-8") as f:
        return json.load(f)
        


if __name__ == "__main__":
    llama = LLMCom()
    
    contest_instructions = "This is a 25-question, multiple choice test. Each question is followed by answers marked A, B, C, D and E. Only one of these is correct. You will receive 6 points for a correct answer, and 0 points for each incorrect answer, for a maximum of 150 points. No aids are permitted other than scratch paper, graph paper, ruler, compass, protractor and erasers (and calculators that are accepted for use on the test if before 2006. No problems on the test will <i>require</i> the use of a calculator). Avoid unecessary brute forcing. Figures are not necessarily drawn to scale."
    
    answer_instructions = "You are to reply your answer with a box, with the label \\boxed{} of the letter corresponding to the answer choice you are most confident with."
    
    full_prompt = f"{contest_instructions} {answer_instructions} The question is given as follows: This is just a test"
    
    response = llama.chat(full_prompt, temperature=0.3) 
    print("Response:", response)

# testing loop

current_score = 0

