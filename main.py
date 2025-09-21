from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers



model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map = "auto",
    trust_remote_code = False,
    revision = "main"
)
#https://huggingface.co/datasets/qwedsacf/competition_math?library=datasets



tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

model.eval()

problem = ""