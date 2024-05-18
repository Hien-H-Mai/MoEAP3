from transformers import (
    
    AutoModelForSequenceClassification, 
    
    AutoTokenizer,
    
    BitsAndBytesConfig,
    
    pipeline,
    
)
import torch
class Grammar:
    
    def __init__(self,
                 model_name_or_path = "cola-deberta-v3-large") -> None:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            )
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, quantization_config=nf4_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pipe = pipeline("text-classification",            
                model = model,  
                tokenizer = tokenizer,
                return_all_scores = True,
                device_map="auto",
                )
        
    def compute(self,
                predictions,
                choice = "acceptable"):
        if choice not in ["acceptable", "unacceptable"]:
            raise ValueError(f"choice must be in {['acceptable', 'unacceptable']}, got {choice}")
        results = self.pipe(predictions,
                            batch_size=len(predictions))
        
        label_map = 1 if choice == "acceptable" else 0
        score = [re[label_map]["score"] for re in results]
        
        return score