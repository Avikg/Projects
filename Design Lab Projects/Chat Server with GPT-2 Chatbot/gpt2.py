import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Suppress logging messages from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

class FAQChatbot:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, question):
        try:
            inputs = self.tokenizer.encode(question, return_tensors="pt")
            attention_mask = inputs.ne(0).float()  # Creating attention mask
            outputs = self.model.generate(inputs, attention_mask=attention_mask, max_length=100, temperature=0.7, do_sample=True)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            print("Error:", e)
            return "Sorry, I couldn't generate a response at the moment."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 Chatbot")
    parser.add_argument("question", type=str, help="Question to ask the chatbot")

    args = parser.parse_args()

    model_name = "openai-community/gpt2-medium"
    faq_bot = FAQChatbot(model_name)

    response = faq_bot.generate_response(args.question)
    print("gpt2bot>", response)
