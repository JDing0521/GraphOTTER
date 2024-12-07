import google.generativeai as genai
from configs import GEMINI_KEYS
import random
import time
from google.api_core.exceptions import ResourceExhausted
class GeminiTool:
    def __init__(self ,key ,args):

        self.key_index = random.randint(0, len(GEMINI_KEYS) - 1)

        self.model_name = args.model

        self.args = args
        # self.key_index = GEMINI_KEYS.index(key)

        # Create the model
        # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUAL",
                "threshold": "BLOCK_NONE",
            },

        ]



    def generate(self ,prompt,system_instruction = 'You are a helpful AI bot.',isrepeated=0.0,response_mime_type=None):
        genai.configure(api_key=GEMINI_KEYS[self.key_index])
        generation_config = {
            "temperature": self.args.temperature + isrepeated if isrepeated > 0.0 else self.args.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": response_mime_type if response_mime_type != None else "text/plain",
        }
        model = genai.GenerativeModel(
            model_name=self.args.reasoning_model_path,
            safety_settings=self.safety_settings,
            generation_config=generation_config,
            system_instruction=system_instruction,
        )

        error = 3
        while error > 0:
            try:
                output = model.generate_content(prompt)
                result = output.text
                break
            except ValueError as v:
                raise UserWarning('unsafe input ' + v.__str__())
            except Exception as e:
                gemini_key_index = self.key_index
                print(GEMINI_KEYS[gemini_key_index], '报错了', e.__str__())
                gemini_key_index = random.randint(0, len(GEMINI_KEYS) - 1)
                genai.configure(api_key=GEMINI_KEYS[gemini_key_index])
                model = genai.GenerativeModel(
                    model_name=self.args.reasoning_model_path,
                    safety_settings=self.safety_settings,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                )
                self.key_index = gemini_key_index
                print('更换Gemini key为', GEMINI_KEYS[gemini_key_index])
                time.sleep(2.0)
                error -= 1
        if error <= 0:
            raise UserWarning('gemini 报错')

        return result
