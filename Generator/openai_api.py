import time
from openai import OpenAI
import openai


class ChatGPTTool(object):
    def __init__(self,args):
        self.API_SECRET_KEY = args.key
        self.BASE_URL = args.base_url
        self.model_name = args.model

        self.args = args
        # chat
        if self.BASE_URL:
            self.client = OpenAI(api_key=self.API_SECRET_KEY, base_url=self.BASE_URL)
        else:
            self.client = OpenAI(api_key=self.API_SECRET_KEY)


    def generate(self,prompt,system_instruction = 'You are a helpful AI bot.',isrepeated=0.0,response_mime_type=None):

        if isrepeated > 0.0:
            temperature = self.args.temperature + isrepeated
        else:
            temperature = self.args.temperature
        error = 3
        while error > 0:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": '\n'.join(prompt)}
                    ],
                    temperature=temperature,
                    seed=42,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    # response_format=response_mime_type,
                )

                break
            except openai.RateLimitError as r:
                print('openai限流了',r.__str__())
                error -= 1
                time.sleep(4.0)
            except openai.InternalServerError as r:
                print('openai奔溃了', r.__str__())
                error -= 1
                time.sleep(2.0)
            except openai.APITimeoutError as a:
                print('openai超时', a.__str__())
                # error -= 1
                # time.sleep(2.0)
                raise UserWarning(f' openai超时 {a.__str__()}')
            except Exception as r:
                print('openai报错了',r.__str__())
                error -= 1
                time.sleep(2.0)
        output = resp.choices[0].message.content

        return output







