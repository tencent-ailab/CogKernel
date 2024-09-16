import hmac
import os
import time
import uuid
import requests
import openai
import json

INFERENCE_SERVER_ENGINE = os.environ.get("INFERENCE_SERVER_ENGINE", "tgi")


class BaseModelConnection:
    def __init__(self, ip):
        "Initialize the base model connection"
        self.ip = ip

    def format_message(message: str, role: str, name: str = None) -> str:
        if role == "system" and name:
            return f"<|im_start|>{role} name={name}\n{message}<|im_end|>"
        else:
            return f"<|im_start|>{role}\n{message}<|im_end|>"

    def _generate_query(self, messages: list()):
        """This function will generate the input messages to be the query we send to the base model.

        Args:
            messages (list): input messages
        """

        message_with_history = ""

        for tmp_round in messages:
            if tmp_round["role"] == "system":
                if "name" in tmp_round:
                    message_with_history += f"<|im_start|>{tmp_round['role']} name={tmp_round['name']}\n{tmp_round['content']}<|im_end|>"
                else:
                    message_with_history += f"<|im_start|>{tmp_round['role']}\n{tmp_round['content']}<|im_end|>"
            elif tmp_round["role"] == "user":
                message_with_history += (
                    "<|im_start|>user\n" + tmp_round["content"] + "<|im_end|>"
                )
            elif tmp_round["role"] == "assistant":
                message_with_history += (
                    "<|im_start|>assistant\n" + tmp_round["content"] + "<|im_end|>"
                )
            else:
                raise NotImplementedError(
                    "Role {} is not implemented".format(tmp_round["role"])
                )
        message_with_history += "<|im_start|>assistant\n"
        return message_with_history

    def get_response(self, messages: list()):
        query = self._generate_query(messages)
        headers = {"Content-Type": "application/json"}
        # print('query:', query)
        if INFERENCE_SERVER_ENGINE == "tgi":
            data = {
                "inputs": query,
                "parameters": {
                    "max_new_tokens": 2048,
                    #    "do_sample":True,
                    "stop": ["<|im_end|>"],
                },
            }
            url = "http://" + self.ip + "/generate"
            response = requests.post(url, headers=headers, data=json.dumps(data))

            if response.status_code == 200:
                return (
                    response.json()["generated_text"]
                    .replace("<|im_start|>", "")
                    .replace("<|im_end|>", "")
                )
            elif response.status_code == 422:
                print("Failed:", response.status_code)
                return "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed:", response.status_code)
                return "Failed:" + str(response.status_code)
        elif INFERENCE_SERVER_ENGINE == "vLLM":
            data = {
                "prompt": query,
                "model": "ck",
                "temperature": 0,
                "max_tokens": 512,
                "stop": ["<|im_end|>"],
            }
            url = "http://" + self.ip + "/v1/completions"
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()["choices"][0]["text"]
            elif response.status_code == 422:
                print("Failed:", response.status_code)
                return "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed happened here:", response.status_code)
                return "Failed:" + str(response.status_code)
        else:
            raise NotImplementedError(
                "Inference server engine {} is not implemented".format(
                    INFERENCE_SERVER_ENGINE
                )
            )

    async def get_response_stream(self, messages: list()):
        print("messages:", messages)
        query = self._generate_query(messages)
        # print(query)
        headers = {"Content-Type": "application/json"}

        if INFERENCE_SERVER_ENGINE == "tgi":
            data = {
                "inputs": query,
                "parameters": {
                    "max_new_tokens": 2048,
                    #    "do_sample":True,
                    "stop": ["<|im_end|>"],
                },
            }
            url = "http://" + self.ip + "/generate_stream"
            response = requests.post(
                url, headers=headers, data=json.dumps(data), stream=True
            )

            counter = 0
            if response.status_code == 200:
                for line in response.iter_lines():
                    counter += 1
                    if line:
                        decoded_line = line.decode("utf-8")
                        decoded_line = json.loads(decoded_line[5:])
                        if not decoded_line["generated_text"]:
                            current_token = decoded_line["token"]["text"]
                            yield current_token

            elif response.status_code == 422:
                print("Failed:", response.status_code)
                yield "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed:", response.status_code)
                yield "Failed:" + str(response.status_code)
        elif INFERENCE_SERVER_ENGINE == "vLLM":
            data = {
                "prompt": query,
                "model": "ck",
                "stream": True,
                "temperature": 0,
                "max_tokens": 1024,
                "stop_token_ids": [128258],
            }
            url = "http://" + self.ip + "/v1/completions"
            response = requests.post(
                url, headers=headers, data=json.dumps(data), stream=True
            )
            if response.status_code == 200:
                start_time = time.time()
                buffer = b""
                all_chunks = list()
                for chunk in response.iter_content(chunk_size=1):
                    buffer += chunk
                    if buffer.endswith(b"\n\n"):
                        buffer = buffer.decode("utf-8")
                        if "[DONE]" not in buffer:
                            try:
                                json_data = json.loads(buffer[6:])
                                new_text = json_data["choices"][0]["text"]
                                print(
                                    f"Received token: {new_text}, time: {time.time() - start_time}"
                                )
                                start_time = time.time()
                            except json.JSONDecodeError as e:
                                print(f"parsing error: {e}")
                        buffer = b""
            elif response.status_code == 422:
                print("Failed:", response.status_code)
                yield "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed:", response.status_code)
                yield "Failed:" + str(response.status_code)
        else:
            raise NotImplementedError(
                "Inference server engine {} is not implemented".format(
                    INFERENCE_SERVER_ENGINE
                )
            )


from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
original_chatgpt_client = OpenAI(api_key=api_key)

client = OpenAI(api_key=api_key)


class ChatGPTConnection:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self._model_name = model_name

    def get_response(self, messages: list()):
        try:
            completion = client.chat.completions.create(
                model=self._model_name, messages=messages
            )
            return completion.choices[0].message.content

        except:
            time.sleep(20)
            completion = client.chat.completions.create(
                model=self._model_name, messages=messages
            )
            return completion.choices[0].message.content
