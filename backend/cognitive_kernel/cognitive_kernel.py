import json
import os
import io
import sys
import copy
import time
import asyncio
import nest_asyncio
import requests
import PyPDF2
import fitz
import docx2txt
import re
import uuid
import shutil
from datetime import datetime
from pdfminer.pdfpage import PDFPage
from pdfminer.high_level import extract_text
from cognitive_kernel.base_model_connection import (
    BaseModelConnection,
    ChatGPTConnection,
)
from functools import partial
from cognitive_kernel.call_web import call_web
from concurrent.futures import ThreadPoolExecutor
from cognitive_kernel.code_executor import ExecutorManager
from database import (
    update_or_create_rawdata,
    get_annotations_for_evaluation,
    get_rawdata_by_session_id,
)

FILE_LOCATIONS = "/app/UploadedFiles"
GLOBAL_DB_LOCATIONS = "/app/Database_global"
DB_LOCATIONS = "/app/Database_local"
CHARACTER_POOL_PATH = "/app/Character_pool"
CUSTOMIZED_CHARACTER_POOL_PATH = "/app/Customized_character_pool"
TOP_K_SENTENCES = 1000

ACTIVATE_KE = os.environ.get("ACTIVATE_KE", "True").lower() in ("true", "1", "t")
ACTIVATE_HISTORY = os.environ.get("ACTIVATE_HISTORY", "True").lower() in (
    "true",
    "1",
    "t",
)
ACTIVATE_SHORT_FEEDBACK = os.environ.get("ACTIVATE_SHORT_FEEDBACK", "True").lower() in (
    "true",
    "1",
    "t",
)
HISTORY_SUMMARY_LENGTH = os.environ.get("HISTORY_SUMMARY_LENGTH", 3)

from cognitive_kernel.memory_kernel.knowledge_engine import (
    KnowledgeEngine,
    KnowledgeEngineConfig,
)

print("ACTIVATE_KE:", ACTIVATE_KE)
print("ACTIVATE_HISTORY:", ACTIVATE_HISTORY)


async def generate_stream(current_connection, messages):
    try:
        full_response = ""
        async for message in current_connection.get_response_stream(messages):
            yield message
        full_response = current_connection.get_response(messages)
        print("full_response:", full_response)
    except AttributeError:
        full_response = current_connection.get_response(messages)
        yield full_response


def load_function_prompt(path, function_name):
    """This function will load the function prompt from the description.txt file
    Args:
        path (str): location of the available functions
        function_name (str): name of the function
    Returns:
        str: the detailed description of the function
    """
    with open(path + "/" + function_name + "/description.txt", "r") as f:
        function_prompt = f.read()
    return function_prompt


def load_function_implementation(path, function_name):
    """This function will load the function implementation from the implementation.py file
    Args:
        path (str): location of the available functions
        function_name (str): name of the function
    Returns:
        str: the detailed implementation of the function
    """
    with open(os.path.join(path, function_name, "implementation.py"), "r") as f:
        function_implementation = f.read()
    return function_implementation


def load_function_example(path, function_name):
    """This function will load the function examples from the examplel.jsons file
    Args:
        path (str): location of the available functions
        function_name (str): location of the available functions

    Returns:
        list: examples of using the target function
    """
    function_exmaples = list()
    with open(os.path.join(path, function_name, "examples.jsonl"), "r") as f:
        for line in f:
            tmp_example = json.loads(line)
            tmp_example[-2]["content"] = (
                "Please generating the acting code for the following query: "
                + tmp_example[-2]["content"]
            )
            function_exmaples += tmp_example
    return function_exmaples


def load_single_character_info(character_pool_path, target_character):
    with open(
        os.path.join(character_pool_path, target_character, "info.json"), "r"
    ) as f:
        basic_info = json.load(f)
        current_character_name = basic_info["name"]
        tmp_character_info = dict()
        tmp_character_info["id"] = basic_info["id"]
        tmp_character_info["shown_title"] = basic_info["shown_title"]
        tmp_character_info["description"] = basic_info["description"]
        tmp_character_info["visible_users"] = basic_info["visible_users"]
        tmp_character_info["head_system_prompt"] = basic_info["head_system_prompt"]
        tmp_character_info["tail_system_prompt"] = basic_info["tail_system_prompt"]
        tmp_character_info["system_prompt_sequence"] = basic_info[
            "system_prompt_sequence"
        ]
        function2implementation = dict()
        function2prompts = dict()
        function2examples = dict()
        available_function_path = os.path.join(
            character_pool_path, target_character, "functions"
        )
        function_dirs = os.listdir(available_function_path)
        if target_character == "cognitiveKernel":
            function_dirs = ["CallWeb", "CallMemoryKernel", "AskLLM"]
        for tmp_function in function_dirs:
            if tmp_function[0] != ".":
                function2implementation[tmp_function] = load_function_implementation(
                    path=available_function_path, function_name=tmp_function
                )
                function2prompts[tmp_function] = load_function_prompt(
                    path=available_function_path, function_name=tmp_function
                )
                function2examples[tmp_function] = load_function_example(
                    path=available_function_path, function_name=tmp_function
                )
        tmp_character_info["function2implementation"] = function2implementation
        tmp_character_info["function2prompts"] = function2prompts
        tmp_character_info["function2examples"] = function2examples
        tmp_character_info["avatar_path"] = os.path.join(
            character_pool_path, target_character, "avatar.png"
        )
        avatar_destination_path = f"/app/static/avatar/{target_character}.png"
        shutil.copy(
            tmp_character_info["avatar_path"],
            avatar_destination_path,
        )
        return tmp_character_info, basic_info["global_db_info"]


def load_character_info(character_pool_path, character_type="global"):
    """This function will load the implementation of the activated functions from the config file.
    Returns:
        dict: a dictionary that maps the function name to the implementation of the function.
    """
    available_characters = os.listdir(character_pool_path)
    character_to_info = dict()
    global_db_info = dict()
    for tmp_character in available_characters:
        try:
            if tmp_character[0] == ".":
                continue
            tmp_character_info, tmp_global_db_info = load_single_character_info(
                character_pool_path=character_pool_path, target_character=tmp_character
            )
            tmp_character_info["character_type"] = character_type
            character_to_info[tmp_character] = tmp_character_info
            global_db_info[tmp_character] = tmp_global_db_info
        except Exception as e:
            print(f"Error in loading {tmp_character}, error: {e}")
            pass

    return character_to_info, global_db_info


class CognitiveKernel(object):
    def __init__(
        self,
        args,
        memory_inference_urls=None,
        model_name="ck",
        service_ip="30.207.99.138:8000",
    ) -> None:
        if model_name == "ck":
            self.LLM_connection = BaseModelConnection(ip=service_ip)
        else:
            self.LLM_connection = ChatGPTConnection(model_name=model_name)
        self.gpt_connection = ChatGPTConnection(model_name="gpt-3.5-turbo")
        self.args = args
        self.model_name = model_name
        # Start to load the activated functions
        self.memory_inference_urls = memory_inference_urls
        self.character_to_info, self.global_db_info = load_character_info(
            character_pool_path=CHARACTER_POOL_PATH, character_type="global"
        )
        self.customized_character_to_info, self.customized_global_db_info = (
            load_character_info(
                character_pool_path=CUSTOMIZED_CHARACTER_POOL_PATH,
                character_type="customized",
            )
        )

        # merge the customized character info to the global character info
        for tmp_character in self.customized_character_to_info:
            self.character_to_info[tmp_character] = self.customized_character_to_info[
                tmp_character
            ]
            self.global_db_info[tmp_character] = self.customized_global_db_info[
                tmp_character
            ]
        self.username2character = dict()
        for tmp_character in self.character_to_info:
            for tmp_user in self.character_to_info[tmp_character]["visible_users"]:
                if tmp_user not in self.username2character:
                    self.username2character[tmp_user] = list()
                self.username2character[tmp_user].append(tmp_character)
        self.knowledge_engine = KnowledgeEngine(
            args=args, inference_urls=self.memory_inference_urls, mode="inference"
        )
        self.executor_manager = ExecutorManager()
        self.setup_global_db()

    def update_character(self, character_name, character_type):
        if character_type == "customized":
            tmp_character_info, tmp_global_db_info = load_single_character_info(
                character_pool_path=CUSTOMIZED_CHARACTER_POOL_PATH,
                target_character=character_name,
            )
            tmp_character_info["character_type"] = character_type
            self.character_to_info[character_name] = tmp_character_info
            self.global_db_info[character_name] = tmp_global_db_info
            for tmp_user in tmp_character_info["visible_users"]:
                if tmp_user not in self.username2character:
                    self.username2character[tmp_user] = list()
                self.username2character[tmp_user].append(character_name)
        else:
            raise NotImplementedError

    def delete_character(self, character_name):
        character_info = self.character_to_info[character_name]
        for tmp_user in character_info["visible_users"]:
            self.username2character[tmp_user].remove(character_name)
            if len(self.username2character[tmp_user]) == 0:
                del self.username2character[tmp_user]
        self.character_to_info.pop(character_name)
        self.global_db_info.pop(character_name)

    def get_all_characters(self, username):
        visible_characters = list()
        if username in self.username2character:
            visible_characters.extend(self.username2character[username])
        visible_characters.extend(self.username2character["all"])
        all_characters = list()
        for tmp_character_name in visible_characters:
            all_characters.append(
                {
                    "name": tmp_character_name,
                    "key": self.character_to_info[tmp_character_name]["id"],
                    "shownTitle": self.character_to_info[tmp_character_name][
                        "shown_title"
                    ],
                    "title": self.character_to_info[tmp_character_name]["description"],
                    "characterType": self.character_to_info[tmp_character_name][
                        "character_type"
                    ],
                }
            )
        return all_characters

    def setup_global_db(self):
        for tmp_character in self.global_db_info:
            for tmp_db in self.global_db_info[tmp_character]:
                tmp_config = KnowledgeEngineConfig(
                    {
                        "db_name": tmp_db,
                        "db_type": "Sqlite",
                        "db_path": f"{GLOBAL_DB_LOCATIONS}/{self.global_db_info[tmp_character][tmp_db]}.db",
                        "chunk_size": 1024,
                        "neighbor_size": 10,
                    }
                )
                self.knowledge_engine.setup_global_knowledge_engine_module(tmp_config)

    def _get_history_db_connection(self, CKStatus):
        current_model_name = CKStatus["current_model"]
        history_id = CKStatus["history_id"]
        tmp_config = KnowledgeEngineConfig(
            {
                "db_name": f"history_{current_model_name}_{history_id}",
                "db_type": "Sqlite",
                "db_path": f"/app/Database_local/history_{current_model_name}_{history_id}.db",
                "chunk_size": 1024,
                "neighbor_size": 10,
            }
        )
        target_ke_module = (
            self.knowledge_engine.get_local_knowledge_engine_module_fresh(
                config_file=tmp_config,
                module_name=f"history_{current_model_name}_{history_id}",
            )
        )
        return target_ke_module

    def _get_online_feedback_db_connection(self, current_model_name, current_user):
        assert current_user != ""
        tmp_config = KnowledgeEngineConfig(
            {
                "db_name": f"online_feedback_{current_model_name}_{current_user}",
                "db_type": "Sqlite",
                "db_path": f"/app/Database_local/online_feedback_{current_model_name}_{current_user}.db",
                "chunk_size": 1024,
                "neighbor_size": 10,
            }
        )
        target_ke_module = (
            self.knowledge_engine.get_local_knowledge_engine_module_fresh(
                config_file=tmp_config,
                module_name=f"online_feedback_{current_model_name}_{current_user}",
            )
        )
        return target_ke_module

    def update_online_feedback_db(self, annotation_input_info):
        if ACTIVATE_SHORT_FEEDBACK:
            user_name = annotation_input_info["username"]
            character_name = annotation_input_info["character_name"]
            annotation = annotation_input_info["messages_in_train_format"]
            messages = json.loads(annotation)
            messages_reverse = messages[::-1]
            user_query = ""
            for tmp_round in messages_reverse:
                if tmp_round["role"] == "user":
                    user_query = tmp_round["content"]
                    break
            current_db_connection = self._get_online_feedback_db_connection(
                character_name, user_name
            )
            current_db_connection.update_from_documents(
                [user_query], [annotation], visualize=False
            )
        else:
            return

    async def _update_history_db(self, messages, target_ke_module):
        start_time = time.time()
        raw_messages = []
        current_message = {"user_query": "", "system_response": ""}
        for tmp_round in messages:
            if tmp_round["role"] == "user":
                if current_message["user_query"] != "":
                    raw_messages.append(copy.deepcopy(current_message))
                current_message["user_query"] = tmp_round["content"]
            elif tmp_round["role"] == "assistant":
                current_message["system_response"] = tmp_round["content"]
        raw_messages.append(copy.deepcopy(current_message))

        current_timestamp = str(datetime.now())
        tmp_round = raw_messages[-1]["user_query"]
        tmp_meta_data = json.dumps(
            {"timestamp": current_timestamp, "content_type": "user_query"}
        )
        target_ke_module.update_from_documents(
            [tmp_round], [tmp_meta_data], visualize=False
        )

    def update_knowledge_engine(self, db_name, db_path, sentences, metadata=None):
        debug_config = KnowledgeEngineConfig(
            {
                "db_name": db_name,
                "db_type": "Sqlite",
                "db_path": db_path,
                "chunk_size": 1024,
                "neighbor_size": 10,
                "retrieval_type": "doc",
            }
        )
        target_ke_module = (
            self.knowledge_engine.get_local_knowledge_engine_module_fresh(
                config_file=debug_config
            )
        )

        target_ke_module.update_from_documents(
            sentences, metadata=metadata, visualize=True
        )
        target_ke_module.setup_retriever()

    def retrieve_feedback(self, query, CKStatus):
        """This function will find the top 1 relevant annotation based on the query.

        Args:
            query (_type_): _description_
            CKStatus (_type_): _description_
        """
        current_model_name = CKStatus["current_model"]
        username = CKStatus["username"]
        db_name = f"online_feedback_{current_model_name}_{username}"
        print("trying to retrieve feedback from db:", db_name)

        current_db_connection = self._get_online_feedback_db_connection(
            current_model_name, username
        )

        _, relevant_meta_data, _ = (
            current_db_connection.find_relevant_knowledge_single_document(
                query=query,
                retrieval_mode="doc",
                sim_threshold=0.5,
                soft_match_top_k=10,
            )
        )
        return relevant_meta_data

    def retrieve_history(self, query, CKStatus):
        """This function will find the relevant history based on the query.
        Args:
            messages (_type_): _description_
        """
        current_model_name = CKStatus["current_model"]
        history_id = CKStatus["history_id"]
        db_name = f"history_{current_model_name}_{history_id}"
        if db_name not in self.knowledge_engine.local_id2module:
            print("We do not have the history db yet.")
            return [], []

        relevant_history, relevant_meta_data, _ = (
            self.knowledge_engine.find_relevant_info_single_db(
                db_name=f"history_{current_model_name}_{history_id}",
                query=query,
                retrieval_mode="doc",
                soft_match_top_k=10,
                sim_threshold=-10000,
            )
        )
        relevant_meta_data = [json.loads(md) for md in relevant_meta_data]
        time2messages = dict()
        time2summaries = dict()
        for i, tmp_history in enumerate(relevant_history):
            if relevant_meta_data[i]["content_type"] == "summary":
                time2summaries[relevant_meta_data[i]["timestamp"]] = tmp_history
            elif relevant_meta_data[i]["content_type"] == "user_query":
                time2messages[relevant_meta_data[i]["timestamp"]] = tmp_history
            else:
                raise NotImplementedError

        sorted_times = sorted(time2messages.keys())
        sorted_messages = []
        for t in sorted_times:
            sorted_messages.append(time2messages[t])

        sorted_times = sorted(time2summaries.keys())
        sorted_summaries = []
        for t in sorted_times:
            sorted_summaries.append(time2summaries[t])

        return sorted_messages, sorted_summaries

    def _get_system_message(self, query, CKStatus):
        """This function will generate the

        Args:
            messages (_type_): _description_
            with_examples (bool, optional): _description_. Defaults to True.
        """
        session_id = CKStatus["session_id"]
        current_character = CKStatus["current_model"]
        system_message = (
            self.character_to_info[current_character]["head_system_prompt"] + "\n"
        )

        for tmp_system_prompt_step in self.character_to_info[current_character][
            "system_prompt_sequence"
        ]:
            if tmp_system_prompt_step["pre_defined"] == True:
                if tmp_system_prompt_step["name"] == "available_functions":
                    system_message += "We have the following available functions:\n"
                    for tmp_function in self.character_to_info[current_character][
                        "function2prompts"
                    ]:
                        system_message += (
                            self.character_to_info[current_character][
                                "function2prompts"
                            ][tmp_function]
                            + "\n"
                        )
                elif tmp_system_prompt_step["name"] == "uploaded_files":
                    if ACTIVATE_KE:
                        file_names = []
                        file_db2descriptions = []
                        for tmp_file_name in CKStatus["uploaded_files"]:
                            file_db2descriptions.append(f"{tmp_file_name}_{session_id}")
                            file_names.append(f"{FILE_LOCATIONS}/{tmp_file_name}")
                        if len(file_db2descriptions) > 0:
                            system_message += (
                                "Avaible DB names and their descriptions:\n"
                            )
                            system_message += "\n".join(file_db2descriptions)
                        system_message += "\n"
                        if len(file_names) > 0:
                            system_message += "Available file paths:\n"
                            system_message += "\n".join(file_names)
                        system_message += "\n"
                        if file_db2descriptions:
                            system_message += "Attention: User just uploaded a file. Please pay attention to that.\n"
                    else:
                        file_names = []
                        for tmp_file_name in CKStatus["uploaded_files"]:
                            file_names.append(f"{FILE_LOCATIONS}/{tmp_file_name}")
                        system_message += "\n"
                        if len(file_names) > 0:
                            system_message += "Available file paths:\n"
                            system_message += "\n".join(file_names)
                        system_message += "\n"
                else:
                    raise NotImplementedError(
                        f"pre_defined step {tmp_system_prompt_step['name']} is not implemented."
                    )
            else:
                if tmp_system_prompt_step["step_type"] == "static":
                    system_message += tmp_system_prompt_step["head"] + "\n"
                    system_message += (
                        "\n".join(tmp_system_prompt_step["content"]) + "\n"
                    )
                elif tmp_system_prompt_step["step_type"] == "dynamic":
                    system_message += tmp_system_prompt_step["head"] + "\n"
                    system_message += (
                        "\n".join(CKStatus[tmp_system_prompt_step["CK_status_key"]])
                        + "\n"
                    )
                else:
                    raise NotImplementedError(
                        f"step type {tmp_system_prompt_step['step_type']} is not implemented."
                    )
        if ACTIVATE_HISTORY:
            start_time = time.time()
            relevent_messages, relevant_summaries = self.retrieve_history(
                query=query, CKStatus=CKStatus
            )
            print("retrieving history time:", time.time() - start_time)
            system_message += "The following sentences are retrieved from the user's previous dialogs. If any of them are relevant to the user's current query then use them when responding to the user: "
            system_message += str(relevent_messages)
            system_message += "\n\n"
        if ACTIVATE_SHORT_FEEDBACK:
            start_time = time.time()
            relevant_feedback = self.retrieve_feedback(query=query, CKStatus=CKStatus)
            if len(relevant_feedback) > 0:
                relevant_feedback = relevant_feedback[:1]
            print("retrieving history time:", time.time() - start_time)
            system_message += "We have the following feedback:\n"
            for tmp_feedback in relevant_feedback:
                system_message += json.dumps(json.loads(tmp_feedback)[1:]) + "\n"
            system_message += "\n"
        system_message += (
            self.character_to_info[current_character]["tail_system_prompt"] + "\n"
        )
        return system_message

    def AskLLM(self, query):
        """This function will ask the base language model with a single query
        Args:
            query (str): input query
        Returns:
            str: returned query
        """
        tmp_messages = [
            {
                "role": "system",
                "name": "head",
                "content": "You are Kirkland, a helpful AI assistant, with no access to external functions.",
            },
            {"role": "user", "content": query},
        ]
        tmp_result = self.LLM_connection.get_response(tmp_messages)
        return tmp_result

    async def planning_execution(self, planning_code, username, session_id, message_id):
        current_id = f"{username}_{session_id}"
        newly_created_executor, current_executor = (
            self.executor_manager.get_or_create_executor(executor_id=current_id)
        )
        if newly_created_executor:
            code = f"""
def CallWeb(query, target_url):
    for tmp_response in my_call_web(
        query=query,
        target_url=target_url,
        session_id='{session_id}',
        message_id='{message_id}',
        username='{username}',
    ):
        print(tmp_response)
    """
            code += planning_code
        else:
            code = planning_code
        global_vars = globals().copy()
        global_vars["CallMemoryKernel"] = self.CallMemoryKernel
        global_vars["AskLLM"] = self.AskLLM
        global_vars["my_call_web"] = self.my_call_web
        current_executor.submit_code(
            code=code,
            global_variable=global_vars,
        )
        async for tmp_output in current_executor.async_output():
            yield tmp_output

    def my_call_web(self, query, target_url, session_id, message_id, username):
        for tmp_response in call_web(
            llm_connection=self.LLM_connection,
            query=query,
            target_url=target_url,
            session_id=session_id,
            message_id=message_id,
            username=username,
        ):
            yield tmp_response

    def CallMemoryKernel(self, query, db_name):
        """This function will call the local memory kernel to get the corresponding information.
        Args:
            query (_type_): _description_
            db_name (_type_): _description_
        """

        retrieved_info, relevant_meta_data, _ = (
            self.knowledge_engine.find_relevant_info_single_db(
                db_name=db_name,
                query=query,
                retrieval_mode="doc",
            )
        )

        response = [
            (meta, text) for meta, text in zip(relevant_meta_data, retrieved_info)
        ]

        return response

    async def generate(
        self,
        messages,
        CKStatus,
        username="test",
        message_id="",
        max_trials=3,
    ):
        """This function is the main generation function of cognitive kernel. It will respond based on the current messages and role.

        Args:
            messages (_type_): the target message
            message_log (_type_): the target message log
            role (str, optional): which role we should as the model to perform. select from ['actor', 'critic', 'improver']. Defaults to 'actor'.
        """
        user_query = messages[-1]["content"]
        local_messages = copy.deepcopy(messages)
        system_message = self._get_system_message(query=user_query, CKStatus=CKStatus)
        local_messages = [
            {"role": "system", "name": "head", "content": system_message}
        ] + local_messages
        local_messages.append(
            {
                "role": "system",
                "name": "actor",
                "content": "Please generate the slow thinking code",
            }
        )
        raw_data_logging = copy.deepcopy(local_messages)
        current_pos = 0
        updated_messages = [
            {
                "group": f"assistant_slow_thinking",
                "pos": current_pos,
                "content": "",
            }
        ]
        current_trial = 0
        continue_generation = True
        previous_raw_planning_code = ""
        while current_trial < max_trials and continue_generation:
            current_trial += 1
            if previous_raw_planning_code == "":
                raw_data_logging.append({"role": "assistant", "content": ""})
                async for tmp_output in generate_stream(
                    current_connection=self.LLM_connection, messages=local_messages
                ):
                    updated_messages[-1]["content"] += tmp_output
                    raw_data_logging[-1]["content"] += tmp_output
                    yield json.dumps(updated_messages)

                raw_planning_code = updated_messages[-1]["content"]
            else:
                raw_planning_code = previous_raw_planning_code

            local_messages.append({"role": "assistant", "content": raw_planning_code})
            planning_decision = "code"
            if "Direct answering" in raw_planning_code:
                planning_decision = "direct"
                planning_code = raw_planning_code
                execution_result = ""
            elif "Additional Info" in raw_planning_code:
                planning_decision = "need_info"
                planning_code = raw_planning_code
                execution_result = ""
            else:
                if "```" in raw_planning_code:
                    planning_decision = "code"
                    parts = raw_planning_code.split("```")
                    planning_code = parts[1] if len(parts) > 1 else ""
                    planning_code = planning_code.replace("python", "")
                    execution_result = ""
                    current_status = "empty"
                    async for tmp_execution_result in self.planning_execution(
                        planning_code=planning_code,
                        username=username,
                        session_id=CKStatus["session_id"],
                        message_id=message_id,
                    ):
                        if "[WEB]" in tmp_execution_result:
                            if current_status == "empty":
                                execution_result = tmp_execution_result
                                current_pos += 1
                                updated_messages.append(
                                    {
                                        "group": f"assistant_web_result",
                                        "pos": current_pos,
                                        "content": execution_result,
                                    }
                                )
                                current_status = "web"
                            elif current_status == "web":
                                execution_result += tmp_execution_result
                                updated_messages[-1]["content"] = execution_result
                            else:
                                current_pos += 1
                                execution_result = tmp_execution_result
                                updated_messages.append(
                                    {
                                        "group": f"assistant_web_result",
                                        "pos": current_pos,
                                        "content": execution_result,
                                    }
                                )
                                current_status = "web"
                        elif "[/WEB]" in tmp_execution_result:
                            assert current_status == "web"
                            web_content = tmp_execution_result.split("[/WEB]")[0]
                            normal_content = tmp_execution_result.split("[/WEB]")[1]
                            execution_result += web_content
                            updated_messages[-1]["content"] = execution_result
                            current_pos += 1
                            execution_result = normal_content
                            updated_messages.append(
                                {
                                    "group": f"assistant_execution_result",
                                    "pos": current_pos,
                                    "content": execution_result,
                                }
                            )
                            current_status = "text"
                        else:
                            if current_status == "empty":
                                current_pos += 1
                                updated_messages.append(
                                    {
                                        "group": f"assistant_execution_result",
                                        "pos": current_pos,
                                        "content": tmp_execution_result,
                                    }
                                )
                                current_status = "text"
                                execution_result = tmp_execution_result
                            else:
                                print("I should not go there")
                                execution_result += tmp_execution_result
                                updated_messages[-1]["content"] = execution_result
                        yield json.dumps(updated_messages)

                else:
                    planning_decision = "direct"
                    planning_code = raw_planning_code
                    execution_result = ""
            if planning_decision == "code":
                cleaned_execution_result = execution_result.replace("stop", "")
                pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
                new_cleaned = []
                for step in cleaned_execution_result.split("\n"):
                    new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
                cleaned_execution_result = "\n".join(new_cleaned)
                local_messages.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                    }
                )
                raw_data_logging.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                    }
                )
            elif planning_decision == "direct":
                cleaned_execution_result = ""
                local_messages.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Directly answer the user query.",
                    }
                )
                raw_data_logging.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Directly answer the user query.",
                    }
                )
            elif planning_decision == "need_info":
                cleaned_execution_result = ""
                local_messages.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Asking for more information from the user.",
                    }
                )
                raw_data_logging.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Asking for more information from the user.",
                    }
                )
            current_pos += 1
            updated_messages.append(
                {
                    "group": f"assistant_final_output",
                    "pos": current_pos,
                    "content": "",
                }
            )
            raw_data_logging.append({"role": "assistant", "content": ""})
            final_output = ""
            async for tmp_output in generate_stream(
                current_connection=self.LLM_connection, messages=local_messages
            ):
                final_output += tmp_output
                raw_data_logging[-1]["content"] += tmp_output
                if tmp_output not in ["<|im_continue|>"]:
                    updated_messages[-1]["content"] += tmp_output
                    yield json.dumps(updated_messages)
            if "<|im_continue|>" in final_output:
                continue_generation = True
                previous_raw_planning_code = final_output
                updated_messages[-1]["content"] = final_output.replace(
                    "<|im_continue|>", ""
                )
                updated_messages[-1]["group"] = "assistant_slow_thinking"
                yield json.dumps(updated_messages)
            else:
                continue_generation = False
            update_or_create_rawdata(
                session_id=CKStatus["session_id"],
                message_id=message_id,
                username=username,
                messages_in_train_format=raw_data_logging,
                updated_time=datetime.now().isoformat(),
            )

        # saving the local history to a history db.
        if ACTIVATE_HISTORY:
            target_history_ke = self._get_history_db_connection(CKStatus=CKStatus)
            task = asyncio.create_task(
                self._update_history_db(raw_data_logging, target_history_ke)
            )

    async def clean_up(self, CKStatus, username):
        current_id = f"{username}_{CKStatus['session_id']}"
        self.executor_manager.cleanup_executor(task_id=current_id)

    async def generate_for_demo(
        self, messages, CKStatus, username, message_id, mode="normal"
    ):
        """This is the main generation function for the demo

        Args:
            messages (_type_): user messages in the chatgpt format
            CKStatus (_type_): all meta information about the current CK conversation
        Returns:
            _type_: the final output.
        """

        print("messages:", messages)
        print("mode:", mode)
        async for updated_message in self.generate(
            messages=messages,
            CKStatus=CKStatus,
            username=username,
            message_id=message_id,
        ):
            yield updated_message

    async def inference_api(self, input_messages, full_info=False):
        current_messages = copy.deepcopy(input_messages)
        user_query = current_messages[-1]["content"]
        role = "actor"
        current_messages.append(
            {
                "role": "system",
                "name": "actor",
                "content": "Please generate the slow thinking code",
            }
        )
        raw_planning_code = self.LLM_connection.get_response(current_messages)
        current_messages.append({"role": "assistant", "content": raw_planning_code})
        planning_decision = "code"
        tmp_session_id = "evaluation_" + uuid.uuid4().hex
        tmp_message_id = "evaluation_" + uuid.uuid4().hex
        if "Direct answering" in raw_planning_code:
            planning_decision = "direct"
            planning_code = raw_planning_code
            execution_result = ""
        elif "Additional Info" in raw_planning_code:
            planning_decision = "need_info"
            planning_code = raw_planning_code
            execution_result = ""
        else:
            if "```" in raw_planning_code:
                planning_decision = "code"
                parts = raw_planning_code.split("```")
                planning_code = parts[1] if len(parts) > 1 else ""
                planning_code = planning_code.replace("python", "")
                execution_result = ""
                current_pos = 1
                current_status = "empty"
                async for tmp_execution_result in self.planning_execution(
                    planning_code=planning_code,
                    username="evaluation",
                    session_id=tmp_session_id,
                    message_id=tmp_message_id,
                ):
                    if "[WEB]" in tmp_execution_result:
                        if current_status == "empty":
                            execution_result = tmp_execution_result
                            current_status = "web"
                        elif current_status == "web":
                            execution_result += tmp_execution_result
                        else:
                            current_pos += 1
                            execution_result = tmp_execution_result
                            current_status = "web"
                    elif "[/WEB]" in tmp_execution_result:
                        assert current_status == "web"
                        web_content = tmp_execution_result.split("[/WEB]")[0]
                        normal_content = tmp_execution_result.split("[/WEB]")[1]
                        execution_result += web_content
                        current_pos += 1
                        execution_result = normal_content
                        current_status = "text"
                    else:
                        if current_status == "empty":
                            execution_result = tmp_execution_result
                            current_status = "text"
                        else:
                            execution_result += tmp_execution_result
            else:
                planning_decision = "direct"
                planning_code = raw_planning_code
                execution_result = ""
        if planning_decision == "code":
            cleaned_execution_result = execution_result.replace("stop", "")
            pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
            new_cleaned = []
            for step in cleaned_execution_result.split("\n"):
                new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
            cleaned_execution_result = "\n".join(new_cleaned)
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                }
            )
        elif planning_decision == "direct":
            cleaned_execution_result = ""
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"Directly answer the user query.",
                }
            )
        elif planning_decision == "need_info":
            cleaned_execution_result = ""
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"Asking for more information from the user.",
                }
            )
        final_output = self.LLM_connection.get_response(current_messages)
        current_messages.append({"role": "assistant", "content": final_output})
        if full_info:
            other_logs = get_rawdata_by_session_id(session_id=tmp_session_id)
            return {"messages": current_messages, "other_logs": other_logs}
        else:
            return current_messages

    async def inference_api_history_retrieval(self, candidate_messages, target_query):
        tmp_history_db_name = f"history_tmp_{uuid.uuid4().hex}"
        tmp_config = KnowledgeEngineConfig(
            {
                "db_name": tmp_history_db_name,
                "db_type": "Sqlite",
                "db_path": f"/app/Database_local/{tmp_history_db_name}.db",
                "chunk_size": 1024,
                "neighbor_size": 10,
            }
        )
        target_ke_module = (
            self.knowledge_engine.get_local_knowledge_engine_module_fresh(
                config_file=tmp_config, module_name=tmp_history_db_name
            )
        )
        for tmp_round in candidate_messages:
            await self._update_history_db(tmp_round, target_ke_module)
        doc, metadata, _ = self.knowledge_engine.find_relevant_info_single_db(
            db_name=tmp_history_db_name,
            query=target_query,
            retrieval_mode="doc",
            soft_match_top_k=10,
            sim_threshold=-10000,
        )

        sorted_res = sorted(
            zip(doc, metadata), key=lambda x: json.loads(x[1])["timestamp"]
        )
        return [i[0] for i in sorted_res]

    async def inference_api_call_web(
        self,
        query,
        target_url,
        session_id,
        message_id,
        username,
        max_steps,
        storage_state,
        geo_location,
    ):
        results = []
        for tmp_response in call_web(
            llm_connection=self.LLM_connection,
            query=query,
            target_url=target_url,
            session_id=session_id,
            message_id=message_id,
            username=username,
            max_steps=max_steps,
            storage_state=storage_state,
            geo_location=geo_location,
            yield_full_message=True,
        ):
            results.append(tmp_response)
        return results
