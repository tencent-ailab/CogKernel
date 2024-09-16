import re
import os
import requests
from datetime import datetime
from database import (
    update_or_create_session,
    update_or_create_rawdata,
    update_or_create_annotation,
)

WEB_IP = os.getenv("WEB_IP", "/web:3000")

def extract_info_from_action(generated_action):
    action_pattern = r"(\w+) \[(\d+)\](?: \[([^\]]*)\])?( \[1\])?"
    match = re.search(action_pattern, generated_action)
    if match:
        action_name, target_id, action_value, enter_marker = match.groups()
        need_enter = enter_marker is not None
        if (action_name == 'type' and action_value is not None) or action_name == 'click':
            return (
                action_name,
                target_id,
                action_value if action_value else None,
                need_enter,
            )
    else:
        scroll_pattern = r"(\w+) \[([^\d\]]*)\]( \[1\])?"
        scroll_match = re.search(scroll_pattern, generated_action)
        if scroll_match:
            action_name, action_value, enter_marker = scroll_match.groups()
            action_value = action_value.replace('direction=', '')
            return action_name, None, action_value, enter_marker is not None
    if generated_action.startswith("type") or generated_action.startswith("click"):
        backup_pattern = r"(\w+)\s*(\[\d+\]|\d+)?\s*([^\[]+)?\s*(\[press_enter_after=\d\]|\[1\]|1)?"
        backup_match = re.search(backup_pattern, generated_action)
        if backup_match:
            action_name, target_id, action_value, enter_marker = backup_match.groups()
            if target_id.startswith("["):
                target_id = target_id[1:-1]
            need_enter = enter_marker is not None
            return action_name, target_id, action_value, need_enter
    elif generated_action.startswith("scroll"):
        action_name = "scroll"
        if len(generated_action.split(" ")) > 1:
            action_value = generated_action.split(" ")[1]
        else:
            action_value = None
        print ('BACKUP SCROLL:', action_value)
        return action_name, None, action_value, False
    return None, None, None, False


def find_target_element_info(current_accessbility_tree, target_id, action_name):
    if target_id is None:
        return None, None, None

    if action_name == "type":
        tree_to_check = current_accessbility_tree.split("\n")[int(target_id) - 1 :]
        for i, line in enumerate(tree_to_check):
            if f"[{target_id}]" in line and ("combobox" in line or "box" not in line):
                num_tabs = len(line) - len(line.lstrip("\t"))
                for j in range(i + 1, len(tree_to_check)):
                    curr_num_tabs = len(tree_to_check[j]) - len(
                        tree_to_check[j].lstrip("\t")
                    )
                    if curr_num_tabs <= num_tabs:
                        break
                    if "textbox" in tree_to_check[j] or "searchbox" in tree_to_check[j]:
                        target_element_id = tree_to_check[j].split("]")[0].strip()[1:]
                        print(
                            "CATCHED ONE MISSED TYPE ACTION, changing the type action to",
                            target_element_id,
                        )
                        target_id = target_element_id
    target_pattern = r"\[" + re.escape(target_id) + r"\] ([a-z]+) '(.*)'"
    matches = re.finditer(target_pattern, current_accessbility_tree, re.IGNORECASE)
    for match in matches:
        target_element_type, target_element_name = match.groups()
        return target_id, target_element_type, target_element_name
    return target_id, None, None


def extract_action_for_web(current_accessbility_tree, raw_action_code, expanded_part):
    try:
        thought = raw_action_code.split("Action:")[0].replace("Thought:", "").strip()
    except:
        thought = None
    action_part = raw_action_code.split("Action:")[1]
    start = action_part.find("```")
    end = action_part.rfind("```")

    if start != -1 and end != -1 and start != end:
        generated_action = action_part[start + 3:end].strip()
    else:
        print("No matching triple backticks found or only one set found.")
    if (
        generated_action.lower().startswith("goback")
        or generated_action.lower().startswith("restart")
        or generated_action.lower().startswith("wait")
        or generated_action.lower().startswith("stop")
    ):
        if generated_action.lower().startswith("goback"):
            action_name = "goback"
        elif generated_action.lower().startswith("restart"):
            action_name = "restart"
        elif generated_action.lower().startswith("wait"):
            action_name = "wait"
        elif generated_action.lower().startswith("stop"):
            action_name = "stop"
        return (
            {
                "action_name": action_name,
                "target_id": None,
                "action_value": None,
                "need_enter": None,
                "target_element_type": None,
                "target_element_name": None,
            },
            generated_action,
            thought,
        )
    else:
        action_name, target_id, action_value, need_enter = extract_info_from_action(
            generated_action
        )
        target_id, target_element_type, target_element_name = find_target_element_info(
            current_accessbility_tree, target_id, action_name
        )
        if expanded_part and int(target_id) in expanded_part:
            expand_target_id, expand_target_type, expand_target_name = expanded_part[int(target_id)]
            print ("Expanded target found", expand_target_id, expand_target_type, expand_target_name, target_element_name)
            return ({
                "action_name": 'select',
                "target_id": expand_target_id,
                "action_value": target_element_name,
                "need_enter": None,
                "target_element_type": expand_target_type,
                "target_element_name": expand_target_name,
            },
            generated_action,
            thought,
            )
        return (
            {
                "action_name": action_name,
                "target_id": target_id,
                "action_value": action_value,
                "need_enter": need_enter,
                "target_element_type": target_element_type,
                "target_element_name": target_element_name,
            },
            generated_action,
            thought,
        )


def get_browser(storage_state, geo_location):
    url = "http://web:3000/getBrowser"
    data = {"storageState": storage_state, "geoLocation": geo_location}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()["browserId"]
        else:
            return None
    except requests.RequestException as e:
        print(f"Request Error: {e}")


def close_browser(browser_id):
    url = "http://web:3000/closeBrowser"
    data = {"browserId": browser_id}
    print(f"Closing browser {browser_id}")
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return None
        else:
            return None
    except requests.RequestException as e:
        print(f"Request Error: {e}")


def open_page(browser_id, target_url):
    url = "http://web:3000/openPage"
    data = {"browserId": browser_id, "url": target_url}

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return True, response.json()["pageId"]
        else:
            print(f"Open page Request failed with status code: {response.status_code}")
            return False, ""
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False, ""


def get_accessibility_tree(browser_id, page_id, current_round):
    url = "http://web:3000/getAccessibilityTree"
    data = {
        "browserId": browser_id,
        "pageId": page_id,
        "currentRound": current_round,
    }

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            res_json = response.json()
            AccessibilityTree = res_json.get("yaml", [])
            curr_url = res_json.get("url", "")
            snapshot = res_json.get("snapshot", "")
            return True, AccessibilityTree, curr_url, snapshot
        else:
            print(
                f"Get accessibility tree Request failed with status code: {response.status_code}"
            )
            return False, None, None, None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False, None, None, None


def action(browser_id, page_id, action):
    url = "http://web:3000/performAction"
    data = {
        "browserId": browser_id,
        "pageId": page_id,
        "actionName": action["action_name"],
        "targetId": action["target_id"],
        "targetElementType": action["target_element_type"],
        "targetElementName": action["target_element_name"],
        "actionValue": action["action_value"],
        "needEnter": action["need_enter"],
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return True
        else:
            print(
                f"Request failed with status code: {response.status_code} {response.text}"
            )
            return False
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False

def is_annoying(current_accessbility_tree):
    if "See results closer to you?" in current_accessbility_tree and len(current_accessbility_tree.split("\n")) <= 10:
        return True
    return False

def get_skip_action(current_accessbility_tree):
    action_name, target_id, action_value, need_enter = extract_info_from_action(
        "click [5]"
    )
    target_id, target_element_type, target_element_name = find_target_element_info(
        current_accessbility_tree, target_id, action_name
    )
    return {
            "action_name": action_name,
            "target_id": target_id,
            "action_value": action_value,
            "need_enter": need_enter,
            "target_element_type": target_element_type,
            "target_element_name": target_element_name,
        }

system_message = """You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current observation (web page's accessibility tree): This is a simplified representation of the webpage, providing key information.
The open tabs: These are the tabs you have open.
The previous actions: You can refer to the conversation history with the user to see the actions you have taken. It may be helpful to track your progress.

The actions you can perform are the following:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.
`wait`: Wait for the page to load, with a duration of 5 seconds.
`goback`: Navigate to the previously viewed page.
`restart`: Navigate to the Google search homepage. When you can't find information in some websites, try starting over from Google search.
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation. For example, you should NOT type into buttons or click on statictext.
2. You should only issue one action at a time.
3. STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.
4. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.

Your reply should strictly follow the format:
Thought: {Your brief thoughts (briefly summarize the info that will help complete the task)} Action: ```{the next action you choose to take}```"""

def call_web(
    llm_connection,
    query,
    target_url,
    session_id,
    message_id,
    username,
    max_steps=12,
    storage_state=None,
    geo_location=None,
    yield_full_message=False,
):
    """Makes an asynchronous GET request to a target URL with a query parameter.

    Args:
        query (str): The query parameter value to be sent with the request.
        target_url (str): The target URL to which the request is made.

    Returns:
        dict: The JSON response from the target URL, if successful.
    """
    # Use aiohttp.ClientSession for making the HTTP request
    browser_id = get_browser(storage_state, geo_location)
    if browser_id is None:
        yield None

    all_messages = [{"role": "system", "name": "head", "content": system_message}]

    _, page_id = open_page(browser_id, target_url)
    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot = get_accessibility_tree(
        browser_id, page_id, 0
    )
    counter = 0
    repeat_count = 0
    prev_obs = current_accessbility_tree
    prev_action = None
    expanded_part = None
    returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] goto {target_url}\n"
    yield returned_info
    action_error = False
    response_error = False
    downloaded_files = []
    while get_accessibility_tree_succeed:

        if counter > max_steps:
            break
        if action_error:
            current_user_message = "The action you have chosen cannot be executed. Please double-check if you have selected the correct element or used correct action format. Then provide the revised Thought and Action."
            action_error = False
        elif response_error:
            current_user_message = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
            response_error = False
        else:
            if repeat_count >= 2:
                print ('Browsing is getting stuck, stop here')
                break
            current_user_message = (
                "OBSERVATION:\n" + current_accessbility_tree + "OBJECTIVE: " + query
            )
        current_messages = list()
        for tmp_message in all_messages:
            if tmp_message["role"] == "user":
                current_messages.append({"role": "user", "content": "<|im_omitted|>"})
            else:
                current_messages.append(tmp_message)
        current_messages.append({"role": "user", "content": current_user_message, 'step_url': step_url})
        all_messages.append({"role": "user", "content": current_user_message})
        next_action = llm_connection.get_response(current_messages)
        print(next_action)
        current_messages.append({"role": "assistant", "content": next_action})
        if yield_full_message:
            yield current_messages
        update_or_create_rawdata(
            session_id=session_id,
            message_id=f"{message_id}@@web@@{counter+1}",
            username=username,
            messages_in_train_format=current_messages,
            updated_time=datetime.now().isoformat(),
        )
        print("RESPONSE", next_action)
        all_messages.append({"role": "assistant", "content": next_action})
        extracted_action, action_string, extracted_thought = extract_action_for_web(
            current_accessbility_tree, next_action, expanded_part
        )
        print("EXTRACTED ACTION", extracted_action)
        counter += 1
        if extracted_action["action_name"] is None:
            get_accessibility_tree_succeed = True
            if extracted_thought is not None:
                action_error = True
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
            else:
                response_error = True
                returned_info = (
                    f"[WEB] [{browser_id}] [{page_id}] [{counter}] {action_string}\n"
                )
            yield returned_info
        else:
            if extracted_action["action_name"] == "stop":
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                yield returned_info
                if len(downloaded_files) == 0:
                    returned_info = f"[/WEB] {action_string}"
                else:
                    returned_info = f"[/WEB] {action_string}\nDownloaded file paths:\n"
                    for path in downloaded_files:
                        returned_info += f"{path}\n"
                yield returned_info
                break

            action_succeed = action(browser_id, page_id, extracted_action)

            if not action_succeed:
                action_error = True
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                yield returned_info
            else:
                get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot = (
                    get_accessibility_tree(browser_id, page_id, counter)
                )
                if 'downloaded the following files:' in current_accessbility_tree:
                    path_str = current_accessbility_tree.split('downloaded the following files:')[1].strip().split('OBJECTIVE:')[0]
                    for path in path_str.split('\n'):
                        downloaded_files.append(path)
                # Google may have pop up asking for enabling location service
                if is_annoying(current_accessbility_tree):
                    skip_this_action = get_skip_action(current_accessbility_tree)
                    action(browser_id, page_id, skip_this_action)
                    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot = (
                        get_accessibility_tree(browser_id, page_id, counter)
                    )
                # try to close cookie popup 
                if "Cookie banner" in current_accessbility_tree:
                    cookie_message = (
                    "OBSERVATION:\n" + current_accessbility_tree + "OBJECTIVE: There is a cookie banner on the page, please accept the cookie banner." 
                    )
                    popup_messages = [{"role": "system", "name": "head", "content": system_message}]
                    popup_messages.append({"role": "user", "content": cookie_message})
                    next_action = llm_connection.get_response(popup_messages)
                    extracted_action, action_string, extracted_thought = extract_action_for_web(
                        current_accessbility_tree, next_action, expanded_part
                    )
                    action(browser_id, page_id, extracted_action)

                    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot = (
                        get_accessibility_tree(browser_id, page_id, counter)
                    )
                current_accessbility_tree, expanded_part = check_if_menu_is_expanded(current_accessbility_tree, snapshot)
                if current_accessbility_tree == prev_obs and prev_action == extracted_action:
                    repeat_count += 1
                else:
                    repeat_count = 0
                prev_obs = current_accessbility_tree
                prev_action = extracted_action
                returned_info = (
                    f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                )
                yield returned_info
    close_browser(browser_id)

def find_node_with_children(node, target_role, target_name):
    # Check if the current node matches the target role and name
    if node.get('role') == target_role and node.get('name') == target_name:
        return node.get('children', None)
    
    # If the node has children, recursively search through them
    children = node.get('children', [])
    for child in children:
        result = find_node_with_children(child, target_role, target_name)
        if result is not None:
            return result
    
    # If no matching node is found, return None
    return None

def check_if_menu_is_expanded(accessibility_tree, snapshot):
    node_to_expand = {}
    lines = accessibility_tree.split("\n")
    for i, line in enumerate(lines):
        if 'hasPopup: menu' in line and 'expanded: true' in line:
            num_tabs = len(line) - len(line.lstrip("\t"))
            next_tabs = len(lines[i+1]) - len(lines[i+1].lstrip("\t"))
            if next_tabs <= num_tabs:
                # In this case, the menu should be expanded but is not present in the tree
                target_pattern = r"\[(\d+)\] ([a-z]+) '(.*)'"
                matches = re.finditer(target_pattern, line, re.IGNORECASE)
                target_id = None
                target_element_type = None
                target_element_name = None
                for match in matches:
                    target_id, target_element_type, target_element_name = match.groups()
                    break
                if target_element_type is not None:
                    # locate the menu items from the snapshot instead
                    children = find_node_with_children(snapshot, target_element_type, target_element_name)
                    if children is not None:
                        node_to_expand[i] = (num_tabs+1, children, target_id, target_element_type, target_element_name)
    new_lines = []
    curr = 1
    if len(node_to_expand) == 0:
        return accessibility_tree, None
    expanded_part = {}
    # add the menu items to the correct location in the tree
    for i, line in enumerate(lines):
        if not line.strip().startswith('['):
            new_lines.append(line)
            continue
        num_tabs = len(line) - len(line.lstrip("\t"))
        content = line.split('] ')[1]
        new_lines.append('\t'*num_tabs+f"[{curr}] {content}")
        curr += 1
        if i in node_to_expand:
            for child in node_to_expand[i][1]:
                child_content = f"{child.get('role', '')} '{child.get('name', '')}' " + ' '.join([f"{k}: {v}" for k, v in child.items() if k not in ['role', 'name']])
                tabs = '\t'*node_to_expand[i][0]
                new_lines.append(f"{tabs}[{curr}] {child_content}")
                expanded_part[curr] = (node_to_expand[i][2], node_to_expand[i][3], node_to_expand[i][4])
                curr += 1
    return '\n'.join(new_lines), expanded_part
