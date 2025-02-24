from yelpsimulator import Simulator
from yelpsimulator.agents.recommendation_agent import RecommendationAgent
import httpx
import re
import ast
from openai import OpenAI
import tiktoken
import time
import os
import json
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

def llm_response(prompt: str, model: str = "deepseek-chat", temperature: float = 0.5) -> str:

    import http.client

    conn = http.client.HTTPSConnection("cloud.infini-ai.com")

    payload_dict = {
        "model": "qwen2.5-72b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    # 转换为 JSON 字符串，json.dumps 会自动处理转义
    payload = json.dumps(payload_dict, ensure_ascii=False)    

    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer sk-dakqyjy2pusruxx7"
    }

        # 尝试发送请求
    conn.request("POST", "/maas/v1/chat/completions", str(payload), headers)
        
    res = conn.getresponse()
    data = res.read()    
    output = data.decode("utf-8")
    response_dict = json.loads(output)
    
    try:
        llm_output = response_dict["choices"][0]["message"]["content"]
    except:
        llm_output = ''

    return llm_output


class PlanningBase():
    def __init__(self, llms_type):
        self.plan = []
        self.llm_type = llms_type[0]
    
    def create_prompt(self, task_type, task_description, ):
        raise NotImplementedError("Subclasses should implement this method")
    
    def __call__(self, task_type, task_description):
        prompt = self.create_prompt(task_type, task_description)
        string = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)

        pattern = r"([a-zA-Z])'([a-zA-Z])"
        string = re.sub(pattern, r"\1\'\2", string)
        dict_strings = re.findall(r"\{[^{}]*\}", string)
        dicts = [ast.literal_eval(ds) for ds in dict_strings]
        self.plan = dicts
        return self.plan
    
class PlanningIO(PlanningBase):
    def create_prompt(self, task_type, task_description):
        prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {{'time': '2023-12-01', 'user': 'user_id_example', 'business': 'business_id_example'}}
sub-task 1: {{'description': 'First I need to find user information', 'reasoning instruction': 'None', 'tool use instruction': user_id_example}}
sub-task 2: {{'description': 'Next, I need to find business information', 'reasoning instruction': 'None', 'tool use instruction': business_id_example}}


Task: {task_description}
'''
        prompt = prompt.format(task_description=task_description, task_type=task_type)
        return prompt
    
class ReasoningBase:
    def __init__(self, profile_type_prompt, llms_type):
        self.profile_type_prompt = profile_type_prompt
        self.llm_type = llms_type[0]
        
class ReasoningIO(ReasoningBase):
    def __call__(self, task_description: str):
        prompt = '''
{task_description}'''
        prompt = prompt.format(task_description=task_description)
        reasoning_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)
        
        return reasoning_result

class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent.
    """
    def __init__(self):

        super().__init__()
        self.planning = PlanningIO(['deepseek-chat'])
        self.reasoning = ReasoningIO('', ['deepseek-chat'])

    def forward(self):
        """
        Simulate user behavior.
        Returns:
            tuple: (star (float), review_text (str), behavior_metrics (tuple))
        """
        task_type = 'user_behavior_simulation'
        # plan = self.planning(task_type=task_type, task_description=str(self.scenario))
        plan = [
         {'description': 'First I need to find user information'},
         {'description': 'Next, I need to find business information'},
         {'description': 'Next, I need to find review information'}
         ]


        for sub_task in plan:
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.scenario['user_id']))
                input_tokens = num_tokens_from_string(user)

                if input_tokens > 21000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:21000])

                # print(user)
            elif 'business' in sub_task['description']:
                business = []
                for n_bus in range(len(self.scenario['candidate_list'])):
                    business.append(self.interaction_tool.get_item(item_id=self.scenario['candidate_list'][n_bus]))
                input_tokens = num_tokens_from_string(str(business))
                if input_tokens > 21000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    business = encoding.decode(encoding.encode(str(business))[:21000])

                # print(business)
            elif 'review' in sub_task['description']:
                history_review = str(self.interaction_tool.get_reviews(user_id=self.scenario['user_id'])) #list[dict,dict]
                # history_review = str([{key: original_dict[key] for key in ['business_id','stars','text']} for original_dict in history_review_ori])
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 21000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:21000])

        print(history_review)
        task_description = f'''
        You are a Yelp user with the following profile: {user}, and your historical business reviews and ratings are as follows: {history_review}.

        Your task is to rank the following 20 candidate {self.scenario['candidate_category']}-type businesses: {self.scenario['candidate_list']}, based on how well they align with your preferences. Detailed information about these businesses is provided: {business}.

        # To ensure an accurate ranking, follow these steps and reason through each stage explicitly before finalizing the ranking:

        # Instructions for your internal thinking (do not output):
        # 1. Preference Extraction:
        # Summarize your preferences based on your historical reviews and ratings. Identify patterns such as frequently visited categories, keywords in positive reviews, and features/services associated with higher ratings (e.g., >4 stars).
        # Example: If you often mention "great service" or "authentic flavors" in 5-star reviews, prioritize businesses exhibiting these traits.
        # 2.Candidate Evaluation:
        # For each candidate business, analyze how well it matches your preferences using the following criteria:
        # Category Match: Does the business fall into a category or feature you have rated highly in the past?
        # Historical Engagement: Have you visited this business before? If yes, prioritize highly rated ones.
        # Location Proximity: Is the business close to areas you frequent (if location data is available)?
        # Reputation: For new businesses, consider aggregated Yelp ratings and reviews.
        # Sentiment Alignment: Compare sentiment trends in your past reviews to the candidate business's characteristics.
        # 3. Step-by-Step Ranking (CoT):
        # Evaluate all candidates step by step. For each candidate:
        # State the reasoning (e.g., "This business aligns with my preference for X and is highly rated, so it ranks higher").
        # Justify decisions explicitly to finalize the ranking.
        # Example reasoning:
        # "Business A matches my preference for authentic dining experiences and has a 4.5-star rating with reviews mentioning 'authentic flavors,' similar to businesses I’ve rated highly in the past. Therefore, it ranks high on my list."
        # Diversity and Context Adjustment:
        # 4.Ensure variety by balancing similar businesses and introducing diversity when highly ranked options are too concentrated within one category.
        # Consider standout contextual details (e.g., unique features or services) that enhance your interest.

        Final Output:
        After reasoning through all candidates in {self.scenario['candidate_list']}, only return the re-ranked list of business IDs without any other content. Use the following format and include only the ranked list:

        ['business_id1', 'business_id2', 'business_id3', ...]        

        '''

        # task_description = f'''
        # Now you are a real user on Yelp (a platform with crowd-sourced reviews about businesses), your basic profile is: {user}, your historical business review text and stars are as follows: {history_review}.
        # Now you need to rank the following 20 businesses: {self.scenario['candidate_list']} according their match degree to your preference.
        # The detailed information of the above 20 candidate business are as follows: {business}.
        # You are encouraged to (1) summarize your interest based on your historical review and stars about visited business (2) rank more front those business you have visited and given a high stars (>=4) and positive review texts (if they are in the candidate list), these business are more likely you are interested.
        # Finally you need to give a ranked business list combined with the above information. Please rank the more interested business more front in your rank list.
        # Return your ranked business list (each element must be a business id, not business real name) of the provided 20 candidate business: {self.scenario['candidate_list']}.
        # Your output should be of the following format (only a ranked business list without any other content):
        # ['business id1', 'business id2', 'business id3', ...]
        # '''

        try:
            # rec_list = [line for line in result.split('\n') if 'ranked business list:' in line][0]   
            # result = rec_list.split(':')[1].strip()
            result = self.reasoning(task_description)
            result = result.strip()
            print(result)
            time.sleep(5)
            return eval(result)

        except:
            print('format error')
            result = '[]'
            time.sleep(5)

            return result


