from yelpsimulator import Simulator
from yelpsimulator.agents import SimulationAgent
import httpx
import re
import ast
from openai import OpenAI
import tiktoken
import json 
import requests

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")  
    tokens = enc.encode(text)
    return len(tokens)


def calculate_price(prompt_tokens: int, response_tokens: int) -> float:
    input_price_per_million_tokens = 0.1  
    output_price_per_million_tokens = 2.0  
    
    prompt_tokens_in_million = prompt_tokens / 1_000_000  
    response_tokens_in_million = response_tokens / 1_000_000  
    
    input_cost = prompt_tokens_in_million * input_price_per_million_tokens
    output_cost = response_tokens_in_million * output_price_per_million_tokens
    
    total_cost = input_cost + output_cost
    
    return round(total_cost, 6)
def sanitize_input(data):
    sanitized_data = re.sub(r'[^\x00-\xFF]', '', data)

    return sanitized_data
    

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

def llm_response(prompt: str, model: str = "deepseek-chat", temperature: float = 0.5) -> str:


    import http.client

    conn = http.client.HTTPSConnection("cloud.infini-ai.com")

    payload_dict = {
        "model": "qwen2.5-72b-instruct",
        "messages": [
            {"role": "user", "content": sanitize_input(prompt)}
        ]
    }

    payload = json.dumps(payload_dict, ensure_ascii=False)    

    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer sk-dakqyjy2pusruxx7"
    }

    conn.request("POST", "/maas/v1/chat/completions", str(payload), headers)
        
    res = conn.getresponse()
    data = res.read()    
    output = data.decode("utf-8")
    response_dict = json.loads(output)
    
    try:
        llm_output = response_dict["choices"][0]["message"]["content"]
    except:
        print(response_dict)
        llm_output = ''

    if '抱歉' in llm_output or len(llm_output) <1:
        print('LLM not answer')
        llm_output = ''
    prompt_tokens = count_tokens(prompt)
    response_tokens = count_tokens(llm_output)
    
    price = calculate_price(prompt_tokens=prompt_tokens, response_tokens=response_tokens)
    # time.sleep(5)
    return llm_output, price


class PlanningBase():
    def __init__(self, llms_type):
        self.plan = []
        self.llm_type = llms_type[0]
    
    def create_prompt(self, task_type, task_description, ):
        raise NotImplementedError("Subclasses should implement this method")

    def __call__(self, task_description):
        self.plan = [
         {'description': 'First I need to find user information'},
         {'description': 'Next, I need to find business information'},
         {'description': 'Next, I need to find historical review'}
         ]

        return self.plan
    
    
class ReasoningBase:
    def __init__(self, profile_type_prompt, llms_type):
        self.profile_type_prompt = profile_type_prompt
        self.llm_type = llms_type[0]
        
class ReasoningIO(ReasoningBase):
    def __call__(self, task_description: str):
        prompt = '''
{task_description}'''
        prompt = prompt.format(task_description=task_description)
        reasoning_result, price = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)
        
        return reasoning_result, price

class MySimulationAgent(SimulationAgent):
    """
    Participant's implementation of SimulationAgent.
    """
    def __init__(self):
        super().__init__()
        self.planning = PlanningBase(['deepseek-chat'])
        self.reasoning = ReasoningIO('', ['deepseek-chat'])

    def forward(self):
        """
        Simulate user behavior.
        Returns:
            tuple: (star (float), review_text (str), behavior_metrics (tuple))
        """
        task_type = 'user_behavior_simulation'
        plan = self.planning(task_description=self.scenario)
        print(self.scenario['user_id'])
        print(self.scenario['business_id'])

        for sub_task in plan:
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.scenario['user_id']))
                input_tokens = num_tokens_from_string(user)

                if input_tokens > 21000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:21000])

                if input_tokens > 21000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:21000])
                    


                # print(user)
            elif 'business' in sub_task['description']:
                business = str(self.interaction_tool.get_item(item_id=self.scenario['business_id']))
                input_tokens = num_tokens_from_string(business)

                if input_tokens > 21000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    business = encoding.decode(encoding.encode(business)[:21000])
                # print(business)
            elif 'review' in sub_task['description']:
                history_review_user = str(self.interaction_tool.get_reviews(user_id=self.scenario['user_id'])) #list[dict,dict]
                # history_review = str([{key: original_dict[key] for key in ['business_id','stars','text']} for original_dict in history_review_ori])
                input_tokens = num_tokens_from_string(history_review_user)
                if input_tokens > 15000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review_user = encoding.decode(encoding.encode(history_review_user)[:15000])
                
                history_review_business = str(self.interaction_tool.get_reviews(item_id=self.scenario['business_id'])) #list[dict,dict]
                # history_review = str([{key: original_dict[key] for key in ['business_id','stars','text']} for original_dict in history_review_ori])
                input_tokens = num_tokens_from_string(history_review_business)
                if input_tokens > 15000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review_business = encoding.decode(encoding.encode(history_review_business)[:15000])

                # print(history_review_business)
        user_summary_prompt = f'''
        Task:
        The input is a user's historical reviews and ratings on Yelp. Your task is to summarize the following details into concise, structured information:

        1. Preferred Business Categories or Features: Identify the types of businesses (e.g., restaurants, gyms, salons) or specific features (e.g., friendly staff, fast service, affordable prices) the user consistently rates highly.
        2. Disliked Aspects: Highlight features (e.g., poor customer service, long wait times, high prices) the user often criticizes or gives low ratings to.
        3. Rating Distribution: Provide an overview of the user's ratings across all reviewed businesses (e.g., how many 5.0s, 4.0s, etc.).
        4. Examples of Favorites and Least Favorites: Extract examples of businesses with their ratings and a brief reason why the user liked or disliked them.

        Input:
        {history_review_user}

        Output Format:

        Preferred Business Categories or Features: ...
        Disliked Aspects: ...
        Rating Number Distribution: [5.0: X, 4.0: Y, 3.0: Z, 2.0: W, 1.0: V]
        Favorite Examples:
        "business_name_1" (Rating: 5.0) - Reason: "Exceptional service and delicious food."
        "business_name_2" (Rating: 4.0) - Reason: "Affordable prices and convenient location."
        Least Favorite Examples:
        "business_name_3" (Rating: 2.0) - Reason: "Unfriendly staff and long wait times."
        '''

        user_summary,price = self.reasoning(user_summary_prompt)

        business_summary_prompt = f'''
        Instruction:
        You are an expert reviewer on Yelp. Given multiple user reviews and ratings for a business, your task is to summarize the key insights into concise and structured information. Focus on extracting the most relevant points about the business's strengths, weaknesses, and overall reception.

        Input:
        Here are the reviews and ratings from other users about the business:
        {history_review_business}

        Task:
        Summarize the reviews into the following key points:

        1. **Overall Sentiment**: Describe the general sentiment expressed in the reviews (positive, neutral, or negative). Use specific adjectives to capture the tone of the reviews (e.g., "friendly," "slow," "amazing").
        2. **Common Praises**: Identify recurring themes or features that users liked about the business (e.g., "excellent customer service," "delicious food," "clean and welcoming atmosphere").
        3. **Common Criticisms**: Identify recurring themes or features that users disliked about the business (e.g., "long wait times," "overpriced," "unfriendly staff").
        4. **Rating Distribution**: Provide a count of how many users gave ratings for each score (e.g., [5.0: X, 4.0: Y, 3.0: Z, 2.0: W, 1.0: V]).
        5. **Representative Reviews**: Extract one or two brief, representative reviews with their ratings to illustrate the sentiment. These should highlight the key praises or criticisms.

        Output Format:
        The response should be structured as follows:
        Overall Sentiment: [Sentiment description].
        Common Praises: [Bullet points summarizing praised features].
        Common Criticisms: [Bullet points summarizing criticized features].
        Rating Distribution: [5.0: X, 4.0: Y, 3.0: Z, 2.0: W, 1.0: V].
        Representative Reviews:
        "[Brief review text]" (Rating: X.X)
        "[Brief review text]" (Rating: X.X)

        '''

        business_summary,price = self.reasoning(business_summary_prompt)

        task_description = f'''
        You are a real human user on Yelp (a platform for reviewing and sharing opinions about businesses). Your profile is as follows: {user}.
        Your summarized preferences on the platform are: {user_summary}. This contains your past reviews, ratings, and preferences, highlighting the business categories, features, and services you typically enjoy or dislike.

        Task:
        Now, you need to perform two tasks for the business:

        1. Write a review text (in English) summarizing your opinion based on the following inputs.
        2. Provide a rating (stars) selected only from [0.0, 1.0, 2.0, 3.0, 4.0, 5.0].

        ### Inputs to Consider Before Responding:
        1. **User Preferences Analysis**:
        - Based on the summarized preferences, determine the key factors the user values most in businesses (e.g., excellent customer service, high-quality food, cleanliness, affordability).
        - Identify any dislikes (e.g., long wait times, unfriendly staff, overpriced services).

        2. **Business's Attributes**:
        - Carefully consider the details of the business {business}.  

        3. **Alignment with Preferences**:
        - If the business aligns with the user's preferences (e.g., services, quality, environment), the rating should reflect enjoyment.
        - If the business conflicts with preferences (e.g., disliked features, poor service), be critical in the review and rating.

        4. **External Reviews**:
        - Evaluate the reviews from other users {business_summary}.
        - Identify recurring themes or criticisms.
        - Compare these to the user’s preferences to determine how the user would likely perceive these aspects.

        ### Reasoning Process (Simulated Thinking):
        You should simulate the user's thought process by answering the following questions step-by-step before completing the task:

        - Does this business fit the user’s favorite categories, services, or qualities?
        - Do the reviews from other users align with or contradict the user's typical opinions?
        - Is the business likely to exceed or fall short of the user’s expectations based on historical data?

        ### Output Format:
        RATING: [A single number reflecting the user’s overall opinion of the business. Avoid bias towards 4.0 and consider lower ratings (1.0 or 2.0) if warranted by conflicts with user preferences].  
        REVIEW TEXT: [A thoughtful and realistic paragraph (at least 50 words) explaining why the user liked or disliked the business. Include specific aspects like service, quality, pricing, or atmosphere. Be critical if necessary].

        ### Example Output:
        RATING: x.0  
        REVIEW TEXT: xxxxxx  
        '''

        result,price = self.reasoning(task_description)
        print('API Cost (yuan):',price)
        try:
            star_line = [line for line in result.split('\n') if 'RATING:' in line][0]
            review_line = [line for line in result.split('\n') if 'REVIEW TEXT:' in line][0]
            match = re.search(r'\d+(\.\d+)?', star_line)
            star = float(match.group()) if match else None
            review_text = review_line.split(':')[1].strip()

        except:
            print('Error:',result)
            star = 3.0
            review_text = ''


        print(star)
        print(review_text)

        return star, review_text


