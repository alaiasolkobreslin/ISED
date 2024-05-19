from openai import OpenAI
import random
import os

client = OpenAI(
  api_key=os.environ["OPENAI_API_KEY"]
)

system_msg = "You are an expert at identifying room types based on the object detected. Give short single responses."
question = "\n What type of room is most likely? Choose among basement, bathroom, bedroom, living room, home lobby, office, lab, kitchen, dining room."
queries = {}

def classify_llm(objects):
  answer = call_llm(objects)
  answer = parse_response(answer)
  return answer

def call_llm(objects):
  objects.sort()
  objects = list(set(objects))
  if 'skip' in objects: objects.remove('skip')
  if 'ball' in objects: objects.remove('ball')
  user_list = ", ".join(objects)
  if len(objects)==1: prompt = f"There is a {user_list}."
  else: prompt = f"There are {user_list}."
  if user_list in queries.keys():
    return queries[user_list]
  response = client.chat.completions.create(
              model="gpt-4o",
              messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt + question}
              ],
              top_p=1e-8
            )
  if response.choices[0].finish_reason == 'stop':
    ans = response.choices[0].message.content.lower()
    print(ans)
    queries[user_list] = ans
    return ans
  raise Exception("LLM failed to provide an answer") 

def parse_response(answer):
  random_scene = ['basement', 'bathroom', 'bedroom', 'dining', 'kitchen', 'lab', 'living', 'lobby', 'office']
  random.shuffle(random_scene)
  for s in random_scene:
    if s in answer: return s
  raise Exception("LLM failed to provide an answer") 
