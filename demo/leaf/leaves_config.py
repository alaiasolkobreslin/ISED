from openai import OpenAI
import os
import json
import re
import random

client = OpenAI(
  api_key=os.environ["OPENAI_API_KEY"]
)

l10_type = ["needle-like", "trifoliate", "palmate", "frond", "finely divided feathery leaves", "pinnately compound"]
l10_margin = ["entire", 'serrate', 'dissected']
l10_texture = ["glossy", "smooth", "leathery"]
l10_labels = ["Acer negundo", "Taxus bacatta", "Eschscholzia californica", "Polypodium vulgare", "Pinus sp",
              "Fraxinus sp", "Aesculus californica", "Chelidonium majus", "Schinus terebinthifolius", "Fragaria vesca", 
              "unknown"]

system_msg = "You are an expert in classifying plant species based on the type, margin, shape, and texture of the leaves. You are designed to output JSON."
question = "\n\nClassify each into one of: "
format = "\n\nGive your answer without explanation."

queries = {}

def classify_texture(prev, textures):
  user_question = question + ", ".join(textures) + ". "
  user_list = "* " + "\n* ".join(prev)
  user_msg = user_list + user_question
  if user_msg in queries.keys():
    return queries[user_msg]
  response = client.chat.completions.create(
              model="gpt-3.5-turbo",  #"gpt-4-1106-preview"
              messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg + format}
              ],
              top_p=0.00000001
            )
  if response.choices[0].finish_reason == 'stop':
    ans = response.choices[0].message.content
    queries[user_msg] = ans
    return ans
  raise Exception("LLM failed to provide an answer") 

def classify_margin(prev, margins):
  user_list = "* " + "\n* ".join(prev)
  user_question = question + ", ".join(margins) + ". "
  user_msg = user_list + user_question
  if user_msg in queries.keys():
    return queries[user_msg]
  response = client.chat.completions.create(
              model="gpt-3.5-turbo",  #"gpt-4-1106-preview"
              messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg + format}
              ],
              top_p=0.00000001
            )
  if response.choices[0].finish_reason == 'stop':
    ans = response.choices[0].message.content
    queries[user_msg] = ans
    return ans
  raise Exception("LLM failed to provide an answer")  

def classify_type(labels, types):
  user_list = "* " + "\n* ".join(labels)
  user_question = question + ", ".join(types) + ". "
  user_msg = user_list + user_question
  if user_msg in queries.keys():
    return queries[user_msg]
  response = client.chat.completions.create(
              model="gpt-3.5-turbo",  #"gpt-4-1106-preview"
              messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg + format}
              ],
              top_p=0.00000001
            )
  if response.choices[0].finish_reason == 'stop':
    ans = response.choices[0].message.content
    queries[user_msg] = ans
    return ans
  raise Exception("LLM failed to provide an answer")

def parse_dict(result, target):
  dict = json.loads(result)
  plants = []
  for plant in dict.keys():
    if dict[plant] == target: plants.append(plant)
  return plants

def classify_llm(type, margin, texture):
  result_type = classify_type(l10_labels, l10_type)
  plants_type = parse_dict(result_type, type)
  if len(plants_type) == 1: return plants_type[0]
  else: 
    result_margin = classify_margin(plants_type, l10_margin)
    plants_margin = parse_dict(result_margin, margin)
    if len(plants_margin) == 0: return 'unknown' # return plants_type[random.randint(0, len(plants_type)-1)]
    if len(plants_margin) == 1: return plants_margin[0]
    else: 
      result_texture = classify_texture(plants_margin, l10_texture)
      plants_texture = parse_dict(result_texture, texture)
      if len(plants_texture) == 0: return 'unknown' # return plants_margin[random.randint(0, len(plants_type)-1)]
      elif len(plants_texture) == 1: return plants_texture[0]
      else: 
        print(len(plants_texture))
        return plants_texture[random.randrange(len(plants_texture))]
    
def classify_llm_expert(type, margin, texture):
  system_message = "You are an expert in classifying leaves based on the type, margin, shape, and texture."
  expert_info = '''* Leaf 1: trifoliate, entire to serrate margin
* Leaf 2: needle-like, glossy texture
* Leaf 3: finely divided feathery leaves
* Leaf 4: fronds
* Leaf 5: needle-like, smooth texture
* Leaf 6: pinnately compound, entire margin, smooth texture
* Leaf 7: palmately compound, entire margin
* Leaf 8: pinnately compound, dissected margin
* Leaf 9: pinnately compound, entire margin, leathery texture
* Leaf 10: trifoliate, toothed margin'''
  instruction = "\n\nWhen identifying leaves, first look at the type, then margin, and then texture."
  user_question = "\n\nWhich leaf is %s, %s, %s? " % (type, margin, texture)
  format = "Give your answer inside <<>> without explanation."
  response = client.chat.completions.create(
              model="gpt-3.5-turbo",  #"gpt-4-1106-preview"
              messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": expert_info + instruction + user_question + format}
              ],
              top_p=0.00000001
            )
  if response.choices[0].finish_reason == 'stop':
    ans = response.choices[0].message.content
    return re.search(r'\<\<(.*?)\>\>',ans).group(1)
  raise Exception("LLM failed to provide an answer")