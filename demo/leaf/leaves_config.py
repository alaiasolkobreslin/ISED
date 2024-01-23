from openai import OpenAI
import os
import json
import random

import llm_configs

client = OpenAI(
  api_key=os.environ["OPENAI_API_KEY"]
)

queries = {}

labels = llm_configs.l11_labels
features_1 = llm_configs.l11_4_one
features_2 = llm_configs.l11_4_two
features_3 = llm_configs.l11_4_three
features_4 = []
system_msg = llm_configs.l11_4_system

def classify_llm(feature1, feature2, feature3):
  result1 = call_llm(labels, features_1)
  plants1 = parse_response(result1, feature1)
  if len(plants1) == 1: return plants1[0]
  elif len(plants1) == 0: 
    plants1 = labels # return 'unknown'
  else:
    results2 = call_llm(plants1, features_2)
    plants2 = parse_response(results2, feature2)
    if len(plants2) == 1: return plants2[0]
    elif len(plants2) == 0: 
      plants2 = plants1  # return 'unknown'
    results3 = call_llm(plants2, features_3)
    plants3 = parse_response(results3, feature3)
    if len(plants3) == 1: return plants3[0]
    elif len(plants3) == 0: return plants2[random.randrange(len(plants2))] # return 'unknown'
    else: return plants3[random.randrange(len(plants3))]

def classify_llm_4(feature1, feature2, feature3, feature4):
  result1 = call_llm(labels, features_1)
  plants1 = parse_response(result1, feature1)
  if len(plants1) == 1: return plants1[0]
  elif len(plants1) == 0: 
    plants1 = labels # return 'unknown'
  else:
    results2 = call_llm(plants1, features_2)
    plants2 = parse_response(results2, feature2)
    if len(plants2) == 1: return plants2[0]
    elif len(plants2) == 0: 
      plants2 = plants1 # return 'unknown'
    results3 = call_llm(plants2, features_3)
    plants3 = parse_response(results3, feature3)
    if len(plants3) == 1: return plants3[0]
    elif len(plants3) == 0:
      plants3 = plants2 # return 'unknown'
    results4 = call_llm(plants3, features_4)
    plants4 = parse_response(results4, feature4)
    if len(plants4) == 1: return plants4[0]
    elif len(plants4) == 0: return plants3[random.randrange(len(plants3))] # 'unknown'
    else: return plants4[random.randrange(len(plants4))]

def call_llm(plants, features):
  user_list = "* " + "\n* ".join(plants)
  question = "\n\nClassify each into one of: " + ", ".join(features) + "."
  format = "\n\nGive your answer without explanation."
  user_msg = user_list + question
  if user_msg in queries.keys():
    return queries[user_msg]
  response = client.chat.completions.create(
              model="gpt-4-1106-preview", # gpt-3.5-turbo
              messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg + format}
              ],
              top_p=0.00000001
            )
  if response.choices[0].finish_reason == 'stop':
    ans = response.choices[0].message.content
    print(ans[7:-3])
    queries[user_msg] = ans[7:-3] # ans
    return ans[7:-3]
  raise Exception("LLM failed to provide an answer") 

def parse_response(result, target):
  dict = json.loads(result)
  plants = []
  for plant in dict.keys():
    if dict[plant] == target: plants.append(plant)
  return plants

def classify_11(margin, shape, texture):
  if margin == 'serrate': return 'Ocimum basilicum'
  elif margin == 'indented': return 'Jatropha curcas'
  elif margin == 'lobed': return 'Platanus orientalis'
  elif margin == 'serrulate': return "Citrus limon"
  elif margin == 'entire':
    if shape == 'ovate': return 'Pongamia Pinnata'
    elif shape == 'lanceolate': return 'Mangifera indica'
    elif shape == 'oblong': return 'Syzygium cumini'
    elif shape == 'obovate': return "Psidium guajava"
    else:
      if texture == 'aristate': return "Alstonia Scholaris"
      elif texture == 'round': return "Terminalia Arjuna"
      elif texture == 'glossy': return "Citrus limon"
      else: return "Punica granatum"
  else:
    if shape == 'elliptical': return 'Terminalia Arjuna'
    elif shape == 'lanceolate': return "Mangifera indica"
    else: return 'Syzygium cumini'

l11_margin = ["entire", "serrate", "lobed", 'indented', 'undulate', 'serrulate']
l11_shape = ["ovate", "oblong", "elliptical", 'obovate', 'lanceolate']
l11_texture = ["aristate", "round", 'glossy', "medium"]
l11_labels = ['Alstonia Scholaris', 'Citrus limon', 'Jatropha curcas', 'Mangifera indica', 'Ocimum basilicum',
              'Platanus orientalis', 'Pongamia Pinnata', 'Psidium guajava', 'Punica granatum', 'Syzygium cumini', 'Terminalia Arjuna']
l11_dim = 2304

def classify_30(margin, shape, texture, venation):
  if margin == 'palmate': return 'Acer palmaturu'
  elif margin == 'dissected': return 'Geranium sp'
  elif margin == 'spiny': return 'Ilex aquifolium'
  elif margin == 'toothed': return 'Urtica dioica'
  elif margin == 'doubly serrate': return 'Populus alba'
  elif margin == 'sinuate': return 'Erodium sp'
  elif margin == 'lobed':
    if venation == 'nearly invisible': return 'Crataegus monogyna'
    return 'Quercus robur'
  if shape == 'cordate': return 'Tilia tomentosa'
  elif shape == 'spear': return 'Arisarum vulgare'
  elif shape == 'deltoid':
    if margin == 'serrate': return 'Betula pubescens'
    else: return 'Populus nigra' 
  elif margin == 'dentate': 
    if shape == 'orbicular' or shape == 'ovate': return 'Euonymus japonicus'
    elif shape == 'elliptical' or shape == 'oblong': return 'Quercus suber'
    else: return 'Primula vulgaris'
  elif margin == 'undulate':
    if shape == 'lanceolate' or shape == 'linear': return 'Magnolia soulangeana'
    elif shape == 'orbicular' or shape == 'ovate': return 'Alnus sp'
    else: return 'Primula vulgaris'
  elif margin == 'serrate': 
    if shape == 'obovate' or shape == 'oblong': return 'Castanea sativa'
    elif shape == 'ovate': return 'Hydrangea sp'
    elif shape == 'orbicular': return 'Alnus sp'
    else: return 'Celtis sp'
  elif margin == 'entire':
    if shape == 'obovate': return 'Salix atrocinerea'
    elif shape == 'lanceolate': return 'Magnolia grandiflora'
    elif shape == 'orbicular':
      if texture == 'thin': return 'Corylus avellana'
      return 'Bougainvillea sp'
    elif shape == 'ovate':
      if texture == 'thin': return 'Corylus avellana'
      return 'Hydrangea sp'
    elif shape == 'linear':
      if texture == 'papery': return 'Pseudosasa japonica'
      else:
        if venation == 'parallel': return 'Nerium oleander'
        else: return 'Podocarpus sp'
    elif shape == 'oblong': return 'Ilex perado ssp azorica'
    elif shape == 'elliptical':
      if texture == 'waxy': return 'Buxus sempervirens'
      else: return 'Acca sellowiana' 

l30_margin = ['palmate', 'dissected', 'spiny', 'toothed', 'doubly serrate', 'sinuate', 'lobed', 'dentate', 'undulate', 'serrate', 'entire']
l30_shape = ['orbicular', 'elliptical', 'lanceolate', 'cordate', 'deltoid', 'obovate', 'ovate', 'spear', 'linear', 'oblong']
l30_venation = ['parallel', 'nearly invisible']
l30_texture = ['smooth', 'leathery', 'thin', 'papery' 'waxy']
l30_labels = ['Acca sellowiana', 'Acer palmaturu', 'Alnus sp', 'Arisarum vulgare', 'Betula pubescens', 'Bougainvillea sp', 
              'Buxus sempervirens', 'Castanea sativa', 'Celtis sp', 'Corylus avellana', 'Crataegus monogyna', 'Erodium sp', 
              'Euonymus japonicus', 'Geranium sp', 'Hydrangea sp', 'Ilex aquifolium', 'Ilex perado ssp azorica', 'Magnolia grandiflora', 
              'Magnolia soulangeana', 'Nerium oleander', 'Podocarpus sp', 'Populus alba', 'Populus nigra', 'Primula vulgaris', 
              'Pseudosasa japonica', 'Quercus robur', 'Quercus suber', 'Salix atrocinerea', 'Tilia tomentosa', 'Urtica dioica']
l30_dim = 3072

def classify_10(arrangement, margin, texture):
  if arrangement == 'pinnate':
    if margin == 'dissected': return "Chelidonium majus"
    elif margin == 'entire': return "Fraxinus sp"
    elif margin == 'serrate': return "Schinus terebinthifolius"
    else: return "Fragaria vesca"
  elif arrangement == 'trifiolate':
    if margin == 'toothed': return "Fragaria vesca"
    elif margin == 'dissected': return "Chelidonium majus"
    else: return "Acer negundo"
  elif arrangement == 'palmate': return "Aesculus californica"
  elif arrangement == 'needle': return "Pinus sp"
  elif arrangement == 'feathery': return "Eschscholzia californica"
  elif arrangement == 'others':
    if texture == 'leathery': return "Polypodium vulgare"
    else: return "Taxus bacatta"

l10_type = ["pinnate", "trifiolate", "palmate", "needle", "others", "feathery"]
l10_margin = ["entire", "toothed", 'dissected', 'serrate']
l10_texture = ["glossy", "leathery"]
l10_labels = ["Acer negundo", "Aesculus californica", "Chelidonium majus", "Eschscholzia californica", "Fragaria vesca", 
                   "Fraxinus sp", "Pinus sp", "Polypodium vulgare", "Schinus terebinthifolius", "Taxus bacatta"]
l10_dim = 3072