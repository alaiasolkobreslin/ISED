from openai import OpenAI
import os
import re
from collections import namedtuple

client = OpenAI(
  api_key='' #os.environ["OPENAI_API_KEY"]
)

def call_llm(labels, margin, shape):
  system_msg = "You are an expert in classifying plant species based on their leaves."
  labels_str = ', '.join(labels)
  expert_knowledge = ''
  question = "Among [%s], which best describes leaf with a %s margin, and %s shape? " % (labels_str, margin, shape)
  format = "Give your answer inside << >> without explanation."
  user_msg = question + format
  response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
              ]
            )
  if response.choices[0].finish_reason == 'stop':
    ans = response.choices[0].message.content
    return re.search(r'\<\<(.*?)\>\>',ans).group(1)
  raise Exception("LLM failed to provide an answer")

Leaf =  namedtuple('Leaf', ['margin', 'shape', 'texture'])
plant_dict = {}

def classify_llm(margin, shape, texture):
  l = Leaf(margin, shape, texture)
  if plant_dict.has_key(l):
    return plant_dict(l)
  else:
    result = call_llm(margin, shape, texture)
    plant_dict[l] = result
    return result

def classify_11(margin, shape, others):
  if margin == 'lobed': return "chinar"
  elif margin == 'shallow': return "jatropha"
  elif margin == 'serrate': return "basil"
  elif margin == 'serrulate': return "lemon"
  elif margin == 'undulate':
    if shape == 'obovate' or shape == 'oblong' or shape == 'ovate':
       return "jamun"
    else:
      if others == 'narrow veins': return "alstonia scholaris"
      elif others == 'leathery': return "mango" 
      else: return 'arjun'
  else: # entire
    if shape == 'ovate': return "pongamia"
    elif shape == 'oblong': return "guava"
    elif shape == 'obovate': return 'jamun'
    else: # elliptical
      if others == 'medium': return "pomegranate"
      elif others == 'serrulate': return "lemon"   
      else: return 'arjun' # lanceolate

l11_margin = ["entire", "serrate", "lobed", "serrulate", "shallow", "undulate"]
l11_shape = ["ovate", "obovate", "oblong", "elliptical"]
l11_texture = ['narrow veins', 'medium', 'leathery', 'smooth', 'serrulate']
l11_labels = ["alstonia scholaris", "arjun", "basil", "chinar", "guava", "jamun", 
              "jatropha", "lemon", "mango", "pomegranate", "pongamia"]
l11_dim = 672*64

def classify_40(type, margin, shape, texture, venation):
  if type == 'palmate':
    return '37. aesculus californica'
  elif type == 'trifoliate':
    if margin == 'toothed' or margin == 'serrate':
      return '40. fragaria vesca'
    elif margin == 'entire' or margin == 'dentate':
      return '16. acer negundo'
    else: # lobate, spiny, others, dissected
      return 'unknown'
  elif type == 'pinnate':
    if margin == 'lobate' or margin == 'dissected':
      return '38. chelidonium majus'
    elif margin == 'entire':
      return '21. fraxinus'
    elif margin == 'serrate' or margin == 'dentate':
      return '39. schinus terebinthifolius' 
    else: # toothed, spiny, others
      return 'unknown'
  elif type == 'fronds':
    if texture == 'waxy':
      return '17. taxus bacatta'
    elif texture == 'leathery':
      return '19. polypodium vulgare'
    else: # wriggle, shiny
      return 'unknown'
  elif type == 'needle':
    ### papaver 
    return '20. pinus'
  elif type == 'incised':
    return '18. papaver'
  else: # simple
    if margin == 'spiny':
      return '7. ilex aquifolium'
    elif margin == 'others':
      return '23. erodium' ### lobate + orbiculate 
    elif margin == 'dissected':
      return '36. geranium'
    elif margin == 'toothed':
      if shape == 'palmate':
        return '15. populus alba'
      elif shape in ['cordate', 'orbiculate']:
        return '30. urtica dioica'
      else: # linear, deltoid, oblong, elliptical, lanceolate, spear
        return 'unknown'
    elif margin == 'lobate':
      if shape == 'spear': ### cordate
        return '25. arisarum vulgare'
      elif shape == 'palmate':
        return '6. crataegus monogyna' ### margin pinnatifid
      elif shape in ['oblong', 'orbiculate', 'elliptical', 'lanceolate']:
        return '5. quercus robur' 
      else: # linear, deltoid, cordate
        return 'unknown'
    elif margin == 'serrate':
      if shape == 'deltoid':
        return '9. betula pubescens'
      elif shape == 'cordate':
        return '10. tilia tomentosa'
      elif shape == 'palmate':
        return '15. populus alba'
      elif shape == 'oblong':
        return '14. castanea sativa'
      elif shape == 'orbiculate': # oval, ovate
        if texture == 'wriggle':
          return '4. alnus'
        elif texture == 'leathery':
          return '33. hydrangea'
        else: # waxy, shiny
          return 'unknown'
      elif shape == 'elliptical':
        return '12. celtis'  
      else: # linear, spear, lanceolate
        return 'unknown'
    elif margin == 'dentate':
      if shape == 'deltoid':
        return '3. populus nigra'
      elif shape == 'cordate':
        return '10. tilia tomentosa'
      elif shape == 'palmate':
        return '11. acer palmaturu'
      elif shape == 'orbiculate':
        if texture == 'wriggle':
          if venation == 'invisible':
            return '4. alnus'
          elif venation == 'pinnate':
            return '13. corylus avellana'
          else:
            return 'unknown'
        elif texture == 'waxy':
          return '26. euonymus japonicus'
        elif texture == 'leathery':
          return '33. hydrangea'
        else:
          return 'unknown'
      elif shape == 'lanceolate':
        return '28. magnolia soulangeana'
      elif shape == 'elliptical' or shape == 'oblong':
        if texture == 'waxy':
          return '26. euonymus japonicus'
        elif texture == 'leathery':
          return '1. quercus suber'
        elif texture == 'wriggle':
          return '22. primula vulgaris'
        else:
          return 'unknown'
      else: # spear, linear
        return 'unknown'
    else: # entire
      if shape == 'spear':
        return '25. arisarum vulgare'
      elif shape == 'linear':
        if venation == 'parallel':
          return '34. pseudossa japonica'
        elif venation == 'invisible':
          return '31. podocarpus'
        else:
          return '8. nerium oleander'
      elif shape == 'deltoid':
        return '3. populus nigra'
      elif shape == 'palmate':
        return '11. acer palmaturu'
      elif shape == 'lanceolate':
        return '35. magnolia grandiflora'
      elif shape == 'orbiculate':
        if texture == 'wriggle':
          return '13. corylus avellana'
        elif texture == 'leathery':
          return '24. bougainvillea'
        else:
          return 'unknown'
      elif shape == 'oblong':
        return '2. salix atrocinerea'
      elif shape == 'elliptical':
        if texture == 'waxy':
          if venation == 'invisible':
            return '29. buxus sempervirens' 
          elif venation == 'pinnate':
            return '27. ilex perado ssp azorica'
          else:
            return 'unknown'
        elif texture == 'leathery':
          return '33. hydrangea'
        elif texture == 'wriggle':
          return '13. corylus avellana'
        else: # shiny
          return '32. acca sellowiana'
      else:
        return 'unknown'

l40_type = ['palmate', 'trifoliate', 'pinnate', 'fronds', 'needle', 'incised', 'simple']
l40_margin = ['toothed', 'serrate', 'entire', 'lobate', 'dentate', 'spiny', 'other', 'dissected'] ### undulate
l40_shape = ['spear', 'linear', 'deltoid', 'cordate', 'oblong', 'orbiculate', 'ovate', 'elliptical', 'lanceolate', 'palmate']
l40_texture = ['waxy', 'leathery', 'wriggle', 'shiny']
l40_venation = ['parallel', 'invisible', 'pinnate']
l40_labels = ["1. quercus suber", "2. salix atrocinerea", "3. populus nigra", "4. alnus", "5. quercus robur",
              "6. crataegus monogyna", "7. ilex aquifolium", "8. nerium oleander", "9. betula pubescens", "10. tilia tomentosa",
              "11. acer palmaturu", "12. celtis", "13. corylus avellana", "14. castanea sativa", "15. populus alba",
              "16. acer negundo", "17. taxus bacatta", "18. papaver", "19. polypodium vulgare", "20. pinus",
              "21. fraxinus", "22. primula vulgaris", "23. erodium", "24. bougainvillea", "25. arisarum vulgare",
              "26. euonymus japonicus", "27. ilex perado ssp azorica", "28. magnolia soulangeana", "29. buxus sempervirens", "30. urtica dioica",
              "31. podocarpus", "32. acca sellowiana", "33. hydrangea", "34. pseudossa japonica", "35. magnolia grandiflora",
              "36. geranium", "37. aesculus californica", "38. chelidonium majus", "39. schinus terebinthifolius", "40. fragaria vesca",
              'unknown']
l40_dim = 850*64
  