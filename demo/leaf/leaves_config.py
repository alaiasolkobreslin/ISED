from openai import OpenAI
import os
import re

client = OpenAI(
  api_key='' #os.environ["OPENAI_API_KEY"]
)

l11_expert = """
              * Leaf 1: lobed margin, heart shape, palmate venation
              * Leaf 2: entire margin, lanceolate shape, glossy texture, pinnate venation
              * Leaf 2: undulate margin, lanceolate shape, glossy texture, pinnate venation
              * Leaf 2: entire margin, lanceolate shape, leathery texture, pinnate venation
              * Leaf 2: undulate margin, lanceolate shape, leathery texture, pinnate venation
              * Leaf 3: incised margin, palmate shape, palmate venation, coarse texture
              * Leaf 4: serrate margin, ovate shape, medium texture, palmate venation
              * Leaf 5: entire margin, obovate shape, rough texture, pinnate venation
              * Leaf 6: entire margin, pinnate venation, elliptical shape, hairy texture
              * Leaf 6: entire margin, pinnate venation, obovate shape, hairy texture
              * Leaf 7: entire margin, ovate shape, pinnate venation
              * Leaf 7: entire margin, elliptical shape, pinnate venation
              * Leaf 8: entire margin, obovate shape, pinnate venation
              * Leaf 8: undulate margin, obovate shape, pinnate venation
              * Leaf 9: serrulate margin, glossy texture, pinnate venation, ovate shape
              * Leaf 10: undulate margin, oblong shape, pinnate venation
              * Leaf 10: entire margin, oblong shape, pinnate venation
              * Leaf 11: entire margin, lanceolate shape, medium texture, pinnate venation
            """

def classify_llm(margin, shape, texture):
  system_msg = "You are an expert in classifying plant species based on their leaves."
  question = "Which best describes leaf with %s margin, %s shape, and %s texture? " % (margin, shape, texture)
  format = "Give the number of the leaf inside << >> without explanation."
  response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": l11_expert + question + format}
              ]
            )
  if response.choices[0].finish_reason == 'stop':
    ans = response.choices[0].message.content
    return re.search(r'\<\<(.*?)\>\>',ans).group(1)
  raise Exception("LLM failed to provide an answer")

def classify_11(margin, shape, others):
  if margin == 'lobed': return "Platanus orientalis"
  elif margin == 'shallow': return "Jatropha curcas"
  elif margin == 'serrate': return "Ocimum basilicum"
  elif margin == 'serrulate': return "Citrus limon"
  elif margin == 'undulate':
    if shape == 'obovate' or shape == 'oblong' or shape == 'ovate':
       return "Syzygium cumini"
    else:
      if others == 'leathery': return "Mangifera indica" 
      else: return 'Terminalia Arjuna'
  else: # entire
    if shape == 'ovate': return "Pongamia Pinnata"
    elif shape == 'oblong': return "Psidium guajava"
    elif shape == 'obovate': return 'Syzygium cumini'
    else: # elliptical
      if others == 'medium': return "Punica granatum"
      elif others == 'serrulate': return "Citrus limon"   
      else: return 'Terminalia Arjuna' # lanceolate

l11_margin = ["entire", "serrate", "lobed", "serrulate", "shallow", "undulate"]
l11_shape = ["ovate", "obovate", "oblong", "elliptical"]
l11_texture = ['medium', 'leathery', 'smooth', 'serrulate']
l11_labels = ['Jatropha curcas', 'Mangifera indica', 'Platanus orientalis', 'Ocimum basilicum', 'Psidium guajava', 
              'Pongamia Pinnata', 'Syzygium cumini', 'Citrus limon', 'Terminalia Arjuna', 'Alstonia Scholaris', 'Punica granatum']
l11_dim = 2304

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
l40_dim = 3072