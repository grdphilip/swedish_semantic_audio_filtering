import json
import ast
import re

def clean_entities(raw_entities):
    cleaned_entities = []

    for entity_string in raw_entities:
        # Fix double escaping issues by replacing `\\"` with `"`, then strip leading/trailing quotes
        entity_string = entity_string.replace('\\"', '"').strip('"')

        # Convert JSON-like string to a Python list
        try:
            entity_list = json.loads(entity_string)
        except json.JSONDecodeError:
            continue  # Skip invalid entries

        # Decode Unicode escape sequences properly and split entities into words
        cleaned_group = [word for entity in entity_list for word in entity.encode().decode("unicode_escape").strip().split()]
        
        cleaned_entities.append(cleaned_group)  # Append the processed group

    return cleaned_entities


test = ['"[\\"Karlshamnsverket\\"]"', '"[\\"Restaurang Terrassen Karlshamn\\"]"', '"[\\"Jwan Ali\\"]"', '"[\\"Jacob Lagercrantz\\"]"', '"[\\"Fredriksson\\", \\"Karlshamn\\"]"', '"[\\"Henrik Johansson\\"]"', '"[\\"Blekinge\\"]"', '"[\\"SMHI\\"]"', '"[\\"Karlskrona\\"]"', '"[\\"Peter Bowin\\", \\"Ronneby\\"]"', '"[\\"Islamiska Shiasamfunden\\", \\"Al Rassol\\"]"', '"[\\"Karlskrona Karlshamn\\"]"', '"[\\"SVT\\"]"', '"[\\"Lena - Marie Bergstr\\u00f6m\\"]"', '"[\\"Klemerstam\\"]"', '"[\\"Magnus Ljungcrantz\\", \\"Sveriges Radio\\"]"', '"[\\"Bergstr\\u00f6m\\"]"', '"[\\"Brunnsparkens\\"]"', '"[\\"Humana Assistans AB : s IVO\\"]"', '"[\\"Kristdemokraterna\\"]"', '"[\\"Fredrik Svennergren\\", \\"S\\u00f6lvesborg\\"]"', '"[\\"Pelle\\"]"', '"[\\"Karlshamnsverket\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Inge Persson\\"]"', '"[\\"Sturesson\\", \\"Karlskrona Kalmar kommun\\"]"', '"[\\"Tyskland\\"]"', '"[\\"Thomas Engman\\"]"', '"[\\"Pelle Nordensson\\", \\"sst\\u00f6dsn\\u00e4mnden\\", \\"Karlskrona kommun\\"]"', '"[\\"E22 M\\u00f6rrum\\"]"', '"[\\"Ronny Mattsson\\"]"', '"[\\"F\\u00f6rsvarsmakten\\"]"', '"[\\"Magnus G\\u00e4rdebring\\", \\"M\\"]"', '"[\\"Hermans\\"]"', '"[\\"Anders Johansson\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Marie Meijer Dahl\\", \\"Karlshamn Blekinge\\"]"', '"[\\"Thomas Johansson\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Syd\\"]"', '"[\\"Ronneby S\\u00f6lvesborg Olofstr\\u00f6m\\"]"', '"[\\"Engman\\"]"', '"[\\"SVT\\"]"', '"[\\"SM\\"]"', '"[\\"l\\u00e4nsstyrelsen\\"]"', '"[\\"Karlshamn Karlskrona\\"]"', '"[\\"Morgan Bengtsson\\"]"', '"[\\"Lotta Idoffson\\"]"', '"[\\"Karl Hamstedt\\"]"', '"[\\"Karlskronas\\"]"', '"[\\"Ringhals Oskarshamn Forsmark\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Litorin\\", \\"a Karlskrona\\"]"', '"[\\"Karlskronas\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Sverige\\"]"', '"[\\"G\\u00e4vleborgs l\\u00e4n\\"]"', '"[\\"Sverige Europa\\"]"', '"[\\"Roger Fredriksson\\", \\"M\\"]"', '"[\\"M\\u00f6rrum Hockey\\"]"', '"[\\"Karlskrona HK Allettan\\", \\"Eric Karlsson\\"]"', '"[\\"International School of Karlskrona\\"]"', '"[\\"Patrik Sommersgaard\\"]"', '"[\\"Karlshamn\\"]"', '"[\\"Kristdemokraterna\\", \\"Karlskrona\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Karlskrona porslinsfabrik\\"]"', '"[\\"Olofstr\\u00f6m\\"]"', '"[\\"stadshuset\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Karlskrona\\"]"', '"[\\"SVT :\\", \\"Sverige\\"]"', '"[\\"Bubblan och Gustav\\"]"', '"[\\"Centerpartiet\\", \\"Blekinge\\"]"', '"[\\"\\u00d6stersj\\u00f6n\\"]"', '"[\\"Martina Ravnb\\u00f6\\"]"', '"[\\"Kristdemokraterna\\"]"', '"[\\"Socialdemokraterna Centerpartiet Kristdemokraterna\\"]"', '"[\\"Christian Svensson\\"]"', '"[\\"M\\u00f6rrum Hockey\\"]"', '"[\\"JVM\\"]"', '"[\\"Karl Hamstedt\\"]"', '"[\\"Kall\\"]"', '"[\\"Riksf\\u00f6rbundet Sveriges\\", \\"museer\\"]"', '"[\\"Karlskrona\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Ambulansf\\u00f6rbundet\\", \\"SOS Alarm\\"]"', '"[\\"Jacob Lagercrantz\\"]"']

test_clean = clean_entities(test)
print(test_clean)
print(len(test_clean))