import json
import ast
import re



def clean_entities(raw_entities):
    cleaned_entities = []

    for entity_string in raw_entities:
        # Remove extra escaping, and strip any leading/trailing quotes
        entity_string = entity_string.replace('\\"', '"').strip('"')

        # Convert JSON-like string to a Python list
        try:
            entity_list = json.loads(entity_string)
        except json.JSONDecodeError:
            continue  # Skip invalid entries

        cleaned_group = []
        for entity in entity_list:
            # We directly decode the entity from the loaded JSON data to handle unicode escapes correctly
            decoded_entity = entity.encode('utf-8').decode('utf-8')

            # Split the entity into individual words
            words = decoded_entity.strip().split()
            cleaned_group.extend(words)

        cleaned_entities.append(cleaned_group)  # Append the processed group

    return cleaned_entities



test = ['"[\\"Karlshamnsverket\\"]"', '"[\\"Restaurang Terrassen Karlshamn\\"]"', '"[\\"Jwan Ali\\"]"', '"[\\"Jacob Lagercrantz\\"]"', '"[\\"Fredriksson\\", \\"Karlshamn\\"]"', '"[\\"Henrik Johansson\\"]"', '"[\\"Blekinge\\"]"', '"[\\"SMHI\\"]"', '"[\\"Karlskrona\\"]"', '"[\\"Peter Bowin\\", \\"Ronneby\\"]"', '"[\\"Islamiska Shiasamfunden\\", \\"Al Rassol\\"]"', '"[\\"Karlskrona Karlshamn\\"]"', '"[\\"SVT\\"]"', '"[\\"Lena - Marie Bergstr\\u00f6m\\"]"', '"[\\"Klemerstam\\"]"', '"[\\"Magnus Ljungcrantz\\", \\"Sveriges Radio\\"]"', '"[\\"Bergstr\\u00f6m\\"]"', '"[\\"Brunnsparkens\\"]"', '"[\\"Humana Assistans AB : s IVO\\"]"', '"[\\"Kristdemokraterna\\"]"', '"[\\"Fredrik Svennergren\\", \\"S\\u00f6lvesborg\\"]"', '"[\\"Pelle\\"]"', '"[\\"Karlshamnsverket\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Inge Persson\\"]"', '"[\\"Sturesson\\", \\"Karlskrona Kalmar kommun\\"]"', '"[\\"Tyskland\\"]"', '"[\\"Thomas Engman\\"]"', '"[\\"Pelle Nordensson\\", \\"sst\\u00f6dsn\\u00e4mnden\\", \\"Karlskrona kommun\\"]"', '"[\\"E22 M\\u00f6rrum\\"]"', '"[\\"Ronny Mattsson\\"]"', '"[\\"F\\u00f6rsvarsmakten\\"]"', '"[\\"Magnus G\\u00e4rdebring\\", \\"M\\"]"', '"[\\"Hermans\\"]"', '"[\\"Anders Johansson\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Marie Meijer Dahl\\", \\"Karlshamn Blekinge\\"]"', '"[\\"Thomas Johansson\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Syd\\"]"', '"[\\"Ronneby S\\u00f6lvesborg Olofstr\\u00f6m\\"]"', '"[\\"Engman\\"]"', '"[\\"SVT\\"]"', '"[\\"SM\\"]"', '"[\\"l\\u00e4nsstyrelsen\\"]"', '"[\\"Karlshamn Karlskrona\\"]"', '"[\\"Morgan Bengtsson\\"]"', '"[\\"Lotta Idoffson\\"]"', '"[\\"Karl Hamstedt\\"]"', '"[\\"Karlskronas\\"]"', '"[\\"Ringhals Oskarshamn Forsmark\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Litorin\\", \\"a Karlskrona\\"]"', '"[\\"Karlskronas\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Sverige\\"]"', '"[\\"G\\u00e4vleborgs l\\u00e4n\\"]"', '"[\\"Sverige Europa\\"]"', '"[\\"Roger Fredriksson\\", \\"M\\"]"', '"[\\"M\\u00f6rrum Hockey\\"]"', '"[\\"Karlskrona HK Allettan\\", \\"Eric Karlsson\\"]"', '"[\\"International School of Karlskrona\\"]"', '"[\\"Patrik Sommersgaard\\"]"', '"[\\"Karlshamn\\"]"', '"[\\"Kristdemokraterna\\", \\"Karlskrona\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Karlskrona porslinsfabrik\\"]"', '"[\\"Olofstr\\u00f6m\\"]"', '"[\\"stadshuset\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Karlskrona\\"]"', '"[\\"SVT :\\", \\"Sverige\\"]"', '"[\\"Bubblan och Gustav\\"]"', '"[\\"Centerpartiet\\", \\"Blekinge\\"]"', '"[\\"\\u00d6stersj\\u00f6n\\"]"', '"[\\"Martina Ravnb\\u00f6\\"]"', '"[\\"Kristdemokraterna\\"]"', '"[\\"Socialdemokraterna Centerpartiet Kristdemokraterna\\"]"', '"[\\"Christian Svensson\\"]"', '"[\\"M\\u00f6rrum Hockey\\"]"', '"[\\"JVM\\"]"', '"[\\"Karl Hamstedt\\"]"', '"[\\"Kall\\"]"', '"[\\"Riksf\\u00f6rbundet Sveriges\\", \\"museer\\"]"', '"[\\"Karlskrona\\"]"', '"[\\"Blekinge\\"]"', '"[\\"Ambulansf\\u00f6rbundet\\", \\"SOS Alarm\\"]"', '"[\\"Jacob Lagercrantz\\"]"']

print(test)
test_clean = clean_entities(test)
print(test_clean)
print(len(test_clean))

# Example of serializing and deserializing:
entities = [["F\u00f6rsvarsmakten"], ["G\u00e4vleborgs", "l\u00e4n"]]

# Simulate saving and loading (json.dumps/loads)
serialized_entities = json.dumps(entities)  # Save as JSON string
print("Serialized:", serialized_entities)

# Simulate loading the data
loaded_entities = json.loads(serialized_entities)
print("Loaded:", loaded_entities)

# Clean entities (after loading them from the file or database)
cleaned_entities = clean_entities([json.dumps(e) for e in loaded_entities])
print("Cleaned:", cleaned_entities)
