import json
import os
from api_key import API_KEY

# TODO
# make openAI API call
# create story assessment prompt

""" unpacks an individual tandom story and returns: (prompt, reading age specific stories) """
def unpack_tandem_story(path: os.PathLike):
    with open(path) as file:
        data = json.load(file)

    # separate out the story prompt only
    prompt = data["generated_prompts"][0]["prompt"]
    story_prompt = prompt.split("\n3)")[0]

    # separate out the reading age specific stories
    story_data = data["storyout"]["stories"]
    stories = [
        [story_data[i]["readingLevel"]] + [line["text"] for line in story_data[i]["pages"]]
        for i in range(len(story_data))
    ]

    return story_prompt, stories

path = "data\\Tandem_Data\\0a9c05f0-d630-403b-a257-7a7e67452c24.json"
prompt, stories = unpack_tandem_story(path)

print(prompt, stories, API_KEY, sep="\n\n")
