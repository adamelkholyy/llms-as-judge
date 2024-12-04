import json
import os
from api_key import API_KEY
from openai import OpenAI
from prompts import system_prompt_3

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
story_prompt, stories = unpack_tandem_story(path)

# generate the user prompt, i.e. the story rating task
def generate_user_prompt(prompt, story, reading_age):
    user_prompt = f"""
    Please rate the following story, which has a suggested reading age of {reading_age}:
    
    Prompt: {prompt}

    Story: {story}
    """
    return user_prompt

# generate user prompt using first story in stories
full_story = "\n".join(stories[0][1:])
reading_age = stories[0][0]
user_prompt = generate_user_prompt(story_prompt, full_story, reading_age)


# gpt4-o1 api call
client = OpenAI(api_key=API_KEY)
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt_3},
        {
            "role": "user",
            "content": user_prompt
        }
    ]
)

print(completion.choices[0].message.content)

