import json
import os
import time
from api_key import API_KEY
from openai import OpenAI
from prompts import system_prompt_3

# TODO
# look at bad stories
# look at asking the LLM to **improve** the story
# provide rating scales for the AI (/7, /5 /10 etc.)
# tweak rating criteria
# nlp analysis of story corpus (word variety, average word length etc.)

''' 
Unpacks an individual tandom story and returns a tuple: (story prompt, title, story, reading level).
Conditional statements are the best method for handling this data due to differing formats within the corpus. 
Returns None if no story is found, as there is therefore nothing to evauluate.
'''
def unpack_tandem_story(path: os.PathLike):
    with open(path) as file:
        data = json.load(file)

    # separate out the story prompt if it exists
    # 'generated_prompts' format
    if 'generated_prompts' in data and data['generated_prompts']:
        prompt = data['generated_prompts'][0]['prompt']
        story_prompt = prompt.split('\n3)')[0]
    # 'prompt' format
    elif 'prompt' in data and data['prompts']:
        story_prompt = data['prompts'][0]['user']
    # no prompt
    else:
        story_prompt = 'No prompt available'

    # separate out the story and story metadata
    # 'storyout' format
    if 'storyout' in data and data['storyout']:
        story_data = data['storyout']
        # 'stories' case
        if 'stories' in story_data and story_data['stories']:
            pages = story_data['stories'][0]['pages']
        # 'pages' case
        elif 'pages' in story_data and story_data['pages']:
            pages = story_data['pages']
        # no story case
        else:
            return None
        story = ' '.join(line['text'] for line in pages)

        # get title and reading level for 'storyout' format
        title = story_data['title']
        reading_level = data['storyin']['reader']['reading_age']

    # 'story' format
    elif 'story' in data and data['story']:
        story = data['story']['story']
        title = data['title']
        reading_level = '(reading age found in prompt)'
    # no story case
    else:
        return None

    return story_prompt, title, story, reading_level


''' generate the user prompt, i.e. format the story rating task '''
def generate_user_prompt(story_data: tuple):
    story_prompt, title, story, reading_age = story_data
    user_prompt = f'''
    Please rate the following story, which has a suggested reading age of {reading_age}:

    Title: {title}

    Prompt: {story_prompt}

    Story: {story}
    '''
    return user_prompt

# TODO: Progress bar
def process_all_stories(dir: os.PathLike):
    print(f'Processing all stories in {dir}')
    start = time.time()
    for file in os.listdir(dir):
        print(file)
        # skip dev files
        if file[:3] == 'dev':
            continue
        unpacked_story = unpack_tandem_story(os.path.join(dir, file))
        # only generate prompts on not None stories
        if unpacked_story:
            user_prompt = generate_user_prompt(unpacked_story)
    print(f'Completed in {time.time() - start}s')


''' make a call to the gpt api '''
def make_gpt_api_call(system_prompt: str, user_prompt: str, model='gpt-4o'):
    client = OpenAI(api_key=API_KEY)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    process_all_stories('data/Tandem_Data')

    ivo_path = 'data\\Tandem_Data\\0a9c05f0-d630-403b-a257-7a7e67452c24.json'
    astronaut_path = 'data\\Tandem_Data\\0a774cd2-ee78-4861-a33e-1a9c2f3cfed4 (1).json'

    astronaut_story = unpack_tandem_story(astronaut_path)
    user_prompt = generate_user_prompt(astronaut_story)
    print(user_prompt)

    # gpt_response = make_gpt_api_call(system_prompt_3, user_prompt)
    # print(gpt_response)
