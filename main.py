from cmd import PROMPT
import os
from dotenv import load_dotenv

import requests
from urllib.parse import urlparse
import base64

import re

from openai import OpenAI, AsyncOpenAI
import anthropic
from groq import Groq, AsyncGroq

import asyncio
import discord
from discord.ext import commands
from discord import app_commands
import sys
import json
from urllib.parse import quote
from transformers import AutoTokenizer
from datetime import datetime
import random

BOT_ON = True

# Load environment variables from .env file
load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(command_prefix="!", intents=intents)
# client = commands.Bot(command_prefix='!', intents=intents)
tree = app_commands.CommandTree(client)

guild_id = int(os.getenv("DISCORD_GUILD_ID"))
channel_last = None

# Load a pre-trained tokenizer (for example, the GPT-2 tokenizer)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

all_models = [
    "gpt-4",
    "gpt-4o",
    "claude-3",
]

parameters = {}

# Drama
subject = 'Nosubject'

client_ego = None
client_superego = None
client_other = None
client_director = None

dialogue_history = []
dialogue_ego = []
dialogue_superego = []
dialogue_other = []
dialogue_director = []

director_prompts = {}

stop_sequences = ["\n", "."]
channel_ids = []

max_size_dialog = 40000
user_refs = []
prompt_suffix = """{0}: {1}
{2}:"""

# Time tracking
initial_time = datetime.now()
formatted_time = None

sleep_counter = 60
make_a_promise_likelihood = 0.5
fulfil_a_promise_likelihood = 0.5
promises = []

bracket_counter = 0


def split_into_chunks(text, max_length=2000):
    chunks = []
    lines = text.split('\n')

    if not lines:
        return chunks

    current_chunk = lines[0]

    for line in lines[1:]:
        if len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += '\n' + line

    chunks.append(current_chunk)
    return chunks


async def send_chunks(message, chunks):
    for chunk in chunks:
        await message.reply(chunk)


async def generate_reply(params, client, system_prompt, messages):
    if "stop_sequences" not in params:
        params["stop_sequences"] = []
    if "temperature" not in params:
        params["temperature"] = 0.5
    if "max_tokens" not in params:
        params["max_tokens"] = 256
    if "frequency_penalty" not in params:
        params["frequency_penalty"] = 1.0
    if "presence_penalty" not in params:
        params["presence_penalty"] = 1.0
    if "top_p" not in params:
        params["top_p"] = 1.0
    if "top_k" not in params:
        params["top_k"] = 1.0
    model = params["model"]
    if "claude" in model:
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages[0].delete(0)
        for m in messages:
            if 'name' in m:
                del m['name']
            # Ensure 'content' is a list
            if isinstance(m['content'], list):                
                to_remove = -1            
                for index, c in enumerate(m['content']):
                    if isinstance(c, dict) and  c['type'] == 'image_url':
                        to_remove = index
                        break
                if to_remove > -1:
                    m['content'].pop(to_remove)
        
        message = client.messages.create(model=model,
                                         system=system_prompt,
                                         messages=messages,
                                         max_tokens=params["max_tokens"],
                                         temperature=params["temperature"],
                                            stop_sequences=params["stop_sequences"])
        reply = message.content[0].text
    else:
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            stop=params["stop_sequences"],
            frequency_penalty=params["frequency_penalty"],
            presence_penalty=params["presence_penalty"]
        )
        if response != 0:
            reply = response.choices[0].message.content

    return reply


def detect_unterminated_strings(code):
    # Regular expression to find all strings
    string_pattern = re.compile(r'(["\'])(?:(?!\1).|\\.)*?\1?')
    matches = string_pattern.finditer(code)
    
    unterminated_strings = []
    for match in matches:
        string = match.group(0)
        if len(string) > 1 and string[-1] not in ('"', "'"):  # Not properly closed
            unterminated_strings.append(match)
    
    return unterminated_strings

def fix_unterminated_strings(code):
    unterminated_strings = detect_unterminated_strings(code)
    if not unterminated_strings:
        return code  # No unterminated strings detected

    fixed_code = []
    last_end = 0
    
    for match in unterminated_strings:
        start, end = match.span()
        fixed_code.append(code[last_end:start])
        string = match.group(0)
        quote_type = string[0]
        
        # Add missing closing quote
        if string[-1] != quote_type:
            fixed_code.append(string + quote_type)
        else:
            fixed_code.append(string)
        
        last_end = end

    fixed_code.append(code[last_end:])
    return ''.join(fixed_code)


async def chat_prompt_internal(prompt, parameters, messages, image_path = None):
    try:
        prompt_for_superego = parameters["superego"]["prompt"]
        prompt_for_superego = eval(f'f"""{prompt_for_superego}"""')
        superego_name = parameters["superego"]["name"]

        content = [{"type": "text", "text": prompt}]

            
        messages = []
        messages.append({"role": "user", "name": superego_name, "content": content})

        return await generate_reply(parameters["superego"]["llm_settings"],
                                    client_superego, prompt_for_superego,
                                    messages)
    except Exception as e:
        print(f"Error in internal chat prompt: {e}")
        return {"role": "system", "content": "Looks like ChatGPT is down!"}


async def chat_prompt(prompt, parameters, image_path = None):
    try:
        prompt_for_ego = parameters["ego"]["prompt"]
        prompt_for_ego = fix_unterminated_strings(prompt_for_ego)
        prompt = fix_unterminated_strings(prompt)
        prompt_for_ego = eval(f'f"""{prompt_for_ego}"""')
        ego_name = parameters["ego"]["name"]
        other_name = parameters["other"]["name"]

        history = build_history(dialogue_ego)
        messages = []
        for item in history:
            messages.append(item)

        content = [{"type": "text", "text": prompt}]

        if image_path is not None:
            # Getting the base64 string
            base64_image = encode_image(image_path)
            content.append({"type": "image_url", 
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
            }})

        messages.append({"role": "user", "name": other_name, "content": content})

        reply = await generate_reply(parameters["ego"]["llm_settings"],
                                     client_ego, prompt_for_ego,
                                     messages)
        
        return reply
    except Exception as e:
        print(f"Error in chat prompt: {e}")
        return {"role": "system", "content": "Looks like ChatGPT is down!"}


async def write_bio():
    global dialogue_history, parameters

    if bracket_counter == 0 or bracket_counter % parameters["ego"]["write_bio_schedule"] != 0:
        return None

    autobiography = await generate_autobiography_from_dialogue()

    return autobiography


async def update_ego_instructions():
    global dialogue_history, dialogue_ego, parameters

    try:
        prompt_for_ego = parameters["ego"]["prompt"]
        ego_name = parameters["ego"]["name"]
        name_for_ego = parameters["ego"]["name"]
        prompt_for_superego = parameters["superego"]["prompt"]
        name_for_superego = parameters["superego"]["name"]
        
        dialog_history = print_history()
        prompt_update = ''

        # Update superego
        messages = []

        prompt_for_superego_rewrite_memory_self = parameters["superego"]["prompt_for_rewrite_memory_self"]
        rewrite_memory_instruction = eval(
            f'f"""{prompt_for_superego_rewrite_memory_self}"""')
        messages.append({
            "role":
            "user",
            "content":
            f"{rewrite_memory_instruction}"
        })

        new_superego_prompt = await generate_reply(
            parameters["superego"]["llm_settings"], 
            client_superego,
            parameters["superego"]["prompt"], 
            messages)
        old_superego_prompt = parameters["superego"]["prompt"]
        parameters["superego"]["prompt"] = new_superego_prompt

        prompt_update = f"{prompt_update}Revised superego:\n {new_superego_prompt}"
        dialogue_superego.append({
            "role":
            "user",
            "content":
            f"{rewrite_memory_instruction}"
        })
        dialogue_superego.append({"role": "assistant", "content": new_superego_prompt})


        # Update ego
        messages = []

        prompt_for_superego_rewrite_memory = parameters["superego"]["prompt_for_rewrite_memory"]
        rewrite_memory_instruction = eval(
            f'f"""{prompt_for_superego_rewrite_memory}"""')
        messages.append({
            "role":
            "user",
            "content":
            f"{rewrite_memory_instruction}"
        })

        # Check these parameters carefully - note the ego model is used here
        new_ego_prompt = await generate_reply(
            parameters["superego"]["llm_settings"], 
            client_superego,
            parameters["superego"]["prompt"], 
            messages)
        old_ego_prompt = parameters["ego"]["prompt"]
        parameters["ego"]["prompt"] = new_ego_prompt

        
        # prompt_update = f"{prompt_update}\n\n\nRevised ego:\n {new_ego_prompt}"
        prompt_update = f"Old ego: {old_ego_prompt}\n\n\nRevised ego:\n {new_ego_prompt}"

        dialogue_superego.append({
            "role":
            "user",
            "content":
            f"{rewrite_memory_instruction}"
        })
        dialogue_superego.append({"role": "assistant", "content": new_ego_prompt})

        # Rewrite chat history
        history = build_history()
        
        # Rewrite history
        messages = []


        return prompt_update

    except Exception as e:
        error_message = f"Error updating ego instructions: {e}"
        print(error_message)
        return error_message


async def revise_memory():
    global bracket_counter, parameters

    if bracket_counter == 0 or bracket_counter % parameters["superego"]["rewrite_memory_schedule"] != 0:
        return None

    reply = await update_ego_instructions()

    # if reply != None:
    #     await send_colored_embed(channel_last, "Instruction Set", reply, [], discord.Color.red())

    return reply


async def generate_autobiography_from_dialogue():
    write_bio_instruction = ''
    try:
        prompt_for_ego = parameters["ego"]["prompt"]
        prompt_for_ego = eval(f'f"""{prompt_for_ego}"""')
        ego_name = parameters["ego"]["name"]
        superego_name = parameters["superego"]["name"]
        other_name = parameters["other"]["name"]

        messages = []

        dialog_history = build_history()

        prompt_for_ego_bio = parameters["ego"]["prompt_for_bio"]
        write_bio_instruction = eval(f'f"""{prompt_for_ego_bio}"""')
        messages.append({"role": "user", "content": write_bio_instruction})

        reply = await generate_reply(parameters["ego"]["llm_settings"],
                                    client_ego, prompt_for_ego,
                                    messages)
        
        return reply

    except Exception as e:
        print(
            f"Error summarising biography: {e}, with {write_bio_instruction}")
        return "Error!"


async def generate_script_from_dialogue(write_bio_reply):
    global parameters, subject

    try:

        history = build_history()

        prompt_for_ego = parameters["ego"]["prompt"]
        name_for_ego = parameters["ego"]["name"]

        builder = f"## Settings\n\n{format_settings()}"
        builder += f'Ego prompt: {prompt_for_ego}\n\n'
        
        for message in history:
            role = message['role']
            username = role
            if 'name' in message:
                username = message['name']
            elif role == 'assistant':
                username = name_for_ego
            content = message['content']
            builder += f"{username}: {content}\n\n"

        builder += f"## Autobiography (as told by the ego)\n\n{write_bio_reply}.\n\n"

        # Write content to file
        file_link = unique_filename('./')
        file = write_markdown_to_file(builder, file_link)
        file_name = os.path.basename(file_link)
        return file_name, file
            
    except Exception as e:
        print(e)
        return f"No history, due to {e}", None


@client.event
async def on_ready():
    await tree.sync(guild=discord.Object(guild_id))
    client.loop.create_task(periodic_task())
    print(
        f'We have logged in as {client.user} to {len(client.guilds)} guilds.')


@tree.command(
    name="get", description="Show parameters", guild=discord.Object(guild_id)
)  #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def get_params(interaction: discord.Interaction):
    response = "Current parameters:\n"
    for key, value in parameters.items():
        response += f"{key}: {value}\n"
    response += f"dialogue_history size: {len(dialogue_history)}\n"
    response += f"stop_sequences: {stop_sequences}\n"
    history = build_history()
    response += f"history: {history}\n"

    chunks = split_into_chunks(response, max_length=1800)

    # Send the first chunk as the initial response
    await interaction.response.send_message(chunks[0])

    # Send the remaining chunks as follow-up messages
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk)


@tree.command(
    name="set", description="Set parameters", guild=discord.Object(guild_id)
)  #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def set_params(interaction,
                     prompt_for_ego: str = None,
                     model_id: int = None,
                     temperature: float = None,
                     top_p: float = None,
                     frequency_penalty: float = None,
                     presence_penalty: float = None,
                     summary_level: int = None,
                     max_size_dialog: int = None,
                     channel_reset: str = None):
    global dialogue_history
    if prompt_for_ego is not None:
        parameters["ego"]["prompt"] = prompt_for_ego
    if model_id is not None:
        model = all_models[model_id]
        parameters["model_id"] = str(model_id)
        parameters["model"] = model
    if temperature is not None:
        parameters["temperature"] = str(temperature)
    if top_p is not None:
        parameters["top_p"] = top_p
    if frequency_penalty is not None:
        parameters["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        parameters["presence_penalty"] = presence_penalty
    if summary_level is not None:
        parameters["summary_level"] = summary_level
    if max_size_dialog is not None:
        dialogue_history = dialogue_history[:max_size_dialog]
        parameters["max_size_dialog"] = max_size_dialog
    if channel_reset is not None:
        parameters["channel_reset"] = channel_reset
    await interaction.response.send_message("Parameters updated.")


@tree.command(
    name="reset", description="Reset memory", guild=discord.Object(guild_id)
)  #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def reset(interaction: discord.Interaction):
    dialogue_history.clear()
    await interaction.response.send_message(
        "Summary and history have been reset!")


@tree.command(
    name="clear_chat",
    description="Clear Chat history",
    guild=discord.Object(guild_id)
)  #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def clear_chat(interaction: discord.Interaction, limit: int = 0):
    channel = client.get_channel(interaction.channel_id)
    await channel.purge(limit=limit)
    await interaction.response.send_message(
        f"{limit} responses cleared from chat.")


@tree.command(name="reload_settings",
              description="Reload settings",
              guild=discord.Object(guild_id))
async def command_reload_settings(interaction: discord.Interaction):
    reload_settings()
    try:
        await interaction.response.send_message(f"Reloading settings...")
    except Exception as e:
        print(f"Exception reloading settings: {e}")


@tree.command(name="generate_biography",
              description="Generates a biography",
              guild=discord.Object(guild_id))
async def generate_biography(interaction: discord.Interaction):
    reply = await generate_autobiography_from_dialogue()
    await interaction.response.send_message("Generating biography now...")
    await send_colored_embed(channel_last, "Autobiography", reply, [],
                             discord.Color.green())

@tree.command(name="generate_prompts",
              description="Generates ego and superego prompts",
              guild=discord.Object(guild_id))
async def generate_prompts(interaction: discord.Interaction):
    prompt_for_ego = parameters["ego"]["prompt"]
    prompt_for_superego = parameters["superego"]["prompt"]
    await send_colored_embed(channel_last, "Memory Revision - ego", prompt_for_ego, [],
                             discord.Color.red())
    await send_colored_embed(channel_last, "Memory Revision - superego", prompt_for_superego, [],
                             discord.Color.red())


# Write the Markdown content to a file
def write_markdown_to_file(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    return open(file_path, 'r', encoding='utf-8')

# Generate a file link (assuming the bot serves files from a known directory)
def generate_file_link(file_path, base_url):
    file_name = os.path.basename(file_path)
    return f"{base_url}/{file_name}"

# Generate a unique file name using timestamp
def unique_filename(base_path, extension='md'):
    global subject
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    return os.path.join(base_path, f'dialogue_{subject}_{timestamp}.{extension}')

@tree.command(name="generate_script",
              description="Generates the conversation as a script",
              guild=discord.Object(guild_id))
async def generate_script(interaction: discord.Interaction):
    write_bio_reply = await write_bio()
    file_name, file = await generate_script_from_dialogue(write_bio_reply)
    await interaction.response.send_message(file=discord.File(file, file_name))
    


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
async def generate_image(prompt):
    try:
        director_name = parameters["director"]["name"]
        llm_settings = parameters["director"]["llm_settings"]
        model = llm_settings["model"]
        
        prompt_for_director_image_generation = parameters["director"]["prompt_for_director_image_generation"]
        prompt_for_director_image_generation = eval(
            f'f"""{prompt_for_director_image_generation}"""')

        messages = []
        messages.append({"role": "system", "content": "You are a creative interpreter of dialogue, and determine whether anything in the respondent's discourse is worth drawing. If so, you will convert that discourse to a detailed prompt for a surrealist image."})
        messages.append({
            "role": "user",
            "name": director_name,
            "content": prompt_for_director_image_generation
        })
        response = await client_director.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=llm_settings["max_tokens"],
            temperature=llm_settings["temperature"],
        )
        if response != 0:
            reply = response.choices[0].message.content

        # Check if reply is empty
        if reply == "" or reply == '""':
            return None
        
        print(f"Image prompt: {reply}")

        response = await client_director.images.generate(
            model="dall-e-3",
            prompt=reply,
            size="1792x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url

        # Parse the URL to get the file name
        parsed_url = urlparse(image_url)
        file_name = os.path.basename(parsed_url.path)
        
        # Create the local path
        local_path = os.path.join('./images/', file_name)
        
        response = requests.get(image_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Open the local file for writing in binary mode
            with open(local_path, 'wb') as file:
                file.write(response.content)
            print(f"File saved to: {local_path}")
        else:
            print(f"Failed to retrieve the file. Status code: {response.status_code}")

        return local_path
    except Exception as e:
        print(f"Error generating image: {e}")
        return None



def build_history(dialogue = None):
    if dialogue is None:
        dialogue = dialogue_history

    # GPT limit, minus max response token size
    token_limit = 16097 - parameters["ego"]["llm_settings"]["max_tokens"]

    # Tokenize the string to get the number of tokens
    tokens_len = 0

    # Iterate through the list in reverse order
    dialogue_limited = []
    for item in reversed(dialogue):
        tokens_item = tokenizer(str(item) + "\n", return_tensors='pt')
        tokens_item_len = tokens_item.input_ids.size(1)
        if tokens_len + tokens_item_len < token_limit:
            tokens_len = tokens_len + tokens_item_len
            dialogue_limited.insert(0, item)
        else:
            break

    # Construct the dialog history
    return dialogue_limited


def print_history():
    history = build_history()
    dialog_history = ''
    for message in history:
        message_role = message['role']
        message_content = message['content']
        dialog_history += f"{message_role}: {message_content}\n"
    return dialog_history


# Method to create and send a colored embed
async def send_colored_embed(channel,
                             title,
                             description,
                             fields,
                             color=discord.Color.blue()):
    # Split the description into chunks of 4096 characters
    chunk_size = 4096
    # description_chunks = [description[i:i+chunk_size] for i in range(0, len(description), chunk_size)]
    description_chunks = []

    if len(description) < chunk_size:
        description_chunks.append(description)
    else:
        while len(description) > chunk_size:
            # Find the last space within the chunk
            last_space_index = description.rfind(' ', 0, chunk_size)
            if last_space_index == -1:
                # No space found, split at chunk_size
                last_space_index = chunk_size
            # Add the chunk to the list and remove it from the description
            description_chunks.append(description[:last_space_index])
            description = description[last_space_index:].lstrip()
    
    # Send each chunk as a separate embed
    for i, chunk in enumerate(description_chunks):
        # Create an embed object
        embed = discord.Embed(title=title if i == 0 else f"{title} (cont.)", description=chunk, color=color)
        
        # Add fields only to the first embed
        # if i == 0:
        for name, value, inline in fields:
            embed.add_field(name=name, value=value, inline=inline)

        # Send the embed
        await channel.send(embed=embed)


async def download_image(url, filename):
    # Send a GET request to the URL
    response = requests.get(url)
    save_path = None

    # Check if the request was successful
    if response.status_code == 200:
        # Define the path where the image will be saved
        save_path = os.path.join('./images/', filename)

        # Write the image content to a file
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Image saved to: {save_path}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
    return save_path

@client.event
async def on_message(message):
    global dialogue_history
    global prompt_for_ego
    global user_refs
    global channel_last
    global bracket_counter
    global parameters

    max_size_dialog = parameters["max_size_dialog"]

    if message.author == client.user:
        return

    if message.channel.id not in channel_ids:
        print(
            f"Channel {message.channel.id} not found in these channel_ids: {channel_ids}"
        )
        return

    data = message.content
    channel = message.channel
    channel_last = message.channel
    author = message.author
    mentions = message.mentions
    author_name = author.name
    if author.nick is not None:
        if author.nick not in user_refs:
            user_refs.append(author.nick)
        author_name = author.nick
    elif author.name not in user_refs:
        user_refs.append(author.name)

    prompt_for_ego = parameters["ego"]["prompt"]
    prompt_for_superego = parameters["superego"]["prompt"]

# Check if the message has attachments
    image_path = None
    if message.attachments:
        for attachment in message.attachments:
            # Check if the attachment is an image
            if attachment.filename.lower().endswith(('png', 'jpg', 'jpeg', 'gif')):
                # Download the image
                image_path = await download_image(attachment.url, attachment.filename)

    # Simulate typing for 3 seconds
    async with channel.typing():
        prompt = data
        member_stops = [x + ":" for x in user_refs]

        # For debugging
        reply = "[NO-GPT RESPONSE]"
        try:
            prompt = f"Frame #{bracket_counter}. User is {author_name}. User says: {prompt}"
            reply = await chat_prompt(prompt, parameters, image_path)

        except Exception as e:
            print(e)
            reply = 'So sorry, dear User! ChatGPT is down.'

        if reply != "":

            chunks = split_into_chunks(reply, max_length=1600)
            await send_chunks(message, chunks)

            result = reply

            content = [{
                "type": "text",
                "text": result
            }]

            content = await check_for_images(result, content)

            # if len(dialogue_history) >= max_size_dialog:
            #     dialogue_history.pop(0)
            #     dialogue_history.pop(0)
            dialogue_history.append({"role": "user", "name": author_name, "content": prompt})
            dialogue_history.append({"role": "assistant", "content": content})

            dialogue_other.append({
                "role": "assistant",
                "content": prompt
            })
            dialogue_other.append({
                "role": "user",
                "name": parameters["ego"]["name"],
                "content": content
            })
        else:
            await message.channel.send("Sorry, couldn't reply")

        reply_revision = await revise_memory()
        if reply_revision != None:
            await send_colored_embed(channel_last, "Memory Revision",
                                     reply_revision, [], discord.Color.red())

        reply_bio = await write_bio()
        if reply_bio != None:
            await send_colored_embed(channel_last, "Autobiography", reply_bio,
                                     [], discord.Color.green())
            file_name, file = await generate_script_from_dialogue(reply_bio)
            await message.channel.send(file=discord.File(file, file_name))

        bracket_counter = bracket_counter + 1

# Check if the following includes a drawing to render
async def check_for_images(result, content):
    if not parameters["ego"]["generate_images"]:
        return content
    image_path = await generate_image(result)
    if image_path is not None:
        # Check if the file exists
        if os.path.exists(image_path):
            # Send the image
            await channel_last.send(file=discord.File(image_path))
        else:
            await channel_last.send('Image file not found.')
        base64_image = encode_image(image_path)
        content.append(
            {"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }})
    return content

def format_settings():
    global parameters

    ego_name = parameters["ego"]["name"]
    superego_name = parameters["superego"]["name"]
    other_name = parameters["other"]["name"]
    director_name = parameters["director"]["name"]

    ego_model = parameters["ego"]["llm_settings"]["model"]
    ego_temperature = parameters["ego"]["llm_settings"]["temperature"]
    ego_write_bio_schedule = parameters["ego"]["write_bio_schedule"]

    superego_model = parameters["superego"]["llm_settings"]["model"]
    superego_temperature = parameters["superego"]["llm_settings"]["temperature"]
    superego_likelihood_to_rewrite_others_prompt = parameters["superego"]["likelihood_to_rewrite_others_prompt"]
    superego_likelihood_to_suggest_alternate_response = parameters["superego"]["likelihood_to_suggest_alternate_response"]
    superego_rewrite_memory_schedule = parameters["superego"]["rewrite_memory_schedule"]

    other_model = parameters["other"]["llm_settings"]["model"]
    other_temperature = parameters["other"]["llm_settings"]["temperature"]
    
    director_model = parameters["director"]["llm_settings"]["model"]
    director_temperature = parameters["director"]["llm_settings"]["temperature"]
    director_intervention = parameters["director"]["intervention"]

    settings_messages = f"""
**Subject**: {subject}

Ego is called: *{ego_name}*.
Ego uses *{ego_model}* with temperature of {ego_temperature}.
Ego writes a bio note every {ego_write_bio_schedule} turns.

Superego is called: *{superego_name}*.
Superego uses *{superego_model}* with temperature of {superego_temperature}.
Superego has a {superego_likelihood_to_rewrite_others_prompt * 100} percent chance of rewriting the other's prompt.
Superego has a {superego_likelihood_to_suggest_alternate_response * 100} percent chance of suggesting an alternate response.
Superego rewrites the ego's system prompt every {superego_rewrite_memory_schedule} turns.

Other character is called: *{other_name}*.
Other character uses *{other_model}* with temperature of {other_temperature}.

Director is called: *{director_name}*.
Director uses *{director_model}* with temperature of {director_temperature}.
Director intervenes every {director_intervention} turns.

"""
    return settings_messages

async def periodic_task():
    global dialogue_history, dialogue_other
    global channel_last, bracket_counter, director_prompts
    global sleep_counter, formatted_time
    global make_a_promise_likelihood, fulfil_a_promise_likelihood

    channel_last_id = int(channel_ids[0])
    channel_last = (client.get_channel(channel_last_id)
                    or await client.fetch_channel(channel_last_id))

    director_intervention = parameters["director"]["intervention"]
    write_bio_schedule = parameters["ego"]["write_bio_schedule"]

    # Loop until the autography is written
    while bracket_counter <= write_bio_schedule:
        # Get the current time
        now = datetime.now()

        elapsed_time = now - initial_time

        # Convert the difference into seconds
        total_seconds = elapsed_time.total_seconds() * 360

        # Convert seconds into hours and minutes
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        elapsed_time_formatted = f"Difference is {int(hours)} hours and {int(minutes)} minutes"

        make_a_promise = make_a_promise_likelihood > random.random()
        fulfil_a_promise = fulfil_a_promise_likelihood > random.random()

        if channel_last is None:
            print("Channel not found")
            return

        message = ''

        revise_memory_reply = await revise_memory()
        if revise_memory_reply != None:
            await send_colored_embed(channel_last, "Memory Revision",
                                        revise_memory_reply, [], discord.Color.red())


        # Load prompts from the JSON file
        prompt = None
        username = None

        ego_name = parameters["ego"]["name"]
        superego_name = parameters["superego"]["name"]
        other_name = parameters["other"]["name"]
        director_name = parameters["director"]["name"]

        if bracket_counter == 0:

            settings_messages = format_settings()
            await send_colored_embed(channel_last, "Drama Machine Settings",
                                    settings_messages, [], discord.Color.yellow())


        if bracket_counter % director_intervention == 0:
            prompt, username = await get_directors_prompt(bracket_counter)

        if prompt is None:
            prompt, username = await get_others_prompt(bracket_counter)
            
        ego_name = parameters["ego"]["name"]
        superego_name = parameters["superego"]["name"]

        if prompt is not None:
            # if make_a_promise:
            #     prompt = "Make promise: Generate a very brief note-to-self that will remind you to reflect on events to date at a future point in time."
            # elif fulfil_a_promise and len(promises) > 0:
            #     random_index = random.randint(0, len(promises) - 1)
            #     prompt = promises[random_index]
            #     prompt = "Fulfil promise: " + prompt
            #     promises.remove(promises[random_index])
            # else:
            #     prompt = elapsed_time_formatted

    
            await channel_last.send(f"**Step {bracket_counter}**")
            prompt_to_display = f"*{username}*: {prompt}"

            await channel_last.send(prompt_to_display[0:2000])
            result = ''
            modified_prompt = prompt
            internal_dialogue = ''


            likelihood_to_rewrite_others_prompt = float(parameters["superego"]["likelihood_to_rewrite_others_prompt"])
            should_we_rewrite_the_prompt = likelihood_to_rewrite_others_prompt > random.random() and bracket_counter % director_intervention != 0

            likelihood_to_suggest_alternate_response = float(parameters["superego"]["likelihood_to_suggest_alternate_response"])
            should_we_suggest_alternate_responset = likelihood_to_suggest_alternate_response > random.random() and bracket_counter % director_intervention != 0

            
            if should_we_rewrite_the_prompt:

                # rewrite the prompt
                prompt_for_rewrite_others_prompt = parameters["superego"]["prompt_for_rewrite_others_prompt"]
                prompt_for_rewrite_others_prompt = eval(f'f"""{prompt_for_rewrite_others_prompt}"""')

                messages = [{"role": "user","content":prompt_for_rewrite_others_prompt}]
                modified_prompt = await generate_reply(
                    parameters["superego"]["llm_settings"], 
                    client_superego,
                    parameters["superego"]["prompt"], 
                    messages)

            try:
                result = await chat_prompt(modified_prompt, parameters)
            except Exception as e:
                print(e)
                result = 'So sorry, dear User! ChatGPT is down.'

            superego_prompt = prompt
            superego_response = result
            if should_we_suggest_alternate_responset:

                prompt_for_suggest_alternate_response = parameters["superego"]["prompt_for_suggest_alternate_response"]
                prompt_for_suggest_alternate_response = eval(f'f"""{prompt_for_suggest_alternate_response}"""')

                # update this logic with 'internal drama'
                superego_prompt = prompt_for_suggest_alternate_response
                messages = [{"role": "user", "content": prompt_for_suggest_alternate_response}]
                superego_response = await generate_reply(
                    parameters["superego"]["llm_settings"], 
                    client_superego,
                    parameters["superego"]["prompt"], 
                    messages)            
                
                internal = f"**Here's what I heard:**\n\n {modified_prompt}\n\n\n**My initial reply: **{result}\n\n\n**My super ego response: **\n\n{superego_response}"
                await send_colored_embed(channel_last, "Internal dialogue",
                                        internal, [], discord.Color.blue())

                prompt_for_reflection_on_alternate_response = parameters["superego"]["prompt_for_reflection_on_alternate_response"]
                prompt_for_reflection_on_alternate_response = eval(f'f"""{prompt_for_reflection_on_alternate_response}"""')

                messages = [{"role": "user", "content": prompt_for_reflection_on_alternate_response}]
                result = await generate_reply(
                    parameters["ego"]["llm_settings"], 
                    client_ego,
                    parameters["ego"]["prompt"], 
                    messages)            
                
            
            ego_name = parameters["ego"]["name"]
            result_to_display = f"*{ego_name}*: {result}"
            chunks = split_into_chunks(result_to_display, max_length=1600)
            for chunk in chunks:
                await channel_last.send(chunk)

            content = [{
                "type": "text",
                "text": result
            }]

            content = await check_for_images(result, content)

            if result != "":
                # if len(dialogue_history) >= max_size_dialog:
                #     dialogue_history.pop(0)
                #     dialogue_history.pop(0)
                dialogue_history.append({
                    "role": "user",
                    "name": username,
                    "content": prompt
                })
                dialogue_history.append({
                    "role": "assistant",
                    "content": result
                })

                dialogue_ego.append({
                    "role": "user",
                    "name": username,
                    "content": modified_prompt
                })
                dialogue_ego.append({
                    "role": "assistant",
                    "content": result
                })

                dialogue_superego.append({
                    "role": "user",
                    "name": ego_name,
                    "content": superego_prompt
                })
                dialogue_superego.append({
                    "role": "assistant",
                    "content": superego_response
                })

                dialogue_other.append({
                    "role": "assistant",
                    "content": prompt
                })
                dialogue_other.append({
                    "role": "user",
                    "name": ego_name,
                    "content": content
                })

                dialogue_director.append({
                    "role": "user",
                    "name": username,
                    "content": prompt
                })
                dialogue_director.append({
                    "role": "assistant",
                    "content": content
                })                

                if make_a_promise:
                    promises.append(result)

        write_bio_reply = await write_bio()
        if write_bio_reply != None:
            await send_colored_embed(channel_last, "Autobiography",
                                        write_bio_reply, [], discord.Color.green())
            file_name, file = await generate_script_from_dialogue(write_bio_reply)
            await channel_last.send(file=discord.File(file, file_name))

        bracket_counter = bracket_counter + 1
        await asyncio.sleep(sleep_counter)  # sleep for 20 seconds


# Function to read Markdown content from a file
def load_markdown_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as md_file:
        return md_file.read()


# Function to load JSON prompts from a file
def load_prompts(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)['prompts']
    except Exception as e:
        print(f"Error loading prompts file: {e}")
        return {}


# Function to get the prompt for the current step
def get_director_script_prompt(prompts, step):
    for prompt in prompts:
        if prompt['step'] == step:
            return prompt['prompt']
    return None

# Function to get the prompt for the current step
async def get_directors_prompt(step):
    try:
        director_name = parameters["director"]["name"]
        ego_name = parameters["ego"]["name"]
        other_name = parameters["other"]["name"]

        messages = []

        dialog_history = print_history()

        prompt_for_director = parameters["director"]["prompt"]
        prompt_for_director = eval(
            f'f"""{prompt_for_director}"""')
        
        prompt_for_director_instruction = parameters["director"]["prompt_for_director_instruction"]
        prompt_for_director_instruction = eval(
            f'f"""{prompt_for_director_instruction}"""')
        
        messages.append({
            "role":
            "user",
            "name": ego_name,
            "content":
            f"{prompt_for_director_instruction}"
        })

        reply = await generate_reply(parameters["director"]["llm_settings"],
                                    client_director, prompt_for_director,
                                    messages)

        return reply, director_name
    
    except Exception as e:
        message = f"Error in generating director message! {e}"
        print(message)
        return message, director_name

# Function to get the prompt for the current step
async def get_others_prompt(step):
    global dialogue_other

    try:
        prompt_for_other = parameters["other"]["prompt"]
        other_name = parameters["other"]["name"]
        ego_name = parameters["ego"]["name"]

        dialog_history = print_history()
        
        last_other_message = None
        last_message_content = ''
        if len(dialogue_other) > 0:
            last_other_message = dialogue_other.pop()
            last_message_content_container = last_other_message["content"]
            if isinstance(last_message_content_container, list):
                for message in last_message_content_container:
                    if message["type"] == "text":
                        last_message_content = message["text"]

        prompt_for_other_message = parameters["other"]["prompt_for_message"]
        prompt_for_other_message = eval(
            f'f"""{prompt_for_other_message}"""')

        if last_other_message == None:
            if 'claude' in parameters["other"]["llm_settings"]["model"]:
                last_other_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", 
                        "text": f"{ego_name}: {prompt_for_other_message}"}
                    ]
                }
            else:
                last_other_message = {
                    "role": "user",
                    "name": ego_name,
                    "content": [
                        {"type": "text", 
                        "text": f"{prompt_for_other_message}"}
                    ]
                }
        else:
            for message in last_other_message["content"]:
                if message["type"] == "text":
                    message["text"] = prompt_for_other_message

        dialogue_other.append(last_other_message)

        reply = await generate_reply(parameters["other"]["llm_settings"],
                                    client_other, prompt_for_other,
                                    dialogue_other)

        return reply, other_name
    
    except Exception as e:
        error_message = f"Error in generating other's message! {e}"
        print(error_message)
        return error_message, other_name

def generate_client(model):
    client = None
    if 'claude-3' in model:
        client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY'))
    elif 'gpt' in model:
        client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"))
    elif 'llama3-' in model:
        client = AsyncGroq(api_key=os.getenv('GROQ_API_KEY'))
    else:
        # Hardwired for the moment
        client = AsyncOpenAI(
            base_url="http://localhost:1234/v1", api_key="lm-studio")
    return client

def reload_settings():
    global client_ego, client_superego, client_other, client_director
    global parameters
    global guild_id, channel_ids
    global sleep_counter, make_a_promise_likelihood, fulfil_a_promise_likelihood
    global director_prompts
    global subject
    global bracket_counter
    global dialogue_history, dialogue_ego, dialogue_superego, dialogue_other, dialogue_director

    # Load the settings file
    try:
        subject_settings_file = f'settings_{subject}.json'
        with open(subject_settings_file, "r") as read_file:
            parameters = json.loads(read_file.read())
    except Exception as e:
        print(f"Error loading settings file: {e}")
        return

    channel_ids = parameters['channel_ids']

    try:
        sleep_counter = int(parameters['sleep_counter'])
        make_a_promise_likelihood = float(
            parameters['make_a_promise_likelihood'])
        fulfil_a_promise_likelihood = float(
            parameters['fulfil_a_promise_likelihood'])
    except Exception as e:
        print(e)
        pass

    parameters["ego"]["prompt"] = load_markdown_content(
        parameters["ego"]["prompt"])
    parameters["ego"]["prompt_for_bio"] = load_markdown_content(
        parameters["ego"]['prompt_for_bio'])

    parameters["superego"]["prompt"] = load_markdown_content(
        parameters["superego"]["prompt"])
    parameters["superego"]["prompt_for_rewrite_memory"] = load_markdown_content(
        parameters["superego"]['prompt_for_rewrite_memory'])
    parameters["superego"]["prompt_for_rewrite_memory_self"] = load_markdown_content(
        parameters["superego"]['prompt_for_rewrite_memory_self'])
    parameters["superego"]["prompt_for_rewrite_others_prompt"] = load_markdown_content(
        parameters["superego"]['prompt_for_rewrite_others_prompt'])
    parameters["superego"]["prompt_for_suggest_alternate_response"] = load_markdown_content(
        parameters["superego"]['prompt_for_suggest_alternate_response'])
    parameters["superego"]["prompt_for_reflection_on_alternate_response"] = load_markdown_content(
        parameters["superego"]['prompt_for_reflection_on_alternate_response'])

    parameters["other"]["prompt"] = load_markdown_content(
        parameters["other"]["prompt"])
    parameters["other"]["prompt_for_message"] = load_markdown_content(
        parameters["other"]['prompt_for_message'])
    parameters["director"]["prompt"] = load_markdown_content(parameters["director"]["prompt"])
    parameters["director"]["prompt_for_director_instruction"] = load_markdown_content(parameters["director"]["prompt_for_director_instruction"])
    parameters["director"]["prompt_for_director_image_generation"] = load_markdown_content(parameters["director"]["prompt_for_director_image_generation"])

    # Reset dialogue
    dialogue_history = []
    dialogue_ego = []
    dialogue_superego = []
    dialogue_other = []
    dialogue_director = []    

    # Add an initial 'user' message to the other's conversation
    dialogue_other.append({"role": "user", "content": "[Start the conversation]"}) 

    # Load prompts from the JSON file

    director_prompts = load_prompts(parameters["director"]['director_script'])

    client_ego = generate_client(parameters["ego"]["llm_settings"]["model"])
    client_superego = generate_client(parameters["superego"]["llm_settings"]["model"])
    client_other = generate_client(parameters["other"]["llm_settings"]["model"])
    client_director = generate_client(parameters["director"]["llm_settings"]["model"])

    bracket_counter = 0


def main():
    global subject

    subject = sys.argv[2]

    reload_settings()

    # Load the Discord token from the environment variable
    discord_token_env_var = parameters['discord_token_env_var']
    discord_token = os.environ.get(discord_token_env_var)

    client.run(discord_token)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 main.py --subject <name-of-subject>")
        sys.exit(1)

    main()
