from cmd import PROMPT
import os
from dotenv import load_dotenv

import requests
from urllib.parse import urlparse
import base64

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

client_character = None
client_narrator = None
client_audience = None
client_director = None

dialogue_history = []
dialogue_narrator = []
dialogue_audience = []
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

bracket_counter = 1


def split_into_chunks(text, max_length=2000):
    chunks = []
    words = text.split()

    if not words:
        return chunks

    current_chunk = words[0]

    for word in words[1:]:
        if len(current_chunk) + len(word) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = word
        else:
            current_chunk += ' ' + word

    chunks.append(current_chunk)
    return chunks


async def send_chunks(message, chunks):
    for chunk in chunks:
        await message.reply(chunk)


async def generate_reply(params, client, system_prompt, messages):
    model = params["model"]
    if model.startswith("claude-3"):
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages[0].delete(0)
        for m in messages:
            del m['name']

        message = client.messages.create(model=model,
                                         max_tokens=params["max_tokens"],
                                         temperature=params["temperature"],
                                         system=system_prompt,
                                         messages=messages)
        reply = message.content[0].text
    else:
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
        )
        if response != 0:
            reply = response.choices[0].message.content

    return reply


async def chat_prompt_internal(prompt, parameters, messages, image_path = None):
    try:
        prompt_for_narrator = parameters["narrator"]["prompt"]
        prompt_for_narrator = eval(f'f"""{prompt_for_narrator}"""')
        narrator_name = parameters["narrator"]["name"]

        content = [{"type": "text", "text": prompt}]

            
        messages = []
        messages.append({"role": "user", "name": narrator_name, "content": content})

        return await generate_reply(parameters["narrator"]["llm_settings"],
                                    client_narrator, prompt_for_narrator,
                                    messages)
    except Exception as e:
        print(f"Error in chat prompt: {e}")
        return {"role": "system", "content": "Looks like ChatGPT is down!"}


async def chat_prompt(prompt, parameters, image_path = None):
    try:
        prompt_for_character = parameters["character"]["prompt"]
        prompt_for_character = eval(f'f"""{prompt_for_character}"""')
        audience_name = parameters["character"]["name"]

        history = build_history()
        messages = []
        for item in history:
            messages.append(item)

        # internal_messages = []
        # for item in history:
        #     internal_messages.append(item)
        # internal_messages.append({"role": "user", "content": f"About me: {prompt_for_character}"})
        # internal_prompt = f"Given what you know about me, what should I say in reply to this '{prompt}'?"
        # internal_reply = await chat_prompt_internal(internal_prompt, parameters, messages)
        # inserted_prompt = f"When you are asked this '{prompt}'?"
        # messages.append({"role": "user", "content": inserted_prompt})
        # messages.append({"role": "assistant", "content": internal_reply})

        content = [{"type": "text", "text": prompt}]
        if image_path is not None:
            # Getting the base64 string
            base64_image = encode_image(image_path)
            content.append({"type": "image_url", 
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
            }})
        messages.append({"role": "user", "name": audience_name, "content": content})

        reply = await generate_reply(parameters["character"]["llm_settings"],
                                     client_character, prompt_for_character,
                                     messages)

        # print("*****")
        # counter = 1
        # for message in messages:
        #     print(str(counter)+": "+message["content"][0:50])
        #     counter = counter + 1
        # print(str(counter)+": "+reply[0:50])

        return reply
    except Exception as e:
        print(f"Error in chat prompt: {e}")
        return {"role": "system", "content": "Looks like ChatGPT is down!"}


async def write_bio():
    global dialogue_history, parameters

    if bracket_counter % parameters["write_bio_schedule"] != 0:
        return None

    autobiography = await generate_autobiography_from_dialogue()

    return autobiography


async def update_character_instructions():
    global dialogue_history, parameters

    try:
        prompt_for_character = parameters["character"]["prompt"]
        prompt_for_narrator = parameters["narrator"]["prompt"]
        

        messages = []

        # history = build_history()
        # write_bio_instruction = parameters["write_bio_instruction"]
        # messages.append({'role': 'user', 'content': f"{write_bio_instruction}: '{history}'."})
        # reply = await generate_reply(parameters["gpt_settings_narrator"], client_narrator, prompt_for_narrator, messages)
        # del messages[0]
        # messages.append({'role': 'assistant', 'content': reply})

        # Rewrite history
        # messages_to_rewrite = int(parameters["rewrite_memory"]) * 2
        # messages = []
        # for message in dialogue_history[-messages_to_rewrite:]:
        #     if message['role'] == 'assistant':
        #         content = message['content']
        #         messages.append({'role': 'user', 'content': f"Review and revise the following: '{content}'. Include nothing but the revised statement."})
        #         reply = await generate_reply(client_character, parameters, parameters, prompt_for_narrator, messages)
        #         # Remove first system message
        #         del messages[0]
        #         messages.append({'role': 'assistant', 'content': reply})
        #         message['content'] = reply

        # print(dialogue_history[-1:]['content'])
        # reply = await generate_reply(client_character, parameters, parameters["model"], prompt_for_narrator, messages)
        # messages.append({"role": "assistant", "content": reply})
        # messages.append({"role": "user", "content": f"Analyse this text: ."})

        # reply2 = await generate_reply(client_character, parameters, parameters["model"], prompt_for_narrator, messages)
        # prompt_for_narrator = reply2
        # parameters["prompt_for_narrator"] = reply2

        # rewrite_memory_instruction = parameters["rewrite_memory_instruction"]

        dialog_history = print_history()

        prompt_for_narrator_rewrite_memory = parameters["narrator"]["prompt_for_rewrite_memory"]
        rewrite_memory_instruction = eval(
            f'f"""{prompt_for_narrator_rewrite_memory}"""')
        messages.append({
            "role":
            "user",
            "content":
            f"{rewrite_memory_instruction}: {prompt_for_character}"
        })

        new_character_prompt = await generate_reply(
            parameters["narrator"]["llm_settings"], client_narrator,
            prompt_for_narrator, messages)
        old_character_prompt = parameters["character"]["prompt"]
        parameters["character"]["prompt"] = new_character_prompt

        prompt_update = f"Previous Instruction:\n {old_character_prompt}\n\nRevised Instruction:\n {new_character_prompt}"

        dialogue_narrator.append({
            "role":
            "user",
            "content":
            f"{rewrite_memory_instruction}: {prompt_for_character}"
        })
        dialogue_narrator.append({"role": "assistant", "content": prompt_update})

        return prompt_update

    except Exception as e:
        print(e)
        return f"Couldn't update intructions! {e}"


async def revise_memory():
    global bracket_counter, parameters

    if bracket_counter % parameters["rewrite_memory_schedule"] != 0:
        return None

    reply = await update_character_instructions()

    # if reply != None:
    #     await send_colored_embed(channel_last, "Instruction Set", reply, [], discord.Color.red())

    return reply


async def generate_autobiography_from_dialogue():
    write_bio_instruction = ''
    try:
        prompt_for_narrator = parameters["narrator"]["prompt"]
        prompt_for_narrator = eval(f'f"""{prompt_for_narrator}"""')

        messages = []

        dialog_history = build_history()

        prompt_for_narrator_bio = parameters["narrator"]["prompt_for_bio"]
        write_bio_instruction = eval(f'f"""{prompt_for_narrator_bio}"""')
        messages.append({"role": "user", "content": write_bio_instruction})

        reply = await generate_reply(parameters["narrator"]["llm_settings"],
                                    client_narrator, prompt_for_narrator,
                                    messages)
        
        return reply

    except Exception as e:
        print(
            f"Error summarising biography: {e}, with {write_bio_instruction}")
        return "Error!"


async def generate_script_from_dialogue():
    global parameters

    try:

        history = build_history()

        prompt_for_character = parameters["character"]["prompt"]
        name_for_character = parameters["character"]["name"]
        builder = f'system: {prompt_for_character}\n\n'
        for message in history:
            role = message['role']
            username = role
            if 'name' in message:
                username = message['name']
            elif role == 'assistant':
                username = name_for_character
            content = message['content']
            builder += f"{username}: {content}\n\n"

        return builder
    except Exception as e:
        print(e)
        return f"No history, due to {e}"


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
                     prompt_for_character: str = None,
                     model_id: int = None,
                     temperature: float = None,
                     top_p: float = None,
                     frequency_penalty: float = None,
                     presence_penalty: float = None,
                     summary_level: int = None,
                     max_size_dialog: int = None,
                     channel_reset: str = None):
    global dialogue_history
    if prompt_for_character is not None:
        parameters["character"]["prompt"] = prompt_for_character
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
async def clear_chat(interaction: discord.Interaction):
    reload_settings()
    await interaction.response.send_message(f"Settings reloaded.")


@tree.command(name="generate_biography",
              description="Generates a biography",
              guild=discord.Object(guild_id))
async def generate_biography(interaction: discord.Interaction):
    reply = await generate_autobiography_from_dialogue()
    await interaction.response.send_message("Generating biography now...")
    await send_colored_embed(channel_last, "Autobiography", reply, [],
                             discord.Color.green())


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
def generate_unique_filename(base_path, extension='md'):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    return os.path.join(base_path, f'dialogue_{timestamp}.{extension}')

@tree.command(name="generate_script",
              description="Generates the conversation as a script",
              guild=discord.Object(guild_id))
async def generate_script(interaction: discord.Interaction):
    reply = await generate_script_from_dialogue()
    
    # await interaction.response.send_message("Generating script now...")

    # Write content to file
    file_link = generate_unique_filename('./')
    file = write_markdown_to_file(reply, file_link)
    file_name = os.path.basename(file_link)
    await interaction.response.send_message(file=discord.File(file, file_name))
    await send_colored_embed(channel_last, "The Dialogue so far...", reply, [],
                             discord.Color.green())


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
async def generate_image(prompt):
    try:
        audience_name = parameters["audience"]["name"]
        llm_settings = parameters["audience"]["llm_settings"]
        model = llm_settings["model"]

        messages = []
        messages.append({"role": "system", "content": "You are a creative assistant to an interpreter, and determine whether anything in the respondent's discourse is worth drawing."})
        messages.append({
            "role": "user",
            "name": audience_name,
            "content":f"If the following text contains an image to be drawn, create a prompt. Otherwise respond with an empty string.:\n\n{prompt}"
        })
        response = await client_audience.chat.completions.create(
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
        
        response = await client_audience.images.generate(
            model="dall-e-3",
            prompt=reply,
            size="1024x1024",
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



def build_history():
    # GPT limit, minus max response token size
    token_limit = 16097 - parameters["character"]["llm_settings"]["max_tokens"]

    # Tokenize the string to get the number of tokens
    tokens_len = 0

    # Iterate through the list in reverse order
    dialogue_history_limited = []
    for item in reversed(dialogue_history):
        tokens_item = tokenizer(str(item) + "\n", return_tensors='pt')
        tokens_item_len = tokens_item.input_ids.size(1)
        if tokens_len + tokens_item_len < token_limit:
            tokens_len = tokens_len + tokens_item_len
            dialogue_history_limited.insert(0, item)
        else:
            break

    # Construct the dialog history
    return dialogue_history_limited


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
        if i == 0:
            for name, value, inline in fields:
                embed.add_field(name=name, value=value, inline=inline)

        # Send the embed
        await channel.send(embed=embed)
    # # Truncate the description if it's too long
    # description = description[:4096]

    # # Create an embed object
    # embed = discord.Embed(title=title, description=description, color=color)

    # # Add fields to the embed
    # for name, value, inline in fields:
    #     embed.add_field(name=name, value=value, inline=inline)

    # # Send the embed
    # await channel.send(embed=embed)

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
    global prompt_for_character
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

    prompt_for_character = parameters["character"]["prompt"]
    prompt_for_narrator = parameters["narrator"]["prompt"]

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
            bracket_counter = bracket_counter + 1

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

            dialogue_audience.append({
                "role": "assistant",
                "content": prompt
            })
            dialogue_audience.append({
                "role": "user",
                "name": parameters["character"]["name"],
                "content": content
            })
        else:
            await message.channel.send("Sorry, couldn't reply")

        reply_bio = await write_bio()
        if reply_bio != None:
            await send_colored_embed(channel_last, "Autobiography", reply_bio,
                                     [], discord.Color.green())
            # chunks = split_into_chunks(reply, max_length=1600)
            # for chunk in chunks:
            #     await channel_last.send(chunk)

        reply_revision = await revise_memory()
        if reply_revision != None:
            await send_colored_embed(channel_last, "Memory Revision",
                                     reply_revision, [], discord.Color.red())
            # chunks = split_into_chunks(reply, max_length=1600)
            # for chunk in chunks:
            #     await channel_last.send(chunk)

# Check if the following includes a drawing to render
async def check_for_images(result, content):
    if not parameters["character"]["generate_images"]:
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


async def periodic_task():
    global dialogue_history, dialogue_audience
    global channel_last, bracket_counter, director_prompts
    global sleep_counter, formatted_time
    global make_a_promise_likelihood, fulfil_a_promise_likelihood

    channel_last_id = int(channel_ids[0])
    channel_last = (client.get_channel(channel_last_id)
                    or await client.fetch_channel(channel_last_id))

    turn_limit = parameters["audience"]["turn_limit"]
    director_intervention = parameters["director"]["intervention"]
    write_bio_schedule = parameters["write_bio_schedule"]

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

        if channel_last:
            message = ''

            # Load prompts from the JSON file
            prompt = None
            username = None

            # prompt = get_director_script_prompt(director_prompts, bracket_counter)
            if bracket_counter % director_intervention == 0:
                prompt, username = await get_director_prompt(bracket_counter)

            if prompt is None:
                prompt, username = await get_audience_prompt(bracket_counter)
                
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

                await channel_last.send(prompt_to_display)
                result = ''
                try:
                    result = await chat_prompt(prompt, parameters)
                except Exception as e:
                    print(e)
                    result = 'So sorry, dear User! ChatGPT is down.'

                character_name = parameters["character"]["name"]
                result_to_display = f"*{character_name}*: {result}"
                chunks = split_into_chunks(result_to_display, max_length=1600)
                for chunk in chunks:
                    await channel_last.send(chunk)
                # await channel_last.send(result_to_display)

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

                    dialogue_audience.append({
                        "role": "assistant",
                        "content": prompt
                    })
                    dialogue_audience.append({
                        "role": "user",
                        "name": parameters["character"]["name"],
                        "content": content
                    })
                    if make_a_promise:
                        promises.append(result)



                reply = await write_bio()
                if reply != None:
                    await send_colored_embed(channel_last, "Autobiography",
                                             reply, [], discord.Color.green())
                    # chunks = split_into_chunks(reply, max_length=1600)
                    # for chunk in chunks:
                    #     await channel_last.send(chunk)

                reply = await revise_memory()
                if reply != None:
                    await send_colored_embed(channel_last, "Memory Revision",
                                             reply, [], discord.Color.red())
                    # chunks = split_into_chunks(reply, max_length=1600)
                    # for chunk in chunks:
                    #     await channel_last.send(chunk)

        else:
            print("No channel_last_id")

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
async def get_director_prompt(step):
    try:
        prompt_for_director = parameters["director"]["prompt"]
        director_name = parameters["director"]["name"]
        character_name = parameters["character"]["name"]

        messages = []

        dialog_history = print_history()

        prompt_for_director_instruction = parameters["director"]["prompt_for_director_instruction"]
        prompt_for_director_instruction = eval(
            f'f"""{prompt_for_director_instruction}"""')
        
        messages.append({
            "role":
            "user",
            "name": character_name,
            "content":
            f"{prompt_for_director_instruction}"
        })

        reply = await generate_reply(parameters["director"]["llm_settings"],
                                    client_director, prompt_for_director,
                                    messages)

        return reply, director_name
    
    except Exception as e:
        print(e)
        return f"Error in generating director message! {e}"

# Function to get the prompt for the current step
async def get_audience_prompt(step):
    global dialogue_audience

    try:
        prompt_for_audience = parameters["audience"]["prompt"]
        audience_name = parameters["audience"]["name"]
        character_name = parameters["character"]["name"]

        dialog_history = print_history()
        
        last_audience_message = None
        last_message_content = ''
        if len(dialogue_audience) > 0:
            last_audience_message = dialogue_audience.pop()
            last_message_content_container = last_audience_message["content"]
            if isinstance(last_message_content_container, list):
                for message in last_message_content_container:
                    if message["type"] == "text":
                        last_message_content = message["text"]

        prompt_for_audience_message = parameters["audience"]["prompt_for_audience_message"]
        prompt_for_audience_message = eval(
            f'f"""{prompt_for_audience_message}"""')
        
        if last_audience_message == None:
            last_audience_message = {
                "role": "user",
                "name": character_name,
                "content": [
                    {"type": "text", 
                     "text": prompt_for_audience_message}
                ]
            }
        else:
            for message in last_audience_message["content"]:
                if message["type"] == "text":
                    message["text"] = prompt_for_audience_message

        dialogue_audience.append(last_audience_message)

        reply = await generate_reply(parameters["audience"]["llm_settings"],
                                    client_audience, prompt_for_audience,
                                    dialogue_audience)

        return reply, audience_name
    
    except Exception as e:
        print(e)
        return f"Error in generating audience message! {e}", audience_name

def generate_client(model):
    client = None
    if 'claude-3' in model:
        client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY'))
    elif 'gpt' in model:
        # client_character = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
        client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        # client_character = Groq(api_key=os.getenv('GROQ_API_KEY'))
        client = AsyncGroq(api_key=os.getenv('GROQ_API_KEY'))
    return client

def reload_settings():
    global client_character, client_narrator, client_audience, client_director
    global parameters
    global guild_id, channel_ids
    global sleep_counter, make_a_promise_likelihood, fulfil_a_promise_likelihood
    global director_prompts
    global subject
    global bracket_counter

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

    parameters["character"]["prompt"] = load_markdown_content(
        parameters["character"]["prompt"])
    parameters["narrator"]["prompt"] = load_markdown_content(
        parameters["narrator"]["prompt"])
    parameters["narrator"]["prompt_for_bio"] = load_markdown_content(
        parameters["narrator"]['prompt_for_bio'])
    parameters["narrator"]["prompt_for_rewrite_memory"] = load_markdown_content(
        parameters["narrator"]['prompt_for_rewrite_memory'])
    parameters["audience"]["prompt"] = load_markdown_content(
        parameters["audience"]["prompt"])
    parameters["audience"]["prompt_for_audience_message"] = load_markdown_content(
        parameters["audience"]['prompt_for_audience_message'])
    parameters["director"]["prompt"] = load_markdown_content(parameters["director"]["prompt"])
    parameters["director"]["prompt_for_director_instruction"] = load_markdown_content(parameters["director"]["prompt_for_director_instruction"])

    # Load prompts from the JSON file
    director_prompts = load_prompts(parameters["director"]['director_script'])

    client_character = generate_client(parameters["character"]["llm_settings"]["model"])
    client_narrator = generate_client(parameters["narrator"]["llm_settings"]["model"])
    client_audience = generate_client(parameters["audience"]["llm_settings"]["model"])
    client_director = generate_client(parameters["director"]["llm_settings"]["model"])

    bracket_counter = 1


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
