from cmd import PROMPT
import os
from dotenv import load_dotenv

import requests
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
client = discord.Client(intents=intents)
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

client_character = None
client_narrator = None

bot_name=""
stop_sequences = ["\n", "."]
channel_ids = []

max_size_dialog = 10000
bot_running_dialog = []
user_refs = []
prompt_suffix = """{0}: {1}
{2}:"""

director_prompts = {}

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
        message = client.messages.create(
            model=model,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            system=system_prompt,
            messages=messages
        )
        reply = message.content[0].text
    else:
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


async def chat_prompt(prompt, parameters):
    try:
        prompt_for_character = parameters["prompt_for_character"]
        prompt_for_character = eval(f'f"""{prompt_for_character}"""')

        messages = []
        history = build_history()
        for item in history:
            messages.append(item)
        messages.append({"role": "user", "content": prompt})

        return await generate_reply(parameters["gpt_settings_character"], client_character, prompt_for_character, messages)
    except Exception as e:
        print(f"Error in chat prompt: {e}")
        return {"role": "system", "content": "Looks like ChatGPT is down!"}

async def write_bio():
    global bot_running_dialog, parameters

    if bracket_counter % parameters["write_bio"] != 0:
        return None
    
    autobiography = await summarise_autobiography()

    return autobiography

async def update_instructions():
    global bot_running_dialog, parameters

    try:
        prompt_for_character = parameters["prompt_for_character"]
        prompt_for_narrator = parameters["prompt_for_narrator"]

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
        # for message in bot_running_dialog[-messages_to_rewrite:]:
        #     if message['role'] == 'assistant':
        #         content = message['content']
        #         messages.append({'role': 'user', 'content': f"Review and revise the following: '{content}'. Include nothing but the revised statement."})
        #         reply = await generate_reply(client_character, parameters, parameters, prompt_for_narrator, messages)
        #         # Remove first system message
        #         del messages[0]
        #         messages.append({'role': 'assistant', 'content': reply})
        #         message['content'] = reply
        
        # print(bot_running_dialog[-1:]['content'])
        # reply = await generate_reply(client_character, parameters, parameters["model"], prompt_for_narrator, messages)
        # messages.append({"role": "assistant", "content": reply})
        # messages.append({"role": "user", "content": f"Analyse this text: ."})

        # reply2 = await generate_reply(client_character, parameters, parameters["model"], prompt_for_narrator, messages)
        # prompt_for_narrator = reply2
        # parameters["prompt_for_narrator"] = reply2

        rewrite_memory_instruction = parameters["rewrite_memory_instruction"]
        messages.append({"role": "user", "content": f"{rewrite_memory_instruction}: {prompt_for_character}"})

        # print(messages)
        new_character_prompt = await generate_reply(parameters["gpt_settings_narrator"], client_narrator, prompt_for_narrator, messages)
        old_character_prompt = parameters["prompt_for_character"]
        parameters["prompt_for_character"] = new_character_prompt

        prompt_update = f"Previous Instruction:\n {old_character_prompt}\n\nRevised Instruction:\n {new_character_prompt}"

        return prompt_update

    except Exception as e:
        print(e)
        return "Couldn't update isntructions!"

async def revise_memory():
    global bracket_counter, parameters

    if bracket_counter % parameters["rewrite_memory"] != 0:
        return None

    reply = await update_instructions()

    if reply != None:
        await send_colored_embed(channel_last, "Instruction Set", reply, [], discord.Color.red())

    return reply




async def summarise_autobiography():
    prompt = ''
    try:
        prompt_for_character = parameters["prompt_for_character"]
        prompt_for_character = eval(f'f"""{prompt_for_character}"""')

        prompt_for_narrator = parameters["prompt_for_narrator"]
        prompt_for_narrator = eval(f'f"""{prompt_for_narrator}"""')

        messages = []
        
        history = build_history()
        builder = ''
        for message in history:
            builder += message['content'] + "\n"

        prompt = f"Convert this into an biographical summary of the character's life to date: \n\n\"{builder}\". Express the narrative in the past tense. \n\n"
        messages.append({"role": "user", "content": prompt})

        return await generate_reply(parameters["gpt_settings_narrator"], client_narrator, prompt_for_narrator, messages)

    except Exception as e:
        print(f"Error summarising autobiography: {e}, with {prompt}")
        return "Error!"
    

async def dump_autobiography():
    try:

        history = build_history()

        builder = ''
        for message in history:
            role = message['role']
            content = message['content']
            builder += f"{role}: {content}\n\n" 

        return builder
    except Exception as e:
        return "No history"

@client.event
async def on_ready():
    await tree.sync(guild=discord.Object(guild_id))
    client.loop.create_task(periodic_task())
    print(f'We have logged in as {client.user} to {len(client.guilds)} guilds.')

    

@tree.command(name = "get", description = "Show parameters", guild=discord.Object(guild_id)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def get_params(interaction: discord.Interaction):
    response = "Current parameters:\n"
    for key, value in parameters.items():
        response += f"{key}: {value}\n"
    response += f"bot_running_dialog size: {len(bot_running_dialog)}\n"
    response += f"stop_sequences: {stop_sequences}\n"
    history = build_history()
    response += f"history: {history}\n"

    chunks = split_into_chunks(response, max_length=1800)

    # Send the first chunk as the initial response
    await interaction.response.send_message(chunks[0])

    # Send the remaining chunks as follow-up messages
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk)


@tree.command(name = "set", description = "Set parameters", guild=discord.Object(guild_id)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
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
    global bot_running_dialog
    if prompt_for_character is not None:
        parameters["prompt_for_character"] = prompt_for_character
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
        bot_running_dialog = bot_running_dialog[:max_size_dialog]
        parameters["max_size_dialog"] = max_size_dialog
    if channel_reset is not None:
        parameters["channel_reset"] = channel_reset
    await interaction.response.send_message("Parameters updated.")


@tree.command(name = "reset", description = "Reset memory", guild=discord.Object(guild_id)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def reset(interaction: discord.Interaction):
    bot_running_dialog.clear()
    await interaction.response.send_message("Summary and history have been reset!")


@tree.command(name = "clear_chat", description = "Clear Chat history", guild=discord.Object(guild_id)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def clear_chat(interaction: discord.Interaction, limit: int = 0):
    channel = client.get_channel(interaction.channel_id)
    await channel.purge(limit=limit)
    await interaction.response.send_message(f"{limit} responses cleared from chat.")


def build_history():
    # GPT limit, minus max response token size
    token_limit = 16097 - parameters["gpt_settings_character"]["max_tokens"]

    # Tokenize the string to get the number of tokens
    tokens_len = 0
    
    # Iterate through the list in reverse order
    bot_running_dialog_limited = []
    for item in reversed(bot_running_dialog):
        tokens_item = tokenizer(str(item) + "\n", return_tensors='pt')
        tokens_item_len = tokens_item.input_ids.size(1)
        if tokens_len + tokens_item_len < token_limit:
            tokens_len = tokens_len + tokens_item_len
            bot_running_dialog_limited.insert(0, item)
        else:
            break
    # Construct the dialog history
    return bot_running_dialog_limited



# Method to create and send a colored embed
async def send_colored_embed(channel, title, description, fields, color=discord.Color.blue()):
    # Create an embed object
    embed = discord.Embed(
        title=title,
        description=description,
        color=color
    )
    
    # Add fields to the embed
    for name, value, inline in fields:
        embed.add_field(name=name, value=value, inline=inline)
    
    # Send the embed
    await channel.send(embed=embed)

@client.event
async def on_message(message):
    global bot_running_dialog
    global prompt_for_character
    global user_refs
    global channel_last
    global bracket_counter
    global parameters

    max_size_dialog = parameters["max_size_dialog"]

    if message.author == client.user:
        return
    
    if message.channel.id not in channel_ids:
        print(f"Channel {message.channel.id} not found in these channel_ids: {channel_ids}")
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

    prompt_for_character = parameters["prompt_for_character"]
    prompt_for_narrator = parameters["prompt_for_narrator"]

    if not BOT_ON:
        return

    # Simulate typing for 3 seconds
    async with channel.typing():
        prompt = data
        member_stops = [x + ":" for x in user_refs]

        # For debugging
        reply = "[NO-GPT RESPONSE]"
        try:

            reply = await revise_memory()
            if reply != None:
                await send_colored_embed(channel, "Memory Revision", reply, [], discord.Color.green())
                # chunks = split_into_chunks(reply, max_length=1600)
                # await send_chunks(message, chunks)

            if prompt.startswith("SUMMARISE"):
                reply = await summarise_autobiography()
            elif prompt.startswith("DUMP"):
                reply = await dump_autobiography()
            elif prompt.startswith("BEN"):
                reply = await dump_autobiography()
            else:
                prompt = f"Frame #{bracket_counter}. User is {author_name}. User says: {prompt}"
                reply = await chat_prompt(prompt, parameters)
        except Exception as e:
            print(e)
            reply = 'So sorry, dear User! ChatGPT is down.'
        
        if reply != "":
            if len(bot_running_dialog) >= max_size_dialog:
                bot_running_dialog.pop(0)
                bot_running_dialog.pop(0)
            bot_running_dialog.append({"role": "user", "content": prompt})
            bot_running_dialog.append({"role": "assistant", "content": reply})

            bracket_counter = bracket_counter + 1

            chunks = split_into_chunks(reply, max_length=1600)
            await send_chunks(message, chunks)

        else:
            await message.channel.send("Sorry, couldn't reply")


async def periodic_task():
    global bot_running_dialog
    global channel_last, bracket_counter, director_prompts
    global sleep_counter, formatted_time
    global make_a_promise_likelihood,  fulfil_a_promise_likelihood
    
    channel_last_id = int(channel_ids[0])
    channel_last = (client.get_channel(channel_last_id) or await client.fetch_channel(channel_last_id))

    while True:
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
            prompt = get_prompt(director_prompts, bracket_counter)

            if prompt is not None:

                reply = await write_bio()
                if reply != None:
                    await send_colored_embed(channel_last, "Autobiography", reply, [], discord.Color.green())
                    # chunks = split_into_chunks(reply, max_length=1600)
                    # for chunk in chunks:
                    #     await channel_last.send(chunk)

                reply = await revise_memory()
                if reply != None:
                    await send_colored_embed(channel_last, "Memory Revision", reply, [], discord.Color.green())
                    # chunks = split_into_chunks(reply, max_length=1600)
                    # for chunk in chunks:
                    #     await channel_last.send(chunk)

                prompt =  f"Step {bracket_counter}: {prompt}"
                
                # if make_a_promise:
                #     prompt = "Make promise: Generate a very brief note-to-self that will remind you to reflect on events to date at a future point in time."
                # elif fulfil_a_promise and len(promises) > 0:
                #     random_index = random.randint(0, len(promises) - 1)
                #     prompt = promises[random_index]
                #     prompt = "Fulfil promise: " + prompt
                #     promises.remove(promises[random_index])
                # else:
                #     prompt = elapsed_time_formatted

                await channel_last.send(prompt)
                result = ''
                try:
                    result = await chat_prompt(prompt, parameters)
                except Exception as e:
                    print(e)
                    result = 'So sorry, dear User! ChatGPT is down.'
                
                if result != "":
                    if len(bot_running_dialog) >= max_size_dialog:
                        bot_running_dialog.pop(0)
                        bot_running_dialog.pop(0)
                    bot_running_dialog.append({"role": "user", "content": prompt})
                    bot_running_dialog.append({"role": "assistant", "content": result})
                    bracket_counter = bracket_counter + 1
                    if make_a_promise:
                        promises.append(result)
                await channel_last.send(result)
        else:
            print("No channel_last_id")

        await asyncio.sleep(sleep_counter)  # sleep for 20 seconds


# Function to read Markdown content from a file
def load_markdown_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as md_file:
        return md_file.read()


# Function to load JSON prompts from a file
def load_prompts(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)['prompts']

# Function to get the prompt for the current step
def get_prompt(prompts, step):
    for prompt in prompts:
        if prompt['step'] == step:
            return prompt['prompt']
    return None



def main():
    global client_character, client_narrator, params_gpt_character, params_gpt_narrator
    global parameters
    global bot_name
    global guild_id, channel_ids
    global sleep_counter, make_a_promise_likelihood, fulfil_a_promise_likelihood
    global director_prompts

    # Load the settings file
    try:
        subject = f'settings_{sys.argv[2]}.json'
        with open(subject, "r") as read_file:
            parameters = json.loads(read_file.read())
    except Exception as e:
        print(f"Error loading settings file: {e}")
        return
    
    # Load the Discord token from the environment variable
    discord_token_env_var = parameters['discord_token_env_var']
    discord_token = os.environ.get(discord_token_env_var)


    bot_name = parameters['name']
    channel_ids = parameters['channel_ids']

    try:
        sleep_counter = int(parameters['sleep_counter'])
        make_a_promise_likelihood = float(parameters['make_a_promise_likelihood'])
        fulfil_a_promise_likelihood = float(parameters['fulfil_a_promise_likelihood'])
    except Exception as e:
        print(e)
        pass

    parameters["prompt_for_character"] = load_markdown_content(parameters['prompt_for_character'])
    parameters["prompt_for_narrator"] = load_markdown_content(parameters['prompt_for_narrator'])
    
    # Load prompts from the JSON file
    director_prompts = load_prompts(parameters['director_script'])

    # params_gpt["summarise_level"] = parameters['summarise_level']
    # params_gpt["max_size_dialog"] = parameters['max_size_dialog']
    # params_gpt["rewrite_memory"] = parameters['rewrite_memory']
    # params_gpt["rewrite_memory_instruction"] = parameters['rewrite_memory_instruction']
    # params_gpt["write_bio"] = parameters['write_bio']
    # params_gpt["write_bio_instruction"] = parameters['write_bio_instruction']

    # params_gpt["character"] = parameters['gpt_settings_character']
    # params_gpt["narrator"] = parameters['gpt_settings_narrator']
    # params_gpt["model"] = gpt_settings['model']
    # params_gpt["temperature"] = gpt_settings['temperature']
    # params_gpt["max_tokens"] = gpt_settings['max_tokens']
    # params_gpt["top_p"] = gpt_settings['top_p']
    # params_gpt["frequency_penalty"] = gpt_settings['frequency_penalty']
    # params_gpt["presence_penalty"] = gpt_settings['presence_penalty']
    
    model_client = parameters["gpt_settings_character"]["model"]
    if model_client == 'claude-3':
        client_character = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    elif 'gpt' in model_client:
        # client_character = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
        client_character = AsyncOpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
    else:
        # client_character = Groq(api_key=os.getenv('GROQ_API_KEY'))    
        client_character = AsyncGroq(api_key=os.getenv('GROQ_API_KEY'))    
    
    model_narrative = parameters["gpt_settings_narrator"]["model"]
    if model_narrative == 'claude-3':
        client_narrator = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    elif 'gpt' in model_narrative:
        # client_narrator = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
        client_narrator = AsyncOpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
    else:
        # client_narrator = Groq(api_key=os.getenv('GROQ_API_KEY'))    
        client_narrator = AsyncGroq(api_key=os.getenv('GROQ_API_KEY'))    
    
    client.run(discord_token)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 main.py --subject <name-of-subject>")
        sys.exit(1)

    main()
    