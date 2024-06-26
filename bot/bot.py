
# Python standard library imports
import os
import time
from datetime import datetime
import asyncio


# Local application imports
from base_modules.sample import GramSampler
from base_modules.config import Config as cfg
from base_modules.argparse import arg_parser

# Third party imports
from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.custom import Button
from telethon import events


# Initialize a GramSampler object for the entire application.
sampler = GramSampler(**vars(arg_parser()))
samplers = {}

async def auto_reply(client):
    """Handles new messages and responds to them.

    This function is a coroutine that takes a TelegramClient object, listens for
    new messages, and responds to them using the GramSampler object associated
    with the chat.

    Args:
        client (TelegramClient): The Telegram client.

    """

    first_message_flags = {}  # A dictionary to keep track of whether it's the first message from each chat.

    @client.on(events.NewMessage(incoming=True))  # Decorator that registers a new event handler.
    async def handle_new_message(event):
        """Handles a new incoming message.

        This function is a coroutine that gets triggered whenever a new message
        event occurs. It responds to the message using the GramSampler associated
        with the chat.

        Args:
            event (NewMessage.Event): The new message event.

        """

        # Only auto-reply to incoming chats.
        if not event.out: 

            # Get sender_id according to the type of chat (private or group)
            sender_id = event.message.peer_id.user_id if event.is_private else event.message.from_id.user_id

            # Retrieve sender and bot info.
            me = await client.get_me()  # get the current user information
            sender_entity = await client.get_entity(sender_id)  # get the entity of the sender

            # Extract the sender's name.
            sender_name = f"{sender_entity.first_name} {sender_entity.last_name or ''}".strip()

            # If the sender doesn't have a name, use their username instead.
            if not sender_name:
                sender_name = sender_entity.username

            # Define the path for the chat file.
            sender_file_path = f"chat_files/{sender_name}.txt"

            # If the directory doesn't exist, create it.
            if not os.path.exists(os.path.dirname(sender_file_path)):
                os.makedirs(os.path.dirname(sender_file_path))

            # If there's no sampler for this conversation, create one.
            if sender_id not in samplers:
                # If the chat file doesn't exist, create it and write the first message.
                if not os.path.isfile(sender_file_path):
                    with open(sender_file_path, "w", encoding="utf-8") as f:
                        f.write(f"{sender_name}: {event.message.text}\n")

                # Mark this as the first message from this chat.
                first_message_flags[sender_id] = True  

            # Get the GramSampler for this chat.
            reply_text = sampler.generate(file=sender_file_path,
                                          temperature=cfg.sampling.temperature,
                                          top_k=cfg.sampling.top_k)
            
            # Send the reply.
            await event.respond(reply_text)

            # Append the last received message and its automatic reply to the sender's chat file.
            with open(sender_file_path, "a", encoding="utf-8") as f:
                # Write the sender's message.
                f.write(f"{sender_name}: {event.message.text}\n")

                # Write the bot's reply.
                f.write(f"{cfg.sampling.user}: {reply_text}\n")

            # Set the flag for first message to False.
            first_message_flags[sender_id] = False

    print(time.asctime(), '-', 'Auto-replying...')
    await client.run_until_disconnected()
    print(time.asctime(), '-', 'Stopped!')

if __name__ == '__main__':
    # Load the environment variables from the .env file.
    load_dotenv()

    # Retrieve the API ID and API Hash from the environment variables.
    api_id = os.getenv('API_ID')
    api_hash = os.getenv('API_HASH')

    # Initialize the Telegram client.
    client = TelegramClient('anon', api_id, api_hash) 

    # Start the client before running the bot.
    client.start()

    # Run the auto-reply coroutine.
    loop = asyncio.get_event_loop()
    loop.run_until_complete(auto_reply(client))

    
    
    
    
    


    
    
    
    
    
