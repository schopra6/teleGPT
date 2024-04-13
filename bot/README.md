The Telegram Auto-Reply Bot, powered by TeleGPT, enables automatic responses in private Telegram chats. Leveraging the TeleGPT model, it generates replies to messages, enhancing the chat experience.

**Key Features:**
- Automatically responds to messages in private Telegram chats.
- Records chat history for each conversation in a text file, facilitating contextually relevant replies.
- Offers a user-friendly command-line interface for bot setup and control.

**Setup:**
To use the bot, create a Telegram API app by logging into your Telegram account via the Telegram app and accessing API Development Tools. Fill out the form to obtain your `API_ID` and `API_HASH`. Store these credentials in a `.env` file in the repository's root directory.

```bash
API_ID=your_api_id
API_HASH=your_api_hash
```

**Usage:**
The bot initiates upon startup and responds to incoming private messages using the TeleGPT model. To adjust bot configurations such as GPT parameters or optimizer settings, refer to the `arg_parser()` function in the [argparser.py](../Telegpt/argparser.py) script for detailed instructions.