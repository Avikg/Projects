# Chat Server with GPT-2 Chatbot README

## Overview
This README file provides instructions for compiling and running the Chat Server application with integrated GPT-2 Chatbot functionality. The server handles multiple clients simultaneously, allowing for message exchange between clients and interactions with a GPT-2 powered chatbot for answering queries.
This Chat Application allows multiple clients to connect to a server to send messages to each other in real-time. It supports basic chat functionalities, including sending messages to specific clients, listing active clients, and managing chat histories (viewing, deleting individual or all histories)

## Dependencies

- GCC (GNU Compiler Collection) for compiling C programs.
- Python 3.x with the following packages installed:
  - `transformers`
  - `torch`
- Hugging Face's `transformers` library is used to interact with the GPT-2 model.

## Installation

### 1.1 Install Python Dependencies

Ensure Python 3.x is installed on your system. You can install the required Python packages using `pip`. It's recommended to use a virtual environment.

```bash
pip install transformers torch
```

### 1.1 Install the libuuid Library
```bash
sudo apt-get update
sudo apt-get install uuid-dev
```

### 2. Compile the Server and Client

Navigate to the directory containing the server and client source code, then use `gcc` to compile:

```bash
gcc chat_server.c -o chat_server -lpthread -luuid
gcc chat_client.c -o chat_client
```

This will generate two executable files: `chat_server` and `chat_client`.

## Running the Application

### Start the Chat Server

First, start the server by running the `chat_server` executable. By default, the server listens on port 8080. Ensure the port is open and not being used by another application.

```bash
./chat_server
```

The server must be running before clients can connect.

### Connect with a Chat Client

Open a new terminal window to start a chat client. Run the `chat_client` executable, and when prompted, enter the server IP address and port number.

```bash
./chat_client
```
When prompted, enter the server IP address (e.g., `127.0.0.1` for localhost) and the server port number (e.g., `8080`).

After connecting, enter your desired username when prompted. You can now start sending messages, interact with the chatbot, and use other available commands.

## Features and Commands

### Chatting

#### Supported Commands

- `/send <client_id> <message>`: Send a message to a specific client by their ID.
- `/active`: List all active clients connected to the server.
- `/chatbot_login`: Login to Chatbot Server for FAQs. FAQs are loaded from 'FAQs.txt'.
- `/chatbot_logout`: Logout from the Chatbot Server.
- `/chatbot_v2 login`: Login to the GPT-2 Enabled Bot.
- `/chatbot_v2 logout`: Logout from GPT-2 Enabled Bot.
- `/history <client_id>`: View the chat history with a specific client by their ID.
- `/history_delete <client_id>`: Delete the chat history with a specific client by their ID.
- `/delete_all`: Delete all chat history associated with the client.
- `/logout`: Disconnect from the server.

- **Send Messages**: Type your message and press enter to send it to all connected clients.
- **Private Messages**: Use `/send <client_uid> <message>` to send a private message to a specific client.

### Chatbot Interaction

- **Activate Chatbot**: Use `/chatbot_v2 login` to activate the GPT-2 chatbot.
- **Deactivate Chatbot**: Use `/chatbot_v2 logout` to deactivate the chatbot.

### Chat History

- **View Chat History**: Use `/history <client_uid>` to view chat history with a specific client.
- **Delete Chat History**: Use `/history_delete <client_uid>` to delete chat history with a specific client.
- **Delete All Chat Histories**: Use `/delete_all` to delete all your chat histories.

### Miscellaneous

- **View Active Clients**: Use `/active` to view a list of active clients.
- **Logout**: Use `/logout` to disconnect from the server.

## Additional Features and Improvements

- **GPT-2 Chatbot Integration**: Implemented integration with GPT-2 for answering queries when the chatbot feature is active. Implemented in 'gpt2.py' module.
- **Dynamic Chat History**: Added functionality for storing, retrieving, and deleting chat history between clients.
- **Concurrent Client Handling**: Improved server architecture to handle multiple clients concurrently without blocking.
- **Security Enhancements**: Included basic input validation to prevent command injection through chat messages.

## Troubleshooting

- Ensure all dependencies are packages.
- Verify the server IP address and port number are correctly entered in tcorrectly installed, especially the Python he client application.
- If the server or client fails to compile, check for missing dependencies such as the `uuid` library or issues with the GCC installation.
