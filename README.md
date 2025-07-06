# Data Omni Chat 

Welcome to Data Omni Chat! This application, built with FastAPI, allows you to engage in private conversations with your CSV files using the power of Large Language Models (LLMs).Leverage powerfull large language models without them seeing your data. Upload your data and start asking questions to gain insights quickly and intuitively. Generate Powerpoint slides from your conversatio to quickly shar ein presentations

[![Watch the video](https://img.youtube.com/vi/Dpa3oYEAJw4/maxresdefault.jpg)](https://youtu.be/Dpa3oYEAJw4)

## Features

* **Interactive Chat Interface:** A user-friendly chat interface to interact with your data.
* **Private Conversations:** Your data remains private and are processed locally in browser.
* **Support for CSV Files:** Directly upload and start chatting with your CSV data.
* **Powerpoint Generation:** Generate Powerpoint slides from your conversation.


Project Structure
The repository is organized into a backend (app) and a frontend (frontend) directory. The compiled frontend code is served from the root dist folder.

```bash
/data_omni_chat
├── app/                  # FastAPI application, routers, logic
├── dist/                 # Compiled React code (auto-generated)
├── frontend/             # React source code (where you edit)
│   ├── src/
│   ├── vite.config.ts
│   └── package.json
├── main.py               # Main FastAPI server entrypoint
├── requirements.txt      # Python dependencies
└── README.md
```
Prerequisites
Before you begin, ensure you have the following installed:

Python 3.8+

Node.js v22.x or another supported LTS version. We recommend using nvm to manage Node versions.

pip for Python package installation.



## 🛠️ Installation

To get the application running on your local machine, please follow these steps.

```bash

1. Clone the Repository

First, clone the repository to your local machine using the following command:


git clone https://github.com/the-data-omni/data_omni_chat
cd data_omni_chat

2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

On macOS and Linux:

python3 -m venv venv
source venv/bin/activate

On Windows:
python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies
Install all the required Python packages from the requirements.txt file:

pip install -r requirements.txt

4. Setup Frontend

# Navigate to the frontend directory
cd frontend

# Install Node.js dependencies
npm install

# build
npm run build

# return to root 
cd ..

```


▶️ How to Run the Application
Once you have completed the installation steps, you can run the application with a single command.


```bash
1. Start the FastAPI Server
From the root directory of the project, run the following command:

python main.py

2. Access the Chat Interface
After the server starts, open your favorite web browser and navigate to the following URL:

http://127.0.0.1:8000/chat

You should now see the chat interface, where you can upload your CSV file and begin your conversation!


```

You will need an OpenAI or Gemini API key to chat with your data

