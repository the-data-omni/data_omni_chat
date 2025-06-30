# Data Omni Chat 

Welcome to Data Omni Chat! This application, built with FastAPI, allows you to engage in private conversations with your CSV files using the power of Large Language Models (LLMs).Leverage powerfull large language models without them seeing your data. Upload your data and start asking questions to gain insights quickly and intuitively. Generate Powerpoint slides from your conversatio to quickly shar ein presentations

## 🚀 Features

* **Interactive Chat Interface:** A user-friendly chat interface to interact with your data.
* **Private Conversations:** Your data remains private and are processed locally in browser.
* **Support for CSV Files:** Directly upload and start chatting with your CSV data.
* **Powerpoint Generation:** Generate Powerpoint slides from your conversation.

## 🛠️ Installation

To get the application running on your local machine, please follow these steps.

```bash

### 1. Clone the Repository

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

