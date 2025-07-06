from abc import ABC, abstractmethod
from typing import List, Dict, Any
import openai
import google.generativeai as genai
import anthropic


class LLMProvider(ABC):
    @abstractmethod
    async def chat_completion(self, model: str, messages: List[Dict[str, Any]], temperature: float, api_key: str) -> str:
        pass

    @abstractmethod
    async def verify_connection(self, model: str, api_key: str) -> tuple[bool, str]:
        """
        Verifies the connection and credentials.
        Returns a tuple of (is_successful, message).
        """
        pass

class OpenAIProvider(LLMProvider):
    async def chat_completion(self, model: str, messages: List[Dict[str, Any]], temperature: float, api_key: str) -> str:
        try:
            print(f"--- Calling OpenAI with model: {model} ---")
            client = openai.AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI returned no content.")
            return content
        except openai.AuthenticationError as e:
            print(f"!!! OpenAI Authentication Error: {e}")
            raise Exception("Invalid OpenAI API key provided.")
        except Exception as e:
            print(f"!!! OpenAI API Error: {e}")
            raise

    async def verify_connection(self, model: str, api_key: str) -> tuple[bool, str]:
        try:
            client = openai.AsyncOpenAI(api_key=api_key) # Use async client
            # Listing models is a low-cost way to verify a key
            await client.models.list()
            return (True, "OpenAI connection successful.")
        except openai.AuthenticationError:
            return (False, "Invalid OpenAI API Key.")
        except Exception as e:
            return (False, f"OpenAI connection failed: {e}")


class GoogleProvider(LLMProvider):
    async def chat_completion(self, model: str, messages: List[Dict[str, Any]], temperature: float, api_key: str) -> str:
        try:
            print(f"Calling Google Gemini with model: {model}")
            genai.configure(api_key=api_key)

            gemini_messages = []
            system_instruction = None
            for msg in messages:
                if msg['role'] == 'system':
                    system_instruction = msg['content']
                    continue
                role = 'model' if msg['role'] == 'assistant' else msg['role']
                gemini_messages.append({'role': role, 'parts': [msg['content']]})

            gemini_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction
            )
            response = await gemini_model.generate_content_async(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            return response.text
        except Exception as e:
            if "api key not valid" in str(e).lower():
                 print(f"!!! Google Gemini API Key Error: {e}")
                 raise Exception("Invalid Google API key provided.")
            print(f"!!! Google Gemini API Error: {e}")
            raise

    async def verify_connection(self, model: str, api_key: str) -> tuple[bool, str]:
        try:
            genai.configure(api_key=api_key)
            # Check if model exists
            # This is not an async operation in the current library, but it's fast.
            await genai.get_model_async(f'models/{model}')
            return (True, "Google Gemini connection successful.")
        except Exception as e:
            return (False, f"Google Gemini connection failed: {e}")

class AnthropicProvider(LLMProvider):

    async def chat_completion(self, model: str, messages: List[Dict[str, Any]], temperature: float, api_key: str) -> str:
        try:
            print(f"Calling Anthropic Claude with model: {model} ")
            client = anthropic.AsyncAnthropic(api_key=api_key)
            
            system_prompt = ""
            claude_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    system_prompt = msg['content']
                else:
                    claude_messages.append(msg)
            
            response = await client.messages.create(
                model=model,
                system=system_prompt,
                messages=claude_messages,
                max_tokens=4096,
                temperature=temperature
            )
            return response.content[0].text
        except anthropic.AuthenticationError as e:
            print(f"Anthropic Authentication Error: {e}")
            raise Exception("Invalid Anthropic API key provided.")
        except Exception as e:
            print(f"!!! Anthropic API Error: {e}")
            raise

    async def verify_connection(self, model: str, api_key: str) -> tuple[bool, str]:
        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            # A simple, low-cost async operation to verify connection.
            # Let's try to create a message with a very short prompt.
            await client.messages.create(
                model="claude-3-haiku-20240307", # Use a fast model
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}]
            )
            return (True, "Anthropic connection successful.")
        except anthropic.AuthenticationError:
            return (False, "Invalid Anthropic API Key.")
        except Exception as e:
            return (False, f"Anthropic connection failed: {e}")