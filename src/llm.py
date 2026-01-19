import os
import sys
import json
import re
import time
import logging
from typing import Dict, List, Optional, Any, Union, Literal, TypeVar, Generic, Type, Tuple
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import aiohttp
from google import genai
from google.genai import types as genai_types 
from pydantic import BaseModel
load_dotenv()

# Set up logging
logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / f"llm_processing_{time.strftime('%Y%m%d')}.log"

# Configure logging
logging.basicConfig(
    level=logging.ERROR,  # Changed from INFO to ERROR - only log errors and critical issues
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger("llm_processing")

# Define type variable for generic pydantic models
T_PydanticModel = TypeVar('T_PydanticModel', bound=BaseModel)


class BaseLLMClient:
    """
    Base class for LLM clients.
    """

    def __init__(self, api_key: str, model_name: str):
        """
        Initializes the LLM client.

        Args:
            api_key (str): The API key for the LLM provider.
            model_name (str): The name of the model to use.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.async_client = None

    async def initialize_client(self):
        """
        Initializes the aiohttp client session.
        """
        self.async_client = aiohttp.ClientSession()

    async def close_client(self):
        """
        Closes the aiohttp client session.
        """
        if self.async_client:
            await self.async_client.close()

    async def generate_text(self, prompt: str) -> str:
        """
        Generates text based on the given prompt.

        Args:
            prompt (str): The prompt to use for text generation.

        Returns:
            str: The generated text.
        """
        raise NotImplementedError

    async def generate_embeddings(self, text: str) -> List[float]:
        """
        Generates embeddings for the given text.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            List[float]: The generated embeddings.
        """
        raise NotImplementedError
        
    def parse(self, messages: List[Dict[str, str]], response_model: Type[T_PydanticModel]) -> T_PydanticModel:
        """
        Parse the LLM response directly into a Pydantic model.
        
        Args:
            messages (List[Dict[str, str]]): List of message dicts with role and content
            response_model (Type[T_PydanticModel]): Pydantic model class to parse the response into
            
        Returns:
            T_PydanticModel: Parsed response in the specified Pydantic model
        """
        raise NotImplementedError("Subclasses must implement parse method")
        
    async def async_parse(self, messages: List[Dict[str, str]], response_model: Type[T_PydanticModel]) -> T_PydanticModel:
        """
        Asynchronously parse the LLM response directly into a Pydantic model.
        
        Args:
            messages (List[Dict[str, str]]): List of message dicts with role and content
            response_model (Type[T_PydanticModel]): Pydantic model class to parse the response into
            
        Returns:
            T_PydanticModel: Parsed response in the specified Pydantic model
        """
        # Default implementation runs the synchronous version in a thread pool
        return await asyncio.to_thread(self.parse, messages, response_model)


class OpenAIClient(BaseLLMClient, Generic[T_PydanticModel]):
    """Client for OpenAI API."""

    def __init__(self, api_key: str, model_name: str):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key (str): The OpenAI API key
            model_name (str): The name of the OpenAI model to use
        """
        super().__init__(api_key, model_name)
        # We'll initialize the client in the initialize_client method
        
    async def initialize_client(self):
        """Initialize the async client session."""
        from openai import AsyncOpenAI
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    async def generate_text(self, prompt: str) -> str:
        """
        Generates text using the OpenAI API.

        Args:
            prompt (str): The prompt to use for text generation.

        Returns:
            str: The generated text.
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text with OpenAI: {e}")
            return ""
            
    def parse(self, messages: List[Dict[str, str]], response_model: Type[T_PydanticModel]) -> T_PydanticModel:
        """
        Parse the OpenAI response directly into a Pydantic model.
        
        Args:
            messages (List[Dict[str, str]]): List of message dicts with role and content
            response_model (Type[T_PydanticModel]): Pydantic model class to parse the response into
            
        Returns:
            T_PydanticModel: Parsed response in the specified Pydantic model
        """
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            response_content = response.choices[0].message.content or "{}"
            parsed_response = json.loads(response_content)
            return response_model.model_validate(parsed_response)
        except Exception as e:
            raise Exception(f"Error parsing OpenAI response: {e}")
            
    async def async_parse(self, messages: List[Dict[str, str]], response_model: Type[T_PydanticModel]) -> T_PydanticModel:
        """
        Asynchronously parse the OpenAI response directly into a Pydantic model.
        
        Args:
            messages (List[Dict[str, str]]): List of message dicts with role and content
            response_model (Type[T_PydanticModel]): Pydantic model class to parse the response into
            
        Returns:
            T_PydanticModel: Parsed response in the specified Pydantic model
        """
        try:
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
            completion = await self.async_client.chat.completions.create(**request_params)
            response_content = completion.choices[0].message.content or "{}"

            parsed_response = json.loads(response_content)
            return response_model.model_validate(parsed_response)
        except Exception as e:
            raise Exception(f"Error asynchronously parsing OpenAI response: {e}. Response content: '{response_content[:200]}...'")

    async def generate_embeddings(self, text: str) -> List[float]:
        """
        Generates embeddings using the OpenAI API.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            List[float]: The generated embeddings.
        """
        try:
            response = await self.async_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embeddings with OpenAI: {e}")
            return []


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""
    
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize Gemini client using genai.Client.
        
        Args:
            api_key (str): Google API key
            model_name (str): Model name to use
        """
        super().__init__(api_key, model_name)
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise Exception(f"Error initializing Gemini client: {e}")

    async def initialize_client(self):
        """
        Initializes the Gemini client.
        No need for additional initialization as it's done in __init__
        """
        # Client is already initialized in __init__
        pass
        
    async def generate_text(self, prompt: str) -> str:
        """
        Generates text using the Gemini API.

        Args:
            prompt (str): The prompt to use for text generation.

        Returns:
            str: The generated text.
        """
        return self.invoke([{"role": "user", "content": prompt}])
        
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """
        Invoke the Gemini model with the given messages.
        
        Args:
            messages (List[Dict[str, str]]): List of message dicts with role and content
            
        Returns:
            str: The content of the model's response
        """
        try:
            is_reasonear = os.getenv("LLM_IS_REASONER_ABSTRACT", "False").lower() == "true"
            prompt_parts = []
            for msg in messages:
                if msg["role"] in ("user", "system"):
                    prompt_parts.append(msg["content"])
            prompt = "\n\n".join(prompt_parts)

            # Check if the model is from the gemini-2.5-flash family to add thinking_config
            model_lower = self.model_name.lower()
            if model_lower.startswith("gemini-2.5-flash") and not is_reasonear:
                config = genai_types.GenerateContentConfig(
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0)
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
            return response.text
        except Exception as e:
            raise Exception(f"Error invoking Gemini model: {e}")
    
    async def async_invoke(self, messages: List[Dict[str, str]]) -> str:
        """
        Asynchronously invoke the Gemini model with the given messages.
        
        Args:
            messages (List[Dict[str, str]]): List of message dicts with role and content
            
        Returns:
            str: The content of the model's response
        """
        # Use asyncio.to_thread to run the synchronous invoke method in a thread pool
        return await asyncio.to_thread(self.invoke, messages)
    
    def parse(self, messages: List[Dict[str, str]], response_model: Type[T_PydanticModel]) -> T_PydanticModel:
        """
        Parse the Gemini response directly into a Pydantic model.
        
        Args:
            messages (List[Dict[str, str]]): List of message dicts with role and content
            response_model (Type[T_PydanticModel]): Pydantic model class to parse the response into
            
        Returns:
            T_PydanticModel: Parsed response in the specified Pydantic model
        """
        try:
            # First, enhance messages to explicitly request JSON output
            enhanced_messages = messages.copy()
            if enhanced_messages and enhanced_messages[-1]["role"] == "user":
                enhanced_messages[-1]["content"] += "\n\nPlease return your response in valid JSON format following the schema provided."
            
            # Get response as plain text
            response_content = self.invoke(enhanced_messages)
            
            # Save raw response content
            save_raw_llm_response(response_content)
            
            # Clean response to extract JSON content
            # Sometimes model returns markdown code block or other formatting
            json_content = response_content
            if "```json" in json_content:
                # Extract JSON from markdown code block if present
                start_idx = json_content.find("```json") + 7
                end_idx = json_content.find("```", start_idx)
                if end_idx > start_idx:
                    json_content = json_content[start_idx:end_idx].strip()
            
            # Parse JSON response
            parsed_response = json.loads(json_content)
            
            # Convert to pydantic model
            try:
                # Try pydantic v2 approach first
                return response_model.model_validate(parsed_response)
            except AttributeError:
                # Fall back to pydantic v1 approach
                return response_model.parse_obj(parsed_response)
                
        except json.JSONDecodeError as e:
            # If JSON parsing fails, show part of the raw response for debugging
            preview = response_content[:200] + "..." if len(response_content) > 200 else response_content
            raise Exception(f"Error parsing Gemini response as JSON: {e}. Response preview: {preview}")
        except Exception as e:
            raise Exception(f"Error parsing Gemini response: {e}")
    
    async def async_parse(self, messages: List[Dict[str, str]], response_model: Type[T_PydanticModel]) -> T_PydanticModel:
        """
        Asynchronously parse the Gemini response directly into a Pydantic model.
        
        Args:
            messages (List[Dict[str, str]]): List of message dicts with role and content
            response_model (Type[T_PydanticModel]): Pydantic model class to parse the response into
            
        Returns:
            T_PydanticModel: Parsed response in the specified Pydantic model
        """
        # Use asyncio.to_thread to run the synchronous parse method in a thread pool
        return await asyncio.to_thread(self.parse, messages, response_model)
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """
        Generates embeddings for the given text.
        Note: Implementation placeholder as the Gemini API may require different handling.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            List[float]: The generated embeddings.
        """
        # Placeholder - implementation would depend on Gemini's specific embedding API
        return []

    async def close_client(self):
        """
        Closes the Gemini client.
        """
        # No explicit close method for Gemini API
        pass


class DeepSeekClient(BaseLLMClient):
    """
    Client for interacting with the DeepSeek API.
    """

    def __init__(self, api_key: str, model_name: str):
        """
        Initializes the DeepSeek client.

        Args:
            api_key (str): The DeepSeek API key.
            model_name (str): The name of the DeepSeek model to use.
        """
        super().__init__(api_key, model_name)
        
    async def initialize_client(self):
        """
        Initializes the client - for DeepSeek we use OpenAI's client.
        """
        from openai import AsyncOpenAI
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")
        )

    async def generate_text(self, prompt: str) -> str:
        """
        Generates text using the DeepSeek API.

        Args:
            prompt (str): The prompt to use for text generation.

        Returns:
            str: The generated text.
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text with DeepSeek: {e}")
            return ""

    async def generate_embeddings(self, text: str) -> List[float]:
        """
        Generates embeddings for the given text.
        Note: Implementation placeholder as specific embedding endpoints may vary.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            List[float]: The generated embeddings.
        """
        # Placeholder - implementation would depend on DeepSeek's specific embedding API
        return []


def save_raw_llm_response(response: str) -> None:
    """
    Save the raw LLM response to a file with a timestamp.

    Args:
        response (str): The raw response content from the LLM.
    """
    timestamp: str = time.strftime("%Y%m%d-%H%M%S")
    # You may adjust the path below to your desired folder.
    # Create data/llm_results directory if it doesn't exist
    output_dir = os.path.join("data", "llm_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use relative path to save the response
    file_path: str = os.path.join(output_dir, f"llm_response_{timestamp}.txt")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response)
    except Exception as e:
        error_msg = f"Error saving LLM response: {e}"
        print(error_msg)
        logger.error(error_msg)
        # Create emergency backup in logs directory
        try:
            emergency_dir = Path("logs/emergency_backups")
            emergency_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Save emergency backups
            with open(emergency_dir / f"llm_response_{timestamp}_raw.txt", "w", encoding="utf-8") as f:
                f.write(response)
            logger.info(f"Created emergency backups in {emergency_dir}")
        except Exception as backup_error:
            logger.critical(f"Failed to create emergency backups: {backup_error}")
        
def clean_and_parse_json(file_content: str) -> Dict[str, Any]:
    """
    Cleans and parses JSON content from raw responses.
    
    Args:
        file_content (str): Response file content
        
    Returns:
        Dict[str, Any]: Dictionary containing the parsed JSON data
    """
    # Check if content is within code blocks
    json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(json_block_pattern, file_content)
    
    if matches:
        # Get the first JSON block
        content = matches[0]
    else:
        # No code blocks, use raw content
        content = file_content
    
    # Additional cleaning
    content = content.replace("```json", "").replace("```", "").strip()
    content = content.strip("[").strip("]")
    
    # Try to parse as JSON
    try:
        dict_json = json.loads(content)
        if len(dict_json) == 1 and "id" not in dict_json:
            new_dict = {}
            for key, value in dict_json.items():
                new_dict["id"] = key
                if not isinstance(value, dict):
                    value = {"value": value}
                new_dict = new_dict | value
            return new_dict
        return dict_json

    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parsing failed: {e}. Attempting to fix JSON format.")
        try:
            # Handle possible trailing commas
            content = re.sub(r',\s*}', '}', content)
            content = re.sub(r',\s*]', ']', content)
            return json.loads(content)
        except json.JSONDecodeError as e2:
            # Log the error with the problematic content
            logger.error(f"JSON parsing failed after cleanup: {e2}. Content preview: {content[:200]}...")
            # Return raw content as fallback
            return {"raw_content": content}


async def convert_to_json_llm(raw_text: str, llm_client: Optional[BaseLLMClient] = None) -> Dict[str, Any]:
    """
    Uses a language model to convert raw text into valid JSON.
    
    Args:
        raw_text (str): Raw text to be converted to JSON
        llm_client (Optional[BaseLLMClient]): Optional LLM client instance to use
        
    Returns:
        Dict[str, Any]: Dictionary with the JSON content parsed from the LLM response
    """
    try:
        # Define a prompt for JSON conversion
        prompt = f"""Convert the string into valid json. json must be unique dictionary, not a list or array,
Return only json without any other text or explanation or comments or any other text.
Do not add any other text or explanation or comments or any other text.

string json to convert:
{raw_text}
"""
        
        # Use the provided LLM client or create a new one
        client_initialized = False
        if llm_client is None:
            # Get environment variables for parsing video
            model_name = os.getenv("OPENAI_PARSE_VIDEO_MODEL_NAME", "gpt-4.1-nano")
            api_key = os.getenv("OPENAI_PARSE_VIDEO_API_KEY")
            
            if not api_key:
                error_msg = "Error: API key for LLM not found"
                logger.error(error_msg)
                return {"error": "API key not found"}
            
            # Initialize a client
            llm_client = OpenAIClient(api_key=api_key, model_name=model_name)
            await llm_client.initialize_client()
            client_initialized = True
        
        # Get the response using the appropriate method
        result_text = await llm_client.generate_text(prompt)
        
        # Close the client if we created it
        if client_initialized:
            await llm_client.close_client()
        
        # Parse JSON from response
        return clean_and_parse_json(result_text)
            
    except Exception as e:
        error_msg = f"Error using LLM for JSON conversion: {e}"
        print(error_msg)
        logger.error(error_msg)
        return {"error": str(e)}


async def process_text_with_llm(llm_client: BaseLLMClient, text: str) -> Tuple[str, str, Dict[str, Any], str]:
    """
    Processes the given text using the specified LLM client and returns structured data.

    Args:
        llm_client (BaseLLMClient): The LLM client to use.
        text (str): The text to process.

    Returns:
        Tuple[str, str, Dict[str, Any], str]: A tuple containing the LLM's client name, the generated text, 
        parsed JSON data, and the actual model name.
    """
    try:
        start_time = time.time()
        generated_text = await llm_client.generate_text(text)
        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"LLM {llm_client.__class__.__name__} processed text in {processing_time:.2f} seconds.")
        
        # Parse the generated text as JSON
        parsed_data = clean_and_parse_json(generated_text)
        
        # If parsing failed or returned minimal data, try using the LLM to convert to JSON
        if "raw_content" in parsed_data:
            logger.info(f"Initial JSON parsing failed, using LLM to convert to JSON")
            parsed_data = await convert_to_json_llm(generated_text, llm_client)

        # Return the model name as well
        model_name = llm_client.model_name
        return llm_client.__class__.__name__, generated_text, parsed_data, model_name

    except Exception as e:
        error_msg = f"Error processing text with LLM {llm_client.__class__.__name__}: {e}"
        print(error_msg)
        logger.error(error_msg)
        return llm_client.__class__.__name__, "", {"error": str(e)}, llm_client.model_name


def save_response_data(llm_name: str, model_name: str, text_id: str, generated_text: str, parsed_data: Dict[str, Any], prompt: str, base_output_dir: str = "data/llm_results") -> Tuple[str, str, str]:
    """
    Saves LLM response, parsed JSON data, and the prompt to files in separate directories.
    Uses model name for the filename instead of timestamp.
    
    Args:
        llm_name (str): Name of the LLM client class
        model_name (str): Actual model name (e.g., gpt-4.1-nano, gemini-pro)
        text_id (str): Identifier for the processed text
        generated_text (str): Raw text generated by the LLM
        parsed_data (Dict[str, Any]): Parsed JSON data from the response
        prompt (str): The prompt used to generate the response
        base_output_dir (str): Base directory to save the output files. Defaults to "data/llm_results".
        
    Returns:
        Tuple[str, str, str]: Paths to the raw text file, JSON file, and prompt file
    """
    try:
        # Create subdirectories within the base output directory
        raw_output_dir = Path(base_output_dir) / "raw"
        json_output_dir = Path(base_output_dir) / "json"
        prompts_output_dir = Path(base_output_dir) / "prompts"
        
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        json_output_dir.mkdir(parents=True, exist_ok=True)
        prompts_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use model name for filenames
        raw_file_path = raw_output_dir / f"{model_name}_raw.txt"
        json_file_path = json_output_dir / f"{model_name}_parsed.json"
        prompt_file_path = prompts_output_dir / f"{model_name}_prompt.txt"
        
        # Save raw text response
        with open(raw_file_path, "w", encoding="utf-8") as f:
            f.write(generated_text)
        
        # Save parsed JSON data
        try:
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            error_msg = f"Error saving JSON data for {model_name}: {e}"
            logger.error(error_msg)
            # Try to save as text if JSON serialization fails
            fallback_path = json_output_dir / f"{model_name}_parsed_error.txt"
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(str(parsed_data))
            logger.info(f"Saved fallback text representation to {fallback_path}")
        
        # Save prompt
        with open(prompt_file_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        logger.info(f"Saved raw response to {raw_file_path}")
        logger.info(f"Saved parsed JSON to {json_file_path}")
        logger.info(f"Saved prompt to {prompt_file_path}")
        
        return str(raw_file_path), str(json_file_path), str(prompt_file_path)
    
    except Exception as e:
        error_msg = f"Error saving response data: {e}"
        print(error_msg)
        logger.error(error_msg)
        # Create emergency backup in logs directory
        try:
            emergency_dir = Path("logs/emergency_backups")
            emergency_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Save emergency backups
            with open(emergency_dir / f"{model_name}_{timestamp}_raw.txt", "w", encoding="utf-8") as f:
                f.write(generated_text)
            with open(emergency_dir / f"{model_name}_{timestamp}_prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            
            logger.info(f"Created emergency backups in {emergency_dir}")
        except Exception as backup_error:
            logger.critical(f"Failed to create emergency backups: {backup_error}")
        
        return "", "", ""


def save_structured_data(
    data: Union[BaseModel, List[BaseModel]],
    output_path: str # Full path including filename.json
):
    """
    Save a single Pydantic BaseModel instance or a list of them to a JSON file.
    Args:
        data (Union[BaseModel, List[BaseModel]]): The Pydantic model instance or list of instances.
        output_path (str): Full path (including filename.json) to save the JSON data.
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, list):
            json_data = [item.model_dump(mode="json") for item in data]
        else:
            json_data = data.model_dump(mode="json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Structured data saved to {output_file}")

    except Exception as e:
        error_msg = f"Error saving structured data to {output_path}: {e}"
        print(error_msg)
        logger.error(error_msg)
        
        # Try to save as text representation as fallback
        try:
            fallback_path = str(output_path).replace(".json", "_error.txt")
            with open(fallback_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
            logger.info(f"Saved fallback text representation to {fallback_path}")
        except Exception as fallback_error:
            logger.error(f"Failed to save fallback text representation: {fallback_error}")

async def run_llm_clients(prompt: str, base_output_dir: str = "") -> None:
    """
    Process a prompt with multiple LLM clients and save the results.
    
    Args:
        prompt (str): The prompt to process
        base_output_dir (str): Base directory to save results
    """
    # Load API keys and model names from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano")
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")

    # Create output directory
    if not base_output_dir:
        base_output_dir = os.getenv("LLM_OUTPUT_DIR", "data/llm_results")
    
    # Initialize LLM clients
    openai_client = OpenAIClient(api_key=openai_api_key, model_name=openai_model_name)
    gemini_client = GeminiClient(api_key=gemini_api_key, model_name=gemini_model_name)
    deepseek_client = DeepSeekClient(api_key=deepseek_api_key, model_name=deepseek_model_name)

    await openai_client.initialize_client()
    await gemini_client.initialize_client()
    await deepseek_client.initialize_client()
    texts = [prompt]

    # Process texts with each LLM
    async def process_with_client(client, text):
        llm_name, generated_text, parsed_data, model_name = await process_text_with_llm(client, text)
        print(f"LLM: {llm_name} (Model: {model_name}), Generated Text: {generated_text[:50]}...")
        
        # Save both raw response, parsed data, and prompt
        raw_path, json_path, prompt_path = save_response_data(
            llm_name=llm_name,
            model_name=model_name,
            text_id=text,
            generated_text=generated_text,
            parsed_data=parsed_data,
            prompt=text,  # Save the original prompt
            base_output_dir=base_output_dir
        )
        
        print(f"Results saved to {raw_path}, {json_path}, and {prompt_path}")

    # Use asyncio.gather to run the processing concurrently
    await asyncio.gather(
        *[process_with_client(openai_client, text) for text in texts],
        *[process_with_client(gemini_client, text) for text in texts],
        *[process_with_client(deepseek_client, text) for text in texts],
    )

    await openai_client.close_client()
    await gemini_client.close_client()
    await deepseek_client.close_client()


async def run_llm_clients_one(prompt: str, base_output_dir: str = "", llm_name: str = "") -> None:
    """
    Process a prompt with a single LLM client specified by name and save the results.
    
    Args:
        prompt (str): The prompt to process
        base_output_dir (str): Base directory to save results
        llm_name (str): Name of the LLM to use ("openai", "gemini", or "deepseek")
    """
    # Normalize the llm_name to lowercase for case-insensitive comparison
    llm_name = llm_name.lower() if llm_name else "openai"  # Default to OpenAI if not specified
    
    # Load API keys and model names from environment variables based on the selected LLM
    client = None
    
    if llm_name == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano")
        client = OpenAIClient(api_key=openai_api_key, model_name=openai_model_name)
    elif llm_name == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")
        client = GeminiClient(api_key=gemini_api_key, model_name=gemini_model_name)
    elif llm_name == "deepseek":
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        deepseek_model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
        client = DeepSeekClient(api_key=deepseek_api_key, model_name=deepseek_model_name)
    else:
        raise ValueError(f"Unsupported LLM: {llm_name}. Choose from: openai, gemini, deepseek")

    # Create output directory
    if not base_output_dir:
        base_output_dir = os.getenv("LLM_OUTPUT_DIR", "data/llm_results")
    
    # Initialize LLM client
    await client.initialize_client()
    
    # Process text with the selected LLM
    llm_name_result, generated_text, parsed_data, model_name = await process_text_with_llm(client, prompt)
    print(f"LLM: {llm_name_result} (Model: {model_name}), Generated Text: {generated_text[:50]}...")
    
    # Save both raw response, parsed data, and prompt
    raw_path, json_path, prompt_path = save_response_data(
        llm_name=llm_name_result,
        model_name=model_name,
        text_id=prompt,
        generated_text=generated_text,
        parsed_data=parsed_data,
        prompt=prompt,  # Save the original prompt
        base_output_dir=base_output_dir
    )
    
    print(f"Results saved to {raw_path}, {json_path}, and {prompt_path}")
    
    # Close the client
    await client.close_client()


if __name__ == "__main__":
    asyncio.run(run_llm_clients(prompt="Conte uma piada sobre programadores em Python."))