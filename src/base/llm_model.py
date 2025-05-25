from langchain_ollama import OllamaLLM

def get_ollama_llm(model_name: str = "llama3.2", **kwargs):
    """
    Function to return an Ollama model instance for language model tasks.

    :param model_name: The name of the Ollama model to load (default is "llama3.2").
    :param kwargs: Additional parameters to be passed to the Ollama model.
    :return: An Ollama model instance.
    """
    # Create and return an Ollama model instance
    llm = OllamaLLM(model=model_name, **kwargs)
    
    return llm
