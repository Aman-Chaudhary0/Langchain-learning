
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
import requests
import os

load_dotenv()


# -----------------------------
# DuckDuckGo Search Tool
# -----------------------------

search_tool = DuckDuckGoSearchRun()


# -----------------------------
# Hugging Face LLM
# -----------------------------

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


# -----------------------------
# Weather Tool
# -----------------------------

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city
    using the Weatherstack API.
    """

    url = "https://api.weatherstack.com/current"

    params = {
        "access_key": WEATHER_API_KEY,
        "query": city
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return f"Weather API request failed: {response.status_code}"

    data = response.json()

    if "error" in data:
        return f"Weather API error: {data['error']}"

    current = data.get("current", {})

    return (
        f"Weather in {city}: "
        f"{current.get('weather_descriptions', ['Unknown'])[0]}, "
        f"temperature {current.get('temperature', 'Unknown')}°C, "
        f"humidity {current.get('humidity', 'Unknown')}%, "
        f"wind speed {current.get('wind_speed', 'Unknown')} km/h."
    )


# -----------------------------
# Create Agent
# -----------------------------

agent = create_agent(
    model=model,
    tools=[search_tool, get_weather_data],
    system_prompt=(
        "You are a helpful research assistant. "
        "Use the search tool when you need factual information "
        "from the internet. Use the weather tool when weather "
        "information is required."
    )
)


# -----------------------------
# Run Agent
# -----------------------------

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find the capital of Madhya Pradesh, "
                    "then find its current weather condition."
                )
            }
        ]
    }
)


# -----------------------------
# Print Final Answer
# -----------------------------

print(response["messages"][-1].content)

