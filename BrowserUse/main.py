from langchain_ollama import ChatOllama

from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
import asyncio
# Configure the browser to connect to your Chrome instance
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"

browser = Browser(
   config=BrowserConfig(
         chrome_instance_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe' 

    )
)


# Initialize the model
llm=ChatOllama(model="mistral", num_ctx=32000)

# Create agent with the model
agent = Agent(
    task="Search for latest news about AI",
    llm=llm,
    browser=browser,
)

async def main():
    await agent.run()

    input('Press Enter to close the browser...')
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())