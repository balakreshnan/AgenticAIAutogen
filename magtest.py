import asyncio
import sys
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
import os
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def main() -> None:
    try:
        model_client = AzureOpenAIChatCompletionClient(model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                                                       azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                                       api_key=os.getenv("AZURE_OPENAI_KEY"),
                                                       azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                                                       api_version=os.getenv("AZURE_API_VERSION"))

        # surfer = MultimodalWebSurfer(
        #     "WebSurfer",
        #     model_client=model_client,
        # )
        surfer = MultimodalWebSurfer(
            "MultimodalWebSurfer",
            model_client=model_client,
            downloads_folder="./downs",
            debug_dir="./debug",
            headless = False,
            to_resize_viewport=True,
            description="A web surfing assistant that can browse and interact with web pages.",
            start_page="https://www.bing.com",  # Optional: Initial page
            animate_actions=True,
            browser_data_dir="./browser_data",
        )

        team = MagenticOneGroupChat([surfer], model_client=model_client)
        await Console(team.run_stream(task="Summarize the top 10 AI papers in arxiv?"))

        # # Note: you can also use  other agents in the team
        # team = MagenticOneGroupChat([surfer, file_surfer, coder, terminal], model_client=model_client)
        # file_surfer = FileSurfer( "FileSurfer",model_client=model_client)
        # coder = MagenticOneCoderAgent("Coder",model_client=model_client)
        # terminal = CodeExecutorAgent("ComputerTerminal",code_executor=LocalCommandLineCodeExecutor())
    except Exception as e:
        print(f"Exception in main: {e}")
    finally:
        # Cancel all running tasks to help cleanup
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())