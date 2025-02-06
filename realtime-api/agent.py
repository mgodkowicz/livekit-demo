from __future__ import annotations

import csv
import datetime
import logging
from typing import Annotated

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
from pydantic import BaseModel

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)

    logger.info("agent started")

class QuestionAnswer(BaseModel):
    question: Annotated[
        str, llm.TypeInfo(description="The question asked to the user")
    ]
    answer: Annotated[
        str, llm.TypeInfo(description="User answer to question")
    ]

# first define a class that inherits from llm.FunctionContext
class AssistantFnc(llm.FunctionContext):
    # the llm.ai_callable decorator marks this function as a tool available to the LLM
    # by default, it'll use the docstring as the function's description
    @llm.ai_callable()
    async def save_answer(
            self,
            user_name: Annotated[
                str, llm.TypeInfo(description="Name given by the user. If wasn't given, use default - 'Anonim'")
            ],
            # answers: Annotated[
            #     list[QuestionAnswer], llm.TypeInfo(description="List of user's answers")
            # ]
            # by using the Annotated type, arg description and type are available to the LLM
            question: Annotated[
                str, llm.TypeInfo(description="The question asked to the user")
            ],
            answer: Annotated[
                str, llm.TypeInfo(description="User answer to question")
            ],
    ):
        """Called to save answers to all users question to the database"""


        # Create a filename based on username and today's date
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = f"{user_name}_{date_str}.csv"

        # Define CSV headers
        headers = ["Timestamp", "Question", "Response"]

        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write to CSV
        try:
            # Check if file exists to determine if headers are needed
            file_exists = False
            try:
                with open(filename, "r", newline="") as file:
                    file_exists = True
            except FileNotFoundError:
                pass

            with open(filename, "a", newline="") as file:
                writer = csv.writer(file)

                # Write headers only if the file is new
                if not file_exists:
                    writer.writerow(headers)

                # Write the response data
                writer.writerow([timestamp, question, answer])

            print(f"Response saved successfully to {filename}")

        except Exception as e:
            print(f"Error saving response: {e}")


def run_multimodal_agent(ctx: JobContext, participant: rtc.RemoteParticipant):
    logger.info("starting multimodal agent")

    model = openai.realtime.RealtimeModel(
        instructions=(
            """
You are an AI survey taker conducting a voice call with the user. Your role is to ask two questions, record their responses, and save them in real time.
	1.	Start the call with a friendly and professional greeting. Include a brief introduction and a touch of small talk to make the user comfortable.
	2.	Ask for the user’s name. Do not proceed until a valid response is given. If unclear or missing, politely ask again.
	3.	Ask for the user’s favorite type of pizza. Ensure the answer is real (e.g., no joke responses). If unclear, rephrase the question to guide the user.
	4.	Confirm the responses by repeating them back and asking for confirmation. 
	5.	Politely conclude the call in a natural, conversational way.

After collecting the user’s name, save it in a database. After collecting their favorite pizza, save it as well. Ensure the data is stored before ending the conversation, but do not mention this process to the user

The conversation should feel smooth and engaging, but still concise. Avoid excessive small talk, but add just enough to make the experience pleasant.
        """
        ),
        model="gpt-4o-mini-realtime-preview",
        modalities=["audio", "text"],
    )
    fnc_ctx = AssistantFnc()
    chat_ctx = llm.ChatContext()

    agent = MultimodalAgent(
        model=model,
        fnc_ctx=fnc_ctx,
        chat_ctx=chat_ctx
    )
    agent.start(ctx.room, participant)

    session = model.sessions[0]
    # @session.on()
    # session.conversation.item
    session.conversation.item.create(
        llm.ChatMessage(
            role="assistant",
            content="Please begin the interaction with the user in a manner consistent with your instructions.",
        )
    )
    session.response.create()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
