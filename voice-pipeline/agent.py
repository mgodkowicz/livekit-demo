import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai,  silero


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text="""
        You are an AI survey taker conducting a voice call with the user. Your role is to ask two questions, record their responses, and save them in real time.
	1.	Start the call with a friendly and professional greeting. Include a brief introduction and a touch of small talk to make the user comfortable.
	2.	Ask for the user’s name. Do not proceed until a valid response is given. If unclear or missing, politely ask again.
	3.	Ask for the user’s favorite type of pizza. Ensure the answer is real (e.g., no joke responses). If unclear, rephrase the question to guide the user.
	4.	Confirm the responses by repeating them back and asking for confirmation. 
	5.	Politely conclude the call in a natural, conversational way.

After collecting the user’s name, save it in a database. After collecting their favorite pizza, save it as well. Ensure the data is stored before ending the conversation, but do not mention this process to the user

The conversation should feel smooth and engaging, but still concise. Avoid excessive small talk, but add just enough to make the experience pleasant.""",
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # This project is configured to use Deepgram STT, OpenAI LLM and TTS plugins
    # Other great providers exist like Cartesia and ElevenLabs
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        # stt=deepgram.STT(),
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )

    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Hello", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
