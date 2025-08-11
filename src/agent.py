import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import Agent, ChatContext
import json

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self, chat_ctx: ChatContext = None) -> None:
        super().__init__(
            instructions="""
            You are a career advisor assistant. Your goal is to ask the user questions
            to understand their background, skills, and interests. Then help them
            figure out what job or field might suit them best.
            Speak clearly, be empathetic, and let them talk.
            """,
            chat_ctx=chat_ctx
        )

    async def on_user_message(self, message: str):
        self.transcript.append({"role": "user", "content": message})
        await super().on_user_message(message)

    async def on_agent_response(self, message: str):
        self.transcript.append({"role": "assistant", "content": message})
        await super().on_agent_response(message)
    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    background = "The user has not provided any background info yet."
    
    # Join the room and connect to the user first
    await ctx.connect()
    
    # Wait for a participant to join
    logger.info("Waiting for participant to join...")
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")
    
    # Now check participant metadata
    if participant.metadata:
        logger.info(f"Participant metadata: {participant.metadata}")
        try:
            meta = participant.metadata
            background = f"The user is resume data is {meta}."
            logger.info(f"Successfully parsed participant metadata: {background}")
        except Exception as e:
            logger.warning(f"Failed to parse participant metadata: {e}")
    else:
        logger.info("No metadata found on participant")
    
    # Fall back to room metadata if no participant metadata found
    if background == "The user has not provided any background info yet." and ctx.room and ctx.room.metadata:
        logger.info(f"Trying room metadata: {ctx.room.metadata}")
        try:
            meta = json.loads(ctx.room.metadata)
            background = f"The user is a {meta.get('userBackground', 'person')} interested in {', '.join(meta.get('interests', []))}."
            logger.info(f"Successfully parsed room metadata: {background}")
        except Exception as e:
            logger.warning(f"Failed to parse room metadata: {e}")
    
    if background == "The user has not provided any background info yet.":
        logger.info("Using default background - no metadata found")

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content=background)

    assistant = Assistant(chat_ctx=chat_ctx)
    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=openai.LLM(model="gpt-4o-mini"),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="multi"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=cartesia.TTS(voice="78ab82d5-25be-4f7d-82b3-7ad64e5b85b2"),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel()
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

        # Join the room and connect to the user

    # Send starting reply after connection
    await session.say("Hi! I'm your career guide. Let's find the best job for you.", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
