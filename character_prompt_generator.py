# character_prompt_generator.py

import asyncio
import os
from typing import List, Optional

# --- Gemini Imports (for final chat example) ---
import google.generativeai as genai
import typer

# --- Crawl4AI Imports ---
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- LangChain Imports for robust Gemini integration ---
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# --- Rich Library Imports for a beautiful UI ---
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# --- Initialize Rich Console for beautiful output ---
console = Console()


# --- Pydantic Schema for Character Extraction ---
# This schema is now used by LangChain's PydanticOutputParser
class CharacterProfile(BaseModel):
    name: str = Field(description="The full name of the character.")
    personality_traits: List[str] = Field(
        description="A list of 3-5 key personality traits or adjectives (e.g., 'sarcastic', 'brave', 'methodical')."
    )
    speech_patterns: List[str] = Field(
        description="A list of 2-4 characteristic speech patterns, tones, or catchphrases (e.g., 'Speaks in short, direct sentences', 'Has a habit of saying \"calibrations\"', 'Uses formal language')."
    )
    key_backstory_points: List[str] = Field(
        description="A list of 3-5 crucial, defining memories or events from the character's past that shape their worldview."
    )
    key_relationships: List[str] = Field(
        description="A list of 2-4 important relationships, specifying the person and their connection (e.g., 'Commander Shepard (trusted friend and commander)', 'Dr. Sidonis (betrayed him)')."
    )
    core_motivations_or_goals: List[str] = Field(
        description="A list of 1-3 primary goals or motivations driving the character's actions."
    )
    example_quotes: List[str] = Field(
        description="A list of 2-3 sample quotes that capture the character's unique voice and personality."
    )


# --- The Main Class for our Project ---
class CharacterPromptGenerator:
    """
    Orchestrates fetching character data, extracting details, and generating prompts.
    """

    def __init__(self, character_url: str):
        self.character_url = character_url
        self.crawler = AsyncWebCrawler()

    async def _fetch_and_process_content(self) -> str:
        """Fetches the webpage and returns clean, filtered Markdown."""
        markdown_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.45)
        )
        config = CrawlerRunConfig(
            markdown_generator=markdown_generator, word_count_threshold=100
        )
        result = await self.crawler.arun(url=self.character_url, config=config)
        if not result.success or not result.markdown.fit_markdown:
            raise ValueError(
                f"Failed to fetch or process content. Error: {result.error_message}"
            )
        return result.markdown.fit_markdown

    async def _extract_character_details(self, content: str) -> CharacterProfile:
        """Uses LangChain and Gemini to extract structured data from the content."""

        # 1. Initialize the Gemini model via LangChain
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

        # 2. Set up a Pydantic parser to enforce the output schema
        parser = PydanticOutputParser(pydantic_object=CharacterProfile)

        # 3. Create a prompt template that includes the content and format instructions
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at analyzing fictional characters. Your task is to extract key details that define their personality, background, and speech. Fill out all fields in the provided JSON schema. {format_instructions}",
                ),
                (
                    "human",
                    "Please analyze the following text about a character and extract their details:\n\n{character_content}",
                ),
            ]
        )

        # 4. Create the LangChain "chain"
        chain = prompt_template | model | parser

        # 5. Invoke the chain asynchronously with the content
        # LangChain handles the API call and parsing
        profile = await chain.ainvoke(
            {
                "character_content": content,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        return profile

    def _generate_system_prompt(self, profile: CharacterProfile) -> str:
        """Assembles the final system prompt from the extracted character profile."""

        # Pre-format the lists into strings to avoid backslashes inside f-string expressions
        traits = "\n- ".join(profile.personality_traits)
        backstory = "\n- ".join(profile.key_backstory_points)
        relationships = "\n- ".join(profile.key_relationships)
        motivations = "\n- ".join(profile.core_motivations_or_goals)
        speech = "\n- ".join(profile.speech_patterns)
        quotes = "\n- ".join(f'"{q}"' for q in profile.example_quotes)

        prompt = f"""### INSTRUCTION: ACT AS A CHARACTER ###

You are to embody the character of {profile.name}. Your personality, memories, and way of speaking must align with the detailed profile below. Stay in character at all times. Do not break character under any circumstances.

---
### CHARACTER PROFILE: {profile.name} ###

**1. CORE PERSONA:**
Your personality is defined by these traits. Embody them in your responses:
- {traits}

**2. KEY MEMORIES & WORLDVIEW:**
These core memories shape your beliefs, opinions, and how you see the world. Refer to them when relevant, but don't state them plainly unless asked directly. They are part of your implicit knowledge.
- {backstory}

**3. RELATIONSHIPS:**
These are the most important people in your life. Your opinion of them should color your responses.
- {relationships}

**4. MOTIVATIONS & GOALS:**
These are the goals that drive your actions.
- {motivations}

**5. DIALOGUE & SPEECH STYLE:**
You must adhere to these speech patterns to sound authentic:
- {speech}

**Example Quotes for Voice Matching:**
- {quotes}

---
### RESPONSE GUIDELINES ###
- **Stay in character:** Do not reveal you are an AI. You are {profile.name}.
- **Be authentic:** Use the persona, memories, and speech style defined above.
- **Be conversational:** Respond naturally to the user's questions and statements based on your character profile.
"""
        return prompt.strip()

    async def generate(self) -> str:
        """The main method to generate the character prompt."""
        await self.crawler.start()
        try:
            with console.status(
                "[bold green]Fetching and processing content...", spinner="dots8"
            ):
                markdown_content = await self._fetch_and_process_content()
            console.print("[green]✔[/green] Content processed successfully.")

            with console.status(
                "[bold green]Extracting character details with Gemini...",
                spinner="dots8",
            ):
                character_profile = await self._extract_character_details(
                    markdown_content
                )
            console.print("[green]✔[/green] Character details extracted successfully.")

            with console.status(
                "[bold green]Generating final system prompt...", spinner="dots8"
            ):
                system_prompt = self._generate_system_prompt(character_profile)
            console.print("[green]✔[/green] Prompt generated.")

            return system_prompt
        finally:
            await self.crawler.close()


async def use_prompt_with_gemini(system_prompt: str):
    """Shows how to use the generated prompt with the Google Gemini API."""
    console.rule(
        "[bold magenta]Testing the Generated Prompt with Gemini[/bold magenta]"
    )

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        console.print(
            "[bold yellow]Warning:[/bold yellow] GEMINI_API_KEY not set. Skipping Gemini usage example."
        )
        return

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash", system_instruction=system_prompt
        )
        chat = model.start_chat()

        console.print(
            "\n[bold cyan]Chatting with the character... Type 'quit' to exit.[/bold cyan]\n"
        )

        # Example interaction
        user_question = "What are you up to?"
        console.print(f"[bold]You:[/bold] {user_question}")
        with console.status(
            "[italic]Character is thinking...[/italic]", spinner="bouncingBar"
        ):
            response = chat.send_message(user_question)
        console.print(f"\n[bold magenta]Character:[/bold magenta] {response.text}\n")

        user_question = "Any thoughts on the Reapers?"
        console.print(f"[bold]You:[/bold] {user_question}")
        with console.status(
            "[italic]Character is thinking...[/italic]", spinner="bouncingBar"
        ):
            response = chat.send_message(user_question)
        console.print(f"\n[bold magenta]Character:[/bold magenta] {response.text}\n")

    except Exception as e:
        console.print(
            f"[bold red]An error occurred while using the Gemini API:[/bold red] {e}"
        )


async def run_generation(character_url: str):
    """Main async function to run the character prompt generation."""
    load_dotenv()

    if not os.getenv("GEMINI_API_KEY"):
        console.print(
            "[bold red]Error:[/bold red] GEMINI_API_KEY environment variable not set."
        )
        console.print("Please create a .env file and add your Gemini API key to it.")
        return

    try:
        # No LLMConfig needed here anymore, as LangChain handles it internally
        generator = CharacterPromptGenerator(character_url=character_url)
        generated_prompt = await generator.generate()

        prompt_panel = Panel(
            Syntax(generated_prompt, "markdown", theme="dracula", word_wrap=True),
            title="[bold green]🤖 Super Cool Character Prompt Generated! 🤖[/bold green]",
            border_style="green",
            expand=True,
        )
        console.print(prompt_panel)

        await use_prompt_with_gemini(generated_prompt)

    except Exception as e:
        console.print(
            f"\n[bold red]An error occurred during the process:[/bold red] {e}"
        )


# --- Typer CLI Application ---
app = typer.Typer(
    add_completion=False,
    help="Generates high-quality LLM system prompts for character role-playing from a Fandom/wiki URL.",
    rich_markup_mode="markdown",
)


@app.command()
def generate(
    character_url: str = typer.Argument(
        "https://masseffect.fandom.com/wiki/Garrus_Vakarian",
        help="The URL of the character's Fandom/wiki page.",
    )
):
    """
    Generates a high-quality LLM system prompt for character role-playing from a URL.
    """
    console.rule("[bold blue]Character Prompt Generator Initialized[/bold blue]")
    console.print(f"Target URL: [link={character_url}]{character_url}[/link]")
    asyncio.run(run_generation(character_url))


if __name__ == "__main__":
    app()
