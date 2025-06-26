```
uv venv
.venv\Scripts\activate

python file.py url
```



| Step | What happens                           | Function                       |
| ---- | -------------------------------------- | ------------------------------ |
| 1    | User runs CLI command with URL         | `generate()`                   |
| 2    | Async runner kicks off                 | `run_generation()`             |
| 3    | Loads environment variables            | `load_dotenv()`                |
| 4    | Initializes generator                  | `CharacterPromptGenerator()`   |
| 5    | Starts crawler                         | `.crawler.start()`             |
| 6    | Scrapes and cleans wiki content        | `_fetch_and_process_content()` |
| 7    | Sends content to Gemini for extraction | `_extract_character_details()` |
| 8    | Builds final system prompt             | `_generate_system_prompt()`    |
| 9    | Closes crawler                         | `.crawler.close()`             |
| 10   | Prints final prompt                    | `console.print()`              |
| 11   | Optional chat with Gemini              | `use_prompt_with_gemini()`     |
