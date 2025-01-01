"""
Agentic sampling loop that supports both Anthropic API and Google's Gemini API, with local implementation of computer use tools.
"""

import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast
import json

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import (
    ToolResultBlockParam,
)
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration

from tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

BETA_FLAG = "computer-use-2024-10-22"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    GEMINI = "gemini"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
    APIProvider.GEMINI: "gemini-exp-1206",
}


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
# SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
# * You are utilizing a macOS Sonoma 15.7 environment using {platform.machine()} architecture with internet access.
# * You can install applications using homebrew with your bash tool. Use curl instead of wget.
# * To open Chrome, please just click on the Chrome icon in the Dock or use Spotlight.
# * Using bash tool you can start GUI applications. GUI apps can be launched directly or with `open -a "Application Name"`. GUI apps will appear natively within macOS, but they may take some time to appear. Take a screenshot to confirm it did.
# * When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
# * When viewing a page it can be helpful to zoom out so that you can see everything on the page. In Chrome, use Command + "-" to zoom out or Command + "+" to zoom in.
# * When using your computer function calls, they take a while to run and send back to you. Where possible/feasible, try to chain multiple of these calls all into one function calls request.
# * The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
# </SYSTEM_CAPABILITY>
# <IMPORTANT>
# * When using Chrome, if any first-time setup dialogs appear, IGNORE THEM. Instead, click directly in the address bar and enter the appropriate search term or URL there.
# * If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext (available via homebrew) to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
# </IMPORTANT>"""
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a macOS Sonoma 15.7 environment using {platform.machine()} architecture with command line internet access.
* Package management:
  - Use homebrew for package installation
  - Use curl for HTTP requests
  - Use npm/yarn for Node.js packages
  - Use pip for Python packages

* Browser automation available via Playwright:
  - Supports Chrome, Firefox, and WebKit
  - Can handle JavaScript-heavy applications
  - Capable of screenshots, navigation, and interaction
  - Handles dynamic content loading

* System automation:
  - cliclick for simulating mouse/keyboard input
  - osascript for AppleScript commands
  - launchctl for managing services
  - defaults for reading/writing system preferences

* Development tools:
  - Standard Unix/Linux command line utilities
  - Git for version control
  - Docker for containerization
  - Common build tools (make, cmake, etc.)

* Output handling:
  - For large output, redirect to tmp files: command > /tmp/output.txt
  - Use grep with context: grep -n -B <before> -A <after> <query> <filename>
  - Stream processing with awk, sed, and other text utilities

* Note: Command line function calls may have latency. Chain multiple operations into single requests where feasible.

* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>"""

async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
):
    """
    Agentic sampling loop for the assistant/tool interaction using either Anthropic or Gemini API.
    """
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )
    system = (
        f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
    )

    while True:
        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(messages, only_n_most_recent_images)

        if provider == APIProvider.GEMINI:
            # Configure Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name=model)

            # Convert tools to Gemini function declarations
            tools = [tool.to_gemini_tool() for tool in tool_collection.tools]

            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, list):
                        # Handle tool results
                        text_content = "\n".join(
                            block.get("text", "") 
                            for block in content 
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                        gemini_messages.append({"role": "user", "parts": [text_content]})
                    else:
                        gemini_messages.append({"role": "user", "parts": [content]})
                elif msg["role"] == "assistant":
                    content = msg["content"]
                    if isinstance(content, list):
                        text_content = "\n".join(
                            block.get("text", "") 
                            for block in content 
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                        gemini_messages.append({"role": "model", "parts": [text_content]})

            # Call Gemini API
            chat = model.start_chat(history=gemini_messages)
            response = chat.send_message(
                system,
                tools=tools,
                generation_config={"max_output_tokens": max_tokens}
            )

            # Convert Gemini response to our format
            content_blocks = []
            if response.text:
                content_blocks.append({
                    "type": "text",
                    "text": response.text
                })

            if response.candidates[0].function_calls:
                for tool_call in response.candidates[0].function_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": f"tool_{len(messages)}",
                        "name": tool_call.name,
                        "input": json.loads(tool_call.args)
                    })

            messages.append({
                "role": "assistant",
                "content": content_blocks
            })

            tool_result_content = []
            for content_block in content_blocks:
                output_callback(content_block)
                if content_block["type"] == "tool_use":
                    result = await tool_collection.run(
                        name=content_block["name"],
                        tool_input=content_block["input"],
                    )
                    tool_result_content.append(
                        _make_api_tool_result(result, content_block["id"])
                    )
                    tool_output_callback(result, content_block["id"])

        else:
            # Original Anthropic logic
            if provider == APIProvider.ANTHROPIC:
                client = Anthropic(api_key=api_key)
            elif provider == APIProvider.VERTEX:
                client = AnthropicVertex()
            elif provider == APIProvider.BEDROCK:
                client = AnthropicBedrock()

            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                system=system,
                tools=tool_collection.to_params(),
                betas=[BETA_FLAG],
            )

            api_response_callback(cast(APIResponse[BetaMessage], raw_response))
            response = raw_response.parse()

            messages.append({
                "role": "assistant",
                "content": cast(list[BetaContentBlockParam], response.content),
            })

            tool_result_content = []
            for content_block in cast(list[BetaContentBlock], response.content):
                print("CONTENT", content_block)
                output_callback(content_block)
                if content_block.type == "tool_use":
                    result = await tool_collection.run(
                        name=content_block.name,
                        tool_input=cast(dict[str, Any], content_block.input),
                    )
                    tool_result_content.append(
                        _make_api_tool_result(result, content_block.id)
                    )
                    tool_output_callback(result, content_block.id)

        if not tool_result_content:
            return messages

        messages.append({"content": tool_result_content, "role": "user"})


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
