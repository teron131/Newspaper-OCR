import os
from multiprocessing import Pool, cpu_count
from typing import Optional

from dotenv import load_dotenv
from google.cloud import documentai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy
from PIL import Image
from pydantic import BaseModel, Field
from rich import print
from tqdm import tqdm

from ..ocr import run_ocr
from ..preprocess import (
    PreprocessOptions,
    image_to_base64,
    image_to_bytes,
    load_preprocess_image,
)
from .llm import run_llm
from .prompt import CHECKER_PROMPT, IMAGE_DESCRIPTION_PROMPT, TEXT_EXTRACTION_PROMPT
from .schema import (
    CRITERIA_TO_FIELDS,
    Criteria,
    ImageContentList,
    NewspaperPage,
    NewspaperText,
)

load_dotenv()

# Previously, use Google Gemini directly instead of through OpenRouter due to some errors in validating structured output with nested Pydantic schemas.
# Now, use Google Gemini for auto resizing the image (base64). Otherwise, it is required to resize the image or it will exceed the token limit.
llm_ocr = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_OCR"),
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY"),
)
llm_checker = ChatOpenAI(
    model=os.getenv("LLM_CHECKER"),
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MAX_CORRECTION = 3
CRITERIA_PERCENTAGE = 99
CRITERIA_THRESHOLD = 7


class GraphState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}  # For PIL.Image
    image_path: str
    preprocess: bool = Field(default=True)
    preprocess_options: PreprocessOptions = Field(default_factory=PreprocessOptions)
    image: Optional[Image.Image] = Field(default=None, exclude=True)  # Exclude from serialization
    image_bytes: Optional[bytes] = Field(default=None, exclude=True)  # Exclude from serialization
    image_base64: Optional[str] = Field(default=None)
    ocr_result: Optional[documentai.Document] = Field(default=None, exclude=True)  # Exclude from serialization
    text_extraction_result: Optional[NewspaperText] = Field(default=None)
    image_description_result: Optional[ImageContentList] = Field(default=None)
    newspaper_page_result: Optional[NewspaperPage] = Field(default=None)
    criteria: Optional[Criteria] = Field(default=None)
    correction_attemps: int = Field(default=0)


def image_preprocessing(state: GraphState) -> dict[str, Image.Image]:
    """Load and preprocess the image."""
    image = load_preprocess_image(
        state.image_path,
        preprocess=state.preprocess,
        preprocess_options=state.preprocess_options,
    )
    # display_resize(image)
    print(f"🖼️ Image Preprocessing complete: {state.image_path}")
    return {"image": image}


def format_conversion(state: GraphState) -> dict[str, bytes | str]:
    """Convert image to bytes and base64."""
    image_bytes = image_to_bytes(state.image)
    image_base64 = image_to_base64(state.image)
    print(f"🔄 Format Conversion complete: {state.image_path}")
    return {"image_bytes": image_bytes, "image_base64": image_base64}


def ocr_text_extraction(state: GraphState) -> dict[str, documentai.Document]:
    """Run OCR on the image."""
    ocr_result = run_ocr(state.image_bytes)
    print(f"🔡 OCR complete: {state.image_path}")
    return {"ocr_result": ocr_result}


def llm_text_extraction(state: GraphState) -> dict[str, NewspaperText]:
    """Run LLM for text extraction."""
    text_extraction_result = run_llm(
        llm=llm_ocr,
        prompt=TEXT_EXTRACTION_PROMPT,
        reference_image=state.image,
        reference_text=state.ocr_result.text,
        schema=NewspaperText,
    )
    print(f"🧠 LLM Text Extraction complete: {state.image_path}")
    return {"text_extraction_result": text_extraction_result}


def llm_image_description(state: GraphState) -> dict[str, ImageContentList]:
    """Run LLM for image description."""
    image_description_result = run_llm(
        llm=llm_ocr,
        prompt=IMAGE_DESCRIPTION_PROMPT,
        reference_image=state.image,
        reference_text=state.text_extraction_result.content,
        schema=ImageContentList,
    )
    print(f"🧠 LLM Image description complete: {state.image_path}")
    return {"image_description_result": image_description_result}


def result_concatenation(state: GraphState) -> dict[str, NewspaperPage]:
    """Concatenate the results."""
    newspaper_page_result = NewspaperPage(
        **state.text_extraction_result.model_dump(),
        images=state.image_description_result.images,
    )
    print(f"🔗 Results Concatenation complete: {state.image_path}")
    return {"newspaper_page_result": newspaper_page_result}


def criteria_checker(state: GraphState) -> dict[str, Criteria]:
    """Check the criteria."""
    print(state.newspaper_page_result)
    result_string = state.newspaper_page_result.as_str
    criteria = run_llm(
        llm=llm_checker,
        prompt=CHECKER_PROMPT,
        reference_image=state.image,
        reference_text=result_string,
        schema=Criteria,
    )
    print(criteria)
    print(f"🔍 Criteria Checker {state.correction_attemps} complete: {state.image_path}")
    return {"criteria": criteria}


def is_criterion(key: str, value: any) -> bool:
    return key != "reasons" and isinstance(value, int)


def corrector(state: GraphState) -> dict[str, NewspaperPage | int]:
    """Correct the results."""
    if state.correction_attemps >= MAX_CORRECTION:
        return {
            "newspaper_page_result": state.newspaper_page_result,
            "correction_attemps": state.correction_attemps,
        }

    # Determine which fields need correction
    fields_to_correct = []
    if state.criteria:
        criteria_dict = state.criteria.model_dump()
        for criterion, score in criteria_dict.items():
            if is_criterion(criterion, score) and score < CRITERIA_THRESHOLD and criterion in CRITERIA_TO_FIELDS:
                fields_to_correct.extend(CRITERIA_TO_FIELDS[criterion])
        # Remove duplicates while preserving order
        fields_to_correct = list(dict.fromkeys(fields_to_correct))

    newspaper_page_result = state.newspaper_page_result.model_copy()

    # If fields need correction, run the LLM to fix them
    if fields_to_correct:
        field_descriptions = []
        for criterion, score in criteria_dict.items():
            if is_criterion(criterion, score) and score < CRITERIA_THRESHOLD:
                description = state.criteria.__class__.model_fields[criterion].description
                field_descriptions.append(f"{description} was found to be inadequate (score: {score}/{CRITERIA_THRESHOLD})")

        # Create targeted correction instructions
        instructions = "\n".join(
            [
                "\n".join(field_descriptions),
                f"Reasons for correction: {state.criteria.reasons if state.criteria else ''}",
            ]
        )

        # Prepare the current result as reference
        result_string = state.newspaper_page_result.as_str

        # Get corrected result
        corrected_result = run_llm(
            llm=llm_ocr,
            prompt=instructions,
            reference_image=state.image,
            reference_text=result_string,
            schema=NewspaperPage,
        )

        # Only update the fields that needed correction
        for field in fields_to_correct:
            if hasattr(corrected_result, field):
                setattr(newspaper_page_result, field, getattr(corrected_result, field))

        correction_attemps = state.correction_attemps + 1
    else:
        # Skip further correction attempts if no fields need correction
        correction_attemps = MAX_CORRECTION + 1

    print(f"📝 Corrector {correction_attemps} complete: {state.image_path}")

    return {
        "newspaper_page_result": newspaper_page_result,
        "correction_attemps": correction_attemps,
    }


def should_continue(state: GraphState) -> str:
    """Check the criteria."""
    # Return the last result if the correction attempts exceed the maximum
    if state.correction_attemps >= MAX_CORRECTION:
        return "valid"

    # Calculate percentage of criteria met using integer scores instead of booleans
    criteria_dict = state.criteria.model_dump()
    valid_scores = [1 if (is_criterion(criterion, score) and score >= CRITERIA_THRESHOLD) else 0 for criterion, score in criteria_dict.items()]

    percentage_met = sum(valid_scores) / len(valid_scores) * 100
    is_valid = percentage_met > CRITERIA_PERCENTAGE

    print(f"✅ Criteria check: {percentage_met:.1f}% met, valid={is_valid}: {state.image_path}")

    return "valid" if is_valid else "invalid"


# Create the graph
builder = StateGraph(GraphState)

builder.add_node("image_preprocessing", image_preprocessing)
builder.add_node("format_conversion", format_conversion)
builder.add_node("ocr_text_extraction", ocr_text_extraction)
builder.add_node("llm_text_extraction", llm_text_extraction, retry=RetryPolicy(max_attempts=3))
builder.add_node("llm_image_description", llm_image_description, retry=RetryPolicy(max_attempts=3))
builder.add_node("result_concatenation", result_concatenation)
builder.add_node("criteria_checker", criteria_checker, retry=RetryPolicy(max_attempts=3))
builder.add_node("corrector", corrector, retry=RetryPolicy(max_attempts=3))

builder.add_edge(START, "image_preprocessing")
builder.add_edge("image_preprocessing", "format_conversion")
builder.add_edge("format_conversion", "ocr_text_extraction")
builder.add_edge("ocr_text_extraction", "llm_text_extraction")
builder.add_edge("llm_text_extraction", "llm_image_description")
builder.add_edge("llm_image_description", "result_concatenation")
builder.add_edge("result_concatenation", "criteria_checker")
builder.add_conditional_edges(
    "criteria_checker",
    should_continue,
    {
        "valid": END,
        "invalid": "corrector",
    },
)
builder.add_edge("corrector", "criteria_checker")

graph = builder.compile()


# Save graph visualization directly to file
# But it is often buggy
if "graph.png" not in os.listdir():
    try:
        graph_image = graph.get_graph().draw_mermaid_png(
            output_file_path="graph.png",
        )
    except Exception as e:
        print(f"Error generating graph image: {e}")
        graph_image = None


def run_graph(image_path: str) -> dict:
    """Run the graph."""
    print(f"🚀 Start processing: {image_path}")
    result = graph.invoke({"image_path": image_path})
    print(f"🎉 Process complete: {image_path}")
    newspaper_page_result: NewspaperPage = result["newspaper_page_result"]
    return newspaper_page_result.model_dump()


def batch_run_graph(image_paths: list[str]) -> list[dict]:
    """Run the graph for a batch of images using multiprocessing."""

    # Create a pool of workers
    with Pool(processes=cpu_count()) as pool:
        # Map the run_graph function to each image path with progress bar
        results = list(tqdm(pool.imap(run_graph, image_paths), total=len(image_paths), desc="Processing images"))

    return results
