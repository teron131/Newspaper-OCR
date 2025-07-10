from datetime import date
from typing import Generator, Literal, Optional

import opencc
from pydantic import BaseModel, Field, field_validator


def s2hk(v: Optional[str]) -> Optional[str]:
    """Convert text to traditional Chinese (Hong Kong standard) if not None.

    Args:
        v (Optional[str]): Text to convert.

    Returns:
        Optional[str]: Converted text or None.
    """
    if v is not None:
        converter = opencc.OpenCC("s2hk")
        return converter.convert(v)
    return v


class TableContent(BaseModel):
    csv_string: str = Field(description="The content of the table in csv format")
    caption: Optional[str] = Field(default=None, description="The caption or description of the table")

    @field_validator("csv_string", "caption", mode="after")
    @classmethod
    def s2hk_validator(cls, v: Optional[str]) -> Optional[str]:
        return s2hk(v)


class ImageContent(BaseModel):
    description: str = Field(description="Detailed description the single image based on the context of the page")
    caption: Optional[str] = Field(default=None, description="The caption or title of the image")

    @field_validator("description", "caption", mode="after")
    @classmethod
    def s2hk_validator(cls, v: Optional[str]) -> Optional[str]:
        return s2hk(v)


# Dummy class for structured output
class ImageContentList(BaseModel):
    images: list[ImageContent] = Field(default_factory=list, description="List of images found on the page")


class NewspaperText(BaseModel):
    # Page metadata
    page_section_letter: Literal["A", "B", "C", "D", "E"] = Field(description="The page section letter (A-E), usually on the top left corner")
    page_section_number: int = Field(description="The page section number, usually on the top left corner", ge=0, le=100)
    page_section_title: str = Field(description="The title of the page section, usually on the top left corner behind the page section letter and number")
    published_date: date | str = Field(description="The publication date of the newspaper")
    author: Optional[str] = Field(default=None, description="The name of the author of the page")
    photographer: Optional[str] = Field(default=None, description="The name of the photographer of the page")

    # Main content
    content: str = Field(
        description="The text content of the newspaper page including the titles / headers, excluding the metadata, organized into coherent paragraphs, where the titles are bolded, seperated by newlines '\n\n'"
    )
    tables: list[TableContent] = Field(default_factory=list, description="List of tables found on the page")

    @field_validator("published_date", mode="before")
    @classmethod
    def parse_date_string(cls, v: any) -> date | str:
        """
        Convert date string to date object if needed.

        Args:
            v: The date string to convert

        Returns:
            The date object
        """
        # Chinese date format
        if isinstance(v, str) and "年" in v and "月" in v and "日" in v:
            try:
                year = int(v.split("年")[0])
                month = int(v.split("年")[1].split("月")[0])
                day = int(v.split("月")[1].split("日")[0])
                return date(year, month, day)
            except ValueError:
                pass

        # Handle slash-separated date format
        elif isinstance(v, str) and ("/" in v or "-" in v):
            parts = v.split("/") if "/" in v else v.split("-")
            try:
                if len(parts) == 3:
                    if len(parts[0]) == 4:
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    else:
                        if int(parts[1]) > 12:
                            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
                        else:
                            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                    return date(year, month, day)
            except ValueError:
                pass

        return v

    @field_validator("page_section_title", "author", "photographer", "content", mode="after")
    @classmethod
    def s2hk_validator(cls, v: Optional[str]) -> Optional[str]:
        return s2hk(v)

    @field_validator("content", mode="after")
    @classmethod
    def format_content(cls, content: str) -> str:
        def concatenate_sentences(text: str) -> Generator[str, None, None]:
            ending_symbols = (".", "。", "!", "?", ":", ";", "」", "』", "）", ")", "》", '"', "'", "*")
            current_sentence = ""
            for line in text.splitlines():
                if not line.strip():
                    if current_sentence:
                        yield current_sentence
                        current_sentence = ""
                    yield ""
                    continue
                if not current_sentence:
                    current_sentence = line
                elif current_sentence[-1] in ending_symbols or line.startswith("**"):
                    yield current_sentence
                    current_sentence = line
                else:
                    current_sentence += line
            if current_sentence:
                yield current_sentence

        concatenated_sentences = list(concatenate_sentences(content))
        return "\n".join(concatenated_sentences)


# Inherit the NewspaperText class and concatenate with the images list
class NewspaperPage(NewspaperText):
    images: list[ImageContent] = Field(default_factory=list, description="List of images found on the page")

    @property
    def as_str(self) -> str:
        """Aggregate the result into a single string."""
        output = [
            "=====METADATA=====",
            f"Page Section Letter: {self.page_section_letter}",
            f"Page Section Number: {self.page_section_number}",
            f"Page Section Title: {self.page_section_title}",
            f"Published Date: {self.published_date}",
            f"Author: {self.author}",
            f"Photographer: {self.photographer}",
            "",
            "=====CONTENT=====",
            f"Content: {self.content}",
        ]

        output.append("")
        output.append("=====TABLES=====")
        for i, table in enumerate(self.tables):
            output.append(f"Table {i+1} Content:\n{table.csv_string}")
            output.append(f"Table {i+1} Caption: {table.caption}")

        output.append("")
        output.append("=====IMAGES=====")
        for i, image in enumerate(self.images):
            output.append(f"Image {i+1} Description: {image.description}")
            output.append(f"Image {i+1} Caption: {image.caption}")

        return "\n".join(output)


class Criteria(BaseModel):
    """Criteria for the results."""

    # Page Metadata criteria
    page_section_letter: int = Field(description="Verify the page section letter is correctly extracted from the top left corner", ge=0, le=10)
    page_section_number: int = Field(description="Verify the page section number is correctly extracted from the top left corner", ge=0, le=10)
    page_section_title: int = Field(description="Verify the page section title is correctly extracted from the top left corner", ge=0, le=10)

    # Date criteria
    published_date: int = Field(description="Confirm the publication date is accurate and in the format of YYYY-MM-DD or DD-MM-YYYY", ge=0, le=10)

    # Attribution criteria
    # author: int = Field(description="Check that the author name are people names (could be empty)", ge=0, le=10)
    # photographer: int = Field(description="Check that the photographer name are people names (could be empty)", ge=0, le=10)

    # Text Content criteria
    text_headers: int = Field(description="Check that all titles or headers are properly identified and formatted with double asterisks (e.g., **Title**)", ge=0, le=10)
    text_content_completeness: int = Field(description="Ensure all text content from the newspaper page is completely extracted without missing any paragraphs or sections", ge=0, le=10)
    text_content_accuracy: int = Field(description="Ensure the text content is free from typos, character recognition errors, or mistranslations", ge=0, le=10)
    text_content_flow: int = Field(description="Confirm text is organized into coherent paragraphs with proper flow", ge=0, le=10)
    text_formatting: int = Field(description="Verify that special formatting like bullet points, numbering, and indentation is preserved appropriately", ge=0, le=10)

    # Table criteria
    tables_included: int = Field(description="Verify all tables from the page are extracted completely without missing any rows or columns (could be empty)", ge=0, le=10)
    tables_structure: int = Field(description="Check that table structure (rows and columns) is correctly preserved in the extraction (could be empty)", ge=0, le=10)
    tables_csv_format: int = Field(description="Confirm table content is properly formatted in CSV format with correct delimiters and escaping (could be empty)", ge=0, le=10)
    tables_caption: int = Field(description="Check that table captions are correctly extracted and associated with the right tables (could be empty)", ge=0, le=10)
    tables_no_extra: int = Field(description="Ensure there is no extra irrelevant content or text in the tables (could be empty)", ge=0, le=10)

    # Image criteria
    images_included: int = Field(description="Verify all images on the page are identified and described without including non-existent ones (could be empty)", ge=0, le=10)
    images_caption: int = Field(description="Check that image captions are correctly extracted and associated with the right images (could be empty)", ge=0, le=10)
    images_description: int = Field(description="Confirm image descriptions are detailed, contextually relevant, and in Traditional Chinese (could be empty)", ge=0, le=10)
    images_no_extra: int = Field(description="Ensure image descriptions do not include table content or any extra irrelevant details (could be empty)", ge=0, le=10)

    # Overall Assessment
    reasons: str = Field(description="Provide detailed reasons for any criteria not met, with specific examples of errors or omissions")


CRITERIA_TO_FIELDS: dict[str, list[str]] = {
    "page_section_letter": ["page_section_letter"],
    "page_section_number": ["page_section_number"],
    "page_section_title": ["page_section_title"],
    "published_date": ["published_date"],
    "author": ["author"],
    "photographer": ["photographer"],
    "text_headers": ["content"],
    "text_content_completeness": ["content"],
    "text_content_accuracy": ["content"],
    "text_content_flow": ["content"],
    "text_formatting": ["content"],
    "tables_included": ["tables"],
    "tables_structure": ["tables"],
    "tables_csv_format": ["tables"],
    "tables_caption": ["tables"],
    "tables_no_extra": ["tables"],
    "images_included": ["images"],
    "images_caption": ["images"],
    "images_description": ["images"],
    "images_no_extra": ["images"],
}
