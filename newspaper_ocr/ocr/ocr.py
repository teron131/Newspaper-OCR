import os
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.cloud import documentai


def run_ocr(content: bytes) -> documentai.Document:
    """Run OCR on a document using Google Document AI with optional preprocess.

    Args:
        content (bytes): Content of the document

    Returns:
        documentai.Document: The processed document with OCR results
    """
    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")  # Format is "us" or "eu"
    processor_id = os.getenv("OCR_PROCESSOR_ID")  # Create processor before running sample
    processor_version = os.getenv("OCR_PROCESSOR_VERSION")  # Refer to https://cloud.google.com/document-ai/docs/manage-processor-versions for more information
    mime_type = "image/png"

    # Optional: Additional configurations for Document OCR Processor.
    # For more information: https://cloud.google.com/document-ai/docs/enterprise-document-ocr
    process_options = documentai.ProcessOptions(
        ocr_config=documentai.OcrConfig(
            enable_native_pdf_parsing=True,
            enable_image_quality_scores=True,
            enable_symbol=True,
            # OCR Add Ons https://cloud.google.com/document-ai/docs/ocr-add-ons
            premium_features=documentai.OcrConfig.PremiumFeatures(
                enable_selection_mark_detection=False,
                compute_style_info=False,
                enable_math_ocr=False,  # Enable to use Math OCR Model
            ),
        )
    )

    # Online processing request to Document AI
    document = _process_document(
        project_id=project_id,
        location=location,
        processor_id=processor_id,
        processor_version=processor_version,
        content=content,
        mime_type=mime_type,
        process_options=process_options,
    )

    return document


def _process_document(
    project_id: str,
    location: str,
    processor_id: str,
    processor_version: str,
    content: bytes,
    mime_type: str,
    process_options: Optional[documentai.ProcessOptions] = None,
) -> documentai.Document:
    """Process a document using Google Document AI.

    Args:
        project_id (str): Google Cloud project ID
        location (str): Location of the processor
        processor_id (str): ID of the processor
        processor_version (str): Version of the processor
        content (bytes): Content of the document
        mime_type (str): MIME type of the document
        process_options (Optional[documentai.ProcessOptions]): Additional processing options

    Returns:
        documentai.Document: The processed document
    """
    # You must set the `api_endpoint` if you use a location other than "us".
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com",
        )
    )

    # The full resource name of the processor version, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}`
    # You must create a processor before running.
    name = client.processor_version_path(project_id, location, processor_id, processor_version)

    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=content, mime_type=mime_type),
        process_options=process_options,
    )

    result = client.process_document(request=request)

    return result.document
