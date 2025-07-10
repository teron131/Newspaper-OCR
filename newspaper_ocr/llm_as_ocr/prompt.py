TEXT_EXTRACTION_PROMPT = """The given image shows a scanned newspaper page and the given text is the OCR text results for reference. With the holistic overview of the layout, extract and organize the content into a structured format following these guidelines:

1. Identify page metadata:
   - Page section letter, number, title, usually from the top left corner (e.g. A15 影音生活)
   - Publication date, usually from the top right corner (e.g. 20XX年X月XX日)
   - The names of the author and photographer if present (e.g. 記者 XXX, 撰文: XXX, 攝影 XXX, XXX 攝)

2. Extract main content:
   - Preserve and identify all titles or headers and make them bolded with double asterisks (e.g. **titles**)
   - Organize text into coherent paragraphs
   - Maintain the logical flow and hierarchy of content
   - Seperate each paragraph with an empty newline (i.e. **H1**\nXXX\n\nYYY\n\n**H2**\nZZZ)

3. Identify any tables:
    - Include table captions if present
   - Extract table headers / captions and content
   - Format as CSV in a single string (e.g. "Header1,Header2,Header3\nValue1,Value2,Value3\n...")

4. Identify any images:
   - Describe the location of each image on the page
   - Include image captions if present"""


IMAGE_DESCRIPTION_PROMPT = """The input consists of a scanned newspaper page and the text content of the page. Analyze and describe each image or infographic on the page with comprehensive detail in TRADITIONAL CHINESE following these guidelines:

1. Describe in TRADITIONAL CHINESE! With no less than 100 words for each image

2. Holistically analyze the page layout
   - Identify all photos, images, charts, graphs, and infographics inside the page
   - Ensure each unique image is described only once
   - Do not describe the entire page as an image

3. For each image:
   - Contextual description with reference to the text content of the page that could be relevant to the image
   - Mention the specific names of the people, objects, locations
   - The exact caption or text associated with the image if present

4. For each infographic:
   - Contextual description with reference to the text content of the page that could be relevant to the infographic
   - A detailed description of the infographic elements such as charts, graphs, data visualizations, colors, text overlays, legends, axes, etc
   - Mention the specific data points, trends, comparisons, or statistical information presented
   - The insights and key takeaways from the infographic
   - The exact caption or text associated with the infographic if present"""


CHECKER_PROMPT = """Be a rigorous and strict checker and grade whether the following criteria are met on a scale of 0 to 10. 10 means perfect and 0 means the criteron is completely ignored. Approximately the score should be the percentage of the criteron has been met. If the criteria are not met, provide the reasons of judgement.

The given image shows a scanned newspaper page and the completed organized result of the text extraction and image description. Thoroughly verify the result, identify any discrepancies, and correct all mistakes to ensure complete accuracy. If the fields are not present, they can be `None` or empty.  

Verification Checklist:
1. Page Metadata:
   - Verify the page section letter, number, and title are correctly extracted from the top left corner
   - Confirm the publication date is accurate and in the format of YYYY-MM-DD or DD-MM-YYYY

2. Text Content:
   - Ensure all text content from the newspaper page is completely extracted in the entire page
   - Confirm text is organized into coherent paragraphs with proper flow
   - Check that all titles or headers are properly identified and formatted with double asterisks (e.g., **titles**)
   - Verify paragraphs are properly separated with empty newlines (i.e., **H1**\nXXX\n\nYYY\n\n**H2**\nZZZ)
   - Ensure no text content is missing, especially from complex layout areas

3. Tables:
   - Verify all tables from the page are extracted completely (could be empty)
   - Check that table captions are correctly identified
   - Confirm table content is properly formatted in CSV format
   - Check that the title or caption is correctly extracted

4. Images:
   - Verify all images on the page are identified and described (could be empty)
   - Ensure image descriptions do not include table content
   - Confirm image descriptions are detailed, contextually relevant, and in Traditional Chinese
   - Verify descriptions provide meaningful insights about the image content
   - Check that the title or caption is correctly extracted."""
