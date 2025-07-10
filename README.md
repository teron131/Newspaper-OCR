# Newspaper-OCR

A robust solution for digitalizing scanned newspaper copies into structured data for RAG (Retrieval Augmented Generation) applications. This tool extracts text, tables, infographics, and images from newspaper scans using a combination of Google Document AI and LLM-as-OCR techniques.

## Features

- **Text Extraction**: High-quality extraction of text with various styles, including text within images.
- **Form/Table Extraction**: Conversion of tabular data into structured CSV format.
- **Image Description**: AI-powered description of images and infographics within newspapers.
- **Preprocessing Pipeline**: Built-in image preprocessing including deskewing and white balance adjustment.
- **Quality Verification**: Automated verification and correction of extracted content using LLM.
- **Structured Output**: Results delivered as Pydantic objects for easy integration.

## LLM-as-OCR vs OCR

| Aspect         | LLM-as-OCR                                                                                                                                                                                                                                                                                       | OCR                                                                                                                                                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Strengths**  | • Good at parsing forms<br>• Simultaneously understands the context and fixes mixing up<br>• Able to describe images<br>• Self-correction capabilities<br>• Potential to swap to next-gen models for better results<br>• Potential to extract images (positions) (failed)<br>• Acts as the brain | • Kind of deterministic<br>• Lower latency<br>• Could parse forms by specialized processor, but format could be wrong<br>• PaddleOCR can parse forms and images, but the overall quality / accuracy is nowhere near Google<br>• Acts as the brawn |
| **Weaknesses** | • Varies as the prompt<br>• Prone to hallucination                                                                                                                                                                                                                                               | • Format could be wrong<br>• Impossible to self-correct, structured outputs to different items, unlike LLMs<br>• No image extraction found in Google Document AI for now                                                                          |

This hybrid approach leverages the strengths of both technologies to produce superior results.

Both Google Document AI and LLM services used in this project are very cost-effective:

- Document AI: ~$1.5 per 1,000 pages
- Gemini 2.0 Flash: ~\$0.1 per 1M tokens input and \$0.4 per 1M tokens output

## Architecture

The system combines traditional OCR with LLM capabilities to achieve superior results:

![Architecture Diagram](graph.png)

### Components

1. **Preprocessing** ([newspaper_ocr/preprocess/](newspaper_ocr/preprocess/)):

   - Deskewing, white balance adjustment, clarity enhancement
   - Optional contrast/brightness/denoising capabilities

2. **OCR** ([newspaper_ocr/ocr/](newspaper_ocr/ocr/)):

   - Google Document AI integration
   - Text extraction with detailed metadata

3. **LLM-as-OCR** ([newspaper_ocr/llm_as_ocr/](newspaper_ocr/llm_as_ocr/)):

   - Prompts, schemas, and graph implementation
   - Text extraction with layout understanding
   - Table extraction and formatting
   - Image description generation
   - Quality verification and correction

4. **Utilities** ([newspaper_ocr/utils.py](newspaper_ocr/utils.py)):
   - Helper functions for file handling
   - Visualization utilities
   - Document processing tools

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Newspaper-OCR.git
cd Newspaper-OCR

# Install dependencies
pip install -r requirements.txt

# Setup your environment variables
cp .env.example .env
# Edit .env with your API keys and configuration

# Setup Google Cloud credentials
```

### Requirements

- Google Cloud credentials and setup(for Document AI)
- API key for Gemini or other LLM service
- Dependencies: Pillow, OpenCV, SciPy, Google Document AI, LangChain, LangGraph, Pydantic, ...

## Usage

```python
from newspaper_ocr.llm_as_ocr.graph import run_graph

image_path = "path_to_image"
result = run_graph(image_path)
```

```python
from newspaper_ocr.llm_as_ocr.graph import batch_run_graph

image_paths = ["path_to_image1", "path_to_image2", ...]
results = batch_run_graph(image_paths)
```

## Example Output

```bash
🚀 Start processing: data/images/baseline.jpg
🖼️ Image Preprocessing complete: data/images/baseline.jpg
🔄 Format Conversion complete: data/images/baseline.jpg
🔡 OCR complete: data/images/baseline.jpg
🧠 LLM Text Extraction complete: data/images/baseline.jpg
🧠 LLM Image Description complete: data/images/baseline.jpg
🔗 Results concatenation complete: data/images/baseline.jpg
🔍 Criteria Checker 0 complete: data/images/baseline.jpg
📝 Corrector 1 complete: data/images/baseline.jpg
🔍 Criteria Checker 1 complete: data/images/baseline.jpg
📝 Corrector 2 complete: data/images/baseline.jpg
🔍 Criteria Checker 2 complete: data/images/baseline.jpg
📝 Corrector 3 complete: data/images/baseline.jpg
🎉 Process complete: data/images/baseline.jpg

NewspaperPage(
    page_section_letter='A',
    page_section_number=2,
    page_section_title='要聞',
    published_date=datetime.date(2025, 2, 18),
    author=None,
    photographer=None,
    content='**內地血濃於水跨境捐心
8月大祈祈成功移植續命**\n內地與香港血濃於水,內地再有器官捐贈給港人續命。心臟衰竭僅剩2成功能、現時8個月大女嬰祈祈,獲
內地跨境器官捐贈,本週日晚完成心臟移植手術,是本港第二宗獲內地捐贈器官個案。醫管局形容手術效果良好,祈祈「好叻女」。\n
\n醫管局昨午召開記者會公佈,上週六接獲內地通知有合適移植的兒童心臟。至本週日上午11時55分,內地醫療團隊開始進行心臟獲
取手術,器官接收程序於下午1時32分在兩地邊境管制站完成,相關心臟於下午2時10分抵達香港兒童醫院,期間有18個內地部門協調行
動,較原本預計提早約1小時送達。移植手術隨即於2時14分展開,至晚上9時完成。\n\n醫管局冀跨境器官移植恆常化\n\n院方指,祈
祈過往病情反覆,未必能在香港尋找合適心臟,今次是她第三次開腔手術,有一定困難和風險,包括需要拆除原本安裝的雙心室輔助器,
將移植的心臟缺血時間減至最短,最終手術效果良好,心臟功能運作理想,順利恢復供血,術後出血量少,醫護人員正持續監察傷口狀況
,以及心臟和其他器官的配合。\n\n醫管局聯網服務總監鄧耀鏗説,自2022年底首宗跨境器官移植個案後,內地與香港建立密切溝通渠
道和流程,完善每個環節,跨境運送器官流程相當複雜,需要處理好當中的細節,縮短將器官運送到港的時間,控制器官缺血情況。未來
將繼續檢視流程,希望為未來制訂跨境器官移植恆常化建立基礎。\n\n女嬰的父親張先生感謝各部門、醫護團隊和捐贈者家屬,形容
女兒獲心續命,是一個奇蹟,這份愛和祝福將成為女兒日後繼續康復的力量。\n\n**北京民營企業座談會
阿里騰訊DeepSeek等參與**\n\n**習近平:要堅決破除公平市場競爭的障礙**\n\n中美貿易戰開打,兩國亦在科技領域激烈競爭。國
家主席習近平昨早在北京會見阿里巴巴、騰訊、小米、DeepSeek、美團、比亞迪等多間巨型科技的創辦人。習近平強調,要堅決破除
公平市場競爭的障礙,亦要大力解決民營企業融資難融資貴的問題,同時強化執法監督,集中整治亂收費、亂罰款、亂檢查、亂查封,
依法保護民營企業合法權益。有經濟學者認為,中國未來將為科企創造更多交流平台,令跨行業可以實施聯動,提高國際競爭力,隨後
應會有更多政策支持,促進民企發展。\n\n中共中央總書記、國家主席、中央軍委主席習近平昨日上午在北京出席民營企業座談會並
發表重要講話,習近平説,新時代新徵程民營經濟發展前景廣闊、大有可為,廣大民營企業和民營企業家大顯身手正當其時,希望他們
胸懷報國志、一心謀發展、守法善經營、先富促共富,為推進中國式現代化作出新的更大的貢獻。\n\n**任正非雷軍等6代表發言**\
n\n習近平強調,當前民營經濟發展面臨一些困難和挑戰,是改革發展、產業轉型升級過程中出現,是局部而不是整體,是暫時而不是長
期,是能夠克服而不是無解。他指出,凡是黨中央定的就要堅決執行,不能打折扣。要堅決破除公平市場競爭的障礙,大力解決民營企
業融資難融資貴問題,解決拖欠民營企業帳款問題。要強化執法監督,集中整治亂收費、亂罰款、亂檢查、亂查封,依法保護民營企業
合法權益。\n\n**學者:跨行業可實施聯動**\n\n中共中央政治局常委、國務院總理李強,中共中央政治局常委、國務院副總理丁薛
祥來美國再打壓中國科企,相信出席座談會。中共中央政治局 國家會直接出面。」他還指,常委、全國政協主席王滬寧主
相信未來國家不僅將對民企科持座談會。座談會上,華為首 企,提供政策支持,還將會為席執行官任正非、比亞迪董事
他們提供交流的平台,令企業長王傳福、新希望董事長劉永 不再是單打獨鬥而是強強聯好、上海韋爾半導體董事長虞
動,有助於企業提高國際競爭仁榮、杭州宇樹科技首席執行 力。官王興興、小米董事長雷軍等\n\n6位民營企業負責人代表先後
上週市場傳出這次民企發言。\n\n浸大會計、經濟及金融學系副教授麥萃才對本報説,這次的座談會顯示國家重視科技行業,農曆新
年DeepSeek的出現,令世界各地的科企都關注中國的人工智能產業,「若未\n\n座談會的消息,一度引起科技股上升。昨天恒指收市微
跌4點,收報22616;科指曾升2.4%,高見5656,創2022年2月18日後近三年高位。個股方面,騰訊升4%;小米升1%;美團跌0.6%;阿里巴巴跌
1.5%;比亞迪跌2.5%。',
    tables=[
        TableContent(
            csv_string='與會民企名單
(部分)\n阿里巴巴創辦人馬雲,騰訊創辦人馬化騰,小米集團董事長雷軍\n華為創辦人任正非,DeepSeek創辦人梁文鋒,美團首席執行
官王興\n,比亞迪董事長王傳福,寧德時代董事長曾毓羣\n,上海韋爾半導體董事長虞仁榮,宇樹科技創始人王興興,恆力集團董事長陳
建華\n,中國飛鶴董事長冷友斌,新希望董事長劉永好,銀河航天創始人徐鳴\n,正泰電器董事長南存輝,寧夏共享董事長彭凡,中磁科
技董事長董清飛',
            caption='None'
        )
    ],
    images=[
        ImageContent(
            description='一名躺在牀上的嬰兒，正在接受醫護人員的照顧。圖片上方文字描述了「醫管局形容手術效果良好，祈
祈「好叻女」」，下方標示「資料圖片」。這張圖片與文章主題「內地血濃於水跨境捐心
8月大祈祈成功移植續命」直接相關，呈現了接受心臟移植手術後的嬰兒祈祈的康復情況。',
            caption='醫管局形容手術效果良好，祈祈「好叻女」。資料圖片'
        ),
        ImageContent(
            description='這是一張集體合照，顯示了中國國家主席習近平與多位中國民營企業家的合影。照片中，習近平站在中
間，周圍是參與座談會的企業家代表。從人物的肢體語言和表情來看，氣氛友好且融洽。圖片下方文字提及「習近平會後與多名出
席的民企巨頭握手，包括DeepSeek創辦人梁文鋒」。這張圖片與文章主題「北京民營企業座談會
阿里騰訊DeepSeek等參與」和「習近平：要堅決破除公平市場競爭的障礙」直接相關，反映了座談會的場景以及參與者。',
            caption='習近平會後與多名出席的民企巨頭握手，包括DeepSeek創辦人梁文鋒。'
        ),
        ImageContent(
            description='這張照片捕捉了北京民營企業座談會的一個場景。圖中可見多位與會者正在鼓掌，包括中國國家主席習
近平，以及阿里巴巴創辦人馬雲。這張照片與主要文章「北京民營企業座談會，阿里騰訊DeepSeek等參與」相關，突出了會議中與
會者對發言的反應。',
            caption='■習近平在民營企業座談會上表示,要依法保護民營企業合法權益。'
        ),
        ImageContent(
            description='一位女士在「要聞焦點」的標題下接受採訪，背景中有一個帶有「P4」字樣和箭頭的標誌，可能表示更
詳細的內容在第四頁。女士面帶微笑，似乎正在講述與「啟德屯門觀塘上水5060夥「簡約公屋」2期2.24起申請」相關的內容。',
            caption='要聞焦點'
        )
    ]
)
```

## Next Steps

- Prompts are not perfect. Switching to better models instantly get better results though.
- Sometimes the texts are still broken throughout different paragraphs.
- Image extraction is not archieved.
- RAG system. It can range from simple to very complex, likely starting with ChromaDB, OpenAI embedding models, LangChain retrievers, ensembled retrievers, LangGraph for agentic RAG with checking relevance, hallucination, etc.
- Scaled application.
  Knowledge graphs, with the metadata, entity extraction on the main content, could be useful to link the newspaper sections across time.
