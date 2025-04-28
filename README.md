# Project Title
**BrandView AI – YouTube Influencer Intelligence Agent**

## Overview of the Idea
BrandView AI is a smart agent built with **Agno** that helps brands evaluate YouTube creators for influencer marketing. It analyzes channels and videos to help marketers find safe, high-performing influencers, faster and more accurately than doing the manual research.

## Project Goal
To build an intelligent agent that allows brands to instantly:
1. Understand a creator’s audience and past performance
2. Assess risk and sentiment around their content
3. Analyze how they handle sponsorships
4. Forecast the performance of the next video
5. Compare creators and optimize influencer choices

## How It Works
Marketers interact with BrandView AI using simple chat prompts. Behind the scenes, the agent (or a team of specialized sub-agents) fetches YouTube data, processes it using advanced tools (audio transcription, visual analysis, ML models, NLP), and delivers clear, actionable results.

Each request can trigger a sophisticated workflow involving data gathering, AI-powered analysis, and tailored output generation—all hidden behind a smooth conversational experience.

### Example User Journeys:

🧠 User: “What is the maximum price we should pay @GamerINF to keep CPM under $30?”<br>
💬 Agent:
- Fetches the influencer’s last 10–15 videos, calculates average and expected next views using ML prediction.
- Calculates the highest price to achieve target CPM.
- Response: <i>“Offer no more than $5,700 to maintain a CPM below $30, based on a projected 190K views.”</i>

🧠 User: “Show me which brands worked with @BeautyGuru over the last year.”<br>
💬 Agent:
- Analyzes video titles, descriptions, and transcripts using sponsorship-related keyword searches.
- Extracts and lists brand names + shows brief context of integrations.
- Response: <i>“Detected partnerships with L'Oréal, Sephora, and Glossier in 7 videos.”</i>

🧠 User: “Generate a detailed report on @UltraPlayer for my team.”<br>
💬 Agent:
- Collects profile data (subs, views, growth trends).
- Analyzes content type, audience sentiment, collaboration history, risk level (politics, scandals).
- Creates a full PDF report with visual charts and KPIs.
- Response: <i>"Here’s your downloadable report, including forecasted ROI estimates and brand safety scores."</i>

🧠 User: “Give me a deep analysis of the latest video from @TechNinja.”<br>
💬 Agent:
- Converts video audio to text (transcript).
- Detects brand mentions and timestamps sponsorship sections.
- Scores thumbnail attractiveness and analyzes engagement around the sponsor message.
- Response: <i>"Sponsor segment starts at 4:35, high viewer retention observed, thumbnail scored 8.4/10."</i>

### How It All Happens Behind the Scenes:
- Calls to YouTube API, transcript generation (ASR), thumbnail vision models, keyword analysis, ML forecasting, and report generation are orchestrated seamlessly.
- Different specialized sub-agents may work in parallel (e.g., transcript agent, sentiment agent, forecasting agent).
- The user experiences it all through one simple chat window — no technical skills needed.

### Core Functionality
- **Channel Health Snapshot** (growth, engagement, sentiment)
- **Brand‑Safety Risk Scan** (flag profanity, politics, scandals)
- **Sponsor Segment Locator** (timestamps + retention drop‑off)
- **90 % Confidence View Forecast** for next upload
- **ROI & CPM Estimator** (“What CPM do we hit if we pay $25K?”)
- **Competitive Benchmarking** (“Compare @TechGamer vs @UltraPlay”)

### Multimodal Elements
- **Text**: chat, transcripts, explanations  
- **Audio→Text**: ASR on video audio  
- **Image**: thumbnail scoring, brand‑logo detection  
- **Video**: sponsor‑segment timestamping  

## Tools Used
| **Purpose**              | **Tool / Library**                                                                 |
|---------------------------|------------------------------------------------------------------------------------|
| Agent Orchestration       | <img src="https://cdn.prod.website-files.com/6796d350b8c706e4533e7e32/6796d350b8c706e4533e8019_Favicon%20small.png" height="16" style="vertical-align:middle;"> **Agno** |
| Backend & API             | **Python**, **FastAPI**                                                            |
| Chat LLM                  | **OpenAI GPT-4o**                                                                  |
| Data Ingestion            | **YouTube Data API**, `yt-dlp`, <img src="https://firecrawl.dev/favicon.ico" height="16" style="vertical-align:middle;"> **Firecrawl** (web scraping for scandals and controversies) |
| Audio to Text             | **OpenAI Whisper (medium)**                                                        |
| Video Analysis            | **Video-LLaMA** (scene parsing, sponsor detection, visual understanding)           |
| Sentiment Analysis        | `textblob`                                                                         |
| Vision & Thumbnails       | **CLIP embeddings**, **OpenCV**                                                    |
| ML Forecasting            | **PyTorch** custom models with bootstrapped prediction intervals                   |
| Report Generation         | `jinja2` (HTML to PDF export)                                                      |
| Frontend (Demo)           | **Streamlit**                                                                      |
| Memory Layer              | <img src="https://mem0.ai/favicon.ico" height="16" style="vertical-align:middle;"> **Mem0** (store influencer profiles and risk assessments for faster, enriched follow-ups) |
| Knowledge Ingestion       | <img src="https://framerusercontent.com/images/KCOWBYLKunDff1Dr452y6EfjiU.png" height="16" style="vertical-align:middle;"> **Graphlit MCP Server** (ingest external documents like press releases and brand guidelines) |

## UI Approach
- **Primary Interface**: Streamlit web app where marketers can chat naturally with BrandView AI to request influencer analysis, forecasts, reports, and recommendations.  
- **Secondary Output**: Auto-generated downloadable PDF/slide deck reports summarizing influencer profiles, health metrics, sponsorship history, risk scores, and forecasts.

## Visuals
*(To be added)*

## Team Information
- **Team Lead**: @mmestrov2000
- **Team Members**: @Abhishek637Saraswat, @LukaMestrovic
- **Background/Experience**: Marin Mestrovic has a Bachelor's degree in Mathematics with strong skills in algorithms, data science, and machine learning. Abhishek Saraswat is a B.Tech student in Electronics Engineering with experience in AI, machine learning, backend development, and projects in stock analysis and speech recognition. Luka Mestrovic has a background in physics, statistics, and mathematics, combined with industry-specific insights in influencer marketing.

## Prize Category (leave blank, to be assigned by judges)
- [ ] Best use of Agno
- [ ] Best use of Firecrawl
- [ ] Best use of Mem0
- [ ] Best use of Graphlit
- [ ] Best use of Browser Use
- [ ] Best Overall Project

## Demo Video Link
*(To be added – 2‑3 min demo)*

## Additional Notes
- **Agent Collaboration**: While BrandView AI presents itself as a single chat agent, it operates as a coordinated system of specialized sub-agents handling different tasks like video processing, data fetching, forecasting, and reporting.
- **Confidence Intervals for Forecasts**: View prediction is not a simple point estimate — it uses bootstrapped simulations on top of a PyTorch model to generate robust 90% confidence intervals, offering brands realistic outcome ranges instead of misleading single predictions.
- **Flexible API Handling**: The agent is designed to dynamically adjust between API retrieval (YouTube Data API) and local scraping (yt-dlp) if necessary to maximize data access and minimize missing information.
- **Sponsorship Detection**: Brand collaborations are detected both through metadata (titles, descriptions) and deep content analysis (transcripts, visual sponsor cue detection using Video-LLaMA).
- **Risk Scoring Model**: Risk assessments blend NLP-based keyword spotting (profanity, politics, controversy) with sentiment analysis from recent comment sections to produce a reliable "Brand Safety Score" for each influencer.
- **Scalable Architecture**: Though the MVP focuses on YouTube, the agent's modular architecture allows easy extension to other platforms like TikTok, Twitch, and Instagram in the future.
- **Streamlit Interface Focus**: For the hackathon demo, Streamlit will be used for simplicity, but the backend is designed to easily plug into more advanced UIs or enterprise integrations if scaled post-hackathon.
- **Future Enhancements**: Planned future improvements include sponsor integration style classification (casual mention vs. dedicated segment) and dynamic audience overlap analysis between multiple influencers.