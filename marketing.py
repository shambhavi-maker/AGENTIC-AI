from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
load_dotenv()

class CampaignSpec(BaseModel):
    product_name: str
    target_audience: str
    campaign_type: str  # e.g., 'social_media', 'email'
    tone: str
    platform: str       # e.g., 'Instagram', 'LinkedIn'

class RawPost(BaseModel):
    content: str

class EnhancedPost(BaseModel):
    content: str
    hashtags: list[str]
    emoji_count: int

class VisualIdea(BaseModel):
    description: str
    recommended_assets: list[str]


# Agent 1: Raw Content Generator
content_generator = Agent(
    'openai:gpt-4o',
    output_type=RawPost,
    system_prompt="Generate an initial social media post for the product and audience."
)

# Agent 2: Content Enhancer
content_enhancer = Agent(
    'huggingface:mistral-7b-instruct',
    output_type=EnhancedPost,
    system_prompt="Enhance the post: add relevant hashtags and emojis for engagement."
)

# Agent 3: Visual Asset Recommender
visual_agent = Agent(
    'openai:gpt-4o',
    output_type=VisualIdea,
    system_prompt="Suggest visual ideas and assets to pair with the social media post."
)

#workflow logic
async def generate_social_campaign(spec: CampaignSpec):
    # Step 1: Generate base content
    raw = await content_generator.run(spec)
    
    # Step 2: Enhance it for engagement
    enhanced = await content_enhancer.run(raw.output)
    
    # Step 3: Suggest visuals
    visuals = await visual_agent.run(spec)

    return {
        "platform": spec.platform,
        "raw_post": raw.output,
        "enhanced_post": enhanced.output,
        "visuals": visuals.output
    }

test_dataset = Dataset[CampaignSpec, EnhancedPost](
    cases=[
        Case(
            name='fitness_campaign_instagram',
            inputs=CampaignSpec(
                product_name="SmartFit Tracker",
                target_audience="fitness enthusiasts",
                campaign_type="social_media",
                tone="energetic and inspiring",
                platform="Instagram"
            ),
            expected_output=None,
            evaluators=[
                LLMJudge(
                    rubric="Ensure the content is inspiring, includes relevant hashtags, and fits the Instagram tone.",
                )
            ],
        )
    ],
    evaluators=[
        LLMJudge(
            rubric="Content should clearly promote the product and include strong calls to action.",
            include_input=True,
        )
    ]
)

# Assume `generate_social_campaign` returns EnhancedPost
report = test_dataset.evaluate_sync(lambda spec: generate_social_campaign(spec)['enhanced_post'])
print(report)
