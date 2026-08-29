from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
import json

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review"
    )

    summary: str = Field(
        description="A brief summary of the review"
    )

    sentiment: Literal["pos", "neg", "neutral"] = Field(
        description="Return sentiment as positive, negative, or neutral"
    )

    pros: Optional[list[str]] = None

    cons: Optional[list[str]] = None

    name: Optional[str] = None


prompt = """
Analyze the following review and return ONLY valid JSON.

The JSON must have exactly these fields:

{
    "key_themes": ["theme1", "theme2"],
    "summary": "summary",
    "sentiment": "pos",
    "pros": ["pro1", "pro2"],
    "cons": ["con1", "con2"],
    "name": "reviewer name"
}

Review:

I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say,
it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes
everything lightning fast—whether I’m gaming, multitasking, or editing
photos. The 5000mAh battery easily lasts a full day even with heavy use,
and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches,
though I don't use it often. What really blew me away is the 200MP camera—
the night mode is stunning, capturing crisp, vibrant images even in low
light. Zooming up to 100x actually works well for distant objects, but
anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed
use. Also, Samsung’s One UI still comes with bloatware. The $1,300 price
tag is also a hard pill to swallow.

Pros:
- Insanely powerful processor
- Stunning 200MP camera
- Long battery life with fast charging
- S-Pen support

Review by Nitish Singh
"""

result = model.invoke(prompt)

data = json.loads(result.content)

review = Review.model_validate(data)

print(review.name)