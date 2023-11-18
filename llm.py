from pydantic import BaseModel, Field, validator

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser


OPEN_API_KEY = "API_KEY_HERE"

template = '''
You are a data scientist tasked to extract insights from review. I am giving you a list of reviews.\
Each element in the list is a review.\
I want you to summarise and return the {no_of_topics} MOST MENTIONED topics (need not separate by sentiment)\
For each key topics, show me 1-2 SHORT PHRASES that you think are most representative of the topic. \
\n
These are the reviews:
{reviews}
\n
ONLY RETURN ME the results in JSON format. DO NOT give me code to do the clustering or your thought process.
\n
Here are the formatting instructions
Make sure the number of topics and the number of lists of sentences ARE THE SAME
'''

class TopicModel(BaseModel):
    topics: list[str] = Field(description="a list of topics")
    sentences: list[list[str]] = Field(description="a nested list of reviews that are most representative of each topic")

topic_parser = PydanticOutputParser(pydantic_object=TopicModel)
format_instructions = topic_parser.get_format_instructions()

CLUSTER_PROMPT = ChatPromptTemplate(
    messages = [
        HumanMessagePromptTemplate.from_template(template + "\n {format_instructions}")
    ],
    input_variables=['reviews', 'no_of_topics'],
    partial_variables={"format_instructions": format_instructions}

)

llm_chain = LLMChain(
    llm=OpenAI(model_name='gpt-3.5-turbo-16k',openai_api_key=OPEN_API_KEY),
    prompt=CLUSTER_PROMPT
)
