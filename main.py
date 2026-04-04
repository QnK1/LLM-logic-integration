import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a semantic parser.
        Convert the sentence into FIRST-ORDER LOGIC.
        Rules:
        - Use standard logical notation
        - Output ONLY the formula
        - No explanation
        - No extra text"""),
    ("human", "{input_sentence}")
])

chain = prompt | model | StrOutputParser()

sentence = "Every human is mortal.."
result = chain.invoke({"input_sentence": sentence})

print(f"Sentence: {sentence}")
print(f"Logic: {result}")