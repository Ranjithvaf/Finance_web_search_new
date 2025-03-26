from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_URL = "http://192.168.0.100:1234/v1"
Local_llm = ChatOpenAI(
    base_url=MODEL_URL,
    api_key="dummy-key",
    streaming=True,
    model="internlm2_5-20b-chat",
)


def QuestionAnswerChat(userQuestion):
    try:
        # List of websites to search within
        websites = "https://www.nseindia.com/"

        # Initialize the website search tool
        website_search_tool = WebsiteSearchTool(websites)

        # Define a general-purpose assistant
        KnowledgeAgent = Agent(
            role="Smart Knowledge Assistant",
            goal=(
                "Answer user queries by searching the given websites and providing "
                "accurate, well-structured, and up-to-date information."
            ),
            backstory=(
                "You are an intelligent assistant capable of retrieving reliable "
                "information from specified sources. You ensure that all responses "
                "are factual, clear, and useful."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[website_search_tool],  # Uses website search tool
        )

        # Define the task to retrieve relevant answers
        AnswerQueryTask = Task(
            description=(
                "Find relevant information for the user's question by searching the specified websites.\n"
                "Steps:\n"
                "1. Use the search tool to find the most relevant information.\n"
                "2. Retrieve and structure the response clearly and concisely.\n"
                "3. Provide relevant source links if available."
            ),
            expected_output=(
                "User Query Response:\n"
                "- Question: {userQuestion}\n"
                "- Answer: A well-structured response based on search results.\n"
                "- Sources: List of links used to gather the information."
            ),
            agent=KnowledgeAgent,
        )

        # Create a crew to handle the task
        crew = Crew(
            agents=[KnowledgeAgent],
            tasks=[AnswerQueryTask],
            verbose=True,
            manager_llm=Local_llm,
        )

        # Get the answer to the user's question
        result = crew.kickoff(inputs={"userQuestion": userQuestion})

        print(result)
        return result

    except Exception as e:
        print(e)


# Get user's question and answer it using the Chatbot
query = input("Question: ")
if query == "":
    query = "What is the current price of Nifty 50?"
QuestionAnswerChat(query)
