from typing import Annotated, List, Dict, Any, Optional, Tuple
from typing_extensions import TypedDict
from datetime import date, datetime, timedelta
from pydantic import BaseModel, Field
import json

from langchain.tools import tool
from langchain_core.callbacks.base import BaseCallbackManager
from langchain.tools import Tool
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_core.runnables.base import Runnable, RunnableConfig
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import AgentExecutor
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
import base64
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import END

from sqlalchemy import create_engine, desc, Column, Integer, String, Date, Float, PickleType, LargeBinary, ARRAY
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.exc import IntegrityError
import os
from dotenv import load_dotenv
import streamlit as st
from io import BytesIO
from PIL import Image
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import re
from pprint import pprint
from langsmith import traceable
from langchain_core.tracers import LangChainTracer

load_dotenv()

Base = declarative_base()

class Wine(BaseModel):
    """
    Pydantic model representing a wine for journaling and AI analysis.
    """
    name: str = Field(description="The name of the wine (e.g., 'Château Margaux 2015').")
    producer: Optional[str] = Field(default=None, description="The producer or winery of the wine.")
    vintage: Optional[int] = Field(default=None, description="The year the wine was produced.")
    wine_type: Optional[str] = Field(default=None, description="The type of wine (e.g., 'Red', 'White', 'Rose', 'Sparkling').")
    grape_varieties: Optional[List[str]] = Field(default=None, description="The grape varieties used (e.g., ['Cabernet Sauvignon', 'Merlot']).")
    region: Optional[str] = Field(default=None, description="The geographical region where the wine was produced (e.g., 'Bordeaux', 'Napa Valley').")
    country: Optional[str] = Field(default=None, description="The country where the wine was produced (e.g., 'France', 'USA').")
    appellation: Optional[str] = Field(default=None, description="The specific appellation (e.g., 'Margaux', 'St. Helena').")
    tasting_date: Optional[date] = Field(default=None, description="The date the wine was tasted.")
    tasting_notes: Optional[str] = Field(default=None, description="Free-form text notes on the tasting experience (aromas, flavors, finish).")
    aromas: Optional[List[str]] = Field(default=None, description="Specific aromas perceived (e.g., 'Black Cherry', 'Vanilla', 'Cedar').")
    flavors: Optional[List[str]] = Field(default=None, description="Specific flavors perceived (e.g., 'Black Currant', 'Chocolate', 'Oak').")
    body: Optional[str] = Field(default=None, description="Body of the wine (e.g., 'Light', 'Medium', 'Full').")
    tannin: Optional[str] = Field(default=None, description="Tannin level (e.g., 'Low', 'Medium', 'High').")
    acidity: Optional[str] = Field(default=None, description="Acidity level (e.g., 'Low', 'Medium', 'High').")
    sweetness: Optional[str] = Field(default=None, description="Sweetness level (e.g., 'Dry', 'Off-dry', 'Sweet').")
    alcohol_content: Optional[float] = Field(default=None, description="Alcohol content (e.g., 13.5).")
    rating: Optional[int] = Field(default=None, description="A rating on a scale (e.g., 1-5 or 1-10)")
    pairing_suggestions: Optional[List[str]] = Field(default=None, description="Suggested food pairings (e.g., ['Steak', 'Lamb', 'Hard Cheese']).")
    price: Optional[float] = Field(default=None, description="The price of the bottle.")
    purchase_location: Optional[str] = Field(default=None, description="Where the wine was purchased (e.g., 'Local Wine Shop', 'Restaurant').")
    notes_metadata: Optional[dict] = Field(default=None, description="Optional metadata to store with the wine notes")
    label_image: Optional[str] = Field(default=None, description="Base64 encoded string of the wine label image.")
    related_urls: Optional[List[str]] = Field(default=None, description="List of URLs related to the wine")

class AgentState(TypedDict):
   """
   Represents the state of our agent
   """
   wine_data: Dict[str, Any] = Field(default={})
   messages: List[BaseMessage] = Field(default=[])
   next_step: Optional[str] = Field(default=None)


class WineTable(Base):
    __tablename__ = 'wines'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    producer = Column(String)
    vintage = Column(Integer)
    wine_type = Column(String)
    grape_varieties = Column(PickleType)
    region = Column(String)
    country = Column(String)
    appellation = Column(String)
    tasting_date = Column(Date)
    tasting_notes = Column(String)
    aromas = Column(PickleType)
    flavors = Column(PickleType)
    body = Column(String)
    tannin = Column(String)
    acidity = Column(String)
    sweetness = Column(String)
    alcohol_content = Column(Float)
    rating = Column(Integer)
    pairing_suggestions = Column(PickleType)
    price = Column(Float)
    purchase_location = Column(String)
    notes_metadata = Column(PickleType)
    label_image = Column(LargeBinary)  # Added to store the base64 encoded string
    related_urls = Column(ARRAY(String))  # Added to store the list of related URLs

def create_wine_table(engine):
    """Creates the Wine table if it doesn't exist."""
    Base.metadata.create_all(engine)


def create_database(db_url: str):
    """Creates a PostgreSQL engine and tables."""
    engine = create_engine(db_url)
    create_wine_table(engine)
    return engine


def add_wine_to_db(wine: Wine, engine) -> dict:
    """Adds wine data to the PostgreSQL database."""
    session = Session(engine)
    try:
        # Encode the base64 image
        label_image_encoded = wine.label_image.encode('utf-8') if wine.label_image else None
        wine_db_entry = WineTable(**wine.model_dump(exclude={"label_image", "related_urls"}), label_image=label_image_encoded, related_urls = wine.related_urls)
        session.add(wine_db_entry)
        session.commit()
        return {"success": True, "wine_name": wine.name}
    except IntegrityError as e:
        session.rollback()
        return {"success": False, "error": str(e)}
    finally:
        session.close()

def iterate_nested_json(json_obj, max_depth: int = 0):
    global prefix
    old_prefix = prefix
    depth = prefix.count("[")
    if max_depth > 0 and depth > depth:
        return
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if isinstance(value, dict) or isinstance(value, list):
                prefix = prefix + f"[{key}]"
                iterate_nested_json(value)
                prefix = old_prefix
            else:
                if prefix.startswith("[vintage]"):
                    print(f"{prefix}[{key}]: {value}")
    elif isinstance(json_obj, list):
        for index, value in enumerate(json_obj):
            if isinstance(value, dict) or isinstance(value, list):
                prefix = prefix + f"[{index}]"
                iterate_nested_json(value)
                prefix = old_prefix
            else:
                if prefix.startswith("[vintage]"):
                    print(f"{prefix}[{index}]{value}")
    else:
        if prefix.startswith("[vintage]"):
           print(f"{prefix}{value}")

@traceable
def analyze_wine_label(image_base64: str) -> dict:
    """
    Analyzes a wine label image using an LLM to extract information and populate
    the wine class attributes.

    Args:
        image_base64 (str): The base64 encoded string of the wine label image

    Returns:
        A dictionary containing the populated wine information if successful, or an empty dictionary if the analysis failed.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=4096, api_key=os.getenv("OPENAI_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are an expert wine taster and sommelier, skilled in analyzing wine labels and extracting information. "
                    "Analyze the image I provide and fill in the missing fields about this wine."
                    "Return the output as a JSON with the wine attributes, do not include any explanatory text"
        ),
        ("user", "Here is the base64 encoded image data: {image_base64} \n\n"
                 "Try to populate the following attributes of the wine, return only a JSON object, do not include any explanatory text:\n"
                 "- name\n"
                 "- producer\n"
                 "- vintage\n"
                 "- wine_type\n"
                 "- grape_varieties\n"
                 "- region\n"
                 "- country\n"
                 "- appellation\n"
        )
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser
    try:
        result = chain.invoke({"image_data": image_base64})
        return result
    except Exception as e:
        print(f"Error analyzing wine label: {e}")
        return {}

@traceable
def search_web(query: str) -> List[str]:
    """
        Searches the web using Tavily API and returns a list of URLs.

        Args:
            query (str): The search query.

        Returns:
            A list of relevant URLs
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("Tavily API Key not set")
        return []
    client = TavilyClient(api_key=tavily_api_key)

    try:
        response = client.search(query=query, include_domains=['vivino.com'], search_depth="basic")
        if (response['results']):
            urls = []
            for result in response['results']:
                if result['score'] < 0.5:
                    print("Score < 0.5, skip")
                    continue
                if "toplists" in result['url']:
                    print("Looks like toplist, skip")
                    continue
                # if all(s not in result['url'] for s in ["vivino.com/US/","vivino.com/FR/"]):
                #     print("Not in US or France, exclude")
                #     continue
                pprint(result)
                urls.append(result['url'])
            return urls
        else:
            return []
    except Exception as e:
        print(f"Error searching the web with Tavily: {e}")
        return []

@traceable
def scrape_wine_data(url: str) -> dict:
    """
        Scrapes detailed wine information from a given URL using BeautifulSoup.

        Args:
            url (str): The URL of the webpage to scrape.

        Returns:
            A dictionary of scraped wine information or an empty dictionary if fails
    """
    # if "vivino.com/US/" not in url and "vivino.com/FR/" not in url:
    #     print("Neither /US/ nor /FR/ found in url")
    #     return {}

    # Replace language with English
    # url = re.sub(r"/US/.*/", "/US/en/", url)
    # url = re.sub(r"/FR/.*/", "/FR/en/", url)

    # Add Accept-Language to header to help Vivino choose translation if available

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        # Step 3: Find all <script> tags
        script_tags = soup.find_all("script")
        global prefix
        prefix = ""
        # Step 4: Search for a specific pattern in each script tag
        target_pattern = r"PageInformation"  # Regex to find any occurrence of 'VariableEnding'

        wine_data = {}
        parsed_data = {}
        for script_tag in script_tags:
            if script_tag.string:  # Ensure the script tag has content
                match = re.search(target_pattern, script_tag.string)
                if match:
                    script_content = script_tag.string
                    # Regex to extract JSON data assigned to 'targetVariable'
                    pattern = r"(\w*PageInformation)\s*=\s*(\{.*?\});"
                    match = re.search(pattern, script_content, re.DOTALL)

                    if match:
                        json_data = match.group(2)  # Extract the JSON string
                        #pprint.pprint(json_data)
                        parsed_data = json.loads(json_data)  # Convert to Python dictionary
                        #iterate_nested_json(parsed_data)
                        #print(parsed_data)
                        break
                    else:
                        print("Target variable not found in the script.")
                else:
                    print("Target pattern not found in the script.")
            else:
                print("No string found in this script.")

        if parsed_data and parsed_data['vintage']:
            print(f"Found Vintage: {parsed_data['vintage']['name']} ... at {url}")
            wine_data['name'] = parsed_data['vintage']['name']
            wine_data['tasting_notes'] = ''
            wine_data['tasting_notes'] += f"Vivino wine description: {parsed_data['vintage']['wine']['description']}"
            # wine_data['vintage'] = parsed_data['vintage']['year']
            wine_data['region'] = parsed_data['vintage']['wine']['region']['name']
            wine_data['country'] = parsed_data['vintage']['wine']['region']['country']['name']
            wine_data['producer'] = parsed_data['vintage']['wine']['winery']['name']
            wine_data['wine_type'] = parsed_data['vintage']['wine']['style']['regional_name'] + parsed_data['vintage']['wine']['style']['varietal_name']
            wine_data['tasting_notes'] += f"\nVivino style description: {parsed_data['vintage']['wine']['style']['description']}"
            wine_data['body'] = parsed_data['vintage']['wine']['style']['body_description'] + ", " + str(parsed_data['vintage']['wine']['style']['baseline_structure']['intensity'])
            wine_data['acidity'] = parsed_data['vintage']['wine']['style']['acidity_description'] + ", " + str(parsed_data['vintage']['wine']['style']['baseline_structure']['acidity'])
            wine_data['sweetness'] = str(parsed_data['vintage']['wine']['style']['baseline_structure']['sweetness'])
            wine_data['tannin'] = str(parsed_data['vintage']['wine']['style']['baseline_structure']['tannin'])
            wine_data['alcohol_content'] = parsed_data['vintage']['wine']['alcohol']
            wine_data['grape_varieties'] = []
            for index, grape in enumerate(parsed_data['vintage']['wine']['style']['grapes']):
                #print(f"Index: {index}, Grape: {grape['name']}")
                wine_data['grape_varieties'].append(grape['name'])
            for index, fact in enumerate(parsed_data['vintage']['wine']['style']['interesting_facts']):
                wine_data['tasting_notes'] += f"\nVivino interesting fact: {fact}"
            pprint(wine_data)
            return wine_data
        elif parsed_data and parsed_data['wine']:
            print(f"Found wine: {parsed_data['wine']['name']} ... at {url}")
            wine_data['name'] = parsed_data['wine']['name']
            wine_data['tasting_notes'] = ''
            wine_data['tasting_notes'] += f"Vivino wine description: {parsed_data['wine']['description']}"
            wine_data['region'] = parsed_data['wine']['region']['name']
            wine_data['country'] = parsed_data['wine']['region']['country']['name']
            wine_data['producer'] = parsed_data['wine']['winery']['name']
            wine_data['wine_type'] = parsed_data['wine']['style']['regional_name'] + parsed_data['wine']['style']['varietal_name']
            wine_data['tasting_notes'] += f"\nVivino style description: {parsed_data['wine']['style']['description']}"
            wine_data['body'] = parsed_data['wine']['style']['body_description'] + ", " + str(parsed_data['wine']['style']['baseline_structure']['intensity'])
            wine_data['acidity'] = parsed_data['wine']['style']['acidity_description'] + ", " + str(parsed_data['wine']['style']['baseline_structure']['acidity'])
            wine_data['sweetness'] = str(parsed_data['wine']['style']['baseline_structure']['sweetness'])
            wine_data['tannin'] = str(parsed_data['wine']['style']['baseline_structure']['tannin'])
            wine_data['alcohol_content'] = parsed_data['wine']['alcohol']
            wine_data['grape_varieties'] = []
            for index, grape in enumerate(parsed_data['wine']['style']['grapes']):
                #print(f"Index: {index}, Grape: {grape['name']}")
                wine_data['grape_varieties'].append(grape['name'])
            for index, fact in enumerate(parsed_data['wine']['style']['interesting_facts']):
                wine_data['tasting_notes'] += f"\nVivino interesting fact: {fact}"
            pprint(wine_data)
            return wine_data
        elif parsed_data:
            print(f"Found parsed data, but no vintage or wine entry found at {url}")
            print("Found this instead:")
            iterate_nested_json(parsed_data, 1)
            return {}
        else:
            print(f"Found no parsed data at {url}")
            return {}

    except requests.exceptions.RequestException as e:
        print(f"Request error scraping URL {url}: {e}")
        return {}
    except Exception as e:
        print(f"Unknown error scraping URL {url}: {e}")
        return {}


def image_analysis_tool(agent_state: AgentState) -> AgentState:
    """
        Analyzes the wine label image using an LLM
    """
    wine_data = agent_state['wine_data']
    try:
        wine = Wine(**wine_data)
        if wine.label_image:
            analysis_result = analyze_wine_label(wine.label_image)
            # Update the attributes from the LLM analysis, but keep the original value of the field if the attribute is not available.
            wine = Wine(**{**wine.model_dump(exclude={"label_image"}), **analysis_result})
            agent_state['wine_data'] = wine
            agent_state['messages'].append(HumanMessage(content=f"Wine label analysis result: {analysis_result}"))
            return agent_state
        else:
            agent_state['wine_data'] = wine_data
            return agent_state
    except Exception as e:
        agent_state['messages'].append(HumanMessage(content=f"Error analyzing wine label: {e}"))
        return agent_state


def web_search_tool(agent_state: AgentState) -> AgentState:
    """
    Searches the web using Tavily API to search for wine related information
    """
    wine_data = agent_state['wine_data']
    try:
        wine = Wine(**wine_data)
        search_query = f"{wine.name} {wine.vintage}"
        if search_query:
            search_results = search_web(query=search_query)
            wine = Wine(**{**wine.model_dump(), "related_urls": search_results})
            agent_state['wine_data'] = wine.model_dump()
            return agent_state
        else:
          agent_state['wine_data'] = wine_data
          return agent_state
    except Exception as e:
        agent_state['messages'].append(HumanMessage(content=f"Error searching the web: {e}"))
        return agent_state

def web_scrape_tool(agent_state: AgentState) -> AgentState:
    """
    Scrapes detailed wine information from a given list of URLs using BeautifulSoup.
    """
    wine_data = agent_state['wine_data']
    try:
        wine = Wine(**wine_data)
        scraped_data_all = {}
        if wine.related_urls:
            for url in wine.related_urls:
                scraped_data = scrape_wine_data(url=url)
                if scraped_data:
                    #Adding all the fields from the scraped data, this might include duplicate fields that can be overwritten by later steps
                    scraped_data_all.update(scraped_data)
                    #break
                    print(f"Accumulated (updated) wine data after scraping {url}")
                    pprint(scraped_data_all)
            wine = Wine(**{**wine.model_dump(), **scraped_data_all})
            print("Current state of wine data model: ")
            pprint(wine.model_dump())
            agent_state['wine_data'] = wine.model_dump()
            return agent_state
        else:
            agent_state['messages'].append(HumanMessage(content=f"No URLs from web search to scrape"))
            return agent_state
    except Exception as e:
        agent_state['messages'].append(HumanMessage(content=f"Error scraping the web: {e}"))
        return agent_state


def add_wine_journal_entry(agent_state: AgentState) -> AgentState:
  """
    Adds an entry of the wine journal to the database
    Args:
        wine_data (dict): The data for the wine entry in a json format
    Returns:
        A dict with the success status of the operation and other metadata
  """
  wine_data = agent_state['wine_data']
  db_url = os.getenv("DATABASE_URL")
  engine = create_database(db_url)
  try:
    wine = Wine(**wine_data)
    result = add_wine_to_db(wine, engine)
    agent_state['messages'].append(HumanMessage(content=f"Wine entry added. Result: {result}"))
    pprint(agent_state['messages'])
    return agent_state
  except Exception as e:
      agent_state['messages'].append(HumanMessage(content=f"Error adding wine entry: {e}"))
      pprint(agent_state['messages'])
      return agent_state

def get_all_wines(engine) -> List[dict]:
    """Retrieves all wine entries from the database."""
    session = Session(engine)
    try:
        wines_db = session.query(WineTable).all()
        wines = []
        for wine in wines_db:
            wine_dict = wine.__dict__
            # Convert the binary data to base64 for display
            if wine_dict['label_image']:
                wine_dict['label_image'] = base64.b64encode(wine_dict['label_image']).decode('utf-8')

            # Convert the lists to regular python lists for display
            if wine_dict['aromas']:
                wine_dict['aromas'] = list(wine_dict['aromas'])

            if wine_dict['flavors']:
                wine_dict['flavors'] = list(wine_dict['flavors'])

            if wine_dict['grape_varieties']:
                wine_dict['grape_varieties'] = list(wine_dict['grape_varieties'])

            if wine_dict['pairing_suggestions']:
                wine_dict['pairing_suggestions'] = list(wine_dict['pairing_suggestions'])

            if wine_dict['related_urls']:
                wine_dict['related_urls'] = list(wine_dict['related_urls'])

            wines.append(wine_dict)
        return wines
    except Exception as e:
        print(f"Error retrieving wine entries from database: {e}")
        return []
    finally:
        session.close()

def get_latest_wines(engine) -> List[dict]:
    """Retrieves all wine entries from the database."""
    session = Session(engine)
    try:
        query = (
            session.query(WineTable)
#            .order_by(desc(WineTable.tasting_date)).limit(25)
            .order_by(desc(WineTable.id)).limit(5)
        )
        wines_db = query.all()

        wines = []
        for wine in wines_db:
            wine_dict = wine.__dict__
            # Convert the binary data to base64 for display
            if wine_dict['label_image']:
                wine_dict['label_image'] = base64.b64encode(wine_dict['label_image']).decode('utf-8')

            # Convert the lists to regular python lists for display
            if wine_dict['aromas']:
                wine_dict['aromas'] = list(wine_dict['aromas'])

            if wine_dict['flavors']:
                wine_dict['flavors'] = list(wine_dict['flavors'])

            if wine_dict['grape_varieties']:
                wine_dict['grape_varieties'] = list(wine_dict['grape_varieties'])

            if wine_dict['pairing_suggestions']:
                wine_dict['pairing_suggestions'] = list(wine_dict['pairing_suggestions'])

            if wine_dict['related_urls']:
                wine_dict['related_urls'] = list(wine_dict['related_urls'])

            wines.append(wine_dict)
        return wines
    except Exception as e:
        print(f"Error retrieving wine entries from database: {e}")
        return []
    finally:
        session.close()

if __name__ == '__main__':
  st.title("Wine Journaling App")

  db_url = os.getenv("DATABASE_URL")
  engine = create_database(db_url)

  # Define the tools
  tools_image_analysis = [image_analysis_tool]
  tools_web_search = [web_search_tool]
  tools_web_scrape = [web_scrape_tool]
  tools_add_db = [add_wine_journal_entry]

  # Define the Langraph Graph
  workflow = StateGraph(AgentState)
  workflow.add_node("image_analysis_node", image_analysis_tool)
  workflow.add_node("web_search_node", web_search_tool)
  workflow.add_node("web_scrape_node", web_scrape_tool)
  workflow.add_node("add_to_database", add_wine_journal_entry)
  workflow.set_entry_point("image_analysis_node")

  workflow.add_edge("image_analysis_node", "web_search_node")
  workflow.add_edge("web_search_node", "web_scrape_node")
  workflow.add_edge("web_scrape_node", "add_to_database")
  app = workflow.compile()

  # Sidebar for adding wine entry
  with st.sidebar:
      st.header("Add New Wine Entry")
      uploaded_file = st.file_uploader("Upload Wine Label Image", type=["jpg", "jpeg", "png"])
      form_data: Dict[str, Any] = {}

      if uploaded_file:
          image = Image.open(uploaded_file)
          st.image(image, caption="Uploaded Wine Label", width=200)
          # Convert image to base64
          buffered = BytesIO()
          image.save(buffered, format="JPEG", optimize=True, quality=50)
          encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
          form_data['label_image'] = encoded_string

      with st.form("add_wine_form"):
        form_data['name'] = st.text_input("Wine Name")
        form_data['producer'] = st.text_input("Producer")
        form_data['vintage'] = st.number_input("Vintage", min_value=1000, max_value=2100, step=1, value=2023)
        form_data['wine_type'] = st.text_input("Wine Type (Red, White, Rose, etc)")
        form_data['grape_varieties'] = st.text_input("Grape Varieties (comma separated)").split(',')
        form_data['region'] = st.text_input("Region")
        form_data['country'] = st.text_input("Country")
        form_data['appellation'] = st.text_input("Appellation")
        form_data['tasting_date'] = st.date_input("Tasting Date")
        form_data['tasting_notes'] = st.text_area("Tasting Notes")
        form_data['aromas'] = st.text_input("Aromas (comma separated)").split(',')
        form_data['flavors'] = st.text_input("Flavors (comma separated)").split(',')
        form_data['body'] = st.selectbox("Body", ["Light", "Medium", "Full"])
        form_data['tannin'] = st.selectbox("Tannin", ["Low", "Medium", "High"])
        form_data['acidity'] = st.selectbox("Acidity", ["Low", "Medium", "High"])
        form_data['sweetness'] = st.selectbox("Sweetness", ["Dry", "Off-Dry", "Sweet"])
        form_data['alcohol_content'] = st.number_input("Alcohol Content", min_value=0.0, max_value=20.0, step=0.1)
        form_data['rating'] = st.number_input("Rating (1-5 or 1-10)", min_value=1, max_value=10, step=1)
        form_data['pairing_suggestions'] = st.text_input("Pairing Suggestions (comma separated)").split(',')
        form_data['price'] = st.number_input("Price")
        form_data['purchase_location'] = st.text_input("Purchase Location")
        form_data['notes_metadata'] = json.loads(st.text_input("Notes Metadata") or '{}')
        form_data['related_urls'] = st.text_input("Related URLs (comma separated)").split(',')
        submitted = st.form_submit_button("Add Wine Entry")

        if submitted:
            with st.spinner("Adding wine entry..."):
                # We are initializing the Agent state
                agent_state = AgentState(wine_data=form_data, messages=[])
                result = app.invoke(agent_state)
                st.success(f"Wine entry added for: {form_data['name']}")

  # Main area for displaying wine data
  st.header("Wine Journal Entries")
  with st.spinner("Loading wine entries..."):
    wines = get_latest_wines(engine)
    if wines:
        for wine in wines:
          st.subheader(wine['name'])
          st.write(f"**Producer:** {wine['producer']}")
          st.write(f"**Vintage:** {wine['vintage']}")
          st.write(f"**Wine Type:** {wine['wine_type']}")
          st.write(f"**Grape Varieties:** {', '.join(wine['grape_varieties']) if wine['grape_varieties'] else 'N/A'}")
          st.write(f"**Region:** {wine['region']}")
          st.write(f"**Country:** {wine['country']}")
          st.write(f"**Appellation:** {wine['appellation']}")
          st.write(f"**Tasting Date:** {wine['tasting_date']}")
          st.write(f"**Tasting Notes:** {wine['tasting_notes']}")
          st.write(f"**Aromas:** {', '.join(wine['aromas']) if wine['aromas'] else 'N/A'}")
          st.write(f"**Flavors:** {', '.join(wine['flavors']) if wine['flavors'] else 'N/A'}")
          st.write(f"**Body:** {wine['body']}")
          st.write(f"**Tannin:** {wine['tannin']}")
          st.write(f"**Acidity:** {wine['acidity']}")
          st.write(f"**Sweetness:** {wine['sweetness']}")
          st.write(f"**Alcohol Content:** {wine['alcohol_content']}")
          st.write(f"**Rating:** {wine['rating']}")
          st.write(f"**Pairing Suggestions:** {', '.join(wine['pairing_suggestions']) if wine['pairing_suggestions'] else 'N/A'}")
          st.write(f"**Price:** {wine['price']}")
          st.write(f"**Purchase Location:** {wine['purchase_location']}")
          st.write(f"**Notes Metadata:** {wine['notes_metadata']}")
          st.write(f"**Related URLs:** {', '.join(wine['related_urls']) if wine['related_urls'] else 'N/A'}")

          if wine['label_image']:
            try:
                image = Image.open(BytesIO(base64.b64decode(wine['label_image'])))
                st.image(image, caption="Wine Label", width=200)
            except Exception as e:
                st.write(f"Error decoding the image for: {wine['name']}")
                st.write(f"Error: {e}")
          st.markdown("---")
    else:
       st.write("No wine entries found in the database.")
