from typing import Annotated, List, Dict, Any, Optional
from datetime import date
from pydantic import BaseModel, Field

from langchain.tools import tool
from langgraph.prebuilt import ToolExecutor
from langchain_core.callbacks.base import BaseCallbackManager
from langchain.tools import Tool
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_core.runnables.base import Runnable, RunnableConfig
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from sqlalchemy import create_engine, Column, Integer, String, Date, Float, PickleType, LargeBinary, ARRAY
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.exc import IntegrityError
import base64
import os
import streamlit as st
from io import BytesIO
from PIL import Image

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
    notes_metadata: Optional[Dict] = Field(default=None, description="Optional metadata to store with the wine notes")
    label_image: Optional[str] = Field(default=None, description="Base64 encoded string of the wine label image.")
    related_urls: Optional[List[str]] = Field(default=None, description="List of URLs related to the wine")


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


def add_wine_to_db(wine: Wine, engine):
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


def analyze_wine_label(image_base64: str) -> dict:
    """
    Analyzes a wine label image using an LLM to extract information and populate
    the wine class attributes.

    Args:
        image_base64 (str): The base64 encoded string of the wine label image

    Returns:
        A dictionary containing the populated wine information if successful, or an empty dictionary if the analysis failed.
    """
    llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are an expert wine taster and sommelier, skilled in analyzing wine labels and extracting information. "
                    "Analyze the image I provide and fill in the missing fields about this wine."
                    "Return the output as a JSON with the wine attributes, do not include any explanatory text"
        ),
        ("user", "Here is the base64 encoded image data: {image_data} \n\n"
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

@tool
def add_wine_journal_entry(wine_data: dict) -> dict:
  """
    Adds an entry of the wine journal to the database
    Args:
        wine_data (dict): The data for the wine entry in a json format
    Returns:
        A dict with the success status of the operation and other metadata
  """
  db_url = st.secrets["DATABASE_URL"]
  engine = create_database(db_url)

  try:
    wine = Wine(**wine_data)
    if wine.label_image:
        analysis_result = analyze_wine_label(wine.label_image)
        # Update the attributes from the LLM analysis, but keep the original value of the field if the attribute is not available.
        wine = Wine(
            **{**wine.model_dump(exclude={"label_image"}), **analysis_result}
        )
    result = add_wine_to_db(wine, engine)
    return result
  except Exception as e:
      return {"success": False, "error": str(e)}


def get_all_wines(engine):
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


if __name__ == '__main__':
    # Initialize Streamlit secrets from environment variables
    import toml
    import os
    
    secrets_path = os.path.join('.streamlit', 'secrets.toml')
    if not os.path.exists(secrets_path):
        os.makedirs('.streamlit', exist_ok=True)
        with open(secrets_path, 'w') as f:
            toml.dump({
                'DATABASE_URL': os.environ.get('DATABASE_URL', ''),
                'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', '')
            }, f)

    st.title("Wine Journaling App")

    db_url = st.secrets["DATABASE_URL"]
    engine = create_database(db_url)
    tools = [add_wine_journal_entry]
    tool_executor = ToolExecutor(tools)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful agent, that will follow user instructions"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("user", "{input}")
    ])

    def format_tool_input(messages):
        return format_to_openai_tool_messages(messages)

    def parse_output(output):
        return output.content

    agent = Runnable(prompt) | tool_executor | StrOutputParser()

    # Define chat history so the agent can keep track of previous prompts and actions
    chat_history = []

    # Sidebar for adding wine entry
    with st.sidebar:
        st.header("Add New Wine Entry")
        uploaded_file = st.file_uploader("Upload Wine Label Image", type=["jpg", "jpeg", "png"])
        #Initialize empty form data to hold the values of the fields
        form_data = {}

        if uploaded_file:
          image = Image.open(uploaded_file)
          st.image(image, caption="Uploaded Wine Label", width=200)
          # Convert image to base64
          buffered = BytesIO()
          image.save(buffered, format="JPEG")
          encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
          form_data['label_image'] = encoded_string


      with st.form("add_wine_form"):
            form_data['name'] = st.text_input("Wine Name")
            form_data['producer'] = st.text_input("Producer")
        form_data['vintage'] = st.number_input("Vintage", min_value=1000, max_value=2100, step=1, value=2023)
        form_data['wine_type'] = st.text_input("Wine Type (Red, White, Rose, etc)")
        form_data['grape_varieties'] = st.text_input("Grape Varieties (comma separated)")
        form_data['region'] = st.text_input("Region")
        form_data['country'] = st.text_input("Country")
        form_data['appellation'] = st.text_input("Appellation")
        form_data['tasting_date'] = st.date_input("Tasting Date")
        form_data['tasting_notes'] = st.text_area("Tasting Notes")
        form_data['aromas'] = st.text_input("Aromas (comma separated)")
        form_data['flavors'] = st.text_input("Flavors (comma separated)")
        form_data['body'] = st.selectbox("Body", ["Light", "Medium", "Full"])
        form_data['tannin'] = st.selectbox("Tannin", ["Low", "Medium", "High"])
        form_data['acidity'] = st.selectbox("Acidity", ["Low", "Medium", "High"])
        form_data['sweetness'] = st.selectbox("Sweetness", ["Dry", "Off-Dry", "Sweet"])
        form_data['alcohol_content'] = st.number_input("Alcohol Content", min_value=0.0, max_value=20.0, step=0.1)
        form_data['rating'] = st.number_input("Rating (1-5 or 1-10)", min_value=1, max_value=10, step=1)
        form_data['pairing_suggestions'] = st.text_input("Pairing Suggestions (comma separated)")
        form_data['price'] = st.number_input("Price")
        form_data['purchase_location'] = st.text_input("Purchase Location")
        form_data['notes_metadata'] = st.text_input("Notes Metadata")
        form_data['related_urls'] = st.text_input("Related URLs (comma separated)")
        submitted = st.form_submit_button("Add Wine Entry")
        if submitted:
          # Convert the comma separated strings into proper lists
          if form_data['grape_varieties']:
             form_data['grape_varieties'] = form_data['grape_varieties'].split(",")
          if form_data['aromas']:
             form_data['aromas'] = form_data['aromas'].split(",")
          if form_data['flavors']:
             form_data['flavors'] = form_data['flavors'].split(",")
          if form_data['pairing_suggestions']:
              form_data['pairing_suggestions'] = form_data['pairing_suggestions'].split(",")
          if form_data['related_urls']:
              form_data['related_urls'] = form_data['related_urls'].split(",")

          with st.spinner("Adding wine entry..."):
              result = agent.invoke({
                "input": "Add a new journal entry to the database with the following values: " + str(form_data),
                "chat_history": chat_history,
                "agent_scratchpad": []
             })
              st.success(f"Wine entry added. Result: {result}")


  # Main area for displaying wine data
  st.header("Wine Journal Entries")
  with st.spinner("Loading wine entries..."):
    wines = get_all_wines(engine)

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