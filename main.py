
from winejournal import *

if __name__ == '__main__':
    st.title("Wine Journaling App")

    db_url = os.getenv("POSTGRES_URL")
    engine = create_database(db_url)
    tools = [add_wine_journal_entry]
    tool_executor = ToolExecutor(tools)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful agent, that will follow user instructions"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("user", "{input}")
    ])

    agent = Runnable(prompt) | tool_executor | StrOutputParser()
    chat_history = []

    # Sidebar for adding wine entry
    with st.sidebar:
        st.header("Add New Wine Entry")
        uploaded_file = st.file_uploader("Upload Wine Label Image", type=["jpg", "jpeg", "png"])
        form_data = {}

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Wine Label", width=200)
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
