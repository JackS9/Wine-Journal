from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, Annotated, Optional
from pydantic import BaseModel, Field
# Define the state with expected input format
class AgentState(BaseModel):
    formdata: Dict[str, Any] = Field(default={})
    extra: Optional[str] = Field(default=None)

# Create the graph
workflow = StateGraph(AgentState)

# Define a function to process the form data
def process_form_data(agent_state: AgentState) -> AgentState:
    # Example processing: Convert dict to string and append a message
    #agent_state['extra'] = "Goodbye, World!" # for TypedDict
    agent_state.extra = "Goodbye, World!" # for BaseModel
    return agent_state

# Add the node to handle the form data
workflow.add_node("process_form", process_form_data)

# Set entry and finish points for the graph
workflow.set_entry_point("process_form")
workflow.set_finish_point("process_form")

# Compile the graph
graph = workflow.compile()

# Example usage
form_data = {"name": "John Doe", "age": 30}
agent_state = AgentState(formdata=form_data, extra="Hello, World!")
result = graph.invoke(agent_state)
print(result)