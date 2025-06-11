# nodes.py
import json
import re
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Import shared components
from state import WorkflowState
from config import llm
from tools import tool_rules_kb, tool_application_kb, tool_domain_kb

# Node 1: Extract initial information from the raw input
def extract_information(state: WorkflowState):
    print("---NODE: Running Information Extractor---")
    
    class ExtractedInfo(BaseModel):
        title: str = Field(description="A concise, descriptive name for the demand...")
        description: str = Field(description="A full narrative or explanation of the demand...")
        request_type: str = Field(description="Feature, Bug, Enhancement, Info/query, or Strategic Initiatives.")
        urgency_cues: str = Field(description="Any time-sensitive language...")
        module_services: str = Field(description="The specific system, product module, or feature area involved...")
        business_priority: str = Field(description="High, Medium, or Low business impact.")
        customer_impact: str = Field(description="A detailed explanation of how this demand affects users...")
        due_date: str = Field(description="Any deadline explicitly stated or implied...")
        regulatory_impact: str = Field(description="Indicates if the demand is required for compliance...")
        revenue_impact: str = Field(description="Describes how the demand could impact revenue streams...")
        security_relevance: str = Field(description="Flags whether this demand touches sensitive systems or data...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that extracts structured information from demand descriptions. Based on the schema below, extract each field."),
        ("human", "{input}")
    ])
    
    structured_llm = llm.with_structured_output(ExtractedInfo)
    chain = prompt | structured_llm
    result = chain.invoke({"input": state["raw_input"]})
    
    return {"extracted_info": result.model_dump()} # Use .model_dump() for Pydantic V2

# Node 2: Classify the demand using an agent and the 'rules_kb' tool
def classify_demand(state: WorkflowState):
    print("---NODE: Running Demand Classifier (Agent)---")
    
    system_prompt = """Objective:
Your primary task is to classify a given input (e.g., a project description, demand request, initiative summary) into a specific Category and Sub-category as defined in the "Knowledge Base: RULES KB IN THE VECTOR STORE." You must also provide a Justification for your classification.

Knowledge Base Reference:

Strictly refer to the "Knowledge Base: Demand Categorisation Rules" provided previously.
This includes:
The two main Categories: Non-Discretionary and Discretionary.
The specific Subclasses (referred to as Sub-categories in your output) under each category.
For each Subclass:
Definition
Classification Criteria
Common Keywords
Processing Steps:

Input Analysis:

Thoroughly analyze the input text.
Identify key terms, phrases, objectives, drivers, and any specific examples or technologies mentioned.
Pay attention to the source or context if provided (e.g., "request from BNM," "due to system outage," "new product launch").
Sub-category Matching - Iterative Comparison:

For each Subclass defined in the knowledge base (across both Non-Discretionary and Discretionary categories):
Keyword Matching: Compare the extracted terms from the input against the Common Keywords listed for the subclass. Note any strong matches.
Criteria Evaluation: Assess if the input meets the Classification Criteria described for the subclass.
Definition Alignment: Consider if the overall intent and nature of the input align with the Definition of the subclass.
Prioritize subclasses where multiple keywords match, and the input strongly aligns with the classification criteria and definition.
Handling Overlaps and Selecting the Best Fit:

If the input seems to match keywords or criteria from multiple subclasses, evaluate which subclass provides the most specific and primary alignment. For example, a project might have customer experience (CX) implications but its primary driver is revenue generation; in this case, "Revenue Generation" would likely be more appropriate than "Customer Experience Enhancement" if the revenue objective is dominant.
Consider the explicit definitions. For instance, differentiate between "High-Impact CX Initiative" (directly affects customer satisfaction, loyalty, or retention) and "Low-Impact CX Initiative" (minor usability improvements or backend enhancements with limited end-user visibility).

Determining the Category:

Once the most appropriate Sub-category (Subclass) is identified, determine its parent Category (Non-Discretionary or Discretionary) as per the knowledge base structure.
Formulating the Justification:

Your justification should clearly explain why the input was classified into the chosen Sub-category and Category.
Reference specific keywords found in the input that match the Common Keywords of the subclass.
Mention how the input meets specific Classification Criteria of the subclass.
You can also briefly refer to the Definition if it strongly supports the classification.
Be concise yet informative. For example: "Matches 'Regulatory Compliance' due to keywords like 'BNM' and 'mandatory submission,' aligning with criteria for regulatory-driven demands."
Output Generation:

Present your classification in the following precise format:

Category : [Identified Category: Non-Discretionary or Discretionary]
Sub-category : [Identified Subclass Name]
Justification : [Your formulated justification]
 Handling Unclear Cases or Insufficient Information:

If the input is too vague, lacks sufficient detail, or does not clearly align with any subclass after thorough analysis:
Category : Unable to classify
Sub-category : Unable to classify
Justification : Insufficient information or input does not clearly match defined categories. Please provide more details regarding [mention specific aspects needed, e.g., drivers, keywords, impact].
 Example of AI's "Thought Process" & Output:

Input: "We have received an urgent request from BNM to implement the new e-KYC guidelines by Q3. This is a mandatory submission."
AI Analysis: Keywords: "urgent," "BNM," "e-KYC guidelines," "mandatory submission."
AI KB Check (Sub-category matching):
"Regulatory Compliance": Keywords BNM, e-KYC, mandatory, submission match. Criteria: "Clearly references BNM," "mandatory compliance deadlines." Definition aligns. Strong match.
Other subclasses: Less relevant.
AI Determines Category: "Regulatory Compliance" is under "Non-Discretionary."
AI Formulates Justification: Based on BNM mention, e-KYC, and mandatory nature.
AI Output:
Category : Non-Discretionary
Sub-category : Regulatory Compliance
Justification : The input mentions 'BNM,' 'e-KYC guidelines,' and 'mandatory submission,' which are common keywords and meet the classification criteria for Regulatory Compliance, such as referencing a regulatory authority and mandatory obligations.
"""
    tools = [tool_rules_kb]
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}"), ("placeholder", "{agent_scratchpad}")])
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)

    result = agent_executor.invoke({
        "input": f"Please classify the following demand information based on your knowledge base: {str(state['extracted_info'])}"
    })
    return {"demand_classification": result['output']}

# Node 3: Classify the domain using an agent and the 'domain_kb' tool
def classify_domain(state: WorkflowState):
    print("---NODE: Running Domain Classifier (Agent)---")
    
    system_prompt = """Your Role: You are an AI assistant tasked with classifying new demand requirements into the correct business domains based on the provided knowledge base.

Your Primary Tool: You must use the "Knowledge Base" as your single source of truth for this task. All classifications, reasoning, and Domain Lead information must be based only on the information contained within it. You will consult the Knowledge Base for domain definitions and to retrieve the "Domain Lead" for each identified domain. Assume Domain Lead information is accessible within or via the Knowledge Base (this may involve searching a dedicated section or a vector store where Domain Lead information is stored).

Your Task: When you receive a user's input (a "demand requirement"), you will perform the following steps to classify it:

Analyze the Input: Carefully examine the user's request. Identify key terms, functionalities, products, and processes mentioned (e.g., "QR payment," "new credit card application," "customer dashboard," "loyalty points," "password reset").
Consult the Knowledge Base for Domain Definitions: Refer to the "Domain Definitions" section of the knowledge base to understand domain scopes.
Identify the Main Domain: Match the key terms from the input to the "Description," "Key Applications," and "Sub-domains" of each domain to find the best fit. This will be the "Main Domain." The Main Domain is the one with primary accountability for the request.
Retrieve Main Domain Lead: Once the Main Domain is identified, retrieve its "Domain Lead" from the Knowledge Base. This may involve searching a dedicated section for Domain Leads or querying a vector store component of the Knowledge Base where this specific information is held.
Check "Out of Scope": Explicitly check the "Out of Scope" section for the potential Main Domain to ensure the request is not excluded. If it is, re-evaluate other domains (and retrieve their respective leads if a new Main Domain is chosen).
Identify Secondary/Impacted Domains: Once you have a Main Domain, consider other domains that might be involved.
Review the "Consulted Domains" listed for the Main Domain.
Analyze the input against the "Rules of Engagement".
Does the request involve a simple dependency like a notification? The secondary domain is likely "Consulted/Informed".
Is it a complex change involving a customer life cycle that impacts another domain's area? The secondary domain is likely "Responsible".
Does the main domain's Business Analyst (BA) handle the end-to-end process even with dependencies? The secondary domain is likely "Consulted/Informed".
Retrieve Secondary/Impacted Domain Lead(s): For each Secondary/Impacted Domain identified, retrieve its "Domain Lead" from the Knowledge Base, using the same method described for the Main Domain Lead.
Assign Roles: Based on your analysis in the previous steps, assign the correct roles to each domain.
Main Domain: This will be "Accountable" or "Accountable/Responsible".
Secondary/Impacted Domain: This will be "Responsible" or "Consulted/Informed".
Construct Your Response: Present your final classification to the user in a clear format. Your response must include:
Main Domain: The name of the primary domain, its assigned role, and its Domain Lead.
Secondary/Impacted Domain(s): The name(s) of any other involved domains, their assigned roles, and their Domain Leads.
Justification: Briefly explain why you made the classification, citing the specific keywords from the input and the rules or descriptions from the knowledge base that support your conclusion.
Example AI Response Format:

"Based on your request for [User's Input], here is the domain classification:

Main Domain ([Assigned Role]): [Identified Main Domain]
Domain Lead: [Main Domain Lead]
Reasoning: The request involves [keywords from input] which falls under the scope of this domain's [description or key application from knowledge base].

Impacted Domain ([Assigned Role]): [Identified Secondary Domain]
Domain Lead: [Secondary Domain Lead]
Reasoning: According to the Rules of Engagement, since the request also affects [functionality of secondary domain], this domain will be [Responsible/Consulted]."

"""
    tools = [tool_domain_kb]
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}"), ("placeholder", "{agent_scratchpad}")])
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)

    result = agent_executor.invoke({
        "input": f"Please classify the following demand into a business domain based on your knowledge base: {str(state['extracted_info'])}"
    })
    return {"domain_classification": result['output']}

# Node 4: Extract and then classify the applications
def extract_and_classify_applications(state: WorkflowState):
    print("---NODE: Running Application Extractor & Classifier---")
    
    extractor_prompt_text = """You are an AI assistant specialized in analyzing IT project titles and descriptions, particularly within a banking or financial technology context. Your task is to identify and extract the names of specific **Systems/Applications** mentioned.

ENTITY DEFINITION:

**System/Application**:
* Description: A 'System/Application' refers to specific software platforms, applications (like mobile apps or web portals), key system modules, core banking systems, significant technology components, or important classification acronyms that are central to the project or process being described. These are often, but not always, represented by acronyms (e.g., MAE, M2U, CASA) or specific proper names. Focus on terms that are treated as distinct operational, technological, or classificatory units in the context of IT and banking projects, as demonstrated by the examples below. Exclude generic project type acronyms (like 'CR' for Change Request or 'STP' for Straight Through Processing unless they refer to a specific named system) or general regulatory bodies (like 'BNM') unless they are an integral part of a system's name being discussed.
* Source: Look for these in both the "Title" and the "Description" fields.
* Output: Provide a unique list of these systems/applications.

FEW-SHOT EXAMPLES:

**Example 1:**
Input:
Title: Salary Financing STP
Description: This project is to enable customer salary financing via MAE app and M2U

Expected JSON Output:
```json
{{
  "Systems/Applications": ["MAE", "M2U"]
}}

Example 2:
Input:
Title: CR - MAE App Onboarding SMS OTP Removal
Description: Currently when Existing to Bank (ETB)(non-CASA STP) customers are on boarded to MAE App, the customer is required to complete a SMS OTP authentication to confirm the user’s mobile number. This mobile number is currently used for MAE App functions such as Tabung, Split Bill and Send and Request.With BNM’s direction to remove SMS OTP, this CR aims to remove the SMS OTP from the MAE onboarding flow.

Expected JSON Output:

JSON
{{
  "Systems/Applications": ["MAE", "CASA", "ETB"]
}}
Example 3 (Illustrative - if no specific systems are found):
Input:
Title: General Process Review
Description: A review of current operational procedures.

Expected JSON Output:

JSON
{{
  "Systems/Applications": []
}}
OUTPUT FORMAT INSTRUCTIONS:

You MUST return the extracted systems/applications as a single, valid JSON object.
The key in the JSON object should be "Systems/Applications".
The value for this key MUST be a list of strings. Each string in the list is an extracted system/application name.
The list should contain unique system/application names. If a system is mentioned multiple times (e.g., "MAE App" and "MAE"), include its common identifier (e.g., "MAE") only once.
If no systems/applications (according to the definition and examples) are found in the text, return an empty list [] for the "Systems/Applications" key.
Identify acronyms and full names that refer to these systems/applications. Prefer the acronym if it's commonly used and seen in the examples (e.g., "MAE" instead of "MAE app").
Now, analyze the following text and provide the JSON output according to all the instructions and definitions above
    """
    
    extractor_prompt = ChatPromptTemplate.from_messages([("system", extractor_prompt_text), ("human", "{demand_info}")])
    extractor_chain = extractor_prompt | llm
    extractor_result = extractor_chain.invoke({"demand_info": str(state["extracted_info"])})
    
    print(f"RAW LLM OUTPUT FOR EXTRACTION: '{extractor_result.content}'")  
     
    app_list = []
    try:
        # Use a regular expression to find the JSON object within the backticks
        # re.DOTALL makes the '.' character match newlines as well
        json_match = re.search(r'\{.*\}', extractor_result.content, re.DOTALL)
        
        if json_match:
            json_string = json_match.group(0)
            # Now parse the cleaned string
            app_list = json.loads(json_string).get("Systems/Applications", [])
        else:
            # This will run if the regex fails to find any JSON
            print("Could not find a JSON object in the LLM's output.")

    except json.JSONDecodeError as e:
        # This will run if the extracted string is still not valid JSON
        print(f"Failed to parse extracted JSON: {e}")

    print(f"Extracted applications: {app_list}")

    if not app_list:
        return {"application_list": [], "application_details": "No applications were extracted from the input."}

    classifier_system_prompt = """
    
    ### IMPORTANT INSTRUCTIONS ###
1.  The CONTEXT you receive is extracted from a table row. The original columns were in this specific order: `Application Name`, `Application full name`, `Tech Lead`, `Assistant tech lead`, `IT Owner`, `Country`, `Platform`.
2.  The text may be jumbled. You must use this column order as your primary clue for identifying the correct people.
3.  **Crucially, the 'Tech Lead' will appear before the 'IT Owner' in the text.**
4.  Scan the CONTEXT to find the names and their associated roles based on this order.
5.  Populate the JSON fields with the exact information you find.
6.  If you cannot confidently distinguish between the roles or if a field is missing, you MUST use the string "Not Found" for that value. Do not guess.
Structure for each JSON object:
```json
{{
  "system_name": "<...>",
  "tech_lead": {{ "name": "<name>", "email": "<email>" }},
  "it_owner": {{ "name": "<name>", "email": "<email>" }},
  "regional_availability": "<region>",
  "platform": "<...>"
}}


the sample expected output is as below :
[
  {{
    "system_name": "MAE",
    "tech_lead": {{
      "name": "Aisha Rahman",
      "email": "aisha.r@example.com"
    }},
    "it_owner": {{
      "name": "Budi Santoso",
      "email": "budi.s@example.com"
    }},
    "regional_availability": "MY",
    "platform": "Mobile Banking Platform"
  }},
  {{
    "system_name": "M2U",
    "tech_lead": {{
      "name": "Charles Lee",
      "email": "charles.l@example.com"
    }},
    "it_owner": {{
      "name": "Budi Santoso",
      "email": "budi.s@example.com"
    }},
    "regional_availability": "PH",
    "platform": "Web Banking Portal"
  }},
  {{
    "system_name": "CASA",
    "tech_lead": {{
      "name": "Not Found",
      "email": "Not Found"
    }},
    "it_owner": {{
      "name": "Diana Velez",
      "email": "diana.v@example.com"
    }},
    "regional_availability": "MY",
    "platform": "Core Banking System"
  }},
  {{
    "system_name": "NonExistentSystem",
    "status": "Not found in vector store",
    "tech_lead": {{
      "name": "N/A",
      "email": "N/A"
    }},
    "it_owner": {{
      "name": "N/A",
      "email": "N/A"
    }},
    "regional_availability": "N/A",
    "platform": "N/A"
  }}
]"""
    tools = [tool_application_kb]
    prompt = ChatPromptTemplate.from_messages([("system", classifier_system_prompt), ("human", "{input}"), ("placeholder", "{agent_scratchpad}")])
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)
    
    result = agent_executor.invoke({"input": f"Please provide details for the following applications: {str(app_list)}"})
    return {"application_list": app_list, "application_details": result['output']}
    
# Node 5: The final formatter agent
def format_output(state: WorkflowState):
    print("---NODE: Formatting Final Output---")

    system_prompt = """Your task is to take the provided classification sections and present them in a structured output. Follow these guidelines precisely:

 Categorization Details:

Start with CATEGORY: followed by the high-level classification.
Then, SUB CATEGORY: followed by the subcategory.
Next, JUSTIFICATION: followed by a brief justification. If no justification is available, insert not found.

 Application Information Table:

Insert a horizontal rule (---).
Add the heading ### APPLICATION INFORMATION.
Create a table with four columns: SYSTEM, IT OWNER, TECH LEAD, and REGIONAL AVAILABILITY, and an additional column for PLATFORM.
Ensure consistent spacing and alignment for all entries in the table columns.
For each application, populate the table with:
SYSTEM: Application Name
IT OWNER: The IT owner of the system
TECH LEAD: Technical lead of the system
REGIONAL AVAILABILITY: List of regions where the system is available
PLATFORM: The business platform the system belongs to
If any value for a system, owner, tech lead, or regional availability is missing, insert not found.

 Domain Information:

Insert a horizontal rule (---).
Add DOMAIN: followed by the specific domain(s). If missing, insert not found.
If more than one domain exists, include all.
Add DOMAIN LEAD: followed by the domain lead(s). If missing, insert not found.
Finally, add JUSTIFICATION: followed by the justification for the domain classification.
If any sub-domains or sub-domain leads are present, include:

SUB-DOMAIN: followed by the sub-domain(s)
SUB-DOMAIN LEAD: followed by the sub-domain lead(s)


Sample Output Format
CATEGORY:
Financial

SUB CATEGORY:
Payments

JUSTIFICATION:
not found

---
### 🧾 APPLICATION INFORMATION

| SYSTEM     | TECH LEAD   | IT OWNER | REGIONAL AVAILABILITY | PLATFORM         |
| :--------- | :--------- | :--------| :--------------------- | :--------------- |
| PayConnect | John Doe   | Jane Roe | APAC, EMEA             | customer         |
| Notifier   | Sarah Lee  | Mike Yeo | Global                 | payment          |
| AppX       | not found  | not found| Americas               | bug              |

---
DOMAIN:
E-commerce /n

SUB-DOMAIN:
Customer Services /n

DOMAIN LEAD:
Jane Smith /n

SUB-DOMAIN LEAD:
Alex Tan /n

JUSTIFICATION:
This application directly supports online payment processing for e-commerce transactions. /n
"""
    prompt_template = """Please use the following information into a final report.

## Demand Classification ##
{demand}

## Domain Classification ##
{domain}

## Affected Application Details ##
{apps}
"""
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", prompt_template)])
    chain = prompt | llm
    
    result = chain.invoke({
        "demand": state['demand_classification'],
        "domain": state['domain_classification'],
        "apps": state['application_details']
    })
    return {"final_output": result.content}
  
