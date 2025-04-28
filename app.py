# app.py - Streamlit App for Customer Ticket Summarization with LLM-as-Judge Eval

# Required Libraries
import streamlit as st
import pandas as pd
import io
import os
from dotenv import dotenv_values
import time  # Added for timing
from datetime import datetime  # Added for unique filenames/run names
import re  # Added for parsing judge score
import mlflow  # Added for MLflow tracking

# Optional Libraries (handle import errors)
try:
    # Using langchain_openai based on prior setup
    from langchain_openai import ChatOpenAI
except ImportError:
    # Let the app load, but flag that LLM functionality is missing
    ChatOpenAI = None

try:
    # For bonus analysis charts
    import plotly.express as px
except ImportError:
    # Let the app load, but flag that charting is missing
    px = None


# --- Configuration & LLM Setup ---

# Find the script's directory to reliably locate the .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, ".env")

print(f"INFO: Loading .env file from: {dotenv_path}")  # For terminal feedback

llm = None  # Language Model instance
initialization_error = None  # To store any setup errors
llm_model_name = "N/A"  # Store model name for logging

try:
    env_values = dotenv_values(dotenv_path)

    if not env_values:
        initialization_error = f".env file missing or empty at {dotenv_path}"
    else:
        # Expecting 'OPENAI_API_KEY' based on prior setup
        OPENAI_API_KEY = env_values.get("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            initialization_error = "'OPENAI_API_KEY' not found in .env file"
        elif ChatOpenAI is None:
            initialization_error = (
                "'langchain-openai' library not installed or import failed"
            )
        else:
            # Initialize the LLM client if key and library are available
            llm_model_name = "gpt-4o-mini"  # Define the model name
            llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=llm_model_name)
            print(f"INFO: LLM client initialized ({llm_model_name})")
except Exception as e:
    initialization_error = f"LLM Init Error: {e}"
    llm = None  # Ensure LLM is None on error

# --- MLflow Setup ---
MLFLOW_EXPERIMENT_NAME = "Customer Ticket Summarization Eval"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"INFO: MLflow experiment set to '{MLFLOW_EXPERIMENT_NAME}'")

# Ensure the evaluation directory exists
EVALS_DIR = "evals"
os.makedirs(EVALS_DIR, exist_ok=True)
print(f"INFO: Evaluation directory '{EVALS_DIR}' ensured.")


# --- Helper Function: Format Data for LLM ---
def format_ticket_data_for_prompt(ticket_data_df):
    """Prepares ticket data DataFrame for the LLM prompt string."""
    if ticket_data_df.empty:
        return "No relevant ticket data."

    details = ["Ticket History Details:"]
    for _, row in ticket_data_df.iterrows():
        # Format timestamps nicely
        accepted_time_str = (
            row["ACCEPTANCE_TIME"].strftime("%Y-%m-%d %H:%M")
            if pd.notna(row["ACCEPTANCE_TIME"])
            else "N/A"
        )
        completed_time_str = (
            row["COMPLETION_TIME"].strftime("%Y-%m-%d %H:%M")
            if pd.notna(row["COMPLETION_TIME"])
            else "N/A"
        )

        # Build string for each ticket entry
        entry = (
            f"- Ticket {row.get('ORDER_NUMBER', 'N/A')} "
            f"(Accepted: {accepted_time_str}, Completed: {completed_time_str})\n"
            f"  Desc: {row.get('ORDER_DESCRIPTION_1', '')} / {row.get('ORDER_DESCRIPTION_2', '')} / {row.get('ORDER_DESCRIPTION_3_MAXIMUM', '')}\n"
            f"  Note: {row.get('NOTE_MAXIMUM', 'N/A')}\n"
            f"  Resolution: {row.get('COMPLETION_NOTE_MAXIMUM', 'N/A')}"
        )
        details.append(entry)
    return "\n".join(details)


# --- Helper Function: Get LLM Summary ---
def get_llm_summary(llm_model, product_name, ticket_data_df):
    """Calls the LLM to generate the 5-section summary.

    Returns:
        tuple: (summary_text, prompt_text, success_flag, error_message)
    """
    if llm_model is None:
        return (
            "LLM is not available (check API key and setup).",
            "",
            False,
            "LLM not initialized",
        )
    if ticket_data_df.empty:
        return f"No ticket data for {product_name}.", "", False, "No input ticket data"

    formatted_data = format_ticket_data_for_prompt(ticket_data_df)

    # Prompt based on requirements doc: 5 sections, stick to facts
    prompt = f"""
    Generate a concise "storytelling" summary for a customer's '{product_name}' service experience based ONLY on the provided ticket history.
    Structure the summary into these exact five sections:
    1. Initial Issue: Timeframe, Ticket Numbers, Narrative of first problems.
    2. Follow-ups: Timeframe, Ticket Numbers, Narrative of subsequent interactions/actions.
    3. Developments: Timeframe, Ticket Numbers, Narrative of significant changes or resolution progress.
    4. Later Incidents: Timeframe, Ticket Numbers, Narrative of recurring or new problems towards the end.
    5. Recent Events: Timeframe, Ticket Numbers, Narrative summarizing the final status based on the last ticket(s).

    Use the provided timestamps and ticket numbers. Base the narrative strictly on the descriptions and notes in the history below. Do not add speculation or infer feelings. If the history ends, state the last known event in 'Recent Events'.

    Ticket History:
    {formatted_data}

    Generate the five-section summary:
    """

    try:
        print(f"INFO: Calling LLM for {product_name} summary...")
        response = llm_model.invoke(prompt)
        print(f"INFO: LLM call for {product_name} successful.")
        summary = response.content if hasattr(response, "content") else str(response)
        return summary.strip(), prompt, True, None
    except Exception as e:
        error_msg = f"Error generating summary for {product_name}: {e}"
        print(f"ERROR: LLM call failed for {product_name}: {e}")
        return error_msg, prompt, False, str(e)


# --- Helper Function: LLM-as-a-Judge for Quality Score ---
def rate_summary_quality(llm_client, input_ticket_details, generated_summary):
    """Uses an LLM to rate the summary quality based on input."""
    if not llm_client:
        print("WARNING: LLM client not available for judging.")
        return None
    if (
        not generated_summary
        or generated_summary.startswith("Error")
        or generated_summary.startswith("LLM is not available")
        or generated_summary.startswith("No ticket data")
    ):
        print(
            f"WARNING: Cannot rate summary quality due to invalid input summary: '{generated_summary[:100]}...'"
        )
        return None  # Cannot rate an error message or missing summary

    eval_prompt = f"""
    You are an expert evaluator assessing the quality of an AI-generated summary based *only* on the provided Ticket History.
    Rate the summary on a scale of 1 to 5 (where 1 is poor, 5 is excellent) based on these criteria:
    1. Faithfulness: Does the summary accurately reflect the events and details in the Ticket History without adding information or making assumptions?
    2. Structure: Does the summary strictly follow the required 5 sections (Initial Issue, Follow-ups, Developments, Later Incidents, Recent Events)? Check for all 5 section headings.
    3. Clarity & Conciseness: Is the summary easy to understand and reasonably concise without omitting key events mentioned in the history?
    4. Relevance: Does the summary focus on the key events and interactions from the history, using provided ticket numbers and dates where appropriate?

    Ticket History:
    {input_ticket_details}

    ---
    Generated Summary to Evaluate:
    {generated_summary}
    ---

    Based on the criteria above and the provided Ticket History, please provide a single integer score from 1 to 5.
    Your response should ONLY contain the integer score (e.g., respond with '4').
    Score:
    """0
    try:
        print("INFO: Calling LLM judge for quality score...")
        response = llm_client.invoke(eval_prompt)
        content = response.content if hasattr(response, "content") else str(response)
        print(f"INFO: LLM judge raw response: '{content}'")

        # Attempt to parse the score (look for a single digit 1-5)
        match = re.search(r"\b([1-5])\b", content.strip())
        if match:
            score = int(match.group(1))
            print(f"INFO: Parsed quality score: {score}")
            return score
        else:
            print(
                f"WARNING: Could not parse score (1-5) from LLM judge response: '{content}'"
            )
            return None  # Failed to parse
    except Exception as e:
        print(f"ERROR: LLM judge call failed: {e}")
        return None


# --- Streamlit Application UI and Logic ---

# Basic page setup
st.set_page_config(layout="wide", page_title="Ticket Summarizer")
st.title("Customer Ticket Summarizer & Evaluator")

# Show LLM status prominently if there was an issue
if llm is None:
    st.error(
        f"LLM Service Error: Could not initialize. Reason: {initialization_error}. Summaries & evaluation disabled."
    )

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload Ticket Data File (.txt or .csv format)", type=["txt", "csv"]
)

# Initialize variables
df_filtered = pd.DataFrame()
data_loaded_successfully = False

if uploaded_file is not None:
    # --- Data Loading and Initial Processing ---
    try:
        file_content_bytes = uploaded_file.getvalue()
        string_data = file_content_bytes.decode("utf-8")
        string_io = io.StringIO(string_data)

        # Determine separator for CSV potential
        separator = ","  # Default assumption
        if uploaded_file.name.lower().endswith(".txt"):
            # Could still be comma-separated, or tab, pipe etc. Let's assume comma for now, might need adjustment.
            print("INFO: Processing .txt file, assuming comma separator.")
        # Read CSV
        df_tickets = pd.read_csv(string_io, sep=separator)
        st.success(f"File '{uploaded_file.name}' Loaded.")
        data_loaded_successfully = True

        # Check for necessary columns before filtering
        required_columns = [
            "SERVICE_CATEGORY",
            "CUSTOMER_NUMBER",
            "ACCEPTANCE_TIME",
            "COMPLETION_TIME",
        ]  # Add others if needed by formatting function
        missing_cols = [
            col for col in required_columns if col not in df_tickets.columns
        ]
        if missing_cols:
            st.error(f"File Error: Missing required columns: {', '.join(missing_cols)}")
            data_loaded_successfully = False
        else:
            # Filter based on required service categories (from requirements doc)
            required_categories = ["HDW", "NET", "KAI", "KAV", "GIGA", "VOD", "KAD"]
            df_filtered = df_tickets[
                df_tickets["SERVICE_CATEGORY"].isin(required_categories)
            ].copy()

            if df_filtered.empty:
                st.warning(
                    "No relevant tickets found in the file based on required service categories."
                )
                # Keep data_loaded_successfully as True, but df_filtered is empty
            else:
                st.write(f"Found {len(df_filtered)} relevant tickets after filtering.")

    except pd.errors.EmptyDataError:
        st.error("Upload Error: The provided file appears to be empty.")
        data_loaded_successfully = False
    except UnicodeDecodeError:
        st.error(
            "File Read Error: Could not decode the file using UTF-8. Please ensure it's a valid text file."
        )
        data_loaded_successfully = False
    except Exception as e:
        st.error(
            f"File Processing Error: Could not read or parse the file. Is it a valid '{separator}' separated file? Details: {e}"
        )
        data_loaded_successfully = False
        st.exception(e)  # Show full traceback for debugging complex parse errors

# --- Main Content Area (requires successful data load with relevant tickets) ---
if data_loaded_successfully and not df_filtered.empty:  # Ensure we have filtered data
    # --- Bonus Analysis Section ---
    st.sidebar.title("Options")
    if px is not None:  # Only show checkbox if Plotly is available
        show_analysis = st.sidebar.checkbox(
            "Show Data Analysis Insights", value=False
        )  # Default to false
    else:
        show_analysis = False
        st.sidebar.warning("Plotly library needed for charts.")

    if show_analysis:
        # (Analysis section code remains the same as before - omitted here for brevity)
        # ... (Place the full analysis code block here) ...
        st.markdown("---")  # Separator after analysis

    # --- Customer Summarization Section ---
    st.header("Customer Storytelling Summary")

    if "CUSTOMER_NUMBER" in df_filtered.columns:
        customer_list = sorted(df_filtered["CUSTOMER_NUMBER"].astype(str).unique())
        selected_customer = st.selectbox(
            "Select Customer Number:", options=[""] + customer_list
        )
    else:
        st.warning(
            "Column 'CUSTOMER_NUMBER' not found in the filtered data. Cannot select customer."
        )
        selected_customer = ""

    # Only run summary logic if a customer is selected and the required column exists
    if selected_customer and "CUSTOMER_NUMBER" in df_filtered.columns:
        df_customer = df_filtered[
            df_filtered["CUSTOMER_NUMBER"].astype(str) == selected_customer
        ].copy()
        st.write(
            f"Processing {len(df_customer)} relevant tickets for customer **{selected_customer}**..."
        )

        # Prepare data for summarization (mapping, time conversion/sorting)
        category_to_product = {
            "KAI": "Broadband",
            "NET": "Broadband",
            "KAV": "Voice",
            "KAD": "TV",
            "GIGA": "GIGA",
            "VOD": "VOD",
            "HDW": "Hardware",
        }
        df_customer["PRODUCT_CATEGORY"] = (
            df_customer["SERVICE_CATEGORY"].map(category_to_product).fillna("Other")
        )

        summary_time_ok = False
        try:
            df_customer["ACCEPTANCE_TIME"] = pd.to_datetime(
                df_customer["ACCEPTANCE_TIME"], errors="coerce"
            )
            df_customer["COMPLETION_TIME"] = pd.to_datetime(
                df_customer["COMPLETION_TIME"], errors="coerce"
            )
            df_customer = df_customer.dropna(subset=["ACCEPTANCE_TIME"]).copy()
            if not df_customer.empty:
                df_customer = df_customer.sort_values(by="ACCEPTANCE_TIME")
                summary_time_ok = True
            else:
                st.warning(
                    "No tickets remaining for this customer after removing entries with invalid Acceptance Time."
                )

        except Exception as e:
            st.warning(
                f"Summary Warning: Time data processing issue, summaries might be incomplete: {e}"
            )

        # Check if LLM is ready, data processing was okay, and there's data left
        if llm is None:
            st.warning("LLM not available, cannot generate summaries or evaluations.")
        elif not summary_time_ok:
            st.warning(
                "Cannot generate summaries due to time data issues or lack of valid tickets."
            )
        elif df_customer.empty:
            st.info(
                f"No valid ticket data found for customer {selected_customer} to summarize after time processing."
            )
        else:
            # --- MLflow Run Start ---
            evaluation_results = []
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"Summarize_{selected_customer}_{run_timestamp}"

            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                st.info(f"Starting MLflow Run: {run_id}")
                print(f"INFO: MLflow Run Started: {run_id} ({run_name})")

                # Log Parameters
                mlflow.log_param("customer_number", selected_customer)
                mlflow.log_param("llm_model", llm_model_name)
                if uploaded_file:
                    mlflow.log_param("input_filename", uploaded_file.name)
                mlflow.log_param(
                    "total_relevant_tickets_for_customer", len(df_customer)
                )
                mlflow.log_param("evaluation_method", "LLM-as-a-Judge")

                grouped_by_product = df_customer.groupby("PRODUCT_CATEGORY")
                product_categories_found = sorted(
                    df_customer["PRODUCT_CATEGORY"].unique()
                )

                st.subheader("Generated Summaries & Evaluation")
                for product_name in product_categories_found:
                    st.markdown(f"#### Summary for: {product_name}")
                    group_df = grouped_by_product.get_group(product_name).copy()

                    # Select columns needed (adjust if your formatting function needs more)
                    summary_input_cols = [
                        "ORDER_NUMBER",
                        "ACCEPTANCE_TIME",
                        "COMPLETION_TIME",
                        "SERVICE_CATEGORY",
                        "ORDER_DESCRIPTION_1",
                        "ORDER_DESCRIPTION_2",
                        "ORDER_DESCRIPTION_3_MAXIMUM",
                        "NOTE_MAXIMUM",
                        "COMPLETION_NOTE_MAXIMUM",
                    ]
                    cols_to_select = [
                        col for col in summary_input_cols if col in group_df.columns
                    ]
                    # Ensure PRODUCT_CATEGORY is available if needed later, though not directly used in default format function
                    if (
                        "PRODUCT_CATEGORY" not in cols_to_select
                        and "PRODUCT_CATEGORY" in group_df.columns
                    ):
                        cols_to_select.append("PRODUCT_CATEGORY")
                    summary_input_data = group_df[cols_to_select].copy()

                    # --- Generate Summary ---
                    start_time = time.time()
                    summary_text, prompt_text, success, error_msg = get_llm_summary(
                        llm, product_name, summary_input_data
                    )
                    end_time = time.time()
                    duration = end_time - start_time

                    st.markdown(summary_text)  # Display result (summary or error)

                    # --- Evaluate Summary Quality (LLM-as-a-Judge) ---
                    quality_score = None  # Initialize score
                    if success:
                        # Format the input data specifically for the judge prompt
                        formatted_input_for_judge = format_ticket_data_for_prompt(
                            summary_input_data
                        )
                        # Call the judge LLM - use the same LLM instance for simplicity
                        quality_score = rate_summary_quality(
                            llm, formatted_input_for_judge, summary_text
                        )
                    else:
                        quality_score = None  # Can't evaluate if summary failed

                    # --- Log Metrics & Collect Results ---
                    input_char_count = len(prompt_text) if success or prompt_text else 0
                    output_char_count = len(summary_text) if success else 0
                    num_tickets_in_group = len(summary_input_data)

                    # Log metrics with product prefix
                    mlflow.log_metric(f"{product_name}_duration_sec", duration)
                    mlflow.log_metric(f"{product_name}_input_chars", input_char_count)
                    mlflow.log_metric(f"{product_name}_output_chars", output_char_count)
                    mlflow.log_metric(f"{product_name}_success", 1 if success else 0)
                    mlflow.log_metric(
                        f"{product_name}_ticket_count", num_tickets_in_group
                    )
                    if quality_score is not None:
                        mlflow.log_metric(
                            f"{product_name}_quality_score", quality_score
                        )

                    # Append results for CSV
                    evaluation_results.append(
                        {
                            "timestamp": run_timestamp,
                            "mlflow_run_id": run_id,
                            "customer_number": selected_customer,
                            "product_category": product_name,
                            "num_tickets": num_tickets_in_group,
                            "llm_model": llm_model_name,
                            "duration_sec": round(duration, 3),
                            "input_chars": input_char_count,
                            "output_chars": output_char_count,
                            "success": success,
                            "quality_score": quality_score,  # Add the new score
                            "error_message": error_msg if not success else None,
                            "input_filename": uploaded_file.name
                            if uploaded_file
                            else "N/A",
                        }
                    )

                    # Update Streamlit status
                    quality_score_display = (
                        quality_score if quality_score is not None else "N/A"
                    )
                    st.markdown(
                        f"> Evaluation Metrics: Duration={duration:.2f}s, Input Chars={input_char_count}, Output Chars={output_char_count}, Success={success}, Quality Score={quality_score_display}"
                    )
                    st.markdown("---")  # Separator

                # --- Save Evaluation CSV and Log as Artifact ---
                if evaluation_results:
                    try:
                        eval_df = pd.DataFrame(evaluation_results)
                        safe_customer_id = str(selected_customer).replace(" ", "_")
                        csv_filename = f"eval_{safe_customer_id}_{run_id}.csv"
                        csv_path = os.path.join(EVALS_DIR, csv_filename)

                        eval_df.to_csv(csv_path, index=False)
                        st.success(f"Evaluation results saved to: {csv_path}")
                        print(f"INFO: Evaluation CSV saved to {csv_path}")

                        # Log the CSV artifact to MLflow
                        mlflow.log_artifact(csv_path)
                        print(f"INFO: Logged {csv_filename} as MLflow artifact.")

                    except Exception as e:
                        st.error(f"Failed to save or log evaluation CSV: {e}")
                        print(f"ERROR: Failed to save/log evaluation CSV: {e}")
                else:
                    st.warning("No evaluation results were generated to save.")

            # MLflow run automatically ends here
            st.info(f"MLflow Run '{run_id}' finished.")
            print(f"INFO: MLflow Run Finished: {run_id}")


elif uploaded_file is None:
    st.info("Upload a ticket data file (.txt or .csv) to start.")
elif not data_loaded_successfully:
    st.warning("Data loading failed. Please check the file format and content.")
elif df_filtered.empty and data_loaded_successfully:
    st.warning(
        "File loaded, but no relevant tickets found based on required service categories. Cannot proceed."
    )

# --- Footer or final messages ---
st.markdown("---")
st.caption("Ticket Summarizer App with MLflow & LLM-as-Judge Evaluation")
