from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import json
import subprocess # To run onedrive.py
import os
import time
from datetime import datetime, timezone

app = Flask(__name__)

# --- Proxy Configuration Information ---
# (Proxy comments remain the same as your previous version)
# To use a proxy (like v2rayng) for API calls:
# 1. Ensure your v2rayng client is running and configured to provide a local
#    SOCKS5 or HTTP proxy. Common defaults:
#    - SOCKS5: 127.0.0.1 port 10808
#    - HTTP:   127.0.0.1 port 10809
# 2. Set the HTTP_PROXY and HTTPS_PROXY environment variables in your terminal
#    BEFORE running this script. The 'openai' library (using 'httpx') and
#    standard libraries like 'requests' (if used in onedrive.py) will
#    automatically use these.
#
#    Example for SOCKS5 proxy on 127.0.0.1:10808:
#    Linux/macOS:
#      export HTTP_PROXY="socks5h://127.0.0.1:10808"
#      export HTTPS_PROXY="socks5h://127.0.0.1:10808"
#
# The vless:// URL you provided is a configuration for your v2ray client,
# not directly usable by Python's requests/httpx libraries.
# -----------------------------------------

# --- Global Statistics for Server Session ---
session_total_queries = 0
session_total_cost = 0.0
session_total_response_time_ms = 0.0 # Sum of durations in milliseconds

class OpenRouterAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        # Using pricing from your last DeepSeekAPI version for demonstration
        self.input_cost_per_1m = 0.55
        self.output_cost_per_1m = 2.19

    def ask_question(self, question, document_content, user_login="User"):
        global session_total_queries, session_total_cost, session_total_response_time_ms # Allow modification

        current_time_utc_start = datetime.now(timezone.utc)
        print(f"\n[{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] --- OpenRouterAPI.ask_question ---")
        # ... (rest of the initial print statements for question, doc length, etc.)
        print(f"    User Login: '{user_login}'")
        print(f"    Received Question: '{question}'")
        print(f"    Received Document Content Length: {len(document_content)} characters.")

        if not document_content or len(document_content) < 10:
            print(f"    [{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] CRITICAL WARNING: Document content is very short or empty (Length: {len(document_content)}). API calls will likely fail or produce poor results.")

        system_prompt_content = "You are a helpful assistant."
        fixed_prefix = "solely generate single sql query to answer user question based on the following explanation:"
        user_prompt_content = f"{fixed_prefix}\nuser question: {question.strip()}\n{document_content}"

        # ... (rest of the print statements for prompts) ...
        print(f"    [{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] Sending request to OpenRouter API...")
        print(f"    --- SYSTEM PROMPT CONTENT (for API messages) ---:\n{system_prompt_content}\n    -----------------------------")
        print(f"    --- USER PROMPT CONTENT (STRUCTURE CHECK for API messages) ---:")
        print(f"        Fixed Prefix: \"{fixed_prefix}\"")
        print(f"        User Question: \"{question.strip()}\"")
        print(f"        Document Content Length Appended: {len(document_content)} characters")
        print(f"    --- USER PROMPT CONTENT (first 400 chars for API messages) ---:\n{user_prompt_content[:400]}...\n    -----------------------------")
        if len(user_prompt_content) > 400:
             print(f"    --- USER PROMPT CONTENT (last 400 chars for API messages if long) ---:\n...{user_prompt_content[-400:]}\n    -----------------------------")
        print(f"    Total length of user_prompt_content being sent: {len(user_prompt_content)} characters.")


        your_site_url = "http://localhost:5000"
        your_site_name = "DocuQuery SQL Bot"

        extra_headers = {
            "HTTP-Referer": your_site_url,
            "X-Title": your_site_name,
        }

        api_call_start_time = time.perf_counter() # For duration measurement

        try:
            response = self.client.chat.completions.create(
                extra_headers=extra_headers,
                model="deepseek/deepseek-r1-0528:free",
                messages=[
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": user_prompt_content}
                ],
                stream=False,
                temperature=0.2,
                max_tokens=2000
            )

            api_call_end_time = time.perf_counter()
            api_duration_ms = (api_call_end_time - api_call_start_time) * 1000
            print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] API Call Duration: {api_duration_ms:.2f} ms")

            answer = ""
            if response.choices and response.choices[0].message:
                answer = response.choices[0].message.content
            else:
                print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] WARNING: API response.choices was empty or message attribute was missing.")
                answer = "[API Call Successful but no message content in choices]"

            print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] API Raw Answer Received: '{answer}'")

            usage_info = {}
            current_query_cost = 0.0
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', prompt_tokens + completion_tokens) # Calculate if not present

                usage_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }

                # Calculate cost for this query
                input_cost = (prompt_tokens / 1_000_000) * self.input_cost_per_1m
                output_cost = (completion_tokens / 1_000_000) * self.output_cost_per_1m
                current_query_cost = input_cost + output_cost
                usage_info["query_cost"] = round(current_query_cost, 6) # Add cost to usage_info

                print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] API Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}, Cost for this query: ${current_query_cost:.6f}")

                # Update global session statistics
                session_total_queries += 1
                session_total_cost += current_query_cost
                session_total_response_time_ms += api_duration_ms

            else:
                print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] API response did not contain standard 'usage' information. Cost cannot be calculated.")
                # Still count as a query, but with 0 cost and recorded duration if successful
                session_total_queries += 1
                session_total_response_time_ms += api_duration_ms


            print(f"[{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] --- OpenRouterAPI.ask_question finished successfully ---")
            return {
                "success": True,
                "answer": answer,
                "model_used": "deepseek/deepseek-r1-0528:free",
                "usage": usage_info, # Contains tokens and query_cost now
                "duration_ms": round(api_duration_ms, 2), # Send duration to frontend
                "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            api_call_end_time = time.perf_counter() # Measure duration even on error if possible
            api_duration_ms = (api_call_end_time - api_call_start_time) * 1000
            print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] API Call Duration (before error): {api_duration_ms:.2f} ms")
            session_total_response_time_ms += api_duration_ms # Add error duration too

            error_msg = str(e)
            print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] OpenRouter API Error: {error_msg}")
            # ... (detailed error logging remains the same)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    err_details = e.response.json()
                    print(f"    API Error Details: {json.dumps(err_details, indent=2)}")
                    error_msg_detail = err_details.get('error', {}).get('message', '')
                    if error_msg_detail: error_msg += f" (API Detail: {error_msg_detail})"
                except json.JSONDecodeError:
                    error_text = e.response.text
                    print(f"    API Error Response (not JSON): {error_text}")
                    error_msg += f" (API Response: {error_text[:200]})"
            print(f"[{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] --- OpenRouterAPI.ask_question finished with error ---")
            return {
                "success": False,
                "error": f"OpenRouter API Error: {error_msg}",
                "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                "duration_ms": round(api_duration_ms, 2)
            }

# --- Global variables and Document Handling ---
OPENROUTER_API_KEY = "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"

if OPENROUTER_API_KEY == "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":
    print("INFO: Using the specific OpenRouter API key provided by the user.")
elif not OPENROUTER_API_KEY or "YOUR_OPENROUTER_API_KEY_HERE" in OPENROUTER_API_KEY: # Generic placeholder check
    print("###################################################################################")
    print("CRITICAL WARNING: OpenRouter API key is NOT SET or is a placeholder!")
    print("Please set your valid OPENROUTER_API_KEY environment variable or define it in the code.")
    print("###################################################################################")
else:
    print(f"INFO: OpenRouter API Key configured (ending with ...{OPENROUTER_API_KEY[-4:] if len(OPENROUTER_API_KEY) > 4 else '****'}).")

open_router_api = OpenRouterAPI(OPENROUTER_API_KEY)

DEFAULT_DOC_FILENAME = "extracted_edit_url.txt"
current_document = ""
document_loaded = False
initial_default_content = ""
source_of_current_document = "None"

def refresh_local_document_content(is_initial_load=False):
    global initial_default_content, current_document, document_loaded, source_of_current_document, DEFAULT_DOC_FILENAME
    log_ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_doc_path = os.path.join(script_dir, DEFAULT_DOC_FILENAME)
        if os.path.exists(default_doc_path):
            with open(default_doc_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            if text_content and text_content.strip():
                if is_initial_load:
                    initial_default_content = text_content
                current_document = text_content
                document_loaded = True
                source_of_current_document = f"Local File ({DEFAULT_DOC_FILENAME})"
                print(f"[{log_ts}] Successfully loaded/refreshed local document: {DEFAULT_DOC_FILENAME} ({len(current_document)} chars)")
                return True
            else:
                print(f"[{log_ts}] Local document '{DEFAULT_DOC_FILENAME}' found but is empty.")
                if is_initial_load:
                    initial_default_content = ""
                    current_document = ""
                    document_loaded = False
                    source_of_current_document = "None (empty default file)"
                return False
        else:
            print(f"[{log_ts}] Local document '{DEFAULT_DOC_FILENAME}' not found.")
            if is_initial_load:
                initial_default_content = ""
                current_document = ""
                document_loaded = False
                source_of_current_document = "None (default file not found)"
            return False
    except Exception as e:
        print(f"[{log_ts}] Error loading/refreshing local document '{DEFAULT_DOC_FILENAME}': {e}")
        if is_initial_load:
            initial_default_content = ""
            current_document = ""
            document_loaded = False
            source_of_current_document = "None (error loading default)"
        return False

refresh_local_document_content(is_initial_load=True)
user_login = "bamavadat"

# --- Flask Routes ---
@app.route('/')
def index_route():
    return render_template('index.html', user_login=user_login)

@app.route('/sync_onedrive_document', methods=['POST'])
def sync_onedrive_document_route():
    # ... (This route remains identical to your previous correct version) ...
    global current_document, document_loaded, source_of_current_document
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onedrive_script_path = os.path.join(script_dir, "onedrive.py")
    log_ts_start = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists(onedrive_script_path):
        print(f"[{log_ts_start}] Error: onedrive.py script not found at {onedrive_script_path}")
        return jsonify({"success": False, "error": "onedrive.py script not found in the application directory."})
    try:
        print(f"[{log_ts_start}] Attempting to run onedrive.py to update '{DEFAULT_DOC_FILENAME}'...")
        current_env = os.environ.copy()
        process = subprocess.run(
            ['python', onedrive_script_path], capture_output=True, text=True,
            check=False, timeout=120, env=current_env
        )
        log_ts_end_script = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        if process.returncode == 0:
            print(f"[{log_ts_end_script}] onedrive.py executed successfully.")
            if process.stdout: print(f"    onedrive.py stdout: {process.stdout.strip()}")
            if process.stderr: print(f"    onedrive.py stderr (may include normal info): {process.stderr.strip()}")
            if refresh_local_document_content():
                return jsonify({
                    "success": True,
                    "message": f"Successfully synced and reloaded '{DEFAULT_DOC_FILENAME}' ({len(current_document)} chars).",
                    "preview": current_document[:500] + ("..." if len(current_document) > 500 else ""),
                    "character_count": len(current_document),
                    "source": source_of_current_document,
                    "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                return jsonify({"success": False, "error": f"onedrive.py ran, but failed to reload/validate '{DEFAULT_DOC_FILENAME}'."})
        else:
            print(f"[{log_ts_end_script}] onedrive.py execution failed. RC: {process.returncode}.")
            if process.stdout: print(f"    onedrive.py stdout: {process.stdout.strip()}")
            if process.stderr: print(f"    onedrive.py stderr: {process.stderr.strip()}")
            return jsonify({"success": False, "error": f"onedrive.py script failed. Error: {process.stderr[:250].strip() if process.stderr else 'No stderr.'}"})
    except Exception as e:
        print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] Error in sync route: {e}")
        return jsonify({"success": False, "error": f"Sync error: {str(e)}"})


@app.route('/ask_question', methods=['POST'])
def ask_question_route():
    # ... (This route remains identical to your previous correct version, calling open_router_api.ask_question) ...
    global current_document, document_loaded
    if not document_loaded or not current_document or not current_document.strip():
        return jsonify({"success": False, "error": "No document loaded or document is empty. Please load or sync a document first."})
    data = request.get_json()
    question_text = data.get('question', '').strip()
    if not question_text: return jsonify({"success": False, "error": "Please provide a question"})

    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] User {user_login} asked: '{question_text}' using document: {source_of_current_document} (Length: {len(current_document)})")
    result = open_router_api.ask_question(question_text, current_document, user_login)
    return jsonify(result)

@app.route('/status')
def status_route():
    # ... (This route remains identical to your previous correct version) ...
    global current_document, document_loaded, source_of_current_document
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    doc_length = len(current_document) if current_document else 0
    is_doc_effectively_loaded = document_loaded and bool(current_document and current_document.strip())
    current_source = source_of_current_document if is_doc_effectively_loaded else "None (No document active)"

    return jsonify({
        "loaded": is_doc_effectively_loaded,
        "characters": doc_length,
        "preview": current_document[:300] + ("..." if doc_length > 300 else "") if current_document else "[No document content currently loaded]",
        "user": user_login, "timestamp": ts,
        "source": current_source
    })

@app.route('/clear')
def clear_route():
    # ... (This route remains identical to your previous correct version) ...
    # Note: "Reset to Initial Local Doc" button was removed from UI,
    # so this route might not be directly callable by user in the current UI design.
    # It still correctly resets to the initial_default_content if available.
    global current_document, document_loaded, initial_default_content, source_of_current_document, DEFAULT_DOC_FILENAME
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    message = ""
    preview_text = ""
    char_count = 0

    if initial_default_content:
        current_document = initial_default_content
        document_loaded = True
        source_of_current_document = f"Local File ({DEFAULT_DOC_FILENAME}) - Reset to Initial"
        message = f"Document reset to initial cached version of '{DEFAULT_DOC_FILENAME}' ({len(current_document)} chars)."
        preview_text = current_document[:500] + ("..." if len(current_document) > 500 else "")
        char_count = len(current_document)
        print(f"[{ts}] User {user_login} cleared. Reverted to initial cache of: {DEFAULT_DOC_FILENAME}")
    else:
        current_document = ""
        document_loaded = False
        source_of_current_document = "None"
        message = "Document cleared. No initial cached version was available to restore."
        print(f"[{ts}] User {user_login} cleared. No initial cache to revert to.")

    return jsonify({
        "success": True, "message": message, "timestamp": ts,
        "character_count": char_count, "loaded": document_loaded,
        "source": source_of_current_document,
        "preview": preview_text
    })


@app.route('/test_api')
def test_api_route():
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # Corrected the API key check:
    is_the_specific_user_key = OPENROUTER_API_KEY == "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"
    is_a_generic_placeholder = "YOUR_OPENROUTER_API_KEY_HERE" in OPENROUTER_API_KEY # A more general placeholder

    if is_a_generic_placeholder: # If it's a generic placeholder, it's definitely wrong.
        print(f"[{ts}] API Test: Using a generic placeholder API key. Test will likely fail authentication.")
        return jsonify({"success": False, "error": "OpenRouter API key is a generic placeholder. Cannot test.", "timestamp": ts})

    if is_the_specific_user_key:
        print(f"[{ts}] API Test: Proceeding with the specific OpenRouter API key provided by user. Ensure it is valid and not rate-limited.")
    elif not OPENROUTER_API_KEY:
         print(f"[{ts}] API Test: OpenRouter API key is not set. Test will fail.")
         return jsonify({"success": False, "error": "OpenRouter API key not configured. Cannot test.", "timestamp": ts})
    # If key is set and not the example one, and not generic placeholder, assume it's a user's custom key.

    sys_prompt = "You are a helpful AI assistant. Respond with a simple greeting."
    usr_prompt = "Hello AI, this is a connectivity test. Are you operational?"
    try:
        response = open_router_api.client.chat.completions.create(
            model="openrouter/auto", # Using auto-routing for test
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
            temperature=0.5, max_tokens=50,
            extra_headers={
                "HTTP-Referer": "http://localhost:5000/test_api_referer", # Updated referer
                "X-Title": "DocuQuery API Test",  # Updated title
            }
        )
        answer = response.choices[0].message.content if response.choices and response.choices[0].message else "[No content in test response]"
        return jsonify({"success": True, "answer": answer, "model_used": "openrouter/auto (or routed)", "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})
    except Exception as e:
        return jsonify({"success": False, "error": f"OpenRouter API test error: {str(e)}", "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})

@app.route('/api_info')
def api_info_route():
    global session_total_queries, session_total_cost, session_total_response_time_ms

    avg_cost = 0
    if session_total_queries > 0:
        avg_cost = session_total_cost / session_total_queries

    avg_response_time_ms = 0
    if session_total_queries > 0:
        avg_response_time_ms = session_total_response_time_ms / session_total_queries

    return jsonify({
        "api_provider": "OpenRouter.ai",
        "default_model_configured": "deepseek/deepseek-r1-0528:free",
        "current_user": user_login,
        "current_time_utc": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        "api_endpoint": "https://openrouter.ai/api/v1",
        "session_stats": {
            "total_queries": session_total_queries,
            "total_cost_usd": round(session_total_cost, 8), # More precision for cost
            "total_response_time_ms": round(session_total_response_time_ms, 2),
            "average_cost_per_query_usd": round(avg_cost, 8),
            "average_response_time_ms_per_query": round(avg_response_time_ms, 2)
        }
    })

if __name__ == '__main__':
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(templates_dir): os.makedirs(templates_dir)
    index_html_path = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_html_path): print(f"WARNING: HTML template '{index_html_path}' not found.")

    print("="*70)
    print("🚀 DocuQuery SQL Bot with OpenRouter API (Enhanced Stats)")
    print("="*70)
    print(f"📅 Current UTC Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 User Login: {user_login}")
    print(f"🤖 AI Model: deepseek/deepseek-r1-0528:free (via OpenRouter)")

    if OPENROUTER_API_KEY == "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":
        print(f"🔑 OpenRouter API Key: Using the specific key provided by user (ending with ...{OPENROUTER_API_KEY[-4:]}).")
        print("     Ensure this key is valid and active.")
    elif "YOUR_OPENROUTER_API_KEY_HERE" in OPENROUTER_API_KEY or not OPENROUTER_API_KEY: # Generic placeholder
        print(f"🔑 OpenRouter API Key: USING GENERIC PLACEHOLDER OR NOT SET! Please set your valid key.")
    else: # User has set some other key
        print(f"🔑 OpenRouter API Key: Configured (ending with ...{OPENROUTER_API_KEY[-4:] if len(OPENROUTER_API_KEY) > 4 else '****'}).")

    print(f"📄 Local Document File: '{DEFAULT_DOC_FILENAME}'")
    if initial_default_content:
        print(f"    ✓ Initial local document '{DEFAULT_DOC_FILENAME}' loaded ({len(initial_default_content)} chars).")
    else:
        print(f"    ✗ Initial local document '{DEFAULT_DOC_FILENAME}' NOT loaded or is empty.")
    print("="*70)
    print("📢 To use a PROXY (like v2rayng):")
    print("   Set HTTP_PROXY and HTTPS_PROXY environment variables before running this script.")
    print("   Example (SOCKS5 on 10808): export HTTPS_PROXY=socks5h://127.0.0.1:10808")
    print("="*70)
    print("📍 Open your browser and go to: http://localhost:5000")
    print("🔧 Test API endpoint: http://localhost:5000/test_api")
    print("📊 API & Session Stats endpoint: http://localhost:5000/api_info")
    print("🔄 Sync Document: Button in UI calls /sync_onedrive_document to run onedrive.py")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=5000)
