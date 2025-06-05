import sys  # Required for sys.stderr
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import json
import subprocess
import os
import time
from datetime import datetime, timezone

print("INFO: app.py - Starting application.", file=sys.stderr)
# Test update <date_time>
app = Flask(__name__)
print("INFO: app.py - Flask app instance created.", file=sys.stderr)


# --- Proxy Configuration Information ---
# (Proxy comments from previous version remain relevant if you set env vars)
# -----------------------------------------

class OpenRouterAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        # Minimal init log, more detailed logging in ask_question
        print(
            f"INFO: OpenRouterAPI __init__ - Initializing with API key (status: {'Set' if api_key and not 'YOUR_ACTUAL' in api_key else 'NOT SET or Placeholder'}).",
            file=sys.stderr)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.input_cost_per_1m = 0.55
        self.output_cost_per_1m = 2.19
        # print("DEBUG: OpenRouterAPI __init__ - OpenAI client initialized for OpenRouter", file=sys.stderr)

    def ask_question(self, question, document_content, user_login="User"):
        global session_total_queries, session_total_cost, session_total_response_time_ms

        current_time_utc_start = datetime.now(timezone.utc)
        print(f"\n[{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] --- OpenRouterAPI.ask_question ---",
              file=sys.stderr)
        print(f"    User Login: '{user_login}'", file=sys.stderr)
        print(f"    Received Question: '{question}'", file=sys.stderr)
        print(f"    Received Document Content Length: {len(document_content)} characters.", file=sys.stderr)

        if not document_content or len(document_content) < 10:
            print(
                f"    [{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Document content is very short or empty (Length: {len(document_content)}).",
                file=sys.stderr)

        system_prompt_content = "You are a helpful assistant."
        fixed_prefix = "solely generate single sql query to answer user question based on the following explanation:"
        user_prompt_content = f"{fixed_prefix}\nuser question: {question.strip()}\n{document_content}"

        print(f"    [{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] Sending request to OpenRouter API...",
              file=sys.stderr)
        # Commenting out full prompt logging unless specifically needed for deep debugging
        # print(f"    --- SYSTEM PROMPT CONTENT (for API messages) ---:\n{system_prompt_content}\n    -----------------------------", file=sys.stderr)
        # print(f"    --- USER PROMPT CONTENT (first 400 chars for API messages) ---:\n{user_prompt_content[:400]}...\n    -----------------------------", file=sys.stderr)
        # if len(user_prompt_content) > 400:
        #      print(f"    --- USER PROMPT CONTENT (last 400 chars for API messages if long) ---:\n...{user_prompt_content[-400:]}\n    -----------------------------", file=sys.stderr)
        print(f"    Total length of user_prompt_content being sent: {len(user_prompt_content)} characters.",
              file=sys.stderr)

        your_site_url = "http://SQLBOT.pythonanywhere.com"
        your_site_name = "Screener SQL Bot"

        extra_headers = {
            "HTTP-Referer": your_site_url,
            "X-Title": your_site_name,
        }

        api_call_start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                extra_headers=extra_headers,
                model="deepseek/deepseek-r1-0528:free",
                messages=[
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": user_prompt_content}
                ],
                stream=False,
                max_tokens=164000
            )

            api_call_end_time = time.perf_counter()
            api_duration_ms = (api_call_end_time - api_call_start_time) * 1000
            print(
                f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] API Call Duration: {api_duration_ms:.2f} ms",
                file=sys.stderr)

            answer = ""
            if response.choices and response.choices[0].message:
                answer = response.choices[0].message.content
            else:
                print(
                    f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] WARNING: API response.choices was empty or message attribute was missing.",
                    file=sys.stderr)
                answer = "[API Call Successful but no message content in choices]"

            print(
                f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] API Raw Answer Received (first 100 chars): '{answer[:100]}...'",
                file=sys.stderr)

            usage_info = {}
            current_query_cost = 0.0
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', prompt_tokens + completion_tokens)

                usage_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }

                input_cost = (prompt_tokens / 1_000_000) * self.input_cost_per_1m
                output_cost = (completion_tokens / 1_000_000) * self.output_cost_per_1m
                current_query_cost = input_cost + output_cost
                usage_info["query_cost"] = round(current_query_cost, 6)

                print(
                    f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] API Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}, Cost for this query: ${current_query_cost:.6f}",
                    file=sys.stderr)

                session_total_queries += 1
                session_total_cost += current_query_cost
                session_total_response_time_ms += api_duration_ms
            else:
                print(
                    f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] API response did not contain standard 'usage' information. Cost cannot be calculated for this query.",
                    file=sys.stderr)
                session_total_queries += 1
                session_total_response_time_ms += api_duration_ms

            # print(f"[{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] --- OpenRouterAPI.ask_question finished successfully ---", file=sys.stderr) # Can be verbose
            return {
                "success": True,
                "answer": answer,
                "model_used": "deepseek/deepseek-r1-0528:free",
                "usage": usage_info,
                "duration_ms": round(api_duration_ms, 2),
                "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            api_call_end_time = time.perf_counter()
            api_duration_ms = (api_call_end_time - api_call_start_time) * 1000
            print(
                f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] API Call Duration (before error): {api_duration_ms:.2f} ms",
                file=sys.stderr)
            session_total_response_time_ms += api_duration_ms

            error_msg = str(e)
            print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] OpenRouter API Error: {error_msg}",
                  file=sys.stderr)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    err_details = e.response.json()
                    print(f"    API Error Details: {json.dumps(err_details, indent=2)}", file=sys.stderr)
                    error_msg_detail = err_details.get('error', {}).get('message', '')
                    if error_msg_detail: error_msg += f" (API Detail: {error_msg_detail})"
                except json.JSONDecodeError:
                    error_text = e.response.text
                    print(f"    API Error Response (not JSON): {error_text}", file=sys.stderr)
                    error_msg += f" (API Response: {error_text[:200]})"
            # print(f"[{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] --- OpenRouterAPI.ask_question finished with error ---", file=sys.stderr) # Can be verbose
            return {
                "success": False,
                "error": f"OpenRouter API Error: {error_msg}",
                "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                "duration_ms": round(api_duration_ms, 2)
            }


# --- Global variables and Document Handling ---
OPENROUTER_API_KEY = "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"
if OPENROUTER_API_KEY == "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":
    print("INFO: app.py - Using the specific OpenRouter API key provided by the user.", file=sys.stderr)
elif not OPENROUTER_API_KEY or "YOUR_OPENROUTER_API_KEY_HERE" in OPENROUTER_API_KEY:
    print("CRITICAL WARNING: app.py - OpenRouter API key is NOT SET or is a placeholder!", file=sys.stderr)
else:
    print(
        f"INFO: app.py - OpenRouter API Key configured (ending with ...{OPENROUTER_API_KEY[-4:] if OPENROUTER_API_KEY and len(OPENROUTER_API_KEY) > 4 else '****'}).",
        file=sys.stderr)

open_router_api = OpenRouterAPI(OPENROUTER_API_KEY)

DEFAULT_DOC_FILENAME = "extracted_edit_url.txt"
current_document = ""
document_loaded = False
initial_default_content = ""
source_of_current_document = "None"

session_total_queries = 0
session_total_cost = 0.0
session_total_response_time_ms = 0.0
print("INFO: app.py - Session statistics initialized.", file=sys.stderr)


def refresh_local_document_content(is_initial_load=False):
    global initial_default_content, current_document, document_loaded, source_of_current_document, DEFAULT_DOC_FILENAME
    log_ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    # print(f"DEBUG: app.py - refresh_local_document_content called. is_initial_load={is_initial_load}", file=sys.stderr) # Less verbose
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
                print(
                    f"INFO: [{log_ts}] Loaded/Refreshed local document: {DEFAULT_DOC_FILENAME} ({len(current_document)} chars)",
                    file=sys.stderr)
                return True
            else:
                print(f"WARNING: [{log_ts}] Local document '{DEFAULT_DOC_FILENAME}' found but is empty.",
                      file=sys.stderr)
                if is_initial_load:
                    initial_default_content = ""
                    current_document = ""
                    document_loaded = False
                    source_of_current_document = "None (empty default file)"
                return False
        else:
            print(f"WARNING: [{log_ts}] Local document '{DEFAULT_DOC_FILENAME}' not found.", file=sys.stderr)
            if is_initial_load:
                initial_default_content = ""
                current_document = ""
                document_loaded = False
                source_of_current_document = "None (default file not found)"
            return False
    except Exception as e:
        print(f"ERROR: [{log_ts}] Error loading/refreshing local document '{DEFAULT_DOC_FILENAME}': {e}",
              file=sys.stderr)
        if is_initial_load:
            initial_default_content = ""
            current_document = ""
            document_loaded = False
            source_of_current_document = "None (error loading default)"
        return False


refresh_local_document_content(is_initial_load=True)

user_login = "bamavadat"
print(f"INFO: app.py - User login set to: {user_login}", file=sys.stderr)


# --- Flask Routes ---
@app.route('/')
def index_route():
    # print(f"DEBUG: app.py - / route accessed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    try:
        return render_template('index.html', user_login=user_login)
    except Exception as e:
        print(f"ERROR in / route (index_route) during render_template: {e}", file=sys.stderr)
        return f"Error rendering template: {str(e)}", 500


@app.route('/sync_onedrive_document', methods=['POST'])
def sync_onedrive_document_route():
    # print(f"DEBUG: app.py - /sync_onedrive_document route accessed", file=sys.stderr)
    global current_document, document_loaded, source_of_current_document
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onedrive_script_path = os.path.join(script_dir, "onedrive.py")
    log_ts_start = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    if not os.path.exists(onedrive_script_path):
        print(f"ERROR: [{log_ts_start}] onedrive.py script not found at {onedrive_script_path}", file=sys.stderr)
        return jsonify({"success": False, "error": "onedrive.py script not found."})
    try:
        print(f"INFO: [{log_ts_start}] Attempting to run onedrive.py to update '{DEFAULT_DOC_FILENAME}'...",
              file=sys.stderr)
        current_env = os.environ.copy()
        process = subprocess.run(
            ['python', onedrive_script_path], capture_output=True, text=True,
            check=False, timeout=120, env=current_env
        )
        log_ts_end_script = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        if process.returncode == 0:
            print(f"INFO: [{log_ts_end_script}] onedrive.py executed successfully.", file=sys.stderr)
            if process.stdout: print(f"    onedrive.py stdout: {process.stdout.strip()}", file=sys.stderr)
            if process.stderr: print(f"    onedrive.py stderr (may include normal info): {process.stderr.strip()}",
                                     file=sys.stderr)
            if refresh_local_document_content():
                return jsonify({
                    "success": True,
                    "message": f"Successfully synced and reloaded '{DEFAULT_DOC_FILENAME}' ({len(current_document)} chars).",
                    "preview": current_document[:500] + ("..." if len(current_document) > 500 else ""),
                    "character_count": len(current_document),
                    "source": source_of_current_document,
                    "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                })
            else:  # refresh_local_document_content failed
                return jsonify({"success": False,
                                "error": f"onedrive.py ran, but failed to reload/validate '{DEFAULT_DOC_FILENAME}' after sync."})
        else:  # onedrive.py script failed
            print(f"ERROR: [{log_ts_end_script}] onedrive.py execution failed. RC: {process.returncode}.",
                  file=sys.stderr)
            if process.stdout: print(f"    onedrive.py stdout (on failure): {process.stdout.strip()}", file=sys.stderr)
            if process.stderr: print(f"    onedrive.py stderr (on failure): {process.stderr.strip()}", file=sys.stderr)
            return jsonify({"success": False,
                            "error": f"onedrive.py script failed. Error: {process.stderr[:250].strip() if process.stderr else 'No stderr.'}"})
    except Exception as e:
        print(f"ERROR: [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] Error in sync route: {e}",
              file=sys.stderr)
        return jsonify({"success": False, "error": f"Sync error: {str(e)}"})


@app.route('/ask_question', methods=['POST'])
def ask_question_route():
    # print(f"DEBUG: app.py - /ask_question route accessed", file=sys.stderr)
    global current_document, document_loaded
    if not document_loaded or not current_document or not current_document.strip():
        return jsonify({"success": False,
                        "error": "No document loaded or document is empty. Please load or sync a document first."})
    data = request.get_json()
    question_text = data.get('question', '').strip()
    if not question_text: return jsonify({"success": False, "error": "Please provide a question"})

    print(
        f"INFO: [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] User {user_login} asked: '{question_text}' using document: {source_of_current_document} (Length: {len(current_document)})",
        file=sys.stderr)
    result = open_router_api.ask_question(question_text, current_document, user_login)
    return jsonify(result)


@app.route('/status')
def status_route():
    # print(f"DEBUG: app.py - /status route accessed", file=sys.stderr)
    global current_document, document_loaded, source_of_current_document
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    doc_length = len(current_document) if current_document else 0
    is_doc_effectively_loaded = document_loaded and bool(current_document and current_document.strip())
    current_source = source_of_current_document if is_doc_effectively_loaded else "None (No document active)"

    return jsonify({
        "loaded": is_doc_effectively_loaded,
        "characters": doc_length,
        "preview": current_document[:300] + (
            "..." if doc_length > 300 else "") if current_document else "[No document content currently loaded]",
        "user": user_login, "timestamp": ts,
        "source": current_source
    })


@app.route('/clear')
def clear_route():
    # print(f"DEBUG: app.py - /clear route accessed", file=sys.stderr)
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
        print(f"INFO: [{ts}] User {user_login} cleared. Reverted to initial cache of: {DEFAULT_DOC_FILENAME}",
              file=sys.stderr)
    else:
        current_document = ""
        document_loaded = False
        source_of_current_document = "None"
        message = "Document cleared. No initial cached version was available to restore."
        print(f"INFO: [{ts}] User {user_login} cleared. No initial cache to revert to.", file=sys.stderr)

    return jsonify({
        "success": True, "message": message, "timestamp": ts,
        "character_count": char_count, "loaded": document_loaded,
        "source": source_of_current_document,
        "preview": preview_text
    })


@app.route('/test_api')
def test_api_route():
    # print(f"DEBUG: app.py - /test_api route accessed", file=sys.stderr)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    is_example_key = OPENROUTER_API_KEY == "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"
    generic_placeholder_check = "YOUR_OPENROUTER_API_KEY_HERE"

    if generic_placeholder_check in OPENROUTER_API_KEY:
        print(f"WARNING: [{ts}] API Test: Using a generic placeholder API key.", file=sys.stderr)
        return jsonify(
            {"success": False, "error": "OpenRouter API key is a generic placeholder. Cannot test.", "timestamp": ts})

    if is_example_key:
        print(f"INFO: [{ts}] API Test: Proceeding with the specific OpenRouter API key provided by user.",
              file=sys.stderr)
    elif not OPENROUTER_API_KEY:
        print(f"ERROR: [{ts}] API Test: OpenRouter API key is not set.", file=sys.stderr)
        return jsonify({"success": False, "error": "OpenRouter API key not configured. Cannot test.", "timestamp": ts})

    sys_prompt = "You are a helpful AI assistant. Respond with a simple greeting."
    usr_prompt = "Hello AI, this is a connectivity test. Are you operational?"
    try:
        response = open_router_api.client.chat.completions.create(
            model="openrouter/auto",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
            temperature=0.5, max_tokens=50,
            extra_headers={
                "HTTP-Referer": "http://SQLBOT.pythonanywhere.com/test_api",
                "X-Title": "Screener Bot API Test",
            }
        )
        answer = response.choices[0].message.content if response.choices and response.choices[
            0].message else "[No content in test response]"
        return jsonify({"success": True, "answer": answer, "model_used": "openrouter/auto (or routed)",
                        "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})
    except Exception as e:
        print(
            f"ERROR: [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] OpenRouter API test error: {str(e)}",
            file=sys.stderr)
        return jsonify({"success": False, "error": f"OpenRouter API test error: {str(e)}",
                        "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})


@app.route('/api_info')
def api_info_route():
    # print(f"DEBUG: app.py - /api_info route accessed", file=sys.stderr)
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
        "session_stats_server_side": {
            "total_queries_since_server_start": session_total_queries,
            "total_cost_usd_since_server_start": round(session_total_cost, 8),
            "total_response_time_ms_since_server_start": round(session_total_response_time_ms, 2),
            "average_cost_per_query_usd": round(avg_cost, 8),
            "average_response_time_ms_per_query": round(avg_response_time_ms, 2)
        }
    })


print("INFO: app.py - Flask routes defined.", file=sys.stderr)

if __name__ == '__main__':
    print("INFO: app.py - Script is being run directly (e.g., local development)", file=sys.stderr)
    templates_dir_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(templates_dir_local):
        os.makedirs(templates_dir_local)
        print(f"INFO: app.py - Created 'templates' directory for local development at {templates_dir_local}",
              file=sys.stderr)

    print("=" * 70, file=sys.stderr)
    print("🚀 DocuQuery SQL Bot with OpenRouter API (Enhanced Stats) - LOCAL DEV MODE", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    app.run(debug=True, host='0.0.0.0', port=5000)

print("INFO: app.py - End of file reached, application object 'app' should be defined and configured.", file=sys.stderr)


cd D:\Python\Screener
git mv Templates templates
git mv Onedrive onedrive
git commit -m "Fix folder name casing"
