import sys # Required for sys.stderr
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from openai import OpenAI
import json
import subprocess
import os
import time
from datetime import datetime, timezone

print("INFO: app.py - Starting application.", file=sys.stderr)

app = Flask(__name__)
print("INFO: app.py - Flask app instance created.", file=sys.stderr)

# --- Global Statistics for Server Session ---
session_total_queries = 0
session_total_cost = 0.0
session_total_response_time_ms = 0.0

class OpenRouterAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        print(f"INFO: OpenRouterAPI __init__ - Initializing with API key (status: {'Set' if api_key and not 'YOUR_ACTUAL' in api_key else 'NOT SET or Placeholder'}).", file=sys.stderr)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.input_cost_per_1m = 0.55
        self.output_cost_per_1m = 2.19
        print("INFO: OpenRouterAPI __init__ - OpenAI client initialized for OpenRouter", file=sys.stderr)

    def _calculate_cost(self, usage_data):
        if not usage_data:
            return 0.0
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)

        input_cost = (prompt_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (completion_tokens / 1_000_000) * self.output_cost_per_1m
        return input_cost + output_cost

    def ask_question_streaming(self, question, document_content, user_login="User"):
        global session_total_queries, session_total_cost, session_total_response_time_ms

        current_time_utc_start = datetime.now(timezone.utc)
        print(f"\n[{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] --- OpenRouterAPI.ask_question_streaming (SQL Bot) ---", file=sys.stderr)
        # ... (other initial prints for question, doc length etc.)

        system_prompt_content = "You are a helpful assistant."
        fixed_prefix = "solely generate single sql query to answer user question based on the following explanation:"
        user_prompt_content = f"{fixed_prefix}\nuser question: {question.strip()}\n{document_content}"

        print(f"    Total length of user_prompt_content being sent: {len(user_prompt_content)} characters.", file=sys.stderr)

        your_site_url = "http://SQLBOT.pythonanywhere.com"
        your_site_name = "Screener SQL Bot"
        extra_headers = {"HTTP-Referer": your_site_url, "X-Title": your_site_name}
        api_call_start_time = time.perf_counter()
        accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            stream = self.client.chat.completions.create(
                extra_headers=extra_headers,
                model="deepseek/deepseek-r1-0528:free",
                messages=[
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": user_prompt_content}
                ],
                stream=True # Ensure streaming
                # max_tokens parameter removed
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_delta = chunk.choices[0].delta.content
                    yield {"type": "content", "delta": content_delta}
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = chunk.usage
                    accumulated_usage["prompt_tokens"] = getattr(usage, 'prompt_tokens', 0) or accumulated_usage["prompt_tokens"]
                    accumulated_usage["completion_tokens"] = getattr(usage, 'completion_tokens', 0) or accumulated_usage["completion_tokens"]

            api_call_end_time = time.perf_counter()
            api_duration_ms = (api_call_end_time - api_call_start_time) * 1000
            print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] SQL Bot Stream ended. API Call Duration: {api_duration_ms:.2f} ms", file=sys.stderr)

            session_total_queries += 1
            session_total_response_time_ms += api_duration_ms

            current_query_cost = 0.0
            final_usage_info = {}
            if accumulated_usage["prompt_tokens"] > 0 or accumulated_usage["completion_tokens"] > 0:
                accumulated_usage["total_tokens"] = accumulated_usage["prompt_tokens"] + accumulated_usage["completion_tokens"]
                current_query_cost = self._calculate_cost(accumulated_usage)
                final_usage_info = {**accumulated_usage, "query_cost": round(current_query_cost, 8)}
                session_total_cost += current_query_cost
                print(f"    SQL Bot Stream Usage (from chunks) - Prompt: {final_usage_info.get('prompt_tokens',0)}, Completion: {final_usage_info.get('completion_tokens',0)}, Total: {final_usage_info.get('total_tokens',0)}, Query Cost: ${final_usage_info.get('query_cost',0):.8f}", file=sys.stderr)
            else:
                print(f"    SQL Bot Stream response did not yield explicit 'usage' information in chunks.", file=sys.stderr)

            yield {"type": "done", "duration_ms": round(api_duration_ms, 2), "model_used": "deepseek/deepseek-r1-0528:free", "usage": final_usage_info}

        except Exception as e:
            api_call_end_time = time.perf_counter()
            api_duration_ms = (api_call_end_time - api_call_start_time) * 1000
            session_total_response_time_ms += api_duration_ms
            error_msg = str(e)
            print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] OpenRouter API Error (ask_question_streaming): {error_msg}", file=sys.stderr)
            # ... (detailed error logging for e.response)
            yield {"type": "error", "delta": f"SQL Bot Stream API Error: {error_msg}", "duration_ms": round(api_duration_ms, 2)}

    def stream_chat_response(self, messages, user_login="User"): # For General Chat
        global session_total_queries, session_total_cost, session_total_response_time_ms

        current_time_utc_start = datetime.now(timezone.utc)
        print(f"\n[{current_time_utc_start.strftime('%Y-%m-%d %H:%M:%S')}] --- OpenRouterAPI.stream_chat_response (General Chat) ---", file=sys.stderr)
        # ... (other initial prints)

        your_site_url = "http://SQLBOT.pythonanywhere.com"
        your_site_name = "Screener General Chat"
        extra_headers = {"HTTP-Referer": your_site_url, "X-Title": your_site_name}
        api_call_start_time = time.perf_counter()
        accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            stream = self.client.chat.completions.create(
                extra_headers=extra_headers,
                model="deepseek/deepseek-r1-0528:free",
                messages=messages,
                stream=True
                # max_tokens parameter removed
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_delta = chunk.choices[0].delta.content
                    yield {"type": "content", "delta": content_delta}
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = chunk.usage
                    accumulated_usage["prompt_tokens"] = getattr(usage, 'prompt_tokens', 0) or accumulated_usage["prompt_tokens"]
                    accumulated_usage["completion_tokens"] = getattr(usage, 'completion_tokens', 0) or accumulated_usage["completion_tokens"]

            api_call_end_time = time.perf_counter()
            api_duration_ms = (api_call_end_time - api_call_start_time) * 1000
            print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] General Chat Stream ended. API Call Duration: {api_duration_ms:.2f} ms", file=sys.stderr)

            session_total_queries += 1
            session_total_response_time_ms += api_duration_ms

            current_query_cost = 0.0
            final_usage_info = {}
            if accumulated_usage["prompt_tokens"] > 0 or accumulated_usage["completion_tokens"] > 0:
                accumulated_usage["total_tokens"] = accumulated_usage["prompt_tokens"] + accumulated_usage["completion_tokens"]
                current_query_cost = self._calculate_cost(accumulated_usage)
                final_usage_info = {**accumulated_usage, "query_cost": round(current_query_cost, 8)}
                session_total_cost += current_query_cost
                print(f"    General Chat Stream Usage (from chunks) - Prompt: {final_usage_info.get('prompt_tokens',0)}, Completion: {final_usage_info.get('completion_tokens',0)}, Total: {final_usage_info.get('total_tokens',0)}, Query Cost: ${final_usage_info.get('query_cost',0):.8f}", file=sys.stderr)
            else:
                print(f"    General Chat Stream response did not yield explicit 'usage' information in chunks.", file=sys.stderr)

            yield {"type": "done", "duration_ms": round(api_duration_ms, 2), "model_used": "deepseek/deepseek-r1-0528:free", "usage": final_usage_info}

        except Exception as e:
            api_call_end_time = time.perf_counter()
            api_duration_ms = (api_call_end_time - api_call_start_time) * 1000
            session_total_response_time_ms += api_duration_ms
            error_msg = str(e)
            print(f"    [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] OpenRouter Stream API Error (General Chat): {error_msg}", file=sys.stderr)
            yield {"type": "error", "delta": f"Stream API Error: {error_msg}", "duration_ms": round(api_duration_ms, 2)}

# --- Global API Key and instance ---
OPENROUTER_API_KEY = "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"
if OPENROUTER_API_KEY == "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":
    print("INFO: app.py - Using the specific OpenRouter API key provided by the user.", file=sys.stderr)
elif not OPENROUTER_API_KEY or "YOUR_OPENROUTER_API_KEY_HERE" in OPENROUTER_API_KEY :
    print("CRITICAL WARNING: app.py - OpenRouter API key is NOT SET or is a placeholder!", file=sys.stderr)
else:
    print(f"INFO: app.py - OpenRouter API Key configured (ending with ...{OPENROUTER_API_KEY[-4:] if OPENROUTER_API_KEY and len(OPENROUTER_API_KEY) > 4 else '****'}).", file=sys.stderr)
open_router_api = OpenRouterAPI(OPENROUTER_API_KEY)

# --- Document Handling ---
DEFAULT_DOC_FILENAME = "extracted_edit_url.txt"
current_document = ""
document_loaded = False
initial_default_content = ""
source_of_current_document = "None"
print("INFO: app.py - Document handling variables initialized.", file=sys.stderr)

def refresh_local_document_content(is_initial_load=False):
    # ... (this function remains IDENTICAL to your previous correct version) ...
    global initial_default_content, current_document, document_loaded, source_of_current_document, DEFAULT_DOC_FILENAME
    log_ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_doc_path = os.path.join(script_dir, DEFAULT_DOC_FILENAME)
        if os.path.exists(default_doc_path):
            with open(default_doc_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            if text_content and text_content.strip():
                if is_initial_load: initial_default_content = text_content
                current_document = text_content
                document_loaded = True
                source_of_current_document = f"Local File ({DEFAULT_DOC_FILENAME})"
                print(f"INFO: [{log_ts}] Loaded/Refreshed: {DEFAULT_DOC_FILENAME} ({len(current_document)} chars)", file=sys.stderr)
                return True
            else:
                print(f"WARNING: [{log_ts}] Local doc '{DEFAULT_DOC_FILENAME}' empty.", file=sys.stderr)
                if is_initial_load: initial_default_content = ""; current_document = ""; document_loaded = False; source_of_current_document = "None (empty default)"
                return False
        else:
            print(f"WARNING: [{log_ts}] Local doc '{DEFAULT_DOC_FILENAME}' not found.", file=sys.stderr)
            if is_initial_load: initial_default_content = ""; current_document = ""; document_loaded = False; source_of_current_document = "None (no default file)"
            return False
    except Exception as e:
        print(f"ERROR: [{log_ts}] Loading/refreshing '{DEFAULT_DOC_FILENAME}': {e}", file=sys.stderr)
        if is_initial_load: initial_default_content = ""; current_document = ""; document_loaded = False; source_of_current_document = "None (load error)"
        return False

refresh_local_document_content(is_initial_load=True)
user_login = "bamavadat"
print(f"INFO: app.py - User login set to: {user_login}", file=sys.stderr)

# --- Flask Routes ---
@app.route('/')
def index_route():
    try:
        return render_template('index.html', user_login=user_login)
    except Exception as e:
        print(f"ERROR in / route (render_template): {e}", file=sys.stderr)
        return f"Error rendering template: {str(e)}", 500

@app.route('/sync_onedrive_document', methods=['POST'])
def sync_onedrive_document_route():
    global current_document, document_loaded, source_of_current_document
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onedrive_script_path = os.path.join(script_dir, "onedrive.py")
    log_ts_start = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists(onedrive_script_path):
        print(f"ERROR: [{log_ts_start}] onedrive.py script not found at {onedrive_script_path}", file=sys.stderr)
        return jsonify({"success": False, "error": "onedrive.py script not found."})
    try:
        print(f"INFO: [{log_ts_start}] Attempting to run onedrive.py to update '{DEFAULT_DOC_FILENAME}'...", file=sys.stderr)
        current_env = os.environ.copy()
        process = subprocess.run(
            ['python', onedrive_script_path], capture_output=True, text=True,
            check=False, timeout=120, env=current_env,
            encoding='utf-8', errors='replace' # Added encoding and error handling
        )
        log_ts_end_script = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        if process.returncode == 0:
            print(f"INFO: [{log_ts_end_script}] onedrive.py executed successfully.", file=sys.stderr)
            if process.stdout: print(f"    onedrive.py stdout: {process.stdout.strip()}", file=sys.stderr)
            if process.stderr: print(f"    onedrive.py stderr: {process.stderr.strip()}", file=sys.stderr)
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
            print(f"ERROR: [{log_ts_end_script}] onedrive.py execution failed. RC: {process.returncode}.", file=sys.stderr)
            if process.stdout: print(f"    onedrive.py stdout (on failure): {process.stdout.strip()}", file=sys.stderr)
            if process.stderr: print(f"    onedrive.py stderr (on failure): {process.stderr.strip()}", file=sys.stderr)
            return jsonify({"success": False, "error": f"onedrive.py script failed. Error: {process.stderr[:250].strip() if process.stderr else 'No stderr.'}"})
    except Exception as e:
        print(f"ERROR: [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] Error in sync route: {e}", file=sys.stderr)
        return jsonify({"success": False, "error": f"Sync error: {str(e)}"})

@app.route('/ask_question', methods=['POST']) # For SQL Bot - NOW STREAMING
def ask_question_route():
    global current_document, document_loaded
    if not document_loaded or not current_document or not current_document.strip():
        return jsonify({"error": "No document loaded or document is empty."}), 400
    data = request.get_json()
    question_text = data.get('question', '').strip()
    if not question_text:
        return jsonify({"error": "Please provide a question"}), 400

    print(f"INFO: [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] User {user_login} asked SQL Bot (streaming): '{question_text}'", file=sys.stderr)

    def generate_stream_sql():
        try:
            for chunk_data in open_router_api.ask_question_streaming(question_text, current_document, user_login):
                yield f"data: {json.dumps(chunk_data)}\n\n"
                time.sleep(0.01)
        except Exception as e:
            print(f"ERROR in generate_stream_sql: {e}", file=sys.stderr)
            error_payload = {"type": "error", "delta": f"SQL Bot Stream generation error: {str(e)}"}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return Response(stream_with_context(generate_stream_sql()), mimetype='text/event-stream')


@app.route('/stream_general_chat', methods=['POST'])
def stream_general_chat_route():
    print(f"INFO: app.py - /stream_general_chat route accessed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    data = request.get_json()
    messages = data.get('messages')

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    def generate_stream_general():
        try:
            for chunk_data in open_router_api.stream_chat_response(messages, user_login=user_login):
                yield f"data: {json.dumps(chunk_data)}\n\n"
                time.sleep(0.01)
        except Exception as e:
            print(f"ERROR in stream_general_chat generate_stream: {e}", file=sys.stderr)
            error_payload = {"type": "error", "delta": f"Stream generation error: {str(e)}"}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return Response(stream_with_context(generate_stream_general()), mimetype='text/event-stream')

@app.route('/status')
def status_route():
    global current_document, document_loaded, source_of_current_document
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    doc_length = len(current_document) if current_document else 0
    is_doc_effectively_loaded = document_loaded and bool(current_document and current_document.strip())
    current_source = source_of_current_document if is_doc_effectively_loaded else "None (No document active)"
    return jsonify({
        "loaded": is_doc_effectively_loaded, "characters": doc_length,
        "preview": current_document[:300] + ("..." if doc_length > 300 else "") if current_document else "[No doc loaded]",
        "user": user_login, "timestamp": ts, "source": current_source
    })

@app.route('/clear')
def clear_route():
    global current_document, document_loaded, initial_default_content, source_of_current_document, DEFAULT_DOC_FILENAME
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    message = ""
    preview_text = ""
    char_count = 0
    if initial_default_content:
        current_document = initial_default_content
        document_loaded = True
        source_of_current_document = f"Local File ({DEFAULT_DOC_FILENAME}) - Reset to Initial"
        message = f"Document reset to initial: '{DEFAULT_DOC_FILENAME}' ({len(current_document)} chars)."
        preview_text = current_document[:500] + ("..." if len(current_document) > 500 else "")
        char_count = len(current_document)
        print(f"INFO: [{ts}] Cleared. Reverted to initial: {DEFAULT_DOC_FILENAME}", file=sys.stderr)
    else:
        current_document = ""; document_loaded = False; source_of_current_document = "None"
        message = "Document cleared. No initial version available."
        print(f"INFO: [{ts}] Cleared. No initial version to revert to.", file=sys.stderr)
    return jsonify({"success": True, "message": message, "timestamp": ts, "character_count": char_count, "loaded": document_loaded, "source": source_of_current_document, "preview": preview_text})

@app.route('/test_api')
def test_api_route():
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    is_example_key = OPENROUTER_API_KEY == "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"
    generic_placeholder_check = "YOUR_OPENROUTER_API_KEY_HERE"
    if generic_placeholder_check in OPENROUTER_API_KEY:
        print(f"WARNING: [{ts}] API Test: Generic placeholder API key.", file=sys.stderr)
        return jsonify({"success": False, "error": "OpenRouter API key placeholder. Cannot test.", "timestamp": ts})
    if is_example_key: print(f"INFO: [{ts}] API Test: Using specific user-provided OpenRouter API key.", file=sys.stderr)
    elif not OPENROUTER_API_KEY: print(f"ERROR: [{ts}] API Test: OpenRouter API key not set.", file=sys.stderr); return jsonify({"success": False, "error": "OpenRouter API key not configured.", "timestamp": ts})

    sys_prompt = "You are a helpful AI assistant. Respond: 'Test API Online'"; usr_prompt = "Hello AI, connectivity test."
    try:
        response = open_router_api.client.chat.completions.create(
            model="openrouter/auto", messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
            max_tokens=50,
            extra_headers={ "HTTP-Referer": "http://SQLBOT.pythonanywhere.com/test_api", "X-Title": "Screener Bot API Test" }
        )
        answer = response.choices[0].message.content if response.choices and response.choices[0].message else "[No content]"
        return jsonify({"success": True, "answer": answer, "model_used": "openrouter/auto (or routed)", "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})
    except Exception as e:
        print(f"ERROR: [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] API test error: {str(e)}", file=sys.stderr)
        return jsonify({"success": False, "error": f"API test error: {str(e)}", "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})

@app.route('/api_info')
def api_info_route():
    global session_total_queries, session_total_cost, session_total_response_time_ms
    avg_cost = (session_total_cost / session_total_queries) if session_total_queries > 0 else 0
    avg_response_time_ms = (session_total_response_time_ms / session_total_queries) if session_total_queries > 0 else 0
    return jsonify({
        "api_provider": "OpenRouter.ai", "default_model_configured": "deepseek/deepseek-r1-0528:free",
        "current_user": user_login, "current_time_utc": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
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
        print(f"INFO: app.py - Created 'templates' directory for local development at {templates_dir_local}", file=sys.stderr)

    print("="*70, file=sys.stderr)
    print("🚀 DocuQuery SQL Bot with OpenRouter API (Enhanced Stats) - LOCAL DEV MODE", file=sys.stderr)
    print(f"📅 Current UTC Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    # ... (other local dev startup messages as before)
    print("="*70, file=sys.stderr)
    app.run(debug=True, host='0.0.0.0', port=5000)

print("INFO: app.py - End of file reached, application object 'app' should be defined and configured.", file=sys.stderr)