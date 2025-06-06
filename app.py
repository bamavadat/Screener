import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from openai import OpenAI

print("INFO: app.py - Starting application.", file=sys.stderr)

app = Flask(__name__)
print("INFO: app.py - Flask app instance created.", file=sys.stderr)


# --- Multi-Provider API Class ---
class MultiProviderAPI:
    def __init__(self):
        print(f"INFO: MultiProviderAPI - Initializing at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
              file=sys.stderr)

        self.providers = {
            "together": {
                "name": "Together.ai",
                "api_key": "99e280d1353e295a4751d9a6b80b9747c85ed68417be3c23c2b403e51259a41e",
                "base_url": "https://api.together.xyz/v1",
                "model": "deepseek-ai/DeepSeek-R1",
                "client": None,
                "active": True,
                "cost_per_1m_input": 0.55,
                "cost_per_1m_output": 2.19,
                "priority": 1
            },
            "siliconflow": {
                "name": "SiliconFlow",
                "api_key": "YOUR_SILICONFLOW_KEY_HERE",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "client": None,
                "active": False,
                "cost_per_1m_input": 0.0,
                "cost_per_1m_output": 0.0,
                "priority": 2
            },
            "deepseek": {
                "name": "DeepSeek Direct",
                "api_key": "YOUR_DEEPSEEK_KEY_HERE",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-reasoner",
                "client": None,
                "active": False,
                "cost_per_1m_input": 0.14,
                "cost_per_1m_output": 0.28,
                "priority": 3
            }
        }

        self._initialize_clients()

        # Statistics
        self.total_queries = 0
        self.total_cost = 0.0
        self.total_response_time_ms = 0.0
        self.provider_stats = {name: {"queries": 0, "failures": 0, "total_time": 0.0}
                               for name in self.providers.keys()}

    def _initialize_clients(self):
        for provider_name, config in self.providers.items():
            if config["active"] and config["api_key"] and "YOUR_" not in config["api_key"]:
                try:
                    config["client"] = OpenAI(
                        base_url=config["base_url"],
                        api_key=config["api_key"],
                        timeout=None,
                        default_headers={
                            "HTTP-Referer": "http://127.0.0.1:5000",
                            "X-Title": f"Screener Bot ({config['name']})"
                        }
                    )
                    print(f"INFO: MultiProviderAPI - {config['name']} client initialized", file=sys.stderr)
                except Exception as e:
                    print(f"ERROR: MultiProviderAPI - Failed to initialize {config['name']}: {e}", file=sys.stderr)
                    config["active"] = False

    def add_provider(self, provider_name, api_key, base_url, model, cost_input=0.0, cost_output=0.0, priority=999):
        self.providers[provider_name] = {
            "name": provider_name,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "client": None,
            "active": True,
            "cost_per_1m_input": cost_input,
            "cost_per_1m_output": cost_output,
            "priority": priority
        }

        try:
            self.providers[provider_name]["client"] = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=None
            )
            self.provider_stats[provider_name] = {"queries": 0, "failures": 0, "total_time": 0.0}
            print(f"INFO: MultiProviderAPI - Added provider: {provider_name}", file=sys.stderr)
            return True
        except Exception as e:
            print(f"ERROR: MultiProviderAPI - Failed to add {provider_name}: {e}", file=sys.stderr)
            self.providers[provider_name]["active"] = False
            return False

    def update_provider_key(self, provider_name, api_key):
        if provider_name in self.providers:
            self.providers[provider_name]["api_key"] = api_key
            self.providers[provider_name]["active"] = True
            try:
                self.providers[provider_name]["client"] = OpenAI(
                    base_url=self.providers[provider_name]["base_url"],
                    api_key=api_key,
                    timeout=None
                )
                print(f"INFO: MultiProviderAPI - Updated {provider_name} API key", file=sys.stderr)
                return True
            except Exception as e:
                print(f"ERROR: MultiProviderAPI - Failed to update {provider_name}: {e}", file=sys.stderr)
                return False
        return False

    def _calculate_cost(self, usage_data, provider_config):
        if not usage_data:
            return 0.0

        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)

        input_cost = (prompt_tokens / 1_000_000) * provider_config["cost_per_1m_input"]
        output_cost = (completion_tokens / 1_000_000) * provider_config["cost_per_1m_output"]

        return input_cost + output_cost

    def stream_chat_completion(self, messages, max_tokens=2000, temperature=0.7, bot_type="SQL Bot"):
        # Get active providers sorted by priority
        active_providers = sorted(
            [(name, config) for name, config in self.providers.items() if config["active"]],
            key=lambda x: x[1]["priority"]
        )

        if not active_providers:
            yield {"type": "error", "delta": "No active providers available"}
            return

        start_time = time.perf_counter()
        self.total_queries += 1

        for provider_name, config in active_providers:
            if not config["client"]:
                continue

            try:
                print(f"INFO: MultiProviderAPI - Streaming from {config['name']} for {bot_type}", file=sys.stderr)

                yield {"type": "status", "message": f"{bot_type} connecting to {config['name']}..."}

                stream = config["client"].chat.completions.create(
                    model=config["model"],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )

                accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0}

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yield {"type": "content", "delta": chunk.choices[0].delta.content}

                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = chunk.usage
                        accumulated_usage["prompt_tokens"] = getattr(usage, 'prompt_tokens', 0) or accumulated_usage[
                            "prompt_tokens"]
                        accumulated_usage["completion_tokens"] = getattr(usage, 'completion_tokens', 0) or \
                                                                 accumulated_usage["completion_tokens"]

                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                cost = self._calculate_cost(accumulated_usage, config)

                self.provider_stats[provider_name]["queries"] += 1
                self.provider_stats[provider_name]["total_time"] += duration_ms
                self.total_cost += cost
                self.total_response_time_ms += duration_ms

                yield {
                    "type": "done",
                    "provider": config['name'],
                    "model": config['model'],
                    "duration_ms": round(duration_ms, 2),
                    "cost": round(cost, 6),
                    "usage": accumulated_usage
                }

                print(f"INFO: MultiProviderAPI - Stream completed using {config['name']}", file=sys.stderr)
                return

            except Exception as e:
                error_msg = str(e)
                print(f"ERROR: MultiProviderAPI - {config['name']} stream failed: {error_msg}", file=sys.stderr)

                self.provider_stats[provider_name]["failures"] += 1

                if self.provider_stats[provider_name]["failures"] >= 3:
                    config["active"] = False
                    print(f"WARNING: MultiProviderAPI - Disabled {config['name']} due to repeated failures",
                          file=sys.stderr)

                yield {"type": "status", "message": f"{config['name']} failed, trying next provider..."}
                continue

        # All providers failed
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        self.total_response_time_ms += duration_ms

        yield {"type": "error", "delta": "All providers failed"}

    def get_status(self):
        return {
            "providers": {
                name: {
                    "name": config["name"],
                    "active": config["active"],
                    "model": config["model"],
                    "has_key": bool(config["api_key"] and "YOUR_" not in config["api_key"]),
                    "priority": config["priority"],
                    "cost_input": config["cost_per_1m_input"],
                    "cost_output": config["cost_per_1m_output"],
                    "stats": self.provider_stats.get(name, {"queries": 0, "failures": 0, "total_time": 0.0})
                }
                for name, config in self.providers.items()
            },
            "global_stats": {
                "total_queries": self.total_queries,
                "total_cost": round(self.total_cost, 6),
                "total_response_time_ms": round(self.total_response_time_ms, 2),
                "average_response_time_ms": round(self.total_response_time_ms / max(self.total_queries, 1), 2)
            },
            "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        }


# --- Global API instance ---
multi_api = MultiProviderAPI()
print("INFO: app.py - Multi-provider API initialized.", file=sys.stderr)

# --- Document handling variables ---
DEFAULT_DOC_FILENAME = "extracted_edit_url.txt"
current_document = ""
document_loaded = False
initial_default_content = ""
source_of_current_document = "None"
print("INFO: app.py - Document handling variables initialized.", file=sys.stderr)


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
                if is_initial_load: initial_default_content = text_content
                current_document = text_content
                document_loaded = True
                source_of_current_document = f"Local File ({DEFAULT_DOC_FILENAME})"
                print(f"INFO: [{log_ts}] Loaded/Refreshed: {DEFAULT_DOC_FILENAME} ({len(current_document)} chars)",
                      file=sys.stderr)
                return True
            else:
                print(f"WARNING: [{log_ts}] Local doc '{DEFAULT_DOC_FILENAME}' empty.", file=sys.stderr)
                if is_initial_load: initial_default_content = ""; current_document = ""; document_loaded = False; source_of_current_document = "None (empty default)"
                return False
        else:
            print(f"WARNING: [{log_ts}] Local doc '{DEFAULT_DOC_FILENAME}' not found at {default_doc_path}.",
                  file=sys.stderr)
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
        print(f"INFO: [{log_ts_start}] Attempting to run onedrive.py to update '{DEFAULT_DOC_FILENAME}'...",
              file=sys.stderr)
        current_env = os.environ.copy()
        current_env['PYTHONIOENCODING'] = 'utf-8'
        process = subprocess.run(
            ['python', onedrive_script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=False,
            timeout=120,
            env=current_env
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
                return jsonify({"success": False,
                                "error": f"onedrive.py ran, but failed to reload/validate '{DEFAULT_DOC_FILENAME}' after sync."})
        else:
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


@app.route('/stream_sql_query', methods=['POST'])
def stream_sql_query_route():
    global current_document, document_loaded
    if not document_loaded or not current_document or not current_document.strip():
        return jsonify({"error": "No document loaded or document is empty."}), 400
    data = request.get_json()
    messages = data.get('messages')
    if not messages:
        return jsonify({"error": "Please provide user question history (in messages format)"}), 400

    print(
        f"INFO: [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] User {user_login} asked SQL Bot (streaming): '{messages[-1]['content'] if messages else 'N/A'}'",
        file=sys.stderr)

    fixed_prefix = "solely generate single sql query to answer user question based on the following explanation:"
    system_prompt_content = (
        f"You are a helpful assistant that generates SQL Server queries. "
        f"{fixed_prefix}\n\n"
        f"Use the following document context to answer all subsequent user questions:\n\n"
        f"--- DOCUMENT CONTEXT ---\n{current_document}\n--- END DOCUMENT CONTEXT ---"
    )
    api_messages = [{"role": "system", "content": system_prompt_content}] + messages

    def generate_stream_sql():
        try:
            for chunk_data in multi_api.stream_chat_completion(api_messages, bot_type="SQL Bot"):
                yield f"data: {json.dumps(chunk_data)}\n\n"
                time.sleep(0.01)
        except Exception as e:
            print(f"ERROR in generate_stream_sql: {e}", file=sys.stderr)
            error_payload = {"type": "error", "delta": f"SQL Bot Stream generation error: {str(e)}"}
            yield f"data: {json.dumps(error_payload)}\n\n"

    response = Response(stream_with_context(generate_stream_sql()), mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    return response


@app.route('/stream_general_chat', methods=['POST'])
def stream_general_chat_route():
    print(
        f"INFO: app.py - /stream_general_chat route accessed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        file=sys.stderr)
    data = request.get_json()
    messages = data.get('messages')

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    system_prompt_content = "You are a helpful assistant."
    api_messages = [{"role": "system", "content": system_prompt_content}] + messages

    def generate_stream_general():
        try:
            for chunk_data in multi_api.stream_chat_completion(api_messages, bot_type="General Chat"):
                yield f"data: {json.dumps(chunk_data)}\n\n"
                time.sleep(0.01)
        except Exception as e:
            print(f"ERROR in stream_general_chat generate_stream: {e}", file=sys.stderr)
            error_payload = {"type": "error", "delta": f"Stream generation error: {str(e)}"}
            yield f"data: {json.dumps(error_payload)}\n\n"

    response = Response(stream_with_context(generate_stream_general()), mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    return response


@app.route('/status')
def status_route():
    global current_document, document_loaded, source_of_current_document
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    doc_length = len(current_document) if current_document else 0
    is_doc_effectively_loaded = document_loaded and bool(current_document and current_document.strip())
    current_source = source_of_current_document if is_doc_effectively_loaded else "None (No document active)"
    return jsonify({
        "loaded": is_doc_effectively_loaded, "characters": doc_length,
        "preview": current_document[:300] + (
            "..." if doc_length > 300 else "") if current_document else "[No doc loaded]",
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
        current_document = "";
        document_loaded = False;
        source_of_current_document = "None"
        message = "Document cleared. No initial version available."
        print(f"INFO: [{ts}] Cleared. No initial version to revert to.", file=sys.stderr)
    return jsonify(
        {"success": True, "message": message, "timestamp": ts, "character_count": char_count, "loaded": document_loaded,
         "source": source_of_current_document, "preview": preview_text})


# --- Provider Management Routes ---
@app.route('/providers')
def providers_route():
    """Provider management dashboard"""
    return render_template('providers.html', user_login=user_login)


@app.route('/api/providers', methods=['GET'])
def get_providers():
    """Get provider status"""
    return jsonify(multi_api.get_status())


@app.route('/api/providers/<provider_name>/update', methods=['POST'])
def update_provider(provider_name):
    """Update provider API key"""
    data = request.get_json()
    api_key = data.get('api_key')

    if not api_key:
        return jsonify({"success": False, "error": "API key required"}), 400

    success = multi_api.update_provider_key(provider_name, api_key)

    if success:
        return jsonify({"success": True, "message": f"{provider_name} updated successfully"})
    else:
        return jsonify({"success": False, "error": f"Failed to update {provider_name}"}), 400


@app.route('/api/providers/add', methods=['POST'])
def add_provider():
    """Add new provider"""
    data = request.get_json()

    required_fields = ['name', 'api_key', 'base_url', 'model']
    for field in required_fields:
        if not data.get(field):
            return jsonify({"success": False, "error": f"{field} is required"}), 400

    success = multi_api.add_provider(
        provider_name=data['name'].lower().replace(' ', '_'),
        api_key=data['api_key'],
        base_url=data['base_url'],
        model=data['model'],
        cost_input=float(data.get('cost_input', 0.0)),
        cost_output=float(data.get('cost_output', 0.0)),
        priority=int(data.get('priority', 999))
    )

    if success:
        return jsonify({"success": True, "message": f"Provider {data['name']} added successfully"})
    else:
        return jsonify({"success": False, "error": f"Failed to add provider {data['name']}"}), 400


@app.route('/test_api')
def test_api_route():
    """Test all active providers - Fixed version"""
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # Test with a simple message
    test_messages = [{"role": "user", "content": "Hello, respond with: API Test Success"}]

    result = {"timestamp": ts, "tests": [], "status": "ok"}

    active_count = 0
    for provider_name, config in multi_api.providers.items():
        if config["active"] and config["client"]:
            active_count += 1
            try:
                print(f"INFO: [{ts}] Testing {config['name']}", file=sys.stderr)

                response = config["client"].chat.completions.create(
                    model=config["model"],
                    messages=test_messages,
                    max_tokens=50,
                    timeout=10
                )

                answer = response.choices[0].message.content if response.choices else "No response"

                result["tests"].append({
                    "provider": config["name"],
                    "success": True,
                    "response": answer[:100],
                    "model": config["model"]
                })

            except Exception as e:
                result["tests"].append({
                    "provider": config["name"],
                    "success": False,
                    "error": str(e)[:100],
                    "model": config["model"]
                })

    if active_count == 0:
        result["message"] = "No active providers to test"
        result["status"] = "no_providers"

    return jsonify(result)


@app.route('/api_info')
def api_info_route():
    global user_login
    return jsonify({
        "api_provider": "Multi-Provider System",
        "current_user": user_login,
        "current_time_utc": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        "provider_status": multi_api.get_status()
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
    print("🚀 DocuQuery SQL Bot with Multi-Provider API - LOCAL DEV MODE", file=sys.stderr)
    print(f"📅 Current UTC Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    print(f"👤 Current User: {user_login}", file=sys.stderr)
    print(f"🔗 Providers: {len([p for p in multi_api.providers.values() if p['active']])} active", file=sys.stderr)
    print(f"🌐 Local URL: http://127.0.0.1:5000/", file=sys.stderr)
    print(f"🛠️ Provider Dashboard: http://127.0.0.1:5000/providers", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    app.run(debug=True, host='0.0.0.0', port=5000)

print("INFO: app.py - End of file reached, application object 'app' should be defined and configured.", file=sys.stderr)
