import sys
import time
from datetime import datetime, timezone

from openai import OpenAI


class MultiProviderAPI:
    def __init__(self):
        print(f"INFO: MultiProviderAPI - Initializing at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
              file=sys.stderr)

        # Provider configurations
        self.providers = {
            "together": {
                "name": "Together.ai",
                "api_key": "99e280d1353e295a4751d9a6b80b9747c85ed68417be3c23c2b403e51259a41e",
                "base_url": "https://api.together.xyz/v1",
                "model": "deepseek-ai/DeepSeek-R1",
                "client": None,
                "active": False,
                "cost_per_1m_input": 0.55,
                "cost_per_1m_output": 2.19
            },

            "siliconflow": {
                "name": "SiliconFlow",
                "api_key": "YOUR_SILICONFLOW_KEY_HERE",  # Add when you get it
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "client": None,
                "active": False,  # Set to True when you add the key
                "cost_per_1m_input": 0.0,  # Free
                "cost_per_1m_output": 0.0  # Free
            },
            "deepseek": {
                "name": "DeepSeek Direct",
                "api_key": "sk-0aec65f2f73b4afab86063184d94cf8f",  # Add when you get it
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-reasoner",
                "client": None,
                "active": True,  # Set to True when you add the key
                "cost_per_1m_input": 0.14,
                "cost_per_1m_output": 0.28
            },
            "openrouter": {
                "name": "OpenRouter",
                "api_key": "YOUR_OPENROUTER_KEY_HERE",  # Backup
                "base_url": "https://openrouter.ai/api/v1",
                "model": "deepseek/deepseek-r1-0528:free",
                "client": None,
                "active": False,  # Set to True if you get credits
                "cost_per_1m_input": 0.55,
                "cost_per_1m_output": 2.19
            }
        }

        # Initialize clients for active providers
        self._initialize_clients()

        # Statistics
        self.total_queries = 0
        self.total_cost = 0.0
        self.total_response_time_ms = 0.0
        self.provider_stats = {name: {"queries": 0, "failures": 0, "total_time": 0.0}
                               for name in self.providers.keys()}

    def _initialize_clients(self):
        """Initialize OpenAI clients for active providers"""
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

    def add_provider(self, provider_name, api_key, base_url, model, cost_input=0.0, cost_output=0.0):
        """Add a new provider dynamically"""
        self.providers[provider_name] = {
            "name": provider_name,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "client": None,
            "active": True,
            "cost_per_1m_input": cost_input,
            "cost_per_1m_output": cost_output
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
        """Update API key for existing provider"""
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
        """Calculate cost for a request"""
        if not usage_data:
            return 0.0

        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)

        input_cost = (prompt_tokens / 1_000_000) * provider_config["cost_per_1m_input"]
        output_cost = (completion_tokens / 1_000_000) * provider_config["cost_per_1m_output"]

        return input_cost + output_cost

    def _try_provider(self, provider_name, messages, max_tokens=2000, temperature=0.7, stream=False):
        """Try a specific provider"""
        config = self.providers[provider_name]

        if not config["active"] or not config["client"]:
            return None, f"{config['name']} not active"

        start_time = time.perf_counter()

        try:
            print(f"INFO: MultiProviderAPI - Trying {config['name']} with model {config['model']}", file=sys.stderr)

            response = config["client"].chat.completions.create(
                model=config["model"],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Update statistics
            self.provider_stats[provider_name]["queries"] += 1
            self.provider_stats[provider_name]["total_time"] += duration_ms

            # Calculate cost if usage data is available
            cost = 0.0
            if hasattr(response, 'usage') and response.usage:
                cost = self._calculate_cost(response.usage.__dict__, config)
                self.total_cost += cost

            print(f"INFO: MultiProviderAPI - {config['name']} succeeded in {duration_ms:.2f}ms, cost: ${cost:.6f}",
                  file=sys.stderr)

            return {
                "response": response,
                "provider": config['name'],
                "model": config['model'],
                "duration_ms": duration_ms,
                "cost": cost
            }, None

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            self.provider_stats[provider_name]["failures"] += 1
            error_msg = str(e)

            print(f"ERROR: MultiProviderAPI - {config['name']} failed in {duration_ms:.2f}ms: {error_msg}",
                  file=sys.stderr)

            # Disable provider if it's consistently failing
            if self.provider_stats[provider_name]["failures"] >= 3:
                print(f"WARNING: MultiProviderAPI - Disabling {config['name']} due to repeated failures",
                      file=sys.stderr)
                config["active"] = False

            return None, error_msg

    def chat_completion(self, messages, max_tokens=2000, temperature=0.7, stream=False):
        """Try providers in order until one succeeds"""
        start_time = time.perf_counter()
        self.total_queries += 1

        # Get active providers sorted by priority (you can customize this order)
        active_providers = [name for name, config in self.providers.items() if config["active"]]

        if not active_providers:
            error_msg = "No active providers available"
            print(f"ERROR: MultiProviderAPI - {error_msg}", file=sys.stderr)
            return None, error_msg

        print(f"INFO: MultiProviderAPI - Starting request with {len(active_providers)} active providers",
              file=sys.stderr)

        last_error = ""

        for provider_name in active_providers:
            result, error = self._try_provider(provider_name, messages, max_tokens, temperature, stream)

            if result:
                end_time = time.perf_counter()
                total_duration = (end_time - start_time) * 1000
                self.total_response_time_ms += total_duration

                print(f"INFO: MultiProviderAPI - Request completed successfully using {result['provider']}",
                      file=sys.stderr)
                return result, None

            last_error = error
            print(f"INFO: MultiProviderAPI - Trying next provider...", file=sys.stderr)

        # All providers failed
        end_time = time.perf_counter()
        total_duration = (end_time - start_time) * 1000
        self.total_response_time_ms += total_duration

        error_msg = f"All providers failed. Last error: {last_error}"
        print(f"ERROR: MultiProviderAPI - {error_msg}", file=sys.stderr)
        return None, error_msg

    def stream_chat_completion(self, messages, max_tokens=2000, temperature=0.7):
        """Stream response with automatic failover"""
        active_providers = [name for name, config in self.providers.items() if config["active"]]

        if not active_providers:
            yield {"type": "error", "message": "No active providers available"}
            return

        start_time = time.perf_counter()
        self.total_queries += 1

        for provider_name in active_providers:
            config = self.providers[provider_name]

            if not config["active"] or not config["client"]:
                continue

            try:
                print(f"INFO: MultiProviderAPI - Streaming from {config['name']}", file=sys.stderr)

                yield {"type": "status", "message": f"Connecting to {config['name']}..."}

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
                    "duration_ms": duration_ms,
                    "cost": cost,
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

                yield {"type": "status", "message": f"{config['name']} failed, trying next provider..."}
                continue

        # All providers failed
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        self.total_response_time_ms += duration_ms

        yield {"type": "error", "message": "All providers failed"}

    def get_status(self):
        """Get current status of all providers"""
        return {
            "providers": {
                name: {
                    "name": config["name"],
                    "active": config["active"],
                    "model": config["model"],
                    "has_key": bool(config["api_key"] and "YOUR_" not in config["api_key"]),
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


# Global instance
multi_api = MultiProviderAPI()
