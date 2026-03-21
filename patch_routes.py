--- routes/chat.py
+++ routes/chat.py
@@ -544,6 +544,8 @@
             for event in run_agent_stream(
                 api_messages,
                 model,
                 max_steps,
                 active_tool_names,
                 fetch_url_token_threshold=fetch_url_token_threshold,
                 fetch_url_clip_aggressiveness=fetch_url_clip_aggressiveness,
             ):
                 if event["type"] == "answer_delta":
                     full_response += event["text"]
+                elif event["type"] == "answer_sync":
+                    full_response = event["text"]
                 elif event["type"] == "reasoning_delta":
