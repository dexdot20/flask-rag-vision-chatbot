--- agent.py
+++ agent.py
@@ -1021,6 +1021,7 @@
     pending_answer_separator = False
     fatal_api_error = None
     trace_id = uuid4().hex[:12]
+    total_clean_content = ""
     fetch_attempt_counts: dict[str, int] = {}
     usage_totals = {
         "prompt_tokens": 0,
@@ -1245,6 +1246,14 @@
             yield {"type": "tool_error", "step": step, "tool": "parser", "error": tool_call_error}
             break
 
+        if content_text:
+            if pending_answer_separator and content_text.strip():
+                total_clean_content += "\n\n"
+                pending_answer_separator = False
+            total_clean_content += content_text
+            if tool_calls:
+                yield {"type": "answer_sync", "text": total_clean_content}
+
         _trace_agent_event(
             "agent_turn_result",
             trace_id=trace_id,
