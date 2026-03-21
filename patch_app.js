--- static/app.js
+++ static/app.js
@@ -2195,6 +2195,11 @@
         updateReasoningPanel(asstGroup, rawReasoning);
         scrollToBottom();
+      } else if (event.type === "answer_sync") {
+        rawAnswer = event.text || "";
+        fullAnswer = rawAnswer;
+        renderBubbleWithCursor(asstBubble, fullAnswer);
+        scrollToBottom();
       } else if (event.type === "answer_delta") {
         rawAnswer += event.text || "";
         fullAnswer = rawAnswer;
