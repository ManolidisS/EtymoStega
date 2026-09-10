**Note about `questions.jsonl`**:

This was generated on OpenRouter with OpenAI's GPT 5.6 Luna on medium reasoning.
The system prompt was empty, and the user prompt was:
````
Create a 1250 question list in .jsonl format.

The following is an example:
```jsonl
{"question":"[QUESTION 1]","correct":"[ANSWER]","incorrect":"[INCORRECT]","number":1}
{"question":"[QUESTION 1]","correct":"[ANSWER]","incorrect":"[INCORRECT]","number":2}
{"question":"[QUESTION 1]","correct":"[ANSWER]","incorrect":"[INCORRECT]","number":3}
```

Make sure to make the questions as diverse as possible, along a vast general-knowledge domain. Plan your answers before hand to ensure that they are varied across many subjects. Do not just collapse to doing the same types of questions. Also, ensure they are very easy to answer and require simple recall and no logic. Most importantly, do not repeat any questions.

Because this might exceed your maximum response size, create this question list in five batches of 250. At the end of each batch, you will be prompted with "continue".
````
followed by 4 "continue"s.

For reproducability, we include an exported version of the chat at `questions-chat.json`.

Regrettably, some questions are repeated, but we believe this dataset was sufficient to be used.