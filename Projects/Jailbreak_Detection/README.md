Currently in progress.

I'm building a pipeline with a simple logistic regression model that predicts if a prompt is jailbreak attempt or not. I will be documenting my success and failures while evaluating the model

The following are the main functions of the ML pipeline that I've built:

- Data prep: this will define the prompts and labels. I developed the prompts using LLM models and findings from a USENIX conference research paper regarding successful jailbreaking strategies.
- Vocabulary: extracts unique words as features
- Vectorizations: convert sentences into word count vectors aka BoW
- Training: using gradient descent with the helper functions to learn the parameters: theta 
- Evaluation: predicts on test data and compared to actuals 
