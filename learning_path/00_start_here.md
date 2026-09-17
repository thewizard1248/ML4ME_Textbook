# From zero to ML4ME

**A guided learning companion to your ETH course book.** Start with ordinary arithmetic and build towards the chapter [Reviewing Supervised Linear Models](https://ideal.umd.edu/ML4ME_Textbook/part1/reviewing_supervised_linear_models.html). Then prepare for the mathematics and ideas used throughout the rest of the book.

This is a personal companion, not an official course handout. The explanations, small datasets, worked examples and exercises here were created for this learning path. The book by Mark Fuge and the IDEAL Lab provides the topic sequence and destination; its original files are unchanged.

## Start today

Open **01_foundations.html** for the reading version, or **01_foundations.ipynb** in JupyterLab for the executable version. The two contain the same lessons. In Jupyter choose **ML4ME (Python 3.11)** and use **Run → Run All Cells** once to load the examples and sliders. Then study one section at a time. The HTML versions contain saved diagrams; use the notebooks for live experiments.

If Jupyter is closed, start it with the existing `KURS_STARTEN.command` in the parent folder. In its file browser open `learning_path/`. No additional packages or accounts are needed.

## What “from zero” means here

You do not need prior machine learning or Python. We introduce the arithmetic, algebra and notation used by our first examples. Later lessons introduce vectors, matrices, derivatives, probability, and neural networks before connecting them to course chapters. If a lesson is unfamiliar, allow several sessions for it. Understanding all the advanced chapters still requires working through their derivations and experiments; this companion supplies the foundations and reading route, not a guarantee of mastery from reading alone.

Your first concrete destination is being able to explain, calculate and experiment with **every main concept in the linked linear-model chapter**: features, fitting, regularization, norms, regression losses, classification margins, and choosing a loss or penalty.

## Your route

| Step | Read in the browser | Work in Jupyter | What you should be able to do afterwards |
|---|---|---|---|
| 1. Foundations | [Reading version](01_foundations.html) | [Notebook](01_foundations.ipynb) | Read a formula, understand a data table, make a prediction, calculate an error, and run basic Python |
| 2. The linked chapter | [Reading version](02_linear_models.html) | [Notebook](02_linear_models.ipynb) | Explain and use linear/polynomial models, validation, loss functions, norms, Ridge, Lasso, and classification |
| 3. The mathematical tools | [Reading version](03_mathematical_tools.html) | [Notebook](03_mathematical_tools.ipynb) | Read matrix notation, understand PCA/SVD, calculate derivatives, and follow gradient descent and backpropagation |
| 4. Probability and inference | [Reading version](04_probability_and_inference.html) | [Notebook](04_probability_and_inference.ipynb) | Distinguish probabilities from densities, use Bayes, and explain likelihood, KL, MCMC and variational inference |
| 5. The rest of the course | [Reading version](05_course_outlook.html) | [Notebook](05_course_outlook.ipynb) | Understand the questions asked by each later chapter and what to learn before attempting it |

An optional planning estimate is **15–25 focused hours for Steps 1–2**, followed by **15–30 hours for Steps 3–5**. These are rough study allocations, not promises; starting mathematics from scratch can take substantially longer. Let the checkpoints determine your pace.

## A learning loop that actually checks understanding

For each section:

1. **Read:** What problem are we solving? Identify the input, output, and unknowns.
2. **Predict:** Before moving a slider or running a calculation, say what you expect.
3. **Experiment:** Change one setting. Explain the result in a sentence.
4. **Retrieve:** Close the explanation and answer the checkpoint on paper.
5. **Repair:** Open the worked answer. Identify the first step where your reasoning differed, then retry with different numbers.

Use a 30–45 minute session for one or two sections. The next day, redo yesterday's checkpoint before reading more. Revisiting an idea is part of the process.

## Your mastery checklist

These boxes are for you to tick in your own notes. They are not automatically inferred from opening a page.

- [ ] I can calculate a prediction and MSE by hand.
- [ ] I can explain why a polynomial can still be a linear model.
- [ ] I can distinguish a fitted parameter from a hyperparameter.
- [ ] I can explain why training error does not measure generalization.
- [ ] I can choose among squared, absolute, Huber, and epsilon-insensitive losses.
- [ ] I can calculate L1, L2, squared L2, and L-infinity penalties.
- [ ] I can distinguish a classification score, a probability, a label, and a margin.
- [ ] I can explain the different effects of Ridge and Lasso.
- [ ] I can interpret the shapes in `X @ w` and one gradient update.
- [ ] I can explain PCA as a projection and differentiate a composition of functions.
- [ ] I can update a prior using evidence and explain what posterior uncertainty means.
- [ ] I can distinguish a predictor, an autoencoder, and a generative model.

## When to return to the official book

After Step 2, read the linked chapter from the beginning. For each figure, describe the experiment before reading the accompanying explanation. Then go to the cross-validation and gradient-descent chapters. Steps 3 and 4 prepare you for the remaining foundations. Step 5 gives the full chapter map, including prerequisites and exit questions.

The official book sometimes uses a different normalization or notation from this companion. Step 2 includes a short “Reading the book precisely” section explaining the consequential differences and a few likely typos. You do not need to edit the official notebooks.

## If you get stuck

- **A formula:** Name every symbol, substitute tiny numbers, and calculate one example.
- **A graph:** Identify both axes, the observations, the fitted curve, and what was changed.
- **Code:** State the shape and meaning of each array. Read the error message before rerunning.
- **A model:** Separate its prediction rule, training objective, optimization method, and evaluation method.

An effective follow-up request is: “Teach me Step 2, section 4. Ask one question at a time and wait for my answer. Use the spring example.” That lets us work through the course together at your pace.

For the existing environment and repository navigation, see [the original setup guide](../START_HIER.md). For the official source, see [the course book](https://ideal.umd.edu/ML4ME_Textbook/).
