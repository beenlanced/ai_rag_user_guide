# Retrieval Augmented Generation (RAG) Q&A Mechanism to Access User Guide Information - Loving my 2010 iPod Shuffle

<p>
  <img alt="Silver iPod Shuffle " src="imgs/ipod.jpeg"/>
</p>

[img source: crutchfield](https://www.crutchfield.com/p_472SHFL1S/Apple-iPod-shuffle-1GB-Silver.html)

## Project Description

Traditional Large Language Models (LLM) are trained on large datasets and can generate human-like text, but their knowledge is limited to the data they were trained upon. Compounding the knowledge limitation, is the fact that these LLMs often produce inaccurate or outdated information. A way to improve these limitations is to use Retrieval Augmented Generation (RAG) to connect the LLM with external data sources (e.g., databases or even pdf files like this project reveals). It is well understood that, RAG improves the accuracy of LLM's because it reduces the risk of LLMs "hallucinating" or generating incorrect information by grounding the LLM's responses in actual factual data from external data sources. With RAG, learning models get access to up-to-date information, eliding the limitation of LLMs being restricted only to training data. LLMs with RAG are now able to understand a user's query better and able to generate more contextually appropriate responses. One additional benefit, is that RAGs are a more cost efficient way to customize LLMs compared with retraining or fine-tuning the LLM which would mean having to calculate new weights and biases based on the new data.

## Problem

I loved my little `Silver iPod Shuffle`. Terrifically Small. Clips On! Weighs about the same as a USD quarter. It was the best! I recently found the little beauty tucked away in a desk drawer and after a little cable searching I was able to resurrect it from its deep coma.

I had some initial weird warning lights on startup and since the hard copy manuals I had for it are long since gone, I quickly hopped onto the Internet to find the manual to diagnosis the issues. It got me thinking, this is a job for Artificial Intelligence. Specifically, I would love to be able to ask a Chatbot, how to diagnosis or fix my problem.

So, in short I decided to design a RAG system to become my "iPod Shuffle AI Assistant"

### My Solution

This project builds a RAG model using LangChain and an initial Llama 3.1 LLM. I inject new knowledge, the iPod user manual, as a vector database, chunking the manual pdf file into various sections. In short, I create a simple question and response application, where the model in the middle is the RAG model with all of its embedding, prompt engineering, and retrieval goodness. The application has the following form:

<p>
  <img alt="Question Response flow " src="imgs/question_response.png"/>
</p>

[img source](https://www.youtube.com/watch?v=Y08Nn23o_mY)

My Rag process flow looks similar to the following:

<p>
  <img alt="RAG Process Flow " src="imgs/rag_process_flow.png"/>
</p>

[img source](https://www.youtube.com/watch?v=Y08Nn23o_mY)

(Add details about working app and maybe steps to get there)

---

## Objective

The project contains the key elements:

- `Chat0llama` instantiates chatbot like feature
- `ChromaDB` open-source vector embedding database making it easy to build LLM apps
- `Deep Learning` for neural network building,
- `Faiss-cpu` CPU only version of Facebook AI Similarity Search used for similarity search and clustering of dense vectors,
- `Git` (version control),
- `Jupyter` python coded notebooks,
- `LangChain`, simplify the creation of applications using LLMs,
- `Natural Language Processing (NLP)` to understand, interpret, and manipulate text,
- `Numpy` for array manipulation,
- `Llama 3.1` ollama simple model providing initial LLM,
- `Pandas` for dataframe usage,
- `Prompt Engineering` to retrieve information
- `Pydf` to manipulate pdf files,
- `Python` the standard modules,
- `Retrieval Augmented Generation (RAG)` connect the LLM with external data sources,
- `Transfer Learning`, to adapt weights and biases to learn on new data for a pre-existing highly built model, and
- `uv` package management including use of `ruff` for linting and formatting

---

## Tech Stack

![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Langchain](https://img.shields.io/badge/langchain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

---

## Getting Started

Here are some instructions to help you set up this project locally.

---

## Installation Steps

The Python version used for this project is `Python 3.12`.

### Clone the Repo

1. Clone the repo (or download it as a zip file):

   ```bash
   git clone https://github.com/beenlanced/combo_encoder_decoder_transformer.git
   ```

2. Create a virtual environment named `.venv` using `uv` Python version 3.12:

   ```bash
   uv venv --python=3.12
   ```

3. Activate the virtual environment: `.venv`

   On macOs and Linux:

   ```bash
   source .venv/bin/activate #mac
   ```

   On Windows:

   ```bash
    # In cmd.exe
    venv\Scripts\activate.bat
   ```

4. Install packages using `pyproject.toml` or (see special notes section)

   ```bash
   uv pip install -r pyproject.toml
   ```

### Install the Jupyter Notebook(s)

1. **Run the Project**

   - Run the Jupyter Notebook(s) in the Jupyter UI or in VS Code.

---

## Data

I use a [subset](https://huggingface.co/datasets/MaartenGr/arxiv_nlp) of the [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) that was created by [Maarten Grootendorst](https://www.linkedin.com/in/mgrootendorst/) specifically for the _[Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/)_ book. The dataset contains 44,949 abstracts published between 1991 and 2024 from ArXiv’s Computation and Language section, aka cs.CL.

---

## Special Notes

- I use the open-source (read FREE) pre-traind SBERT model: [gte-small model](https://huggingface.co/thenlper/gte-small) from Hugging Face. This transformer model converts the abstracts into embeddings.

- I use the open-source (read FREE) [Flan-t5-small](https://huggingface.co/docs/transformers/model_doc/flan-t5) LLM. It is a smaller, efficient version of the FLAN-T5 large language model, developed by Google. It's a fine-tuned version of the T5 (Text-to-Text Transfer Transformer) model, specifically designed for various natural language processing (NLP) tasks. It excels in zero-shot, few-shot, and chain-of-thought learning scenarios.

---

### Final Words

Thanks for visiting.

Give the project a star (⭐) if you liked it or if it was helpful to you!

You've `beenlanced`! 😉

---

## Acknowledgements

I would like to extend my gratitude to all the individuals and organizations who helped in the development and success of this project. Your support, whether through contributions, inspiration, or encouragement, have been invaluable. Thank you.

Specifically, I would like to acknowledge:

- https://www.youtube.com/watch?v=6ExFTPcJJFs and https://github.com/AarohiSingla/Generative_AI/blob/main/L-6/app1.py

- The folks at Apple, Inc. for their [iPod Shuffle (4th generation) user guide](https://support.apple.com/en-us/docs/ipod/133017)

- [Hema Kalyan Murapaka](https://www.linkedin.com/in/hemakalyan) and [Benito Martin](https://martindatasol.com/blog) for sharing their README.md templates upon which I have derieved my README.md.

- The folks at Astral for their UV [documentation](https://docs.astral.sh/uv/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details
