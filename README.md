# End-to-end-PDF-Chatbot-using-Llama2

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n pdfchatbot python=3.8 -y
```

```bash
conda activate pdfchatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your credentials as follows:(optional)

```ini
API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q8_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```



```bash
# Finally run the following command
streamlit run .\app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- streamlit 
- Meta Llama2
- FAISS


