# OUTLINE: Fine-tuning Llama 2 for JSON Function Calling

 ##  **I. INTRODUCTION**
  - **context**: Last summer, Meta announced the release of Llama 2, an open-source large language model (LLM) available for research and commercial use cases. 
       A few days later, we published a [guide](https://scale.com/blog/fine-tune-llama-2) outlining how to use [Scale’s LLM Engine](https://github.com/scaleapi/llm-engine) to fine-tune Llama 2 for optimum performance on language tasks. 

- But what if you wanted to go beyond text generation?
   - You can expand an LLM’s functionality by fine-tuning it for [function calling](https://openai.com/index/function-calling-and-other-api-updates/), enabling it to respond to human input with structured JSON objects. 
  - This JSON can then be passed along as function arguments.
 
- **why that matters**: Function calling _vastly extends_ the capabilities of LLMs by allowing them to interact with external APIs, databases, and tools.
    - **example**: For example, an LLM can leverage function calling to analyze a user’s question about the week’s weather forecast and return a JSON object containing arguments to call a weather API’s `get_weather_forecast` function.

- Now let’s look at how you can use the LLM Engine to fine-tune Llama 2 for function calling.

## II. Install dependencies and dataset

- Before we get started let's make sure we install install Scale's LLM engine and the `datasets` package from HuggingFace
  
      pip install scale-llm-engine
      pip install datasets
      
  
- Next we need to load an appropriate dataset that can be used for function calling fine tuning.
    - Because LLM Engine supports fine-tuning on prompt-completion pairs, we chose a dataset that both contains high quality data and is structured in a way where formatting it into a CSV file with `prompt` and `response` columns is straight forward.
      
    - With that in mind, we decided to us a dataset structures into two features: prompts in Llama-2 Chat format (`input`), and expected JSON responses and completions (`output`).
      
 
  ```python
  from datasets import load_dataset
  dataset = load_dataset("marclove/llama_functions")
  dataset["train"].features
  ```

  - **example datapoint**:
   ```
  {'input': '[INST] <<SYS>>\n<<API>>\nYou may answer by calling one of the following functions. If you can infer a function\'s required arguments, respond with a the function name and arguments. If multiple functions can be called, pick the one that best answers the question.\n\n```\n{"functions":[{"name":"calculate_cooking_time","description":"Calculates the cooking time of a recipe given the recipe name and serving size","parameters":[{"description":"The name of the recipe for which the cooking time is to be calculated.","name":"recipeName","type":"string"},{"description":"The serving size of the recipe for which the cooking time is to be calculated.","name":"servingSize","optional":true,"type":"number"}]},{"name":"computeQuizScore","description":"Calculates the score of a quiz based on the answers provided by the user.","parameters":[{"description":"An array containing the user\'s answers.","items":{"description":"An object representing a single answer.","properties":{"questionId":{"description":"The ID of the question being answered.","minimum":1,"type":"number"},"selectedOption":{"description":"The option selected by the user for the given question.","enum":["A","B","C","D"],"type":"string"}},"type":"object"},"name":"answers","type":"array"},{"description":"The total number of questions in the quiz.","minimum":1,"name":"totalQuestions","optional":true,"type":"number"}]}]}\n```\n<</API>>\n<</SYS>>\n\nHey, how\'s it going? I just finished a quiz with 2 questions and I\'m really curious what my final score is. Can you help me out and calculate it? My answers were B for question 1 and C for question 2. [/INST]', 'output': ' {"function_call":{"name":"computeQuizScore","arguments":{"answers":[{"questionId":1,"selectedOption":"B"},{"questionId":2,"selectedOption":"C"}],"totalQuestions":2}}} '}
  
  ```
## III. Format dataset
- Now that we've loaded the dataset, let's convert it into the supported format (CSV file with a "prompt" and "response" comlumn)
- Since this dataset is already structured as a series of `inputs` and `outputs`, we simply need to parse through the data and sort each input of the dataset into the **prompt** column and each output into the **response** column of a CSV file we'll create.
  
  ``` python
  import csv
  
  # Define the CSV file path
  csv_file_path = 'prompt_response.csv'

  # Write to CSV
  with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['prompt', 'response']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

  # Write CSV header
    writer.writeheader()

  # Write each data point to CSV
    for data_point in dataset["train"]:
        writer.writerow({'prompt': data_point['input'], 'response': data_point['output']})
  
  print(f"CSV file created successfully at: {csv_file_path}")
  ```

- Next we need to store the formatted dataset into a publicly accessible URL so that LLM Engine can read them.
   - In this example, we've uploaded the CSV file into a publically available [Gist](https://gist.github.com/yebhub/0128b200be4711361179e16ae0e7dfc9#file-data-csv).
      - (**NOTE**: make sure you have given your github access token permission to create gists from API calls)
 
     ```python
     import requests

     # GitHub Gist API endpoint
     api_url = 'https://api.github.com/gists'
     
     # Personal access token for authentication
     access_token = 'YOUR_ACCESS_TOKEN'
     
     csv_file_path = 'prompt_response.csv'
     
     with open(csv_file_path, 'r', encoding='utf-8') as file:
       csv_content = file.read()
     
     # Create Gist payload
     payload = {
       'description': 'CSV file uploaded via API',
       'public': True,
       'files': {
           'data.csv': {
               'content': csv_content
           }
       }
     }
     
     # Create Gist using GitHub API
     headers = {
       'Authorization': f'token {access_token}',
       'Accept': 'application/vnd.github.v3+json'
     }
     response = requests.post(api_url, headers=headers, json=payload)
     
     # Check if the Gist was successfully created
     if response.status_code == 201:
       gist_url = response.json()['html_url']
       print(f"Gist created successfully: {gist_url}")
     else:
       print(f"Failed to create Gist. Status code: {response.status_code}, Message: {response.text}")
     ```
## IV Fine-tune

- With our dataset saved on CSV file hosted on a public Gist, we can move on to fine-tuning.
- Set Scale API Key (instructions [here](https://github.com/scaleapi/llm-engine#-quick-start))
  ```python
     import os
     os.environ['SCALE_API_KEY'] = 'your_api_key'
  ```
- Fine tune Llama 2 with a _single API call_
   - Note that while LLM Engine supports training and fine-tuning with a training and validation dataset, here we are only providing a training dataset. In this case, LLM Engine will randomly split 10% of the data into a validation dataset to prevent overfitting.
     
  ```python
  train_url = "https://gist.githubusercontent.com/yebhub/0128b200be4711361179e16ae0e7dfc9/raw/762363cedce547bdb57ee4f3ab667dd525433267/data.csv"
  
  from llmengine import FineTune
  response = FineTune.create(
      model="llama-2-7b",
      training_file=train_url,
      hyperparameters={
          'lr':2e-4,
      },
      suffix='function-call-llama'
  )
  run_id = response.id
  ```

  ## V inference and evaluation

  - make sure model exists
    ```python
    ft_model = FineTune.get(run_id).fine_tuned_model
    ```
