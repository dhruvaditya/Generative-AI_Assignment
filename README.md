# Generative-AI_Assignment
Main.py is the main file where the actual assignment is located. 
also added streamlit.py which i have created by the help of youtube video : https://youtu.be/uus5eLz6smA?si=-YXebaMESxU-0sX2
The core concept of this project was build a question answer application with langchain and using paLM API.
so firstly the pdf is uploaded to the local store by fast api method then it is processed whether it is in pdf or not if it is in pdf then 
send it for next processing like extracting texts from multiple pages in the pdf and after extracting text from pdf
it is passed in embedding function by using api of google and after embedding we store the embedded text in chroma db.

Now User can ask question from the pdf . once user added any question by post method query is again converted in embedded form and
processed for similarity search and the one which is more similar will be returned as an output. 
This is the main idea behind this project.
Here is the System architecture:

![architecture](https://github.com/dhruvaditya/Generative-AI_Assignment/assets/89244720/edbe369f-6bad-4c04-b3fc-ef070767d0b9)
