import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ollama


url =r"D:\AI\28th, 29th- EDA Automation Mistral, gradio\titanic_ dataset_final.csv"
df=pd.read_csv(url)


def genrate_insights(summary):
    prompt=f"Analyse the insights:\n\n{summary}"
    response=ollama.chat(model="mistral",messages=[{"role":"user","content":prompt}])
    return response['message']['content']
summary1=df.describe().to_string
insights=genrate_insights(summary1)
print("\n AI GENERATED:\n",insights)