import google.generativeai as genai

genai.configure(api_key="AIzaSyAQ4cubW5eUt6vu4t_Gc-8EHWharEedJWA")

models = genai.list_models()

for model in models:
    print(model.name)