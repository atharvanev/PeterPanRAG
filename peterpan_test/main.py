import ollama
import time
import os
import json
import numpy as np
import re

#paring algo from CHAT GPT
def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        text = f.read()

    # Split the text by two or more newlines, which typically indicate a new paragraph
    paragraphs = re.split(r'\n\s*\n', text.strip())

    return paragraphs

# def parse_file(filename):
#     with open(filename, encoding = "utf-8-sig") as f:
#         paragraphs = []
#         buffer = []
#         for line in f.readlines():
#             line = line.strip() #get lines that are only newline characters and only spaces
#             if line:
#                 buffer.append(line) #if line exisits add it to the buffer
#             elif len(buffer): # the line is empty and the buffer has something then
#                 paragraphs.append(" ".join((buffer)))
#                 buffer = []
#         if len(buffer): #one last check if anything is left in buffer
#             paragraphs.append (" ".join(buffer) )
#         return paragraphs
    
def save_embeddings(filename,embeddings):
    if not os.path.exists("embeddings"): # makes an embeddings direcertoy if it doesnt exist
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json","w") as f: #"w" means write or overwrite
        json.dump(embeddings, f)

def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"): #check if the file exist
        return False
    with open(f"embeddings/{filename}.json","r") as f: #"r for read"
        return json.load(f)

def get_embeddings(filename,modelname, chunks):
    if(embeddings:= load_embeddings(filename)) is not False:
        return embeddings #if it is alr saved simply return it
    
    #otherwise create the embedding
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
          for chunk in chunks
          ]
    
    save_embeddings(filename,embeddings) #save it for later use
    return embeddings #return


def find_most_similar(needle, haystack):
    needle = np.array(needle)
    haystack = np.array(haystack)
    
    # Calculate the dot product of the needle and each item in the haystack
    similarity_scores = np.dot(haystack, needle) / (np.linalg.norm(haystack, axis=1) * np.linalg.norm(needle))


    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)




def main():
    SYSTEM_PROMPT = """ You area a helpful reading assistant who answers questions
    based on snippets of text provided in contex. Answer only using the context provided,
    being as concise as possible. If you're unsure, just say that you don't know
    Context: 
    
    """


    filename = "peterpan2.txt"
    print("hello")
    paragraphs = parse_file(filename)
    start = time.perf_counter()
    embeddings = get_embeddings(filename,'mistral',paragraphs)
    print(time.perf_counter()-start)

    print(len(embeddings))

    myprompt = input("what do you want to know?-->")#the prompt has to be an embedding to see which embedding from the paragaphs is closest in relation to our prompt
    prompt_embedding = ollama.embeddings(model='mistral',prompt=myprompt)["embedding"]
    #print(prompt_embedding)
    #finding most similar
    most_similar_chunks = find_most_similar(prompt_embedding,embeddings)[:5]
    
    #print(paragraphs[:10])

    # for item in most_similar_chunks:
    #     print(item[0],paragraphs[item[1]])

    response = ollama.chat(
        model="mistral",
        messages=[
            {
            "role":"system",
            "content":SYSTEM_PROMPT
            +"\n".join(paragraphs[item[1]] for item in most_similar_chunks),
        },
        {"role":"user","content": myprompt},
        ]
    )

    print("\n\n")
    print(response["message"]["content"])

if __name__ == "__main__":
    main()

