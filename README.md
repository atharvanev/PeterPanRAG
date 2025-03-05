# Text Assistant with Embedding Search (Retrival Augmented Generation Pipeline)

This project allows you to query specific pieces of text from a file by using embeddings and cosine similarity. It processes text, generates embeddings, and finds the most relevant paragraphs based on a user prompt.

## Setup

### Install Required Libraries

First, install the required libraries:

```bash
pip install ollama numpy
```
Make sure to install the weight for **mistral** using ollama also.

### Embedding Folder
When you run the script, an `embeddings/` folder will be created automatically if it doesn't exist. This folder stores the embeddings generated from the input text file as `.json` files. These embeddings are saved to disk so that they can be reused in subsequent runs, avoiding the need to regenerate them.

If the embeddings file already exists for the text file, the script will load the existing embeddings instead of regenerating them. This improves performance for subsequent runs, as it avoids unnecessary calculations.

### How to Use
1. Place your text file (e.g., `your-text-file.txt`) in the same directory as the script. (I have used `peterpan2.txt`)
2. Modify the `filename` variable in the script to point to your text file.
3. Run the script, and it will process the text, generate embeddings, and save them in the `embeddings/` folder.
4. Enter your prompt when asked, and the script will return the most relevant sections from the text.

### Note
- The embeddings are stored in JSON format in the `embeddings/` folder.

### Example Output

![Screenshot 2025-03-04 at 7 56 02 PM](https://github.com/user-attachments/assets/80aeb201-ddda-46bc-af6c-301373cba25f)
