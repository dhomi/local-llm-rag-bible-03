# Local Bible AI Agent With RAG

## install required packages 
activate your python venv

then install requirements ```pip install -r requirements.txt```

## Run the code
python main.py

The first time you run it, it will build the vector database from the provided file. This may take a while.

Then you can ask questions about the Bible and it will use the context from the vector database to answer them.

Debugging tips:
the first time you run it, it will 
"chromadb.errors.InternalError: ValueError: Batch size of 31102 is greater than max batch size of 5461"
just rerun it and it should work.