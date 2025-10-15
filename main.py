from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retriever
import re

model = OllamaLLM(model="codeqwen")

# Prompt updated to request bracketed citations and a References section.
prompt = ChatPromptTemplate.from_template(
    "You are an expert on the Bible. Use the Context to answer the Question.\n\n"
    "The Context contains numbered snippets like [1], [2], ... When you use information from the Context,"
    " cite the snippet number inline (for example: [1]). At the end of your answer include a 'References' section"
    " that lists each referenced number and the corresponding source (as given in the reference mapping).\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n"
)
chain = prompt | model

retriever = get_retriever()  # builds index on first run if needed

def docs_to_context(docs, max_chars=1500):
    """
    Turn retrieved docs into a numbered context string the model can cite, and return a references list.

    Returns:
      context_str: the combined text with numbered snippet prefixes like "[1] snippet..."
      references: list of dicts [{ "idx": 1, "desc": "filename (chapter:verse)" }, ...]
    """
    parts = []
    total = 0
    references = []

    for idx, d in enumerate(docs, start=1):
        text = getattr(d, "page_content", "") or getattr(d, "content", "")
        snippet = text[:1000].replace("\n", " ").strip()
        # Build a readable source description from metadata
        meta = getattr(d, "metadata", {}) or {}
        source = meta.get("source") or meta.get("Book") or meta.get("book") or "unknown"
        chapter = meta.get("chapter")
        verse = meta.get("verse")
        ref_desc = source
        if chapter is not None or verse is not None:
            ch = str(chapter) if chapter is not None else ""
            v = str(verse) if verse is not None else ""
            if ch and v:
                ref_desc += f" ({ch}:{v})"
            elif ch:
                ref_desc += f" (chapter {ch})"
            elif v:
                ref_desc += f" (verse {v})"

        references.append({"idx": idx, "desc": ref_desc})
        parts.append(f"[{idx}] {snippet}")
        total += len(snippet)
        if total >= max_chars:
            break

    context = "\n\n".join(parts)
    return context, references

if __name__ == "__main__":
    while True:
        q = input("Ask a question (q to quit): ").strip()
        if q.lower() == "q":
            break

        try:
            docs = retriever.get_relevant_documents(q)
        except AttributeError:
            docs = retriever.retrieve(q)

        context, references = docs_to_context(docs)
        # run the chain; pass context and question
        out = chain.invoke({"context": context, "question": q})
        # ensure string
        out_text = str(out)

        print("\n=== Answer ===\n")
        print(out_text)
        print("\n=== References used ===\n")

        # Try to detect which bracketed citation numbers the model actually used in its answer.
        cited_numbers = set(int(n) for n in re.findall(r"\[(\d+)\]", out_text))
        if cited_numbers:
            # print only the ones that were cited
            for ref in references:
                if ref["idx"] in cited_numbers:
                    print(f"[{ref['idx']}] {ref['desc']}")
        else:
            # fallback: print all references available
            print("No explicit bracketed citations detected in model output. Showing all candidate references:")
            for ref in references:
                print(f"[{ref['idx']}] {ref['desc']}")

        print("\n")
