import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DB_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "inwi_collection")
FAQ_DIR = "data/faq"
OTHERS_DIR = "data/others"


def get_header_level(line):
    match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if match:
        return len(match.group(1)), match.group(2).strip()
    return None, None


# ════════════════════════════════════════
#  FAQ PARSER
# ════════════════════════════════════════

def parse_faq_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    chunks = []
    current_title = ""
    current_question = ""
    current_answer_lines = []

    def save_chunk():
        answer = "\n".join(current_answer_lines).strip()
        if current_question and answer:
            chunk_text = f"## {current_question}\n\n{answer}"
            if current_title:
                chunk_text = f"# {current_title}\n\n{chunk_text}"
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": os.path.basename(filepath),
                    "title": current_title,
                    "question": current_question,
                    "doc_type": "faq"
                }
            })

    for line in lines:
        level, text = get_header_level(line)
        if level == 1:
            save_chunk()
            current_title = text
            current_question = ""
            current_answer_lines = []
        elif level == 2:
            save_chunk()
            current_question = text
            current_answer_lines = []
        else:
            current_answer_lines.append(line)

    save_chunk()
    return chunks


def load_all_faq(faq_dir):
    all_chunks = []
    if not os.path.isdir(faq_dir):
        print(f"  ⚠️ Dossier FAQ introuvable: {faq_dir}")
        return all_chunks

    for file in sorted(os.listdir(faq_dir)):
        if not file.endswith(".md"):
            continue
        filepath = os.path.join(faq_dir, file)
        chunks = parse_faq_file(filepath)
        all_chunks.extend(chunks)
        print(f"  ✅ {file:<45} → {len(chunks):>3} chunks")

    return all_chunks


# ════════════════════════════════════════
#  GENERAL FILES PARSER (esim, conditions, brochure)
# ════════════════════════════════════════

def parse_general_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    chunks = []
    headers = {f"h{i}": "" for i in range(1, 7)}
    current_lines = []

    def save_chunk():
        text = "\n".join(current_lines).strip()
        if len(text) < 30:
            return
        parts = [headers[f"h{i}"] for i in range(1, 7) if headers[f"h{i}"]]
        chunks.append({
            "text": text,
            "metadata": {
                "source": os.path.basename(filepath),
                "h1": headers["h1"],
                "h2": headers["h2"],
                "h3": headers["h3"],
                "h4": headers["h4"],
                "h5": headers["h5"],
                "h6": headers["h6"],
                "breadcrumb": " > ".join(parts),
                "doc_type": "general"
            }
        })

    for line in lines:
        level, title = get_header_level(line)
        if level:
            save_chunk()
            headers[f"h{level}"] = title
            for i in range(level + 1, 7):
                headers[f"h{i}"] = ""
            current_lines = [line]
        else:
            current_lines.append(line)

    save_chunk()
    return chunks


def merge_small_chunks(chunks, min_size=50):
    if not chunks:
        return chunks
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        if len(prev["text"]) < min_size:
            merged[-1] = {
                "text": prev["text"] + "\n\n" + chunk["text"],
                "metadata": chunk["metadata"]
            }
        else:
            merged.append(chunk)
    return merged


def split_large_chunks(chunks, max_size=1500):
    result = []
    for chunk in chunks:
        text = chunk["text"]
        if len(text) <= max_size:
            result.append(chunk)
            continue
        paragraphs = re.split(r"\n\n+", text)
        current_part = []
        current_size = 0
        for para in paragraphs:
            if current_size + len(para) > max_size and current_part:
                result.append({
                    "text": "\n\n".join(current_part),
                    "metadata": dict(chunk["metadata"])
                })
                current_part = []
                current_size = 0
            current_part.append(para)
            current_size += len(para)
        if current_part:
            result.append({
                "text": "\n\n".join(current_part),
                "metadata": dict(chunk["metadata"])
            })
    return result


def load_all_general(others_dir):
    all_chunks = []
    files = ["esim.md", "conditions_generales.md", "Brochure_Tarifaire.md"]

    for file in files:
        filepath = os.path.join(others_dir, file)
        if not os.path.exists(filepath):
            print(f"  ⚠️ Fichier introuvable: {file}")
            continue
        raw = parse_general_file(filepath)
        raw = merge_small_chunks(raw)
        raw = split_large_chunks(raw)
        all_chunks.extend(raw)
        print(f"  ✅ {file:<45} → {len(raw):>3} chunks")

    return all_chunks


# ════════════════════════════════════════
#  EMBEDDING + INDEXATION
# ════════════════════════════════════════

def generate_embeddings(chunks, model):
    texts = [f"passage: {c['text']}" for c in chunks]

    print(f"\n🧠 Génération des embeddings ({len(texts)} chunks)...")
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True)
        all_embeddings.extend(embeddings.tolist())
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return all_embeddings


def store_in_chromadb(chunks, embeddings):
    client = chromadb.PersistentClient(path=DB_PATH)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"\n🗑️  Collection '{COLLECTION_NAME}' supprimée")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            ids=[c["id"] for c in chunks[i:end]],
            documents=[c["text"] for c in chunks[i:end]],
            metadatas=[c["metadata"] for c in chunks[i:end]],
            embeddings=embeddings[i:end]
        )
        print(f"  Stocké {end}/{len(chunks)}")

    return collection


# ════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 Chunking + Indexation inwi")
    print("=" * 60)

    print(f"\n📁 Chargement des FAQ ({FAQ_DIR}):")
    faq_chunks = load_all_faq(FAQ_DIR)

    print(f"\n📁 Chargement des fichiers généraux ({OTHERS_DIR}):")
    general_chunks = load_all_general(OTHERS_DIR)

    all_chunks = faq_chunks + general_chunks

    for i, chunk in enumerate(all_chunks):
        chunk["id"] = f"chunk_{i:04d}"

    print(f"\n{'='*60}")
    print(f"📊 RÉSUMÉ")
    print(f"{'='*60}")
    print(f"  FAQ:      {len(faq_chunks)} chunks")
    print(f"  Général:  {len(general_chunks)} chunks")
    print(f"  TOTAL:    {len(all_chunks)} chunks")

    source_counts = Counter(c["metadata"]["source"] for c in all_chunks)
    print(f"\n  Par fichier:")
    for source, count in sorted(source_counts.items()):
        print(f"    {source:<45} → {count:>3}")

    sizes = [len(c["text"]) for c in all_chunks]
    print(f"\n  Tailles: Min={min(sizes)} | Max={max(sizes)} | Avg={sum(sizes)//len(sizes)}")

    print(f"\n🧠 Chargement du modèle: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = generate_embeddings(all_chunks, model)

    print(f"\n💾 Stockage dans ChromaDB ({DB_PATH}):")
    collection = store_in_chromadb(all_chunks, embeddings)

    print(f"\n{'='*60}")
    print(f"✅ {collection.count()} chunks indexés dans '{COLLECTION_NAME}'")
    print(f"   Modèle: {EMBEDDING_MODEL}")
    print(f"   Préfixe documents: 'passage: ...'")
    print(f"   Préfixe requêtes:  'query: ...'  (dans pipeline.py)")
    print(f"   Distance: cosine")
    print(f"{'='*60}")

    print(f"\n🧪 Test rapide:")
    query = "Comment activer la 5G ?"
    query_emb = model.encode(f"query: {query}", normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=3, include=["documents", "distances"])

    for i in range(len(results["ids"][0])):
        sim = (1 - results["distances"][0][i]) * 100
        print(f"  #{i+1} | {sim:.1f}% | {results['documents'][0][i][:80]}...")