import io
import pandas as pd

def parse_file(filename: str, file_bytes: bytes) -> list[str]:
    ext = filename.lower().split(".")[-1]

    if ext == "txt":
        text = file_bytes.decode("utf-8", errors="ignore")
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 40]
        return chunks

    elif ext == "pdf":
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        chunks = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
                chunks.extend(paragraphs)
        return chunks

    elif ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        chunks = df.apply(lambda row: " | ".join(str(v) for v in row.values if pd.notna(v)), axis=1).tolist()
        return [c for c in chunks if len(c) > 40]

    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .pdf, .csv")