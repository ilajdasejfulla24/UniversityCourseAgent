from app.rag import build_vectorstore

if __name__ == "__main__":
    vs = build_vectorstore()
    print("Vector store built successfully.")
