from app.agent import build_graph


def main():
    app = build_graph()
    print("University Course Selection Assistant")
    print("Type 'exit' to quit.\n")
    while True:
        question = input("Student: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        result = app.invoke({"question": question})
        print("\nAssistant:\n" + result["final_answer"] + "\n")


if __name__ == "__main__":
    main()
