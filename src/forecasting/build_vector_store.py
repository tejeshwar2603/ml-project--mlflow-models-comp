from .rag import build_vector_store_from_environment


def main() -> None:
    path = build_vector_store_from_environment()
    print(f"Vector store written to {path}")


if __name__ == "__main__":
    main()

