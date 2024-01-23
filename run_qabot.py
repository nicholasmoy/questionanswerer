import argparse
import os

from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import (
    RetrieverQueryEngine,
    CitationQueryEngine,
)
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.indices.postprocessor import SimilarityPostprocessor

from llama_index import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage, 
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.schema import NodeWithScore


parser = argparse.ArgumentParser(description="Run QABot")

parser.add_argument("--corpus_dir", type=str)
parser.add_argument("--force_reindex", type=bool)
parser.add_argument("--chat_model", type=str, default="gpt-3.5-turbo")
parser.add_argument("--embed_model", type=str, default="text-embedding-ada-002")

def node_to_string(node_with_score: NodeWithScore):
    formatted_string = f"{node_with_score.metadata['file_path']}\n:{node_with_score.get_text()[:100]}...\n\n"
    return formatted_string


def main():
    print("Hello World")
    print(f"{args.corpus_dir}")
    persist_dir = os.path.join(args.corpus_dir, "storage")

    
    documents = SimpleDirectoryReader(args.corpus_dir).load_data()

    # No existing persisted dir, so we need to create a new index
    if not os.path.exists(persist_dir) or args.force_reindex:
        embedding_service_context = ServiceContext.from_defaults()
        index = VectorStoreIndex.from_documents(documents, show_progress=True, service_context=embedding_service_context)
        index.storage_context.persist(persist_dir = persist_dir)
    else: 
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )
    # configure response synthesizer
    synthesizer_service_context = ServiceContext.from_defaults(
        system_prompt = None,
    )
    response_synthesizer = get_response_synthesizer (
        service_context=synthesizer_service_context,
        response_mode = ResponseMode.SIMPLE_SUMMARIZE,
    )

    # assemble query engine
    query_engine = CitationQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ]
    )

    user_input = ""
    while user_input != "q":
        user_input = input("Submit your query (q to quit):")
        if user_input == "q":
            break
        response = query_engine.query(user_input)
        print(f"Bot:\n{response}")
        print(f"Used \n{len(response.source_nodes)} sources:")
        user_input = input("i to inspect sources, other key to submit a new query (q to quit):")
        if user_input == "i":
            counter = 0
            user_input = "c"
            while counter * 5 <= len(response.source_nodes) and user_input == "c": 
                for node_with_score in response.source_nodes[counter * 5: (counter + 1) *5]:
                    print(node_to_string(node_with_score))
                counter += 1
                user_input = input("c to continue, other key to submit a new query (q to quit):")

    
    

if __name__ == "__main__":
    args = parser.parse_args()
    main()


