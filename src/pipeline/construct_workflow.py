from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document


class State(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]


class WorkFlow:
    def __init__(self, retriever, generate_model, web_search_tool,
                 retrieval_grader, hallucination_grader,
                 answer_grader):

        # Assert if any of the input is none
        assert retriever is not None, "Retriever cannot be None"
        assert generate_model is not None, "Generating model cannot be None"
        assert web_search_tool is not None, "Web search tool cannot be None"
        assert retrieval_grader is not None, "Retrieval grader cannot be None"
        assert hallucination_grader is not None, "Hallucination grader cannot be None"
        assert answer_grader is not None, "Answer grader cannot be None"

        self.retriever = retriever
        self.generate_model = generate_model
        self.web_search_tool = web_search_tool
        self.retrieval_grader = retrieval_grader
        self.hallucination_grader = hallucination_grader
        self.answer_grader = answer_grader

        # Define the workflow
        self.workflow = StateGraph(State)

        # Define all nodes and edges - LangGraph
        self.workflow.add_node("websearch", self.web_search)
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("generate", self.generate)

        self.workflow.set_entry_point('retrieve')
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("websearch", "generate")
        self.workflow.add_conditional_edges(
            "generate",
            self.grade_hallucination,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
            },
        )

        self.app = self.workflow.compile()

    def retrieve(self, state: State):
        question_ = state["question"]
        documents = self.retriever.invoke(question_)
        state["documents"] = documents
        return state

    def generate(self, state: State):
        question_ = state["question"]
        documents = state["documents"]

        generation = self.generate_model.invoke({"context": documents, "question": question_})
        return {"documents": documents, "question": question_, "generation": generation}

    def grade_documents(self, state: State):
        question_ = state["question"]
        documents = state["documents"]

        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question_, "document": d.page_content})
            grade = score["score"]
            if grade.lower() == "yes":
                filtered_docs.append(d)
            else:
                web_search = "Yes"
        state["documents"] = filtered_docs
        state["web_search"] = web_search
        return state

    def web_search(self, state: State):
        question_ = state["question"]
        documents = state["documents"]

        docs = self.web_search_tool.invoke({"query": question_})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        state["documents"] = documents
        return state

    def grade_hallucination(self, state: State):
        question_ = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score["score"]

        if grade == "yes":
            score = self.answer_grader.invoke({"question": question_, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                return "useful"
            else:
                return "not useful"
        else:
            return "not supported"

    def decide_to_generate(self, state: State):
        web_search = state["web_search"]
        if web_search == "Yes":
            return "websearch"
        else:
            return "generate"

    def invoke(self, question: str):
        return self.app.invoke({"question" : question})
