import unittest

from src.model.vectorstore import MyDatabase
from src.pipeline.text_loader import Loader


class MyTestCase(unittest.TestCase):
    def test_retriever(self):
        sample_text = [
            "Artificial Intelligence (AI) is a branch of computer science focused on building machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. AI technologies, such as machine learning, deep learning, and natural language processing, are revolutionizing industries by automating complex processes and enabling data-driven decision making.",

            "Machine learning, a subset of AI, involves training algorithms on large datasets to recognize patterns and make predictions. Supervised learning, unsupervised learning, and reinforcement learning are the primary types of machine learning. Deep learning, a more advanced form, utilizes neural networks with multiple layers to model complex relationships. These technologies power applications like image and speech recognition, recommendation systems, and autonomous vehicles.",

            "AI's impact on society is profound, with potential benefits and challenges. It promises to enhance productivity, improve healthcare through precision medicine, and create new job opportunities. However, concerns about job displacement, ethical considerations, and the need for transparent AI systems must be addressed. Responsible AI development, with a focus on fairness, accountability, and transparency, is crucial for maximizing benefits while minimizing risks."
        ]
        documents = Loader().load_text(sample_text)
        print(documents)
        mydb = MyDatabase(documents=documents)
        retriever = mydb.get_retriever()
        results = retriever.invoke('What is AI?')
        print(len(results))
        # pass with no error
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
