import unittest
from src.pipeline.text_loader import load_pdf
from src.pipeline.text_loader import Loader


def test_class_Loader():
    loader = Loader()
    docs = loader.load_dir('tests')
    assert len(docs) > 0
    print(docs)


class MyTestCase(unittest.TestCase):
    def test_load_pdf(self):
        res = load_pdf('test.pdf')
        # test failed if res is none
        self.assertIsNotNone(res)
        print(res)


if __name__ == '__main__':
    # unittest.main()
    test_class_Loader ()
