import unittest
from src.pipeline.text_loader import load_pdf


class MyTestCase(unittest.TestCase):
    def test_load_pdf(self):
        res = load_pdf('test.pdf')
        # test failed if res is none
        self.assertIsNotNone(res)
        print(res)


if __name__ == '__main__':
    unittest.main()
