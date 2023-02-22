import unittest
from data_preparation import TokenClassificationTagIndexer


class TestTokenClassificationTagIndexer(unittest.TestCase):
    def setUp(self) -> None:
        self.indexer_obj = TokenClassificationTagIndexer(
            vocabs=["[SEP]", "ehsan", "part", "hello", "[PAD]"],
            pad_index=0, cls_index=1, sep_index=2
        )

    def test__remove_duplicate_special_token(self):
        self.assertEqual(len(self.indexer_obj.vocabs), 3)
        self.assertEqual("[SEP]" not in self.indexer_obj.vocabs, True)
        self.assertEqual("[PAD]" not in self.indexer_obj.vocabs, True)


if __name__ == "__main__":
    unittest.main()
