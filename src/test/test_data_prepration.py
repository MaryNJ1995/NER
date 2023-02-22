import unittest
from data_preparation import add_special_tokens, pad_sequence,\
    truncate_sequence, create_attention_masks


class TestHelper(unittest.TestCase):
    def setUp(self) -> None:
        return None

    def test_add_special_tokens(self):
        tokens_1 = ["hi", "i", "am", "Ehsan"]
        labels_1 = ["O", "O", "O", "B-PER"]
        expected_output_1 = ([["[CLS]", "hi", "i", "am", "Ehsan", "[SEP]"]],
                             [["[CLS]", "O", "O", "O", "B-PER", "[SEP]"]])
        self.assertEqual(add_special_tokens([tokens_1], [labels_1], cls_token="[CLS]", sep_token="[SEP]"),
                         expected_output_1)

    def test_pad_sequence(self):
        data = [["[CLS]", "item_1", "item_2", "item_3", "[SEP]"], ["[CLS]", "item_1", "item_2", "[SEP]"]]
        max_length_1 = 4
        max_length_2 = 5
        max_length_3 = 6
        expected_output_1 = [["[CLS]", "item_1", "item_2", "item_3", "[SEP]"],
                             ["[CLS]", "item_1", "item_2", "[SEP]"]]
        expected_output_2 = [["[CLS]", "item_1", "item_2", "item_3", "[SEP]"],
                             ["[CLS]", "item_1", "item_2", "[SEP]", "[PAD]"]]
        expected_output_3 = [["[CLS]", "item_1", "item_2", "item_3", "[SEP]", "[PAD]"],
                             ["[CLS]", "item_1", "item_2", "[SEP]", "[PAD]", "[PAD]"]]

        self.assertEqual(pad_sequence(data, max_length_1), expected_output_1)
        self.assertEqual(pad_sequence(data, max_length_2), expected_output_2)
        self.assertEqual(pad_sequence(data, max_length_3), expected_output_3)

    def test_truncate_sequence(self):
        data = [["[CLS]", "item_1", "item_2", "item_3", "[SEP]"], ["[CLS]", "item_1", "item_2", "[SEP]"]]
        max_length_1 = 5
        max_length_2 = 4

        expected_output_1 = [["[CLS]", "item_1", "item_2", "item_3", "[SEP]"],
                             ["[CLS]", "item_1", "item_2", "[SEP]"]]
        expected_output_2 = [["[CLS]", "item_1", "item_2", "[SEP]"],
                             ["[CLS]", "item_1", "item_2", "[SEP]"]]

        self.assertEqual(truncate_sequence(data, max_length_1), expected_output_1)
        self.assertEqual(truncate_sequence(data, max_length_2), expected_output_2)

    def test_create_attention_masks(self):
        data = [["item_1", "item_2", "item_3", "pad_item"], ["item_1", "pad_item", "pad_item", "pad_item"]]
        expected_output = [[1, 1, 1, 0], [1, 0, 0, 0]]
        self.assertEqual(create_attention_masks(data, "pad_item"), expected_output)


if __name__ == "__main__":
    unittest.main()
