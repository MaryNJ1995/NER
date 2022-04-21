import unittest

from utils import find_max_length_in_list, convert_index_to_tag, ignore_pad_index


class TestHelper(unittest.TestCase):
    def setUp(self) -> None:
        return None

    def test_find_max_length_in_list(self):
        data = [["item_1", "item_2", "item_3", "item_4"], ["item_1", "item_2"]]
        expected_value = 4
        self.assertEqual(find_max_length_in_list(data), expected_value)

    def test_ignore_pad_index(self):
        true_labels = [["item_1", "item_2", "pad", "pad"], ["item_3", "pad", "pad", "pad"]]
        pred_labels = [["item_1", "item_2", "item_3", "pad"], ["item_1", "pad", "item_2", "item_3"]]
        pad_token = "pad"
        expected_output_1 = [["item_1", "item_2"], ["item_3"]]
        expected_output_2 = [["item_1", "item_2"], ["item_1"]]
        self.assertEqual(ignore_pad_index(true_labels, pred_labels, pad_token),
                         (expected_output_1, expected_output_2))

    def test_convert_index_to_tag(self):
        data = [["item_1", "item_2", "item_3"], ["item_1", "item_2"]]
        idx2tag = {"pad_item": 0, "item_1": 1, "item_2": 2, "item_3": 3}
        expected_value = [[1, 2, 3], [1, 2]]
        self.assertEqual(convert_index_to_tag(data, idx2tag), expected_value)


if __name__ == "__main__":
    unittest.main()
