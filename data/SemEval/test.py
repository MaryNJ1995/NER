def read_text(path: str) -> list:
    """

    :param path:
    :return:
    """
    with open(path, "r") as file:
        data = file.readlines()
    return data


RAW_VAL_DATA = read_text("fa_dev.conll")[:10]
print(RAW_VAL_DATA)
for item in RAW_VAL_DATA:
    if not item.startswith("#"):
        print(item)
        best = item.split("_ _")
        print(best[1])
