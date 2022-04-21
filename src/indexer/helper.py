import json


def save_json(data, path: str) -> None:
    """

    :param data:
    :param path:
    :return:
    """
    with open(path, "w") as file:
        json.dump(data, file)


def load_json(path: str):
    """

    :param path:
    :return:
    """
    with open(path) as file:
        data = json.load(file)
    return data


def save_text(data: list, path: str) -> None:
    """

    :param data:
    :param path:
    :return:
    """
    with open(path, "w") as file:
        file.write("\n".join(data))


def load_text(path: str) -> list:
    """

    :param path:
    :return:
    """
    with open(path, "r") as file:
        data = file.readlines()
    return data
