import os
import re

DOCS_FOLDER = os.path.dirname(__file__)
CONVERTED_NOTEBOOK_FOLDER = os.path.join(DOCS_FOLDER, "converted_notebooks")

IMAGE_FILE_REGEX = re.compile("\\[png\\]\\((.*)\\)")

PATH_PREFIX = os.path.join(os.pardir, "docs", "converted_notebooks")


def fix_paths_in_markdown(file_content: str) -> str:
    def replace(match: re.Match) -> str:
        path = match.groups()[0]
        fixed_path = os.path.join(PATH_PREFIX, path)
        return f"[png]({fixed_path})"

    return IMAGE_FILE_REGEX.sub(string=file_content, repl=replace)


if __name__ == "__main__":
    markdown_files = [
        file for file in os.listdir(CONVERTED_NOTEBOOK_FOLDER) if file.endswith(".md")
    ]
    for markdown_file in markdown_files:
        file_path = os.path.join(CONVERTED_NOTEBOOK_FOLDER, markdown_file)
        with open(file_path, "r") as f:
            content = f.read()
        fixed_content = fix_paths_in_markdown(content)
        with open(file_path, "w") as f:
            f.write(fixed_content)
        print(f"fixed paths in {file_path}")
