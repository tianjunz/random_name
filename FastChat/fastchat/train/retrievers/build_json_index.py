import json
import re
from typing import Any, Generator, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document

class JSONLReader(BaseReader):
    def __init__(
        self, levels_back: Optional[int] = None, collapse_length: Optional[int] = None
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.levels_back = levels_back
        self.collapse_length = collapse_length

    def load_data(self, input_file: str) -> List[Document]:
        """Load data from the input file."""
        data = []
        with open(input_file, "r") as f:
            for line in f:
                #TODO: need to decide whether to use entire line
                data.append(str(json.loads(line)))
        if self.levels_back is None:
            # If levels_back isn't set, we just format and make each
            # line an embedding
            # json_output = json.dumps(data, indent=0)
            # lines = json_output.split("\n")
            # useful_lines = [
            #     line for line in lines if not re.match(r"^[{}\[\],]*$", line)
            # ]
            # return [Document("\n".join(useful_lines))]
            return [Document(_data) for _data in data]
        elif self.levels_back is not None:
            # If levels_back is set, we make the embeddings contain the labels
            # from further up the JSON tree
            lines = [
                *_depth_first_yield(
                    data, self.levels_back, self.collapse_length, []
                )
            ]
            return [Document("\n".join(lines))]
