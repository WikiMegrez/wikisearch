from typing import List, Optional
from dataclasses import dataclass
import torch


@dataclass
class Document:
    name: str
    title: str
    raw_main_desc: str
    main_image: str
    other_images: List[str]
    out_links: List[str]
    main_desc: Optional[torch.LongTensor] = None
    main_desc_words: Optional[List[str]] = None
