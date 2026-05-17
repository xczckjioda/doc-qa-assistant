from pydantic import BaseModel
from typing import Optional


class SourceItem(BaseModel):
    file_name: Optional[str] = None
    page: Optional[int] = None