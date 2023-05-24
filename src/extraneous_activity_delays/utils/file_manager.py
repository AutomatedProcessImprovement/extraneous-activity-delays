import datetime
import os
import shutil
import uuid
from pathlib import Path


def delete_folder(folder_path: str):
    shutil.rmtree(folder_path, ignore_errors=True)


def create_new_tmp_folder(base_path: Path) -> Path:
    # Get non existent folder name
    output_folder = base_path.joinpath(
        datetime.datetime.today().strftime("%Y%m%d_") + str(uuid.uuid4()).upper().replace("-", "_")
    )
    while not create_folder(output_folder):
        output_folder = base_path.joinpath(
            datetime.datetime.today().strftime("%Y%m%d_") + str(uuid.uuid4()).upper().replace("-", "_")
        )
    # Return P  ath to new folder
    return output_folder


def create_folder(path: Path) -> bool:
    if os.path.exists(path):
        return False
    else:
        os.makedirs(path)
        return True
