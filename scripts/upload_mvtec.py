"""Upload an MVTec AD category to a new Viam dataset, tagged normal/anomalous.

Usage:
  VIAM_API_KEY=... VIAM_API_KEY_ID=... .venv/bin/python scripts/upload_mvtec.py <category_dir> <part_id> <org_id> <dataset_name>
e.g. category_dir = .spike/MVTecAD/bottle (from the HF mirror foersben/mvtec-ad; anomalib's own MVTec URL returns 404)
"""

import asyncio
import os
import sys
from pathlib import Path

from viam.app.viam_client import ViamClient
from viam.rpc.dial import DialOptions


async def main(category_dir: str, part_id: str, org_id: str, dataset_name: str) -> None:
    client = await ViamClient.create_from_dial_options(
        DialOptions.with_api_key(os.environ["VIAM_API_KEY"], os.environ["VIAM_API_KEY_ID"])
    )
    data = client.data_client
    dataset_id = await data.create_dataset(name=dataset_name, organization_id=org_id)
    root = Path(category_dir)
    ids = []
    for tag, sub in (("normal", "train/good"), ("anomalous", "test/broken_large")):
        for p in sorted((root / sub).glob("*.png")):
            ids.append(await data.file_upload_from_path(part_id=part_id, filepath=str(p), tags=[tag]))
    await data.add_binary_data_to_dataset_by_ids(binary_ids=ids, dataset_id=dataset_id)
    print(f"dataset '{dataset_name}' ({dataset_id}): {len(ids)} images")
    client.close()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    asyncio.run(main(*sys.argv[1:5]))
