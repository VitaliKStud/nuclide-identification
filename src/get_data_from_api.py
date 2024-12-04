import requests


def get_nuclide_data(nuclid_id: str = None):
    if nuclid_id is not None:
        url = f"https://www-nds.iaea.org/relnsd/v1/data?fields=decay_rads&nuclides={nuclid_id}&rad_types=g"
        with requests.get(url, stream=True) as r:
            with open(f"data\\nuclides\\{nuclid_id}.csv", 'wb') as f:
                for chunk in r.iter_content():
                    f.write(chunk)
        print(f"{nuclid_id} saves")



