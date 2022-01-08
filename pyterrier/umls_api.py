import requests
import json

from lxml.html import fromstring


class Authentication:
    def __init__(self, api_key):
        self.api_key = api_key
        self.uri = "https://utslogin.nlm.nih.gov"
        self.auth_endpoint = "/cas/v1/api-key"
        self.service = "http://umlsks.nlm.nih.gov"

    def get_tgt(self):
        params = {"apikey": self.api_key}
        h = {
            "Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain",
            "User-Agent": "python",
        }
        r = requests.post(f"{self.uri}{self.auth_endpoint}", data=params, headers=h)
        response = fromstring(r.text)
        tgt = response.xpath("//form/@action")[0]
        return tgt

    def get_st(self, tgt):
        params = {"service": self.service}
        h = {
            "Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain",
            "User-Agent": "python",
        }
        r = requests.post(tgt, data=params, headers=h)
        st = r.text
        return st


class Umls:
    def __init__(self, api_key) -> None:
        """
        Params
        ------
        api_key: str

        Follow https://documentation.uts.nlm.nih.gov/rest/authentication.html?fbclid=IwAR3EbAatpIHIuX2OCqIw9CPmhLGD878mPIUcyn93d9JUA9Oy-N36pYDhcBA
        Step 1 to get the api_key of your UMLS account.
        """
        self.auth_client = Authentication(api_key)
        self.tgt = self.auth_client.get_tgt()
        self.uri = "https://uts-ws.nlm.nih.gov"
        self.version = "current"

    def _get(self, content_endpoint: str):
        r = requests.get(
            f"{self.uri}{content_endpoint}",
            params={"ticket": self.auth_client.get_st(self.tgt)},
        )
        r.encoding = "utf-8"
        return json.loads(r.text)["result"]

    def retrieve_entity(self, cui: str):
        """
        Sample output:
        {
            "classType": "Concept",
            "ui": "C0009044",
            "suppressible": false,
            "dateAdded": "09-30-1990",
            "majorRevisionDate": "03-16-2016",
            "status": "R",
            "semanticTypes": [
                {
                    "name": "Injury or Poisoning",
                    "uri": "https://uts-ws.nlm.nih.gov/rest/semantic-network/2021AB/TUI/T037"
                }
            ],
            "atomCount": 71,
            "attributeCount": 0,
            "cvMemberCount": 0,
            "atoms": "https://uts-ws.nlm.nih.gov/rest/content/2021AB/CUI/C0009044/atoms",
            "definitions": "https://uts-ws.nlm.nih.gov/rest/content/2021AB/CUI/C0009044/definitions",
            "relations": "https://uts-ws.nlm.nih.gov/rest/content/2021AB/CUI/C0009044/relations",
            "defaultPreferredAtom": "https://uts-ws.nlm.nih.gov/rest/content/2021AB/CUI/C0009044/atoms/preferred",
            "relationCount": 5,
            "name": "Closed fracture of carpal bone"
        }
        """
        content_endpoint = f"/rest/content/{self.version}/CUI/{cui}"
        return self._get(content_endpoint)

    def retrieve_relations(self, cui: str):
        content_endpoint = f"/rest/content/{self.version}/CUI/{cui}/relations"
        return self._get(content_endpoint)
