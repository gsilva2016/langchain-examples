import requests
from concurrent.futures import ThreadPoolExecutor

MILVUS_SIM_SEARCH_ENDPOINT="http://127.0.0.1:8000/search"

def search_in_milvus(query_txt):
    with ThreadPoolExecutor() as pool:
        try:
            response = requests.get(url=MILVUS_SIM_SEARCH_ENDPOINT, params={"query": query_txt})
            if response.status_code != 200:
                err_msg = {"error": f"Error in search_in_milvus function {response.status_code}: {response.content.decode('utf-8')}"}
                print("Error message inside search func:", err_msg)
                return err_msg

            return {"result": response.content.decode("utf-8")}

        except requests.exceptions.RequestException as e:
            print(f"Search in Milvus: Request failed: {e}")
            return {"error": str(e)}
