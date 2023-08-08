import requests
import time
import json

# Load data from external JSON file
with open("test_input_ex.json", "r") as json_file:
    data = json.load(json_file)

url = "https://api.runpod.ai/v2/vuusai5vaeia2x/runsync"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer T1530GW4R81PRD3MHSQGHJOI9D3GZG5Y4K7BTK0G"
}

response = requests.post(url, json=data, headers=headers)
response_data = response.json()
print("Response Data",response_data)

# if "run_id" in response_data:
#     run_id = response_data["run_id"]
#     print(f"Successfully started run with ID: {run_id}")
    
#     # Now, let's wait for the result
#     while True:
#         status_response = requests.get(f"{url}/{run_id}", headers=headers)
#         status_data = status_response.json()
        
#         if status_data.get("status") == "finished":
#             print("Run finished!")
#             # You can access the result in status_data["result"]
#             break
#         elif status_data.get("status") == "failed":
#             print("Run failed.")
#             break
        
#         time.sleep(5)  # Wait for 5 seconds before checking again

# else:
#     print("Failed to start the run.")
