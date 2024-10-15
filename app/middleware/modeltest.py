from tavily import TavilyClient
import json

tavily = TavilyClient(api_key="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")

def model_test(prompt):

    response = tavily.search(query=prompt)
    response1 = json.dumps(response)
    response2 = json.loads(response1)
    print(response2['results'])
    return response2['results'][0]['title'] + "\ncontent " + response2['results'][0]['content'] + "\nurl: " + response2['results'][0]['url']

