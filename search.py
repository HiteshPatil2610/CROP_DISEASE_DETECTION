import urllib.request, json
try:
    r = urllib.request.urlopen("https://huggingface.co/api/datasets?search=plant%20disease&sort=downloads")
    data = json.loads(r.read())
    print("Top Plant Disease Datasets:")
    for d in data[:10]:
        print("-", d['id'])
except Exception as e:
    print("Error:", e)
